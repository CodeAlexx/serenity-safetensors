//! Uncommitted mmap for safetensors weight loading.
//!
//! Maps safetensors files into virtual address space WITHOUT committing
//! physical RAM (MAP_NORESERVE). The OS page cache manages residency —
//! pages are brought in on access and can be reclaimed freely under memory
//! pressure because they're clean, file-backed pages.
//!
//! This means:
//! - RAM usage appears high when plenty is available (pages cached for speed)
//! - OS can reclaim instantly without pagefile involvement
//! - Re-accessing reclaimed pages transparently re-reads from disk
//!
//! Linux only. Compile-errors on other platforms.

#[cfg(not(target_os = "linux"))]
compile_error!("mmap module requires Linux (MAP_NORESERVE, madvise)");

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::io::AsRawFd;

use pyo3::exceptions::{PyIOError, PyKeyError};
use pyo3::prelude::*;

/// A memory-mapped region of a safetensors file.
///
/// The mapped memory is uncommitted (MAP_NORESERVE) — the OS manages
/// physical RAM as a page cache. Dropping this struct unmaps the region.
pub struct MmapRegion {
    /// Pointer to the requested region start (may not be page-aligned)
    ptr: *mut u8,
    /// Length of the requested region
    len: usize,
    /// Actual mmap base pointer (page-aligned)
    mmap_base: *mut u8,
    /// Actual mmap length (page-aligned)
    mmap_len: usize,
}

// SAFETY: MmapRegion is a read-only view into a file.
// Multiple threads can read concurrently (file-backed, PROT_READ only).
unsafe impl Send for MmapRegion {}
unsafe impl Sync for MmapRegion {}

impl MmapRegion {
    /// Map a region of a file into uncommitted memory.
    ///
    /// SIGBUS protection: verifies file size before mapping.
    /// NOTE: TOCTOU — if the file is truncated after this check but before
    /// page access, SIGBUS will crash the process. This is inherent to all
    /// mmap-based approaches and is not practically exploitable for read-only
    /// model weight files. No runtime mitigation attempted.
    pub fn new(file: &File, offset: usize, len: usize) -> Result<Self, MmapError> {
        if len == 0 {
            return Err(MmapError::ZeroLength);
        }

        // Verify file is large enough (SIGBUS prevention)
        let file_len = file.metadata()
            .map_err(MmapError::Io)?
            .len() as usize;
        if offset + len > file_len {
            return Err(MmapError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "File is {file_len} bytes but mapping requires {} bytes (offset={offset})",
                    offset + len,
                ),
            )));
        }

        // SAFETY: sysconf(_SC_PAGESIZE) is always valid on Linux.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };

        let page_offset = offset % page_size;
        let aligned_offset = offset - page_offset;
        let aligned_len = len + page_offset;

        let fd = file.as_raw_fd();

        // SAFETY: mmap with PROT_READ | MAP_PRIVATE | MAP_NORESERVE.
        // fd is a valid file descriptor (from File::open).
        // MAP_PRIVATE: COW (we never write, so effectively read-only).
        // MAP_NORESERVE: don't reserve swap space — uncommitted.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                aligned_len,
                libc::PROT_READ,
                libc::MAP_PRIVATE | libc::MAP_NORESERVE,
                fd,
                aligned_offset as libc::off_t,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(MmapError::Io(std::io::Error::last_os_error()));
        }

        Ok(MmapRegion {
            ptr: unsafe { (ptr as *mut u8).add(page_offset) },
            len,
            mmap_base: ptr as *mut u8,
            mmap_len: aligned_len,
        })
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Advise the OS to prefetch pages into the page cache (MADV_WILLNEED).
    pub fn prefetch_range(&self, region_offset: usize, region_len: usize) {
        if region_offset + region_len > self.len {
            return;
        }
        // SAFETY: sysconf is always valid on Linux.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        let abs_ptr = unsafe { self.ptr.add(region_offset) };
        let aligned_ptr = (abs_ptr as usize & !(page_size - 1)) as *mut libc::c_void;
        let aligned_len = region_len + (abs_ptr as usize - aligned_ptr as usize);
        // SAFETY: aligned_ptr/aligned_len are within the mmap'd region.
        unsafe {
            libc::madvise(aligned_ptr, aligned_len, libc::MADV_WILLNEED);
        }
    }

    /// Advise the OS that pages can be reclaimed (MADV_DONTNEED).
    /// Data is NOT lost — re-access re-reads from disk.
    pub fn release_to_os(&self) {
        // SAFETY: mmap_base/mmap_len are the exact range from mmap().
        unsafe {
            libc::madvise(
                self.mmap_base as *mut libc::c_void,
                self.mmap_len,
                libc::MADV_DONTNEED,
            );
        }
    }
}

impl Drop for MmapRegion {
    fn drop(&mut self) {
        // SAFETY: mmap_base/mmap_len are from a successful mmap() call.
        unsafe {
            libc::munmap(self.mmap_base as *mut libc::c_void, self.mmap_len);
        }
    }
}

/// Tensor reference within an mmap'd safetensors data segment.
pub struct TensorRef {
    pub offset: usize,
    pub size: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
}

/// Mmap'd safetensors file with tensor index.
pub struct MmapFile {
    pub region: MmapRegion,
    pub tensors: HashMap<String, TensorRef>,
    _file: File,
}

impl MmapFile {
    pub fn open(path: &str) -> Result<Self, MmapError> {
        let mut file = File::open(path).map_err(MmapError::Io)?;

        // Read header length (first 8 bytes)
        let mut header_len_buf = [0u8; 8];
        file.read_exact(&mut header_len_buf).map_err(MmapError::Io)?;
        let header_len = u64::from_le_bytes(header_len_buf) as usize;

        if header_len > 100 * 1024 * 1024 {
            return Err(MmapError::Other("Header too large (>100MB)".into()));
        }

        // Read header JSON
        let mut header_buf = vec![0u8; header_len];
        file.read_exact(&mut header_buf).map_err(MmapError::Io)?;
        let header: serde_json::Value =
            serde_json::from_slice(&header_buf).map_err(MmapError::Json)?;

        let data_offset = 8 + header_len;

        // File size minus header = data segment
        let file_len = file.metadata().map_err(MmapError::Io)?.len() as usize;
        let data_len = file_len.saturating_sub(data_offset);

        if data_len == 0 {
            return Err(MmapError::Other("Empty data segment".into()));
        }

        // Map the data segment with MAP_NORESERVE
        let region = MmapRegion::new(&file, data_offset, data_len)?;

        // Build tensor index
        let mut tensors = HashMap::new();
        if let Some(obj) = header.as_object() {
            for (name, info) in obj {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = info.get("dtype")
                    .and_then(|d| d.as_str())
                    .unwrap_or("F32")
                    .to_string();
                let shape: Vec<usize> = info.get("shape")
                    .and_then(|s| s.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
                    .unwrap_or_default();
                let (start, end) = info.get("data_offsets")
                    .and_then(|o| o.as_array())
                    .map(|arr| {
                        let vals: Vec<usize> = arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as usize))
                            .collect();
                        (vals.first().copied().unwrap_or(0), vals.get(1).copied().unwrap_or(0))
                    })
                    .unwrap_or((0, 0));

                tensors.insert(name.clone(), TensorRef {
                    offset: start,
                    size: end.saturating_sub(start),
                    dtype,
                    shape,
                });
            }
        }

        Ok(MmapFile { region, tensors, _file: file })
    }

    pub fn tensor_ptr(&self, name: &str) -> Option<*const u8> {
        self.tensors.get(name).map(|t| {
            // SAFETY: t.offset is within the mmap'd data segment.
            unsafe { self.region.as_ptr().add(t.offset) }
        })
    }

    pub fn prefetch_tensor(&self, name: &str) {
        if let Some(t) = self.tensors.get(name) {
            self.region.prefetch_range(t.offset, t.size);
        }
    }
}

// ── PyO3 bindings ───────────────────────────────────────────────────────────

/// Python-visible uncommitted mmap handle for safetensors files.
#[pyclass(name = "MmapFile")]
pub struct PyMmapFile {
    inner: MmapFile,
}

#[pymethods]
impl PyMmapFile {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let inner = MmapFile::open(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(PyMmapFile { inner })
    }

    /// Get the raw pointer to a tensor's data as an integer.
    fn tensor_ptr(&self, name: &str) -> PyResult<usize> {
        self.inner.tensor_ptr(name)
            .map(|p| p as usize)
            .ok_or_else(|| PyKeyError::new_err(format!("Tensor '{name}' not found")))
    }

    /// Get tensor metadata: (offset, size, dtype, shape).
    fn tensor_info(&self, name: &str) -> PyResult<(usize, usize, String, Vec<usize>)> {
        self.inner.tensors.get(name)
            .map(|t| (t.offset, t.size, t.dtype.clone(), t.shape.clone()))
            .ok_or_else(|| PyKeyError::new_err(format!("Tensor '{name}' not found")))
    }

    /// List all tensor names.
    fn tensor_names(&self) -> Vec<String> {
        self.inner.tensors.keys().cloned().collect()
    }

    /// Prefetch a tensor into the OS page cache (MADV_WILLNEED).
    fn prefetch_tensor(&self, name: &str) {
        self.inner.prefetch_tensor(name);
    }

    /// Prefetch a byte range.
    fn prefetch_range(&self, offset: usize, len: usize) {
        self.inner.region.prefetch_range(offset, len);
    }

    /// Release all pages to the OS (MADV_DONTNEED).
    fn release_to_os(&self) {
        self.inner.region.release_to_os();
    }

    /// Total data segment size in bytes.
    fn data_size(&self) -> usize {
        self.inner.region.len()
    }
}

// ── Error type ──────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum MmapError {
    ZeroLength,
    Io(std::io::Error),
    Json(serde_json::Error),
    Other(String),
}

impl std::fmt::Display for MmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MmapError::ZeroLength => write!(f, "Cannot mmap zero-length region"),
            MmapError::Io(e) => write!(f, "mmap I/O error: {e}"),
            MmapError::Json(e) => write!(f, "Header parse error: {e}"),
            MmapError::Other(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for MmapError {}
