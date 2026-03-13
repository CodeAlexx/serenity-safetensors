//! Python-visible QuantizedTensor — lazy dequant wrapper for GGUF tensors.

use crate::gguf::GgufQuantType;
use crate::gguf_dequant;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// A quantized tensor from a GGUF file. Holds a reference to the raw mmap
/// data and can dequantize to a BF16 torch.Tensor on demand.
#[pyclass]
pub struct QuantizedTensor {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) quant_type: GgufQuantType,
    pub(crate) param_count: usize,
    pub(crate) byte_size: usize,
    /// Python object holding a reference to the backing bytes (mmap slice).
    pub(crate) data: Py<PyAny>,
}

#[pymethods]
impl QuantizedTensor {
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[getter]
    fn quant_type_name(&self) -> &str {
        self.quant_type.name()
    }

    #[getter]
    fn is_quantized(&self) -> bool {
        self.quant_type.is_quantized()
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.byte_size
    }

    #[getter]
    fn nbytes_dequant(&self) -> usize {
        self.param_count * 2 // BF16 = 2 bytes per weight
    }

    #[getter]
    fn compression_ratio(&self) -> f32 {
        self.nbytes_dequant() as f32 / self.byte_size.max(1) as f32
    }

    /// Dequantize to a BF16 torch.Tensor.
    fn dequant(&self, py: Python<'_>) -> PyResult<PyObject> {
        // Extract bytes from the stored Python object
        let data_ref = self.data.bind(py);
        let data_bytes: &[u8] = data_ref
            .extract::<&[u8]>()
            .map_err(|_| PyValueError::new_err("cannot extract bytes from tensor data backing"))?;

        let quant_type = self.quant_type;
        let param_count = self.param_count;

        // Release GIL during dequantization
        let bf16_bytes = py
            .allow_threads(move || {
                let bf16_vec = gguf_dequant::dequant_to_bf16(data_bytes, quant_type, param_count)?;
                // Convert Vec<bf16> to raw bytes
                let raw: &[u8] = bytemuck::cast_slice(&bf16_vec);
                Ok::<Vec<u8>, String>(raw.to_vec())
            })
            .map_err(|e| PyValueError::new_err(e))?;

        // Build torch.Tensor from raw bytes
        let torch = py.import_bound("torch")?;
        let bf16_dtype = torch.getattr("bfloat16")?;
        let py_bytes = pyo3::types::PyBytes::new_bound(py, &bf16_bytes);
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", bf16_dtype)?;
        let tensor = torch.call_method("frombuffer", (py_bytes,), Some(&kwargs))?;

        // Clone to own the data (frombuffer returns a view of the bytes)
        let tensor = tensor.call_method0("clone")?;

        // Reshape
        let shape_tuple = PyTuple::new_bound(py, &self.shape);
        let tensor = tensor.call_method1("reshape", (shape_tuple,))?;
        Ok(tensor.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantizedTensor(name='{}', shape={:?}, quant_type='{}', compressed={}, full={})",
            self.name,
            self.shape,
            self.quant_type.name(),
            self.byte_size,
            self.nbytes_dequant()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_tensor_properties() {
        // We can't test PyO3 methods without a Python runtime, but we can
        // test the struct construction and basic logic.
        let qt = GgufQuantType::Q4_0;
        assert!(qt.is_quantized());
        assert_eq!(qt.name(), "Q4_0");

        // Compression ratio: Q4_0 has 18 bytes per 32 weights
        // Full BF16: 32 * 2 = 64 bytes
        // Ratio: 64 / 18 = 3.555...
        let byte_size = qt.compute_byte_size(32);
        assert_eq!(byte_size, 18);
        let nbytes_dequant = 32 * 2;
        let ratio = nbytes_dequant as f32 / byte_size as f32;
        assert!((ratio - 3.555).abs() < 0.01);
    }
}
