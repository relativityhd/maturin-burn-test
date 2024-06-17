use burn::backend::{ndarray::NdArrayDevice, wgpu::WgpuDevice, NdArray, Wgpu};
use burn::tensor::backend::Backend;
use burn::tensor::{Data, Tensor};
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

fn add<B>(a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 1>
where
    B: Backend,
{
    a + b
}

#[pyfunction]
fn pyadd<'py>(
    py: Python<'py>,
    gpu: bool,
    a: PyReadonlyArray1<'py, f32>,
    b: PyReadonlyArray1<'py, f32>,
) -> Bound<'py, PyArray1<f32>> {
    if gpu {
        let device = WgpuDevice::default();
        println!("GPU device: {:?}", device);
        let a = Data::<f32, 1>::from(a.as_slice().unwrap());
        let a = Tensor::<Wgpu, 1>::from_data(a, &device);
        let b = Data::<f32, 1>::from(b.as_slice().unwrap());
        let b = Tensor::<Wgpu, 1>::from_data(b, &device);
        let result = add(a, b);
        let result = Array1::from_vec(result.into_data().value);
        result.into_pyarray_bound(py)
    } else {
        let device = NdArrayDevice::default();
        println!("CPU device: {:?}", device);
        let a = Data::<f32, 1>::from(a.as_slice().unwrap());
        let a = Tensor::<NdArray, 1>::from_data(a, &device);
        let b = Data::<f32, 1>::from(b.as_slice().unwrap());
        let b = Tensor::<NdArray, 1>::from_data(b, &device);
        let result = add(a, b);
        let result = Array1::from_vec(result.into_data().value);
        result.into_pyarray_bound(py)
    }
}

/// Prints a message.
#[pyfunction]
fn hello() -> PyResult<String> {
    Ok("Hello from maturin-burn-test!".into())
}

/// A Python module implemented in Rust.
#[pymodule]
fn _lowlevel(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(pyadd, m)?)?;
    Ok(())
}
