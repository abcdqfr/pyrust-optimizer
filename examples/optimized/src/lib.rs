
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;

#[pyfunction]
fn memory_opt_0(size: usize) -> PyResult<Vec<i32>> {
    // Pre-allocated vector for better memory performance
    let mut data = Vec::with_capacity(size);

    // Efficient memory operations
    for i in 0..size {
        data.push(i as i32);
    }

    Ok(data)
}

#[pyfunction]
fn memory_opt_0_append(mut data: Vec<i32>, value: i32) -> PyResult<Vec<i32>> {
    // Optimized append operation
    data.push(value);
    Ok(data)
}

#[pyfunction]
fn math_opt_1(x: f64) -> PyResult<f64> {
    // Optimized mathematical computation using Rust's fast math
    let result = x.sin() * x.cos() + x.tan().sqrt();
    Ok(result)
}

#[pyfunction]
fn math_opt_1_vectorized(values: Vec<f64>) -> PyResult<Vec<f64>> {
    // Vectorized mathematical operations
    let results: Vec<f64> = values
        .iter()
        .map(|&x| x.sin() * x.cos() + x.tan().sqrt())
        .collect();

    Ok(results)
}

#[pyfunction]
fn string_opt_2(text: &str, pattern: &str) -> PyResult<Vec<String>> {
    // Optimized string processing using Rust's efficient string handling
    let results: Vec<String> = text
        .split(pattern)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    Ok(results)
}

#[pyfunction]
fn string_opt_2_join(parts: Vec<&str>, separator: &str) -> PyResult<String> {
    // Efficient string joining
    Ok(parts.join(separator))
}

#[pyfunction]
fn math_opt_3(x: f64) -> PyResult<f64> {
    // Optimized mathematical computation using Rust's fast math
    let result = x.sin() * x.cos() + x.tan().sqrt();
    Ok(result)
}

#[pyfunction]
fn math_opt_3_vectorized(values: Vec<f64>) -> PyResult<Vec<f64>> {
    // Vectorized mathematical operations
    let results: Vec<f64> = values
        .iter()
        .map(|&x| x.sin() * x.cos() + x.tan().sqrt())
        .collect();

    Ok(results)
}

#[pyfunction]
fn math_opt_4(x: f64) -> PyResult<f64> {
    // Optimized mathematical computation using Rust's fast math
    let result = x.sin() * x.cos() + x.tan().sqrt();
    Ok(result)
}

#[pyfunction]
fn math_opt_4_vectorized(values: Vec<f64>) -> PyResult<Vec<f64>> {
    // Vectorized mathematical operations
    let results: Vec<f64> = values
        .iter()
        .map(|&x| x.sin() * x.cos() + x.tan().sqrt())
        .collect();

    Ok(results)
}


/// PyRust Optimizer - Optimized Python functions in Rust
#[pymodule]
fn optimized(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(memory_opt_0, m)?)?;
    m.add_function(wrap_pyfunction!(memory_opt_0_append, m)?)?;
    m.add_function(wrap_pyfunction!(math_opt_1, m)?)?;
    m.add_function(wrap_pyfunction!(math_opt_1_vectorized, m)?)?;
    m.add_function(wrap_pyfunction!(string_opt_2, m)?)?;
    m.add_function(wrap_pyfunction!(string_opt_2_join, m)?)?;
    m.add_function(wrap_pyfunction!(math_opt_3, m)?)?;
    m.add_function(wrap_pyfunction!(math_opt_3_vectorized, m)?)?;
    m.add_function(wrap_pyfunction!(math_opt_4, m)?)?;
    m.add_function(wrap_pyfunction!(math_opt_4_vectorized, m)?)?;
    Ok(())
}
