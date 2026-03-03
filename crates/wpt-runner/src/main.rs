/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
use std::collections::{BTreeMap, HashMap};
use std::io::{self, BufRead, Write};

use half::f16;
#[cfg(all(target_os = "macos", feature = "backend-coreml"))]
use rustnn::executors::coreml::{CoremlInput, CoremlOutput, run_coreml_with_inputs_with_weights};

#[cfg(any(feature = "backend-trtx", feature = "backend-trtx-mock"))]
use rustnn::executors::trtx::{TrtxInput, TrtxOutputWithData, run_trtx_with_inputs};

use rustnn::executors::onnx::{OnnxInput, OnnxOutputWithData, TensorData, run_onnx_with_inputs};
use rustnn::{ContextProperties, ConverterRegistry, GraphError, GraphInfo, GraphValidator};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use webnn_graph::ast::GraphJson;

#[derive(Debug, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum Request {
    ExecuteGraph {
        id: String,
        graph: GraphJson,
        inputs: BTreeMap<String, InputTensor>,
        #[serde(default)]
        expected_outputs: BTreeMap<String, ExpectedOutput>,
        #[serde(default)]
        context_options: ContextOptions,
    },
}

#[derive(Debug, Deserialize, Default)]
struct ContextOptions {
    #[serde(default)]
    backend: Option<String>,
    #[serde(rename = "deviceType", default)]
    device_type: Option<String>,
}

#[derive(Debug, Clone, Copy)]
enum Backend {
    Onnx,
    Coreml,
    Trtx,
}

impl Backend {
    fn from_context(options: &ContextOptions) -> Result<Self, RunnerError> {
        let selected = options.backend.as_deref().unwrap_or("onnx");
        match selected.trim().to_ascii_lowercase().as_str() {
            "" | "onnx" | "ort" => Ok(Self::Onnx),
            "coreml" => Ok(Self::Coreml),
            "trtx" | "trt" | "tensorrt" => Ok(Self::Trtx),
            other => Err(RunnerError::BadRequest(format!(
                "unknown backend '{other}'. Supported: onnx, coreml, trtx"
            ))),
        }
    }
}

#[derive(Debug, Deserialize)]
struct TensorDescriptor {
    #[serde(rename = "dataType")]
    data_type: String,
    shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct InputTensor {
    descriptor: TensorDescriptor,
    data: Vec<Value>,
}

#[derive(Debug, Deserialize)]
struct ExpectedOutput {
    descriptor: TensorDescriptor,
    #[serde(default)]
    data: Vec<Value>,
}

#[derive(Debug, Serialize)]
struct Response {
    id: String,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    outputs: Option<BTreeMap<String, OutputTensor>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<ErrorPayload>,
}

#[derive(Debug, Serialize)]
struct OutputTensor {
    descriptor: TensorDescriptorOut,
    data: Vec<Value>,
}

#[derive(Debug, Serialize)]
struct TensorDescriptorOut {
    #[serde(rename = "dataType")]
    data_type: String,
    shape: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct ErrorPayload {
    kind: String,
    message: String,
}

#[derive(Debug, Clone)]
struct RuntimeOutput {
    name: String,
    shape: Vec<usize>,
    data: Vec<f64>,
    int64_data: Option<Vec<i64>>,
    uint64_data: Option<Vec<u64>>,
}

#[derive(Debug, Error)]
enum RunnerError {
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("graph validation failed: {0}")]
    GraphValidation(String),
    #[error("graph conversion failed: {0}")]
    GraphConversion(String),
    #[error("runtime execution failed: {0}")]
    RuntimeExecution(String),
}

fn parse_i64(v: &Value) -> Result<i64, RunnerError> {
    if let Some(n) = v.as_i64() {
        return Ok(n);
    }
    if let Some(s) = v.as_str() {
        return s
            .parse::<i64>()
            .map_err(|_| RunnerError::BadRequest(format!("invalid int64 value: {s}")));
    }
    Err(RunnerError::BadRequest(format!("invalid int64 value: {v}")))
}

fn parse_u64(v: &Value) -> Result<u64, RunnerError> {
    if let Some(n) = v.as_u64() {
        return Ok(n);
    }
    if let Some(s) = v.as_str() {
        return s
            .parse::<u64>()
            .map_err(|_| RunnerError::BadRequest(format!("invalid uint64 value: {s}")));
    }
    Err(RunnerError::BadRequest(format!(
        "invalid uint64 value: {v}"
    )))
}

fn parse_f32(v: &Value) -> Result<f32, RunnerError> {
    if let Some(n) = v.as_f64() {
        return Ok(n as f32);
    }
    if let Some(n) = v.as_i64() {
        return Ok(n as f32);
    }
    if let Some(n) = v.as_u64() {
        return Ok(n as f32);
    }
    if let Some(s) = v.as_str() {
        return s
            .parse::<f32>()
            .map_err(|_| RunnerError::BadRequest(format!("invalid float value: {s}")));
    }
    Err(RunnerError::BadRequest(format!("invalid float value: {v}")))
}

fn shape_element_count(shape: &[usize]) -> Result<usize, RunnerError> {
    let mut count = 1usize;
    for &dim in shape {
        count = count.checked_mul(dim).ok_or_else(|| {
            RunnerError::BadRequest(format!(
                "shape element count overflow for shape {:?}",
                shape
            ))
        })?;
    }
    Ok(count.max(1))
}

fn normalize_input_values(
    descriptor: &TensorDescriptor,
    data: &[Value],
) -> Result<Vec<Value>, RunnerError> {
    let expected = shape_element_count(&descriptor.shape)?;
    let actual = data.len();

    if actual == expected {
        return Ok(data.to_vec());
    }
    if actual == 1 && expected > 1 {
        return Ok(vec![data[0].clone(); expected]);
    }

    Err(RunnerError::BadRequest(format!(
        "input data length mismatch: expected {} values for shape {:?}, got {}",
        expected, descriptor.shape, actual
    )))
}

fn to_tensor_data(
    descriptor: &TensorDescriptor,
    data: &[Value],
) -> Result<TensorData, RunnerError> {
    let normalized = normalize_input_values(descriptor, data)?;
    match descriptor.data_type.as_str() {
        "float32" => Ok(TensorData::Float32(
            normalized
                .iter()
                .map(parse_f32)
                .collect::<Result<Vec<_>, _>>()?,
        )),
        "float16" => {
            let bits = normalized
                .iter()
                .map(|v| Ok(f16::from_f32(parse_f32(v)?).to_bits()))
                .collect::<Result<Vec<u16>, RunnerError>>()?;
            Ok(TensorData::Float16(bits))
        }
        "int8" => Ok(TensorData::Int8(
            normalized
                .iter()
                .map(|v| parse_i64(v).map(|x| x as i8))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        "uint8" | "uint4" => Ok(TensorData::Uint8(
            normalized
                .iter()
                .map(|v| parse_u64(v).map(|x| x as u8))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        "int32" | "int4" => Ok(TensorData::Int32(
            normalized
                .iter()
                .map(|v| parse_i64(v).map(|x| x as i32))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        "uint32" => Ok(TensorData::Uint32(
            normalized
                .iter()
                .map(|v| parse_u64(v).map(|x| x as u32))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        "int64" => Ok(TensorData::Int64(
            normalized
                .iter()
                .map(parse_i64)
                .collect::<Result<Vec<_>, _>>()?,
        )),
        "uint64" => Ok(TensorData::Uint64(
            normalized
                .iter()
                .map(parse_u64)
                .collect::<Result<Vec<_>, _>>()?,
        )),
        other => Err(RunnerError::BadRequest(format!(
            "unsupported input dataType: {other}"
        ))),
    }
}

#[cfg(any(feature = "backend-trtx", feature = "backend-trtx-mock"))]
fn tensor_data_to_le_bytes(data: TensorData) -> Vec<u8> {
    match data {
        TensorData::Float32(values) => values.into_iter().flat_map(f32::to_le_bytes).collect(),
        TensorData::Float16(values) => values.into_iter().flat_map(u16::to_le_bytes).collect(),
        TensorData::Int8(values) => values.into_iter().map(|v| v as u8).collect(),
        TensorData::Uint8(values) => values,
        TensorData::Int32(values) => values.into_iter().flat_map(i32::to_le_bytes).collect(),
        TensorData::Uint32(values) => values.into_iter().flat_map(u32::to_le_bytes).collect(),
        TensorData::Int64(values) => values.into_iter().flat_map(i64::to_le_bytes).collect(),
        TensorData::Uint64(values) => values.into_iter().flat_map(u64::to_le_bytes).collect(),
    }
}

#[cfg(all(target_os = "macos", feature = "backend-coreml"))]
fn to_f32_values(descriptor: &TensorDescriptor, data: &[Value]) -> Result<Vec<f32>, RunnerError> {
    let normalized = normalize_input_values(descriptor, data)?;
    normalized.iter().map(parse_f32).collect()
}

fn onnx_outputs_to_runtime(outputs: Vec<OnnxOutputWithData>) -> Vec<RuntimeOutput> {
    outputs
        .into_iter()
        .map(|output| RuntimeOutput {
            name: output.name,
            shape: output.shape,
            data: output.data,
            int64_data: output.int64_data,
            uint64_data: output.uint64_data,
        })
        .collect()
}

#[cfg(all(target_os = "macos", feature = "backend-coreml"))]
fn coreml_output_shape_to_usize(shape: &[i64]) -> Result<Vec<usize>, RunnerError> {
    shape
        .iter()
        .map(|&dim| {
            if dim < 0 {
                return Err(RunnerError::RuntimeExecution(format!(
                    "coreml output shape contains negative dimension: {shape:?}"
                )));
            }
            Ok(dim as usize)
        })
        .collect()
}

#[cfg(all(target_os = "macos", feature = "backend-coreml"))]
fn coreml_outputs_to_runtime(
    outputs: Vec<CoremlOutput>,
) -> Result<Vec<RuntimeOutput>, RunnerError> {
    outputs
        .into_iter()
        .map(|output| {
            let shape = coreml_output_shape_to_usize(&output.shape)?;
            let data = output
                .data
                .into_iter()
                .map(|v| v as f64)
                .collect::<Vec<_>>();
            Ok(RuntimeOutput {
                name: output.name,
                shape,
                data,
                int64_data: None,
                uint64_data: None,
            })
        })
        .collect()
}

#[cfg(any(feature = "backend-trtx", feature = "backend-trtx-mock"))]
fn read_le_chunks<const N: usize>(bytes: &[u8]) -> Result<Vec<[u8; N]>, RunnerError> {
    if bytes.len() % N != 0 {
        return Err(RunnerError::RuntimeExecution(format!(
            "invalid tensor byte length {} for element width {}",
            bytes.len(),
            N
        )));
    }

    Ok(bytes
        .chunks_exact(N)
        .map(|chunk| {
            let mut arr = [0u8; N];
            arr.copy_from_slice(chunk);
            arr
        })
        .collect())
}

#[cfg(any(feature = "backend-trtx", feature = "backend-trtx-mock"))]
fn trtx_output_to_runtime(output: TrtxOutputWithData) -> Result<RuntimeOutput, RunnerError> {
    let data_type = output.data_type.to_ascii_lowercase();
    let (data, int64_data, uint64_data) = match data_type.as_str() {
        "float32" => {
            let values = read_le_chunks::<4>(&output.data)?
                .into_iter()
                .map(|b| f32::from_le_bytes(b) as f64)
                .collect::<Vec<_>>();
            (values, None, None)
        }
        "float16" => {
            let values = read_le_chunks::<2>(&output.data)?
                .into_iter()
                .map(|b| f16::from_bits(u16::from_le_bytes(b)).to_f32() as f64)
                .collect::<Vec<_>>();
            (values, None, None)
        }
        "int8" => {
            let values = output
                .data
                .iter()
                .map(|b| (*b as i8) as f64)
                .collect::<Vec<_>>();
            (values, None, None)
        }
        "uint8" | "bool" => {
            let values = output.data.iter().map(|b| *b as f64).collect::<Vec<_>>();
            (values, None, None)
        }
        "int32" => {
            let values = read_le_chunks::<4>(&output.data)?
                .into_iter()
                .map(|b| i32::from_le_bytes(b) as f64)
                .collect::<Vec<_>>();
            (values, None, None)
        }
        "uint32" => {
            let values = read_le_chunks::<4>(&output.data)?
                .into_iter()
                .map(|b| u32::from_le_bytes(b) as f64)
                .collect::<Vec<_>>();
            (values, None, None)
        }
        "int64" => {
            let values = read_le_chunks::<8>(&output.data)?
                .into_iter()
                .map(i64::from_le_bytes)
                .collect::<Vec<_>>();
            (
                values.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                Some(values),
                None,
            )
        }
        "uint64" => {
            let values = read_le_chunks::<8>(&output.data)?
                .into_iter()
                .map(u64::from_le_bytes)
                .collect::<Vec<_>>();
            (
                values.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                None,
                Some(values),
            )
        }
        other => {
            return Err(RunnerError::RuntimeExecution(format!(
                "unsupported TensorRT output data type: {other}"
            )));
        }
    };

    Ok(RuntimeOutput {
        name: output.name,
        shape: output.shape,
        data,
        int64_data,
        uint64_data,
    })
}

fn cast_output_data(
    data: &[f64],
    int64_data: Option<&[i64]>,
    uint64_data: Option<&[u64]>,
    dtype: &str,
) -> Vec<Value> {
    fn float_value(x: f64) -> Value {
        if x.is_nan() {
            Value::String("NaN".to_string())
        } else if x.is_infinite() {
            if x.is_sign_positive() {
                Value::String("Infinity".to_string())
            } else {
                Value::String("-Infinity".to_string())
            }
        } else {
            Value::from(x)
        }
    }

    match dtype {
        "float32" => data.iter().map(|x| float_value(*x)).collect(),
        "float16" => data
            .iter()
            .map(|x| float_value(f16::from_f32(*x as f32).to_f32() as f64))
            .collect(),
        "int8" => data
            .iter()
            .map(|x| Value::from((*x as i8) as i64))
            .collect(),
        "uint8" | "uint4" => data
            .iter()
            .map(|x| Value::from((*x as u8) as u64))
            .collect(),
        "int4" => data
            .iter()
            .map(|x| Value::from(((*x as i8).clamp(-8, 7)) as i64))
            .collect(),
        "int32" => data
            .iter()
            .map(|x| Value::from((*x as i32) as i64))
            .collect(),
        "uint32" => data
            .iter()
            .map(|x| Value::from((*x as u32) as u64))
            .collect(),
        "int64" => {
            if let Some(values) = int64_data {
                values
                    .iter()
                    .map(|x| Value::String(x.to_string()))
                    .collect()
            } else {
                data.iter()
                    .map(|x| Value::String((*x as i64).to_string()))
                    .collect()
            }
        }
        "uint64" => {
            if let Some(values) = uint64_data {
                values
                    .iter()
                    .map(|x| Value::String(x.to_string()))
                    .collect()
            } else {
                data.iter()
                    .map(|x| Value::String((*x as u64).to_string()))
                    .collect()
            }
        }
        _ => data.iter().map(|x| float_value(*x)).collect(),
    }
}

fn cast_output_data_compact(
    data: &[f64],
    int64_data: Option<&[i64]>,
    uint64_data: Option<&[u64]>,
    dtype: &str,
    expected_len: usize,
) -> Vec<Value> {
    if expected_len == 1 && !data.is_empty() {
        return cast_output_data(
            &data[..1],
            int64_data.map(|v| &v[..1]),
            uint64_data.map(|v| &v[..1]),
            dtype,
        );
    }
    cast_output_data(data, int64_data, uint64_data, dtype)
}

fn classify_graph_error(err: &GraphError) -> RunnerError {
    let msg = err.to_string();
    if msg.contains("validation") || msg.contains("input") || msg.contains("output") {
        RunnerError::GraphValidation(msg)
    } else if msg.contains("onnx")
        || msg.contains("coreml")
        || msg.contains("trtx")
        || msg.contains("tensorrt")
        || msg.contains("convert")
        || msg.contains("conversion")
    {
        RunnerError::GraphConversion(msg)
    } else {
        RunnerError::RuntimeExecution(msg)
    }
}

fn execute_onnx_backend(
    graph_info: &GraphInfo,
    inputs: &BTreeMap<String, InputTensor>,
) -> Result<Vec<RuntimeOutput>, RunnerError> {
    let converted = ConverterRegistry::with_defaults()
        .convert("onnx", graph_info)
        .map_err(|e| RunnerError::GraphConversion(e.to_string()))?;

    let mut onnx_inputs = Vec::with_capacity(inputs.len());
    for (name, input) in inputs {
        onnx_inputs.push(OnnxInput {
            name: name.clone(),
            shape: input.descriptor.shape.clone(),
            data: to_tensor_data(&input.descriptor, &input.data)?,
        });
    }

    let outputs =
        run_onnx_with_inputs(&converted.data, onnx_inputs).map_err(|e| classify_graph_error(&e))?;
    Ok(onnx_outputs_to_runtime(outputs))
}

#[cfg(all(target_os = "macos", feature = "backend-coreml"))]
fn execute_coreml_backend(
    graph_info: &GraphInfo,
    inputs: &BTreeMap<String, InputTensor>,
) -> Result<Vec<RuntimeOutput>, RunnerError> {
    let converted = ConverterRegistry::with_defaults()
        .convert("coreml", graph_info)
        .map_err(|e| RunnerError::GraphConversion(e.to_string()))?;

    let mut coreml_inputs = Vec::with_capacity(inputs.len());
    for (name, input) in inputs {
        coreml_inputs.push(CoremlInput {
            name: name.clone(),
            shape: input.descriptor.shape.clone(),
            data: to_f32_values(&input.descriptor, &input.data)?,
        });
    }

    let attempts = run_coreml_with_inputs_with_weights(
        &converted.data,
        converted.weights_data.as_deref(),
        coreml_inputs,
    )
    .map_err(|e| classify_graph_error(&e))?;

    let outputs = attempts
        .into_iter()
        .find_map(|attempt| attempt.result.ok())
        .ok_or_else(|| {
            RunnerError::RuntimeExecution(
                "coreml runtime failed: all compute unit attempts failed".to_string(),
            )
        })?;

    coreml_outputs_to_runtime(outputs)
}

#[cfg(not(all(target_os = "macos", feature = "backend-coreml")))]
fn execute_coreml_backend(
    _graph_info: &GraphInfo,
    _inputs: &BTreeMap<String, InputTensor>,
) -> Result<Vec<RuntimeOutput>, RunnerError> {
    Err(RunnerError::RuntimeExecution(
        "backend 'coreml' is unavailable; rebuild runner with feature backend-coreml on macOS"
            .to_string(),
    ))
}

#[cfg(any(feature = "backend-trtx", feature = "backend-trtx-mock"))]
fn execute_trtx_backend(
    graph_info: &GraphInfo,
    inputs: &BTreeMap<String, InputTensor>,
) -> Result<Vec<RuntimeOutput>, RunnerError> {
    let converted = ConverterRegistry::with_defaults()
        .convert("trtx", graph_info)
        .map_err(|e| RunnerError::GraphConversion(e.to_string()))?;

    let mut trtx_inputs = Vec::with_capacity(inputs.len());
    for (name, input) in inputs {
        trtx_inputs.push(TrtxInput {
            name: name.clone(),
            data: tensor_data_to_le_bytes(to_tensor_data(&input.descriptor, &input.data)?),
        });
    }

    let outputs =
        run_trtx_with_inputs(&converted.data, trtx_inputs).map_err(|e| classify_graph_error(&e))?;
    outputs.into_iter().map(trtx_output_to_runtime).collect()
}

#[cfg(not(any(feature = "backend-trtx", feature = "backend-trtx-mock")))]
fn execute_trtx_backend(
    _graph_info: &GraphInfo,
    _inputs: &BTreeMap<String, InputTensor>,
) -> Result<Vec<RuntimeOutput>, RunnerError> {
    Err(RunnerError::RuntimeExecution(
        "backend 'trtx' is unavailable; rebuild runner with feature backend-trtx or backend-trtx-mock"
            .to_string(),
    ))
}

fn execute_backend(
    backend: Backend,
    graph_info: &GraphInfo,
    inputs: &BTreeMap<String, InputTensor>,
) -> Result<Vec<RuntimeOutput>, RunnerError> {
    match backend {
        Backend::Onnx => execute_onnx_backend(graph_info, inputs),
        Backend::Coreml => execute_coreml_backend(graph_info, inputs),
        Backend::Trtx => execute_trtx_backend(graph_info, inputs),
    }
}

fn execute_graph(
    graph: GraphJson,
    inputs: BTreeMap<String, InputTensor>,
    expected_outputs: BTreeMap<String, ExpectedOutput>,
    context_options: ContextOptions,
) -> Result<BTreeMap<String, OutputTensor>, RunnerError> {
    let graph_info = rustnn::webnn_json::from_graph_json(&graph)
        .map_err(|e| RunnerError::GraphValidation(e.to_string()))?;

    let validator = GraphValidator::new(&graph_info, ContextProperties::default());
    let _artifacts = validator
        .validate()
        .map_err(|e| RunnerError::GraphValidation(e.to_string()))?;

    if std::env::var("RUSTNNPT_DEBUG").as_deref() == Ok("1") {
        eprintln!("[RUNNER] inputs (BTreeMap iteration order):");
        for (name, input) in &inputs {
            eprintln!(
                "  {} shape={:?} data.len()={}",
                name,
                input.descriptor.shape,
                input.data.len()
            );
        }
    }

    let _requested_device = context_options.device_type.as_deref().unwrap_or("cpu");
    let backend = Backend::from_context(&context_options)?;
    let outputs = execute_backend(backend, &graph_info, &inputs)?;

    let by_name: HashMap<String, _> = outputs.into_iter().map(|o| (o.name.clone(), o)).collect();

    let mut out = BTreeMap::new();
    if expected_outputs.is_empty() {
        for (name, output) in by_name {
            out.insert(
                name,
                OutputTensor {
                    descriptor: TensorDescriptorOut {
                        data_type: "float32".to_string(),
                        shape: output.shape,
                    },
                    data: cast_output_data(
                        &output.data,
                        output.int64_data.as_deref(),
                        output.uint64_data.as_deref(),
                        "float32",
                    ),
                },
            );
        }
    } else {
        for (name, expected) in &expected_outputs {
            let output = by_name.get(name).ok_or_else(|| {
                RunnerError::RuntimeExecution(format!("missing output from runtime: {name}"))
            })?;
            let expected_element_count = shape_element_count(&expected.descriptor.shape)?;
            let actual_len = output.data.len();
            if actual_len != expected_element_count {
                return Err(RunnerError::RuntimeExecution(format!(
                    "output {name}: runtime returned {actual_len} elements but expected {} (shape {:?})",
                    expected_element_count,
                    expected.descriptor.shape
                )));
            }
            out.insert(
                name.clone(),
                OutputTensor {
                    descriptor: TensorDescriptorOut {
                        data_type: expected.descriptor.data_type.clone(),
                        shape: output.shape.clone(),
                    },
                    data: cast_output_data_compact(
                        &output.data,
                        output.int64_data.as_deref(),
                        output.uint64_data.as_deref(),
                        &expected.descriptor.data_type,
                        expected.data.len(),
                    ),
                },
            );
        }
    }

    Ok(out)
}

fn error_kind(err: &RunnerError) -> String {
    match err {
        RunnerError::BadRequest(_) => "BadRequestError",
        RunnerError::GraphValidation(_) => "GraphValidationError",
        RunnerError::GraphConversion(_) => "GraphConversionError",
        RunnerError::RuntimeExecution(_) => "RuntimeExecutionError",
    }
    .to_string()
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let raw = match line {
            Ok(l) => l,
            Err(e) => {
                let _ = writeln!(
                    stdout,
                    "{{\"id\":\"unknown\",\"ok\":false,\"error\":{{\"kind\":\"BadRequestError\",\"message\":\"{}\"}}}}",
                    e
                );
                let _ = stdout.flush();
                continue;
            }
        };

        if raw.trim().is_empty() {
            continue;
        }

        let parsed: Result<Request, _> = serde_json::from_str(&raw);
        let response = match parsed {
            Ok(Request::ExecuteGraph {
                id,
                graph,
                inputs,
                expected_outputs,
                context_options,
            }) => match execute_graph(graph, inputs, expected_outputs, context_options) {
                Ok(outputs) => Response {
                    id,
                    ok: true,
                    outputs: Some(outputs),
                    error: None,
                },
                Err(err) => Response {
                    id,
                    ok: false,
                    outputs: None,
                    error: Some(ErrorPayload {
                        kind: error_kind(&err),
                        message: err.to_string(),
                    }),
                },
            },
            Err(err) => Response {
                id: "unknown".to_string(),
                ok: false,
                outputs: None,
                error: Some(ErrorPayload {
                    kind: "BadRequestError".to_string(),
                    message: format!("invalid json request: {err}"),
                }),
            },
        };

        match serde_json::to_string(&response) {
            Ok(json) => {
                let _ = writeln!(stdout, "{json}");
                let _ = stdout.flush();
            }
            Err(err) => {
                let _ = writeln!(
                    stdout,
                    "{{\"id\":\"unknown\",\"ok\":false,\"error\":{{\"kind\":\"BadRequestError\",\"message\":\"response encode failed: {}\"}}}}",
                    err
                );
                let _ = stdout.flush();
            }
        }
    }
}


