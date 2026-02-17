use std::collections::{BTreeMap, HashMap};
use std::io::{self, BufRead, Write};

use half::f16;
use rustnn::executors::onnx::{OnnxInput, TensorData, run_onnx_with_inputs};
use rustnn::{ContextProperties, ConverterRegistry, GraphError, GraphValidator};
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
        context_options: Value,
    },
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
    } else if msg.contains("onnx") || msg.contains("convert") || msg.contains("conversion") {
        RunnerError::GraphConversion(msg)
    } else {
        RunnerError::RuntimeExecution(msg)
    }
}

fn execute_graph(
    graph: GraphJson,
    inputs: BTreeMap<String, InputTensor>,
    expected_outputs: BTreeMap<String, ExpectedOutput>,
) -> Result<BTreeMap<String, OutputTensor>, RunnerError> {
    let graph_info = rustnn::webnn_json::from_graph_json(&graph)
        .map_err(|e| RunnerError::GraphValidation(e.to_string()))?;

    let validator = GraphValidator::new(&graph_info, ContextProperties::default());
    let _artifacts = validator
        .validate()
        .map_err(|e| RunnerError::GraphValidation(e.to_string()))?;

    let converted = ConverterRegistry::with_defaults()
        .convert("onnx", &graph_info)
        .map_err(|e| RunnerError::GraphConversion(e.to_string()))?;

    let mut onnx_inputs = Vec::with_capacity(inputs.len());
    for (name, input) in &inputs {
        onnx_inputs.push(OnnxInput {
            name: name.clone(),
            shape: input.descriptor.shape.clone(),
            data: to_tensor_data(&input.descriptor, &input.data)?,
        });
    }

    let outputs =
        run_onnx_with_inputs(&converted.data, onnx_inputs).map_err(|e| classify_graph_error(&e))?;

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
            }) => {
                let _ = context_options;
                match execute_graph(graph, inputs, expected_outputs) {
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
                }
            }
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
