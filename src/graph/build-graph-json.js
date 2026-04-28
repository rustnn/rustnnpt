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

function normalizeOpName(name) {
  return name;
}

// Pool2d-style ops use MLPool2dOptions; rustnn expects WebNN IDL camelCase (outputShapeRounding).
const POOL2D_LIKE_OPS = new Set([
  'averagePool2d',
  'maxPool2d',
  'l2Pool2d',
  'globalAveragePool',
  'globalMaxPool'
]);

// Option keys that are MLOperands (stored in options as indices, not pushed to inputs).
// WebNN camelCase; matches option names from the API (e.g. recurrentBias, initialHiddenState).
const OPTION_OPERAND_KEYS = {
  batchNormalization: ['scale', 'bias'],
  conv2d: ['bias'],
  convTranspose2d: ['bias'],
  gemm: ['c'],
  gru: ['bias', 'recurrentBias', 'initialHiddenState'],
  gruCell: ['bias', 'recurrentBias'],
  instanceNormalization: ['scale', 'bias'],
  layerNormalization: ['scale', 'bias'],
  lstm: ['bias', 'recurrentBias', 'peepholeWeight', 'initialHiddenState', 'initialCellState'],
  lstmCell: ['bias', 'recurrentBias', 'peepholeWeight']
};

function isOperandOption(opName, optKey) {
  const keys = OPTION_OPERAND_KEYS[opName];
  return keys && keys.includes(optKey);
}

function normalizeOptionKey(opName, key) {
  const op = normalizeOpName(opName);
  if (op === 'cast' && key === 'type') return 'to';
  if (POOL2D_LIKE_OPS.has(op) && key === 'roundingType') {
    return 'outputShapeRounding';
  }
  return key;
}

function normalizeValue(v) {
  if (typeof v === 'number' && !Number.isFinite(v)) {
    if (Number.isNaN(v)) return 'NaN';
    return v > 0 ? 'Infinity' : '-Infinity';
  }
  if (typeof v === 'bigint') return v.toString();
  if (Array.isArray(v)) return v.map(normalizeValue);
  if (v && typeof v === 'object') {
    const out = {};
    for (const [k, val] of Object.entries(v)) out[k] = normalizeValue(val);
    return out;
  }
  return v;
}

function flattenInputReferences(argumentValue, operandNames) {
  if (typeof argumentValue === 'string') {
    return operandNames.has(argumentValue) ? [argumentValue] : [];
  }
  if (Array.isArray(argumentValue) && argumentValue.every((x) => typeof x === 'string')) {
    return argumentValue.every((x) => operandNames.has(x)) ? argumentValue : [];
  }
  return [];
}

function normalizeOptions(optionsObj) {
  const out = {};
  for (const [key, value] of Object.entries(optionsObj)) {
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      out[key] = normalizeValue(value);
    } else if (Array.isArray(value) || value === null || typeof value === 'object') {
      out[key] = normalizeValue(value);
    }
  }
  return out;
}

function shapeElementCount(shape) {
  if (!shape || shape.length === 0) return 1;
  return shape.reduce((a, b) => a * b, 1);
}

function parseNumericLoose(v) {
  if (typeof v === 'number') return v;
  if (typeof v === 'bigint') return Number(v);
  if (typeof v === 'string') {
    const t = v.trim();
    if (t === 'NaN') return NaN;
    if (t === 'Infinity' || t === '+Infinity') return Infinity;
    if (t === '-Infinity') return -Infinity;
    const noN = t.endsWith('n') ? t.slice(0, -1) : t;
    return Number(noN);
  }
  return Number(v);
}

function calculateBytesPerElement(dt) {
  switch (dt) {
    case 'float32':
    case 'int32':
    case 'uint32':
      return 4;
    case 'float16':
      return 2;
    case 'int64':
    case 'uint64':
      return 8;
    default:
      return 1;
  }
}

const LARGE_SCALAR_INLINE_BYTES_THRESHOLD = 8 * 1024 * 1024;

function shapeByteSize(dt, shape) {
  const n = Math.max(1, shapeElementCount(shape ?? []));
  return n * calculateBytesPerElement(dt);
}

function constantRawLength(raw) {
  if (Array.isArray(raw)) return raw.length;
  return raw == null ? 0 : 1;
}

function shouldInlineConstant(input) {
  if (input.constant !== true) return false;
  const dt = input?.descriptor?.dataType ?? 'float32';
  const shape = input?.descriptor?.shape ?? [];
  const rawLen = constantRawLength(input.data);
  const scalarFillLike = rawLen <= 1;
  const estBytes = shapeByteSize(dt, shape);
  return !(scalarFillLike && estBytes >= LARGE_SCALAR_INLINE_BYTES_THRESHOLD);
}

/**
 * Pack tensor values to little-endian bytes for rustnn ConstInit::InlineBytes / webnn-graph-json.
 * @param {{ descriptor: { dataType: string, shape: number[] }, data?: unknown }} input
 * @returns {number[]}
 */
function packConstantInlineBytes(input) {
  const dt = input.descriptor.dataType;
  const shape = input.descriptor.shape ?? [];
  const n = Math.max(1, shapeElementCount(shape));
  let raw = input.data;
  if (raw == null || (Array.isArray(raw) && raw.length === 0)) {
    raw = n === 1 ? [0] : new Array(n).fill(0);
  }

  const arr = Array.isArray(raw) ? raw : [raw];
  const getNorm = (i) => normalizeValue(arr.length === 1 ? arr[0] : arr[i]);

  switch (dt) {
    case 'float32': {
      const ta = new Float32Array(n);
      for (let i = 0; i < n; i++) ta[i] = parseNumericLoose(getNorm(i));
      return [...new Uint8Array(ta.buffer)];
    }
    case 'float16': {
      const ta = new Float16Array(n);
      for (let i = 0; i < n; i++) ta[i] = parseNumericLoose(getNorm(i));
      return [...new Uint8Array(ta.buffer)];
    }
    case 'int8': {
      const ta = new Int8Array(n);
      for (let i = 0; i < n; i++) ta[i] = parseNumericLoose(getNorm(i)) | 0;
      return [...new Uint8Array(ta.buffer)];
    }
    case 'uint8':
    case 'uint4':
    case 'int4': {
      const out = new Uint8Array(n);
      for (let i = 0; i < n; i++) out[i] = parseNumericLoose(getNorm(i)) & 0xff;
      return [...out];
    }
    case 'int32': {
      const ta = new Int32Array(n);
      for (let i = 0; i < n; i++) ta[i] = parseNumericLoose(getNorm(i)) | 0;
      return [...new Uint8Array(ta.buffer)];
    }
    case 'uint32': {
      const ta = new Uint32Array(n);
      for (let i = 0; i < n; i++) ta[i] = parseNumericLoose(getNorm(i)) >>> 0;
      return [...new Uint8Array(ta.buffer)];
    }
    case 'int64': {
      const ta = new BigInt64Array(n);
      for (let i = 0; i < n; i++) {
        const v = getNorm(i);
        ta[i] = typeof v === 'bigint' ? v : BigInt(Math.trunc(parseNumericLoose(v)));
      }
      return [...new Uint8Array(ta.buffer)];
    }
    case 'uint64': {
      const ta = new BigUint64Array(n);
      for (let i = 0; i < n; i++) {
        const v = getNorm(i);
        const bi =
          typeof v === 'bigint'
            ? v
            : BigInt.asUintN(64, BigInt(Math.trunc(parseNumericLoose(v))));
        ta[i] = bi;
      }
      return [...new Uint8Array(ta.buffer)];
    }
    default: {
      const ta = new Float32Array(n);
      for (let i = 0; i < n; i++) ta[i] = parseNumericLoose(getNorm(i));
      return [...new Uint8Array(ta.buffer)];
    }
  }
}

export function buildGraphJson(graphResources) {
  const operandNames = new Set(Object.keys(graphResources.inputs ?? {}));
  const graph = {
    format: 'webnn-graph-json',
    version: 2,
    name: 'wpt_graph',
    quantized: false,
    inputs: {},
    consts: {},
    nodes: [],
    outputs: {}
  };

  for (const [name, input] of Object.entries(graphResources.inputs ?? {})) {
    const descriptor = {
      dataType: input.descriptor.dataType,
      shape: input.descriptor.shape
    };

    if (shouldInlineConstant(input)) {
      graph.consts[name] = {
        dataType: input.descriptor.dataType,
        shape: (input.descriptor.shape ?? []).map((d) => Number(d)),
        init: {
          kind: 'inlineBytes',
          bytes: packConstantInlineBytes(input)
        }
      };
    } else {
      graph.inputs[name] = descriptor;
    }
  }

  // Operand name -> index map matching rustnn order. rustnn uses BTreeMap for inputs/consts so
  // iteration is sorted by key; we must use the same order so option indices (e.g. scale=2) match.
  let nextIdx = 0;
  const nameToIndex = new Map();
  for (const name of Object.keys(graph.inputs).sort()) nameToIndex.set(name, nextIdx++);
  for (const name of Object.keys(graph.consts).sort()) nameToIndex.set(name, nextIdx++);

  (graphResources.operators ?? []).forEach((op, index) => {
    const inputs = [];
    const options = {};
    const opName = normalizeOpName(op.name);

    for (const arg of op.arguments ?? []) {
      for (const [key, value] of Object.entries(arg)) {
        if (key === 'options' && value && typeof value === 'object') {
          for (const [optKey, optValue] of Object.entries(value)) {
            const optRefs = flattenInputReferences(optValue, operandNames);
            if (optRefs.length > 0) {
              // MLOperand options stay in options as indices; others become positional inputs.
              if (isOperandOption(opName, optKey)) {
                const idx = nameToIndex.get(optRefs[0]);
                if (idx !== undefined) {
                  options[normalizeOptionKey(op.name, optKey)] = idx;
                }
              } else {
                inputs.push(...optRefs);
              }
              continue;
            }
            options[normalizeOptionKey(op.name, optKey)] = normalizeValue(optValue);
          }
          continue;
        }

        // Positional arguments: operand refs go to inputs, except MLOperand options (e.g. scale/bias) which stay in options only.
        const refs = flattenInputReferences(value, operandNames);
        if (refs.length > 0) {
          if (isOperandOption(opName, key)) {
            const idx = nameToIndex.get(refs[0]);
            if (idx !== undefined) {
              options[normalizeOptionKey(op.name, key)] = idx;
            }
          } else {
            inputs.push(...refs);
          }
          continue;
        }

        options[normalizeOptionKey(op.name, key)] = normalizeValue(value);
      }
    }

    const outputs = Array.isArray(op.outputs) ? op.outputs : [op.outputs];
    for (const out of outputs) {
      if (typeof out === 'string' && out.length > 0) {
        operandNames.add(out);
        if (!nameToIndex.has(out)) nameToIndex.set(out, nextIdx++);
      }
    }

    graph.nodes.push({
      id: `op_${index}`,
      op: normalizeOpName(op.name),
      inputs,
      options,
      outputs
    });
  });

  for (const name of Object.keys(graphResources.expectedOutputs ?? {})) {
    graph.outputs[name] = name;
  }

  return graph;
}

export function buildRuntimeInputs(graphResources) {
  const inputs = {};
  for (const [name, input] of Object.entries(graphResources.inputs ?? {})) {
    if (input.constant === true && shouldInlineConstant(input)) {
      continue;
    }
    let data = Array.isArray(input.data) ? input.data : (input.data == null ? [0] : [input.data]);
    if (data.length === 0) data = [0];
    inputs[name] = {
      descriptor: {
        dataType: input.descriptor.dataType,
        shape: input.descriptor.shape
      },
      data: data.map(normalizeValue)
    };
  }
  return inputs;
}

export function buildExpectedOutputs(graphResources) {
  const outputs = {};
  for (const [name, output] of Object.entries(graphResources.expectedOutputs ?? {})) {
    const data = Array.isArray(output.data) ? output.data : [output.data];
    outputs[name] = {
      descriptor: {
        dataType: output.descriptor.dataType,
        shape: output.descriptor.shape
      },
      data: data.map(normalizeValue)
    };
  }
  return outputs;
}
