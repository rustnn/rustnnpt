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
  if (opName === 'cast' && key === 'type') return 'to';
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

    if (input.constant === true) {
      // Keep constants inline as scalar/array values would require typed packing.
      // For WPT execution we pass these as regular runtime inputs.
      graph.inputs[name] = descriptor;
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
    const data = Array.isArray(input.data) ? input.data : [input.data];
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
