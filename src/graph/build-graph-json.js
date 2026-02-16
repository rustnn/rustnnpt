function normalizeOpName(name) {
  return name;
}

function normalizeValue(v) {
  if (typeof v === 'bigint') return v.toString();
  if (Array.isArray(v)) return v.map(normalizeValue);
  if (v && typeof v === 'object') {
    const out = {};
    for (const [k, val] of Object.entries(v)) out[k] = normalizeValue(val);
    return out;
  }
  return v;
}

function flattenInputReferences(argumentValue) {
  if (typeof argumentValue === 'string') {
    return [argumentValue];
  }
  if (Array.isArray(argumentValue) && argumentValue.every((x) => typeof x === 'string')) {
    return argumentValue;
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

  (graphResources.operators ?? []).forEach((op, index) => {
    const inputs = [];
    const options = {};

    for (const arg of op.arguments ?? []) {
      for (const [key, value] of Object.entries(arg)) {
        if (key === 'options' && value && typeof value === 'object') {
          Object.assign(options, normalizeOptions(value));
          continue;
        }

        const refs = flattenInputReferences(value);
        if (refs.length > 0) {
          inputs.push(...refs);
          continue;
        }

        options[key] = normalizeValue(value);
      }
    }

    const outputs = Array.isArray(op.outputs) ? op.outputs : [op.outputs];

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
      data: data.map((v) => (typeof v === 'bigint' ? v.toString() : v))
    };
  }
  return inputs;
}

export function buildExpectedOutputs(graphResources) {
  const outputs = {};
  for (const [name, output] of Object.entries(graphResources.expectedOutputs ?? {})) {
    outputs[name] = {
      descriptor: {
        dataType: output.descriptor.dataType,
        shape: output.descriptor.shape
      }
    };
  }
  return outputs;
}
