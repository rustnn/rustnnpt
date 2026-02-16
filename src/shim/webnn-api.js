import { buildExpectedOutputs, buildRuntimeInputs } from '../graph/build-graph-json.js';

function product(shape) {
  return shape.reduce((a, b) => a * b, 1);
}

function typedArrayCtor(dataType) {
  switch (dataType) {
    case 'float32': return Float32Array;
    case 'float16': return Uint16Array;
    case 'int8': return Int8Array;
    case 'uint8':
    case 'uint4': return Uint8Array;
    case 'int32':
    case 'int4': return Int32Array;
    case 'uint32': return Uint32Array;
    case 'int64': return BigInt64Array;
    case 'uint64': return BigUint64Array;
    default: return Float32Array;
  }
}

function makeZeroData(descriptor) {
  const Ctor = typedArrayCtor(descriptor.dataType);
  const size = Math.max(product(descriptor.shape), 1);
  return new Ctor(size);
}

function toPlainArray(typed, dataType) {
  if (dataType === 'int64' || dataType === 'uint64') {
    return Array.from(typed, (v) => v.toString());
  }
  return Array.from(typed);
}

class MLOperand {
  constructor(name, descriptor) {
    this.name = name;
    this.dataType = descriptor.dataType;
    this.shape = descriptor.shape.slice();
  }
}

class MLGraph {
  constructor(graphJson, outputDescriptors) {
    this.graphJson = graphJson;
    this.outputDescriptors = outputDescriptors;
  }
}

class MLTensor {
  constructor(descriptor) {
    this.descriptor = { dataType: descriptor.dataType, shape: descriptor.shape.slice() };
    this.data = makeZeroData(this.descriptor);
    this.pending = null;
  }
}

export class MLContext {
  constructor(runnerClient, options = {}) {
    this.runnerClient = runnerClient;
    this.options = options;
  }

  async createTensor(descriptor) {
    return new MLTensor(descriptor);
  }

  writeTensor(tensor, data) {
    if (ArrayBuffer.isView(data)) {
      tensor.data = new (typedArrayCtor(tensor.descriptor.dataType))(data);
    } else if (data instanceof ArrayBuffer) {
      tensor.data = new (typedArrayCtor(tensor.descriptor.dataType))(data);
    } else {
      tensor.data = new (typedArrayCtor(tensor.descriptor.dataType))(data);
    }
  }

  async readTensor(tensor) {
    if (tensor.pending) {
      await tensor.pending;
    }
    return tensor.data.buffer.slice(0);
  }

  dispatch(graph, inputs, outputs) {
    const inputPayload = {};
    for (const [name, tensor] of Object.entries(inputs)) {
      inputPayload[name] = {
        descriptor: tensor.descriptor,
        data: toPlainArray(tensor.data, tensor.descriptor.dataType)
      };
    }

    const expectedOutputs = {};
    for (const [name, tensor] of Object.entries(outputs)) {
      expectedOutputs[name] = { descriptor: tensor.descriptor };
    }

    const pending = this.runnerClient.executeGraph({
      graph: graph.graphJson,
      inputs: inputPayload,
      expectedOutputs,
      contextOptions: this.options
    }).then((result) => {
      for (const [name, tensor] of Object.entries(outputs)) {
        const out = result[name];
        const Ctor = typedArrayCtor(tensor.descriptor.dataType);
        tensor.data = new Ctor(out.data);
      }
    });

    for (const tensor of Object.values(outputs)) {
      tensor.pending = pending;
    }
  }

  opSupportLimits() {
    const types = ['float32', 'float16', 'int32', 'uint32', 'int8', 'uint8', 'int64', 'uint64', 'int4', 'uint4'];
    return {
      input: { dataTypes: types },
      output: { dataTypes: types },
      cast: {
        input: { dataTypes: types },
        output: { dataTypes: types }
      }
    };
  }
}

export class MLGraphBuilder {
  constructor(context) {
    this.context = context;
    this.inputs = {};
    this.consts = {};
    this.nodes = [];
    this.tempId = 0;

    return new Proxy(this, {
      get: (target, prop, receiver) => {
        if (typeof prop === 'string' && !(prop in target)) {
          return (...args) => target._op(prop, ...args);
        }
        return Reflect.get(target, prop, receiver);
      }
    });
  }

  input(name, descriptor) {
    this.inputs[name] = { data: [], descriptor, constant: false };
    return new MLOperand(name, descriptor);
  }

  constant(descriptor, data) {
    const name = `const_${this.tempId++}`;
    this.inputs[name] = {
      data: ArrayBuffer.isView(data) ? Array.from(data) : Array.from(data),
      descriptor,
      constant: true
    };
    return new MLOperand(name, descriptor);
  }

  _op(name, ...args) {
    const opArgs = [];
    const inputRefs = [];
    let options = null;

    for (const arg of args) {
      if (arg instanceof MLOperand) {
        inputRefs.push(arg.name);
      } else if (Array.isArray(arg) && arg.every((x) => x instanceof MLOperand)) {
        inputRefs.push(arg.map((x) => x.name));
      } else if (arg && typeof arg === 'object') {
        options = arg;
      }
    }

    inputRefs.forEach((ref, index) => {
      if (Array.isArray(ref)) {
        opArgs.push({ inputs: ref });
      } else {
        const key = index === 0 ? 'a' : 'b';
        opArgs.push({ [key]: ref });
      }
    });
    if (options) {
      opArgs.push({ options });
    }

    const outName = `tmp_${this.tempId++}`;
    this.nodes.push({
      id: `op_${this.nodes.length}`,
      op: name,
      arguments: opArgs,
      outputs: outName
    });

    return new MLOperand(outName, { dataType: 'float32', shape: [] });
  }

  async build(namedOutputs) {
    const graphResources = {
      inputs: this.inputs,
      operators: this.nodes.map((node) => ({ name: node.op, arguments: node.arguments, outputs: node.outputs })),
      expectedOutputs: {}
    };

    for (const [name, operand] of Object.entries(namedOutputs)) {
      graphResources.expectedOutputs[name] = {
        data: [],
        descriptor: {
          dataType: operand.dataType,
          shape: operand.shape
        }
      };
      if (operand.name !== name && graphResources.operators.length > 0) {
        graphResources.operators[graphResources.operators.length - 1].outputs = name;
      }
    }

    const graphJson = {
      format: 'webnn-graph-json',
      version: 2,
      name: 'ml_builder_graph',
      quantized: false,
      inputs: {},
      consts: {},
      nodes: [],
      outputs: {}
    };

    for (const [name, input] of Object.entries(graphResources.inputs)) {
      graphJson.inputs[name] = input.descriptor;
    }

    for (const [index, op] of graphResources.operators.entries()) {
      const inputs = [];
      const options = {};
      for (const arg of op.arguments) {
        for (const [k, v] of Object.entries(arg)) {
          if (k === 'options') {
            Object.assign(options, v);
          } else if (Array.isArray(v)) {
            inputs.push(...v);
          } else {
            inputs.push(v);
          }
        }
      }
      graphJson.nodes.push({ id: `op_${index}`, op: op.name, inputs, options, outputs: [op.outputs] });
    }

    for (const name of Object.keys(graphResources.expectedOutputs)) {
      graphJson.outputs[name] = name;
    }

    const expectedOutputs = buildExpectedOutputs(graphResources);
    return new MLGraph(graphJson, expectedOutputs);
  }
}

export class ML {
  constructor(runnerClient) {
    this.runnerClient = runnerClient;
  }

  async createContext(options = {}) {
    return new MLContext(this.runnerClient, options);
  }
}

export function installWebNNGlobal(runnerClient, globalObj = globalThis) {
  if (!globalObj.navigator) {
    globalObj.navigator = {};
  }
  globalObj.navigator.ml = new ML(runnerClient);
  globalObj.MLGraphBuilder = MLGraphBuilder;
}

export function graphResourcesToProtocol(graphResources) {
  return {
    inputs: buildRuntimeInputs(graphResources),
    expectedOutputs: buildExpectedOutputs(graphResources)
  };
}
