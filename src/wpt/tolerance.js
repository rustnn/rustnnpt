function float32Bits(v) {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = v;
  return u32[0];
}

function ulpDistanceF32(a, b) {
  if (Object.is(a, b)) return 0;
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    return a === b ? 0 : Number.POSITIVE_INFINITY;
  }
  const aBits = float32Bits(a);
  const bBits = float32Bits(b);
  const toOrdered = (bits) => (bits & 0x80000000 ? 0x80000000 - (bits & 0x7fffffff) : bits + 0x80000000);
  return Math.abs(toOrdered(aBits) - toOrdered(bBits));
}

const OP_ULP = {
  add: 1,
  sub: 1,
  mul: 1,
  div: 2,
  relu: 0,
  sigmoid: 34,
  tanh: 16,
  softmax: 256,
  matmul: 512,
  exp: 4,
  log: 4,
  sqrt: 2,
  reduce_sum: 8,
  reduce_mean: 16,
  reduce_max: 0,
  reduce_min: 0,
  reduce_product: 32,
  reduce_l1: 8,
  reduce_l2: 16,
  reduce_log_sum: 16,
  reduce_log_sum_exp: 32,
  reduce_sum_square: 16
};

const OP_ABS_TOL = {
  cos: { float32: 2 ** -10, float16: 2 ** -7 },
  sin: { float32: 2 ** -11, float16: 2 ** -7 }
};

function castActual(value, dataType) {
  if (dataType === 'float16') {
    // Runner returns uint16 bits for float16. Compare against expected numeric by converting expected to f32 tolerance path.
    return Number(value);
  }
  if (dataType === 'int64' || dataType === 'uint64') {
    return BigInt(value);
  }
  return Number(value);
}

function castExpected(value, dataType) {
  if (dataType === 'int64' || dataType === 'uint64') {
    return BigInt(value);
  }
  return Number(value);
}

export function assertOutputClose({ operatorName, outputName, expected, actual }) {
  const dataType = expected.descriptor.dataType;
  const expectedData = Array.isArray(expected.data) ? expected.data : [expected.data];
  const actualData = actual.data ?? [];

  if (expectedData.length !== actualData.length) {
    throw new Error(`length mismatch for ${outputName}: expected ${expectedData.length}, got ${actualData.length}`);
  }

  const expectedShape = JSON.stringify(expected.descriptor.shape);
  const actualShape = JSON.stringify(actual.descriptor.shape);
  if (expectedShape !== actualShape) {
    throw new Error(`shape mismatch for ${outputName}: expected ${expectedShape}, got ${actualShape}`);
  }

  if (dataType === 'int64' || dataType === 'uint64') {
    for (let i = 0; i < expectedData.length; i += 1) {
      const a = castActual(actualData[i], dataType);
      const e = castExpected(expectedData[i], dataType);
      if (a !== e) {
        throw new Error(`value mismatch for ${outputName}[${i}]: expected ${e}, got ${a}`);
      }
    }
    return;
  }

  const ulpTol = OP_ULP[operatorName] ?? 4;
  for (let i = 0; i < expectedData.length; i += 1) {
    const a = Number(actualData[i]);
    const e = Number(expectedData[i]);

    if (Number.isNaN(e) && Number.isNaN(a)) continue;
    if (!Number.isFinite(e) || !Number.isFinite(a)) {
      if (e !== a) {
        throw new Error(`value mismatch for ${outputName}[${i}]: expected ${e}, got ${a}`);
      }
      continue;
    }

    const absDiff = Math.abs(a - e);
    const absTol = OP_ABS_TOL[operatorName]?.[dataType] ?? (dataType.startsWith('float') ? 1e-4 : 0);
    const ulp = ulpDistanceF32(a, e);

    if (absDiff > absTol && ulp > ulpTol) {
      throw new Error(
        `value mismatch for ${outputName}[${i}]: expected ${e}, got ${a}, absDiff=${absDiff}, ulp=${ulp}, ulpTol=${ulpTol}`
      );
    }
  }
}
