// IEEE float bits reinterpreted as integers for ordered ULP distance (see Hamming / ordering trick).
const F32_SIGN_MASK = 0x8000_0000;
/** Exponent + fraction: all bits except the sign bit. */
const F32_NOT_SIGN_MASK = 0x7fff_ffff;
const F16_SIGN_MASK = 0x8000;
/** Exponent + fraction: all bits except the sign bit. */
const F16_NOT_SIGN_MASK = 0x7fff;

/** `f32` and `u32` share one 4-byte buffer; do not reuse concurrently across overlapping calls. */
/** @param {number} v @param {{ f32: Float32Array, u32: Uint32Array }} scratch */
function float32Bits(v, scratch) {
  scratch.f32[0] = v;
  return scratch.u32[0];
}

/** @param {number} a @param {number} b @param {{ f32: Float32Array, u32: Uint32Array }} scratch */
function ulpDistanceF32(a, b, scratch) {
  if (Object.is(a, b)) return 0;
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    return a === b ? 0 : Number.POSITIVE_INFINITY;
  }
  const aBits = float32Bits(a, scratch);
  const bBits = float32Bits(b, scratch);
  const toOrdered = (bits) =>
    bits & F32_SIGN_MASK ? F32_SIGN_MASK - (bits & F32_NOT_SIGN_MASK) : bits + F32_SIGN_MASK;
  return Math.abs(toOrdered(aBits) - toOrdered(bBits));
}

/** `f16` and `u16` share one 2-byte buffer; do not reuse concurrently across overlapping calls. */
/** @param {number} v @param {{ f16: Float16Array, u16: Uint16Array }} scratch */
function float16Bits(v, scratch) {
  scratch.f16[0] = v;
  return scratch.u16[0];
}

/** ULP distance in IEEE binary16; required for float16 outputs (f32 ULP inflates ~1 f16 step to thousands). */
/** @param {number} a @param {number} b @param {{ f16: Float16Array, u16: Uint16Array }} scratch */
function ulpDistanceF16(a, b, scratch) {
  if (Object.is(a, b)) return 0;
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    return a === b ? 0 : Number.POSITIVE_INFINITY;
  }
  const aBits = float16Bits(a, scratch);
  const bBits = float16Bits(b, scratch);
  const toOrdered = (bits) =>
    bits & F16_SIGN_MASK ? F16_SIGN_MASK - (bits & F16_NOT_SIGN_MASK) : bits + F16_SIGN_MASK;
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

  let ulpTol = OP_ULP[operatorName];
  if (ulpTol === undefined) ulpTol = 4;
  // Subgraphs use the last op for tolerance (e.g. conv2d + relu → relu). relu/reduce_* use 0 = "exact"
  // for that op, but float16 error still comes from earlier ops; allow small f16 ULP in that case.
  if (dataType === 'float16' && ulpTol === 0) ulpTol = 4;

  let f32BitScratch;
  let f16BitScratch;
  if (dataType === 'float16') {
    const f16 = new Float16Array(1);
    f16BitScratch = { f16, u16: new Uint16Array(f16.buffer) };
  } else {
    const f32 = new Float32Array(1);
    f32BitScratch = { f32, u32: new Uint32Array(f32.buffer) };
  }

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
    let ulp;
    if (dataType === 'float16') {
      ulp = ulpDistanceF16(a, e, f16BitScratch);
    } else {
      ulp = ulpDistanceF32(a, e, f32BitScratch);
    }

    if (absDiff > absTol && ulp > ulpTol) {
      throw new Error(
        `value mismatch for ${outputName}[${i}]: expected ${e}, got ${a}, absDiff=${absDiff}, ulp=${ulp}, ulpTol=${ulpTol}`
      );
    }
  }
}
