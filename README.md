# rustnnpt

Thin JavaScript WebNN test shim on top of RustNN for upstream WPT conformance execution.

## What this is

- Uses upstream WPT WebNN conformance files (`webnn/conformance_tests/*.https.any.js`)
- Extracts raw test cases from those JS files
- Sends each graph execution to a Rust subprocess (`wpt-runner`) backed by RustNN + ONNX Runtime
- Compares outputs with lightweight tolerance checks

This is intentionally test-oriented and not a production WebNN runtime.

## Prerequisites

- Node.js >= 20
- Rust toolchain
- RustNN checked out at `../rustnn` relative to this repo (`/Users/tarek/Dev/rustnn`)

## Commands

```bash
# Fetch/update WPT into .cache/wpt
npm run test:wpt:fetch

# Build runner once
npm run build:runner

# Run one operation file (recommended smoke)
npm run test:wpt:run -- --op add --limit-tests 20 --variants cpu

# Run a specific file
npm run test:wpt:run -- --file add.https.any.js --variants cpu,gpu,npu

# Generate machine-readable + HTML conformance report
npm run test:wpt:report -- --op add --limit-tests 20 --variants cpu
```

Report outputs:
- `reports/conformance.json` (full structured execution data)
- `reports/conformance.html` (styled dashboard with summary/failures/skips)

## Notes

- `cpu/gpu/npu` variants are all mapped to the same CPU-backed execution path in v1.
- Conformance-only scope in v1 (validation tests are out of scope).
- Numeric comparison currently uses pragmatic tolerances, not full WPT testharness parity.
