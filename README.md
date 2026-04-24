# rustnnpt

Thin JavaScript WebNN test shim on top of RustNN for upstream WPT conformance execution.

## What this is

- Uses upstream WPT WebNN conformance files (`webnn/conformance_tests/*.https.any.js`)
- Extracts raw test cases from those JS files
- Sends each graph execution to a Rust subprocess (`wpt-runner`) backed by RustNN backends
- Compares outputs with lightweight tolerance checks

This is intentionally test-oriented and not a production WebNN runtime.

## Prerequisites

- Node.js >= 22
- Rust toolchain
- RustNN checked out at `../rustnn` relative to this repo (`/Users/tarek/Dev/rustnn`)

## Commands

```bash
# Fetch/update WPT into .cache/wpt
npm run test:wpt:fetch

# Build runner once
npm run build:runner

# Run one operation file (default backend+device = onnx+cpu)
npm run test:wpt:run -- --op add --limit-tests 20

# Run CoreML (macOS) using backend feature flag
npm run test:wpt:run -- --op add --backend coreml --variants npu --runner-features backend-onnx,backend-coreml

# Run TensorRT mock backend
npm run test:wpt:run -- --op add --backend trtx --variants gpu --runner-features backend-onnx,backend-trtx-mock

# Generate machine-readable + HTML conformance report
npm run test:wpt:report -- --op add --limit-tests 20 --backends onnx,coreml --variants cpu,npu --runner-features backend-onnx,backend-coreml
```

Report outputs:
- `reports/conformance.json` (full structured execution data)
- `reports/conformance.html` (styled dashboard with summary/failures/skips)

## Notes

- Defaults are `backend=onnx` and `variant=cpu` when flags are omitted.
- Available backend names: `onnx`, `coreml`, `trtx`.
- `--runner-features` controls which Rust backend features are enabled when spawning `cargo run`.
- Conformance-only scope in v1 (validation tests are out of scope).
- Numeric comparison currently uses pragmatic tolerances, not full WPT testharness parity.
