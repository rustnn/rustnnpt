#!/usr/bin/env node

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

import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

import { RunnerClient } from '../bridge/runner-client.js';
import { executeGraphResources } from '../shim/webnn-shim.js';
import { extractTestsFromSource } from './extract-tests.js';
import { renderConformanceHtmlReport } from './render-report-html.js';
import { assertOutputClose } from './tolerance.js';

/** Format a tensor (descriptor + data) for debug logging; truncate long arrays. */
function formatTensorForLog(label, obj, maxElements = 40) {
  if (!obj) return `${label}: (none)`;
  const desc = obj.descriptor ?? {};
  const data = Array.isArray(obj.data) ? obj.data : (obj.data != null ? [obj.data] : []);
  const shape = desc.shape ?? [];
  const shapeStr = JSON.stringify(shape);
  const total = data.length;
  let dataStr;
  if (total <= maxElements) {
    dataStr = JSON.stringify(data);
  } else {
    const head = data.slice(0, Math.min(12, total));
    const tail = data.slice(-8);
    dataStr = `[${head.join(', ')} ... (${total} total) ... ${tail.join(', ')}]`;
  }
  return `${label} shape=${shapeStr} (${total} elements)\n  data: ${dataStr}`;
}

/** On assertion failure, log test name, inputs, expected outputs, and actual outputs to stderr. */
function logFailureDetail(testName, graph, outputs) {
  const inputs = graph?.inputs ?? {};
  const expectedOutputs = graph?.expectedOutputs ?? {};
  console.error('\n--- FAILURE DETAIL ---');
  console.error(`test: ${testName ?? '(unknown)'}`);
  console.error('\n--- inputs ---');
  for (const [name, input] of Object.entries(inputs)) {
    const data = Array.isArray(input?.data) ? input.data : (input?.data != null ? [input.data] : []);
    console.error(formatTensorForLog(`input "${name}"`, { descriptor: input?.descriptor, data }));
  }
  console.error('\n--- expected outputs ---');
  for (const [name, expected] of Object.entries(expectedOutputs)) {
    console.error(formatTensorForLog(`expected "${name}"`, expected));
  }
  console.error('\n--- actual outputs ---');
  for (const [name, actual] of Object.entries(outputs ?? {})) {
    console.error(formatTensorForLog(`actual "${name}"`, actual));
  }
  console.error('---\n');
}

/**
 * @typedef {{ file: string, backend: string, variant: string, testPrefix: string }} SkiplistEntry
 */

/**
 * Parse one skiplist line: `file.js :: backend/variant :: test name...`
 * @param {string} line
 * @returns {SkiplistEntry | null}
 */
function parseSkiplistLine(line) {
  const idx1 = line.indexOf(' :: ');
  if (idx1 === -1) return null;
  const idx2 = line.indexOf(' :: ', idx1 + 4);
  if (idx2 === -1) return null;
  const file = line.slice(0, idx1).trim();
  const bv = line.slice(idx1 + 4, idx2).trim();
  const testPrefix = line.slice(idx2 + 4).trim();
  const slash = bv.indexOf('/');
  if (slash === -1) return null;
  const backend = bv.slice(0, slash).trim();
  const variant = bv.slice(slash + 1).trim();
  if (!file || !backend || !variant || !testPrefix) return null;
  return { file, backend, variant, testPrefix };
}

/**
 * @param {SkiplistEntry} entry
 * @param {string} fileName basename of WPT file
 * @param {string} backend
 * @param {string} variant
 * @param {string} testName
 */
function matchesSkiplistEntry(entry, fileName, backend, variant, testName) {
  if (fileName !== entry.file) return false;
  if (backend !== entry.backend || variant !== entry.variant) return false;
  return testName === entry.testPrefix || testName.startsWith(entry.testPrefix);
}

/**
 * Load explicit test skips from `test-skiplist.txt` (or a given path).
 * @param {string} filePath
 * @param {{ required?: boolean }} options
 * @returns {Promise<SkiplistEntry[]>}
 */
async function loadTestSkiplist(filePath, { required = false } = {}) {
  if (!existsSync(filePath)) {
    if (required) {
      throw new Error(`skiplist not found: ${filePath}`);
    }
    return [];
  }
  const text = await readFile(filePath, 'utf8');
  /** @type {SkiplistEntry[]} */
  const entries = [];
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const e = parseSkiplistLine(trimmed);
    if (!e) {
      console.warn(`[skiplist] ignoring malformed line: ${trimmed}`);
      continue;
    }
    entries.push(e);
  }
  return entries;
}

/** @param {SkiplistEntry[]} entries */
function skiplistReasonForTest(entries, fileName, backend, variant, testName) {
  for (const e of entries) {
    if (matchesSkiplistEntry(e, fileName, backend, variant, testName)) {
      return `explicit skiplist: ${e.file} :: ${e.backend}/${e.variant} :: ${e.testPrefix}`;
    }
  }
  return null;
}

function parseArgs(argv) {
  const opts = {
    wptDir: process.env.WPT_DIR ?? path.join(process.cwd(), '.cache', 'wpt'),
    op: null,
    file: null,
    limitTests: Number.POSITIVE_INFINITY,
    limitFiles: Number.POSITIVE_INFINITY,
    backends: ['onnx'],
    variants: ['cpu'],
    runnerFeatures: null,
    skipUnimplemented: false,
    stopOnFail: false,
    reportJson: null,
    reportHtml: null,
    exitZero: false,
    /** Max failure lines printed after the run; Infinity = no cap. */
    failureSummaryMax: 20,
    /** If set, load this skiplist file (must exist). If null, use env or default path when present. */
    skiplistPath: null
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--wpt-dir') opts.wptDir = argv[++i];
    else if (arg === '--op') opts.op = argv[++i];
    else if (arg === '--file') opts.file = argv[++i];
    else if (arg === '--limit-tests') opts.limitTests = Number(argv[++i]);
    else if (arg === '--limit-files') opts.limitFiles = Number(argv[++i]);
    else if (arg === '--backend') opts.backends = [String(argv[++i]).trim()].filter(Boolean);
    else if (arg === '--backends') opts.backends = argv[++i].split(',').map((s) => s.trim()).filter(Boolean);
    else if (arg === '--variants') opts.variants = argv[++i].split(',').map((s) => s.trim()).filter(Boolean);
    else if (arg === '--runner-features') opts.runnerFeatures = argv[++i].split(',').map((s) => s.trim()).filter(Boolean);
    else if (arg === '--stop-on-fail') opts.stopOnFail = true;
    else if (arg === '--skip-unimplemented') opts.skipUnimplemented = true;
    else if (arg === '--report-json') opts.reportJson = argv[++i];
    else if (arg === '--report-html') opts.reportHtml = argv[++i];
    else if (arg === '--exit-zero') opts.exitZero = true;
    else if (arg === '--all-failures') opts.failureSummaryMax = Number.POSITIVE_INFINITY;
    else if (arg === '--failure-summary-limit') {
      const n = Number(argv[++i]);
      opts.failureSummaryMax =
        !Number.isFinite(n) || n <= 0 ? Number.POSITIVE_INFINITY : Math.floor(n);
    } else if (arg === '--skiplist') opts.skiplistPath = argv[++i];
    else if (arg === '--help') {
      console.log(
        'Usage: node src/wpt/run-conformance.js [options]\n' +
          '  [--wpt-dir PATH] [--op NAME] [--file FILE] [--limit-tests N] [--limit-files N]\n' +
          '  [--backend onnx|coreml|trtx] [--backends LIST] [--variants cpu,gpu,npu]\n' +
          '  [--runner-features LIST] [--skip-unimplemented] [--stop-on-fail]\n' +
          '  [--report-json PATH] [--report-html PATH] [--exit-zero]\n' +
          '  [--all-failures | --failure-summary-limit N]  (default: first 20 failures; N<=0 means all)\n' +
          '  [--skiplist PATH]  (optional; default: ./test-skiplist.txt if present, or RUSTNNPT_TEST_SKIPLIST)\n' +
          '  [--debug]'
      );
      process.exit(0);
    }
    else if (arg === '--debug') {
      process.env.RUSTNNPT_DEBUG = '2';
      process.env.RUSTNN_DEBUG = '2';
      // ONNX debug dump (when RUSTNN_DEBUG=2) writes to this folder
      process.env.RUSTNN_DEBUG_ONNX_DIR = process.env.RUSTNN_DEBUG_ONNX_DIR ?? 'C:\\git\\rustnn-workspace\\rustnnpt';
    }
  }

  if (opts.backends.length === 0) opts.backends = ['onnx'];
  if (opts.variants.length === 0) opts.variants = ['cpu'];

  return opts;
}

async function listConformanceFiles(wptDir) {
  const { readdir } = await import('node:fs/promises');
  const base = path.join(wptDir, 'webnn', 'conformance_tests');
  const names = await readdir(base);
  return names.filter((n) => n.endsWith('.https.any.js')).sort().map((n) => path.join(base, n));
}

const SUPPORTED_DTYPES = new Set([
  'float32', 'float16', 'int8', 'uint8', 'int32', 'uint32', 'int64', 'uint64', 'int4', 'uint4'
]);

// Keep this set aligned with rustnn's implementation-status docs.
const UNIMPLEMENTED_OPS = new Set([
  'is_nan',
  'l2_pool2d',
  // Intentionally deferred in rustnn.
  'gru',
  'gru_cell',
  'lstm',
  'lstm_cell'
]);

function collectUnimplementedOps(test) {
  const ops = Array.isArray(test?.graph?.operators) ? test.graph.operators : [];
  const missing = new Set();
  for (const op of ops) {
    const normalized = normalizeOpName(op?.name ?? '');
    if (UNIMPLEMENTED_OPS.has(normalized)) {
      missing.add(normalized);
    }
  }
  return [...missing];
}

function shouldSkipTest(test, opts) {
  const inputs = Object.values(test.graph?.inputs ?? {});
  const outputs = Object.values(test.graph?.expectedOutputs ?? {});
  const tensors = [...inputs, ...outputs];

  for (const t of tensors) {
    const dt = t?.descriptor?.dataType;
    if (!SUPPORTED_DTYPES.has(dt)) {
      return `unsupported dataType: ${dt}`;
    }
  }

  if (opts.skipUnimplemented) {
    const missingOps = collectUnimplementedOps(test);
    if (missingOps.length > 0) {
      return `unimplemented op(s): ${missingOps.join(', ')}`;
    }
  }
  return null;
}

function normalizeOpName(opName) {
  return opName.replace(/([a-z0-9])([A-Z])/g, '$1_$2').toLowerCase();
}

function contextOptionsForRun(backend, variant) {
  return { backend, deviceType: variant };
}

function isRunnerCrashError(err) {
  const msg = String(err?.message ?? '');
  return msg.includes('runner exited')
    || msg.includes('stream was destroyed')
    || msg.includes('failed to send request to runner')
    || msg.includes('runner stdin error')
    || msg.includes('EPIPE')
    || msg.includes('SIGTRAP');
}

async function runSingleTest({ runner, test, backend, variant, opts, testName }) {
  const graph = test.graph;
  const skipReason = shouldSkipTest(test, opts);
  if (skipReason) {
    return { status: 'skip', reason: skipReason };
  }

  const outputs = await executeGraphResources(runner, graph, contextOptionsForRun(backend, variant));
  const lastOp = normalizeOpName(graph?.operators?.[graph.operators.length - 1]?.name ?? 'unknown');
  const graphOperatorNames = (graph.operators ?? []).map((o) => normalizeOpName(o?.name ?? ''));

  try {
    for (const [name, expected] of Object.entries(graph.expectedOutputs ?? {})) {
      const actual = outputs[name];
      if (!actual) {
        throw new Error(`missing output: ${name}`);
      }
      assertOutputClose({
        operatorName: lastOp,
        graphOperatorNames,
        outputName: name,
        expected,
        actual
      });
    }
  } catch (err) {
    logFailureDetail(testName, graph, outputs);
    throw err;
  }

  return { status: 'pass' };
}

function serializeOptions(opts) {
  const numberOrNull = (value) => Number.isFinite(value) ? value : null;
  return {
    wptDir: opts.wptDir,
    op: opts.op,
    file: opts.file,
    limitTests: numberOrNull(opts.limitTests),
    limitFiles: numberOrNull(opts.limitFiles),
    backends: opts.backends,
    variants: opts.variants,
    runnerFeatures: opts.runnerFeatures,
    skipUnimplemented: opts.skipUnimplemented,
    stopOnFail: opts.stopOnFail,
    reportJson: opts.reportJson,
    reportHtml: opts.reportHtml,
    exitZero: opts.exitZero,
    failureSummaryMax: numberOrNull(opts.failureSummaryMax),
    skiplistResolvedPath: opts.skiplistResolvedPath ?? null,
    skiplistEntryCount: opts.skiplistEntryCount ?? 0
  };
}

function rustnnMetaFromEnv() {
  const commit = process.env.RUSTNN_GIT_SHA ?? null;
  const commitUrl = process.env.RUSTNN_GIT_URL ?? null;
  return { commit, commitUrl };
}

async function writeReportFile(filePath, content) {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, content, 'utf8');
}

async function main() {
  const opts = parseArgs(process.argv);

  const defaultSkiplistPath = path.join(process.cwd(), 'test-skiplist.txt');
  const skiplistExplicit = opts.skiplistPath;
  const skiplistPath =
    skiplistExplicit
    ?? (process.env.RUSTNNPT_TEST_SKIPLIST
      ? path.resolve(process.cwd(), process.env.RUSTNNPT_TEST_SKIPLIST)
      : defaultSkiplistPath);
  const skiplistEntries = await loadTestSkiplist(skiplistPath, { required: Boolean(skiplistExplicit) });
  opts.skiplistResolvedPath = existsSync(skiplistPath) ? skiplistPath : null;
  opts.skiplistEntryCount = skiplistEntries.length;
  if (skiplistEntries.length > 0) {
    console.log(`[skiplist] loaded ${skiplistEntries.length} entr${skiplistEntries.length === 1 ? 'y' : 'ies'} from ${skiplistPath}`);
  }

  if (!existsSync(path.join(opts.wptDir, 'webnn'))) {
    console.error(`WPT not found at ${opts.wptDir}. Run: npm run test:wpt:fetch`);
    process.exit(2);
  }

  let files = await listConformanceFiles(opts.wptDir);
  if (opts.file) {
    files = files.filter((f) => f.endsWith(opts.file));
  }
  if (opts.op) {
    files = files.filter((f) => path.basename(f).startsWith(`${opts.op}.`) || path.basename(f).startsWith(`${opts.op}_`));
  }
  files = files.slice(0, opts.limitFiles);

  if (files.length === 0) {
    console.error('No matching conformance files.');
    process.exit(2);
  }

  let runner = new RunnerClient({ runnerFeatures: opts.runnerFeatures ?? [] });

  let passed = 0;
  let failed = 0;
  let skipped = 0;
  const failures = [];
  const startedAt = new Date().toISOString();
  const report = {
    meta: {
      startedAt,
      endedAt: null,
      options: serializeOptions(opts),
      cwd: process.cwd(),
      rustnn: rustnnMetaFromEnv()
    },
    summary: {
      passed: 0,
      failed: 0,
      skipped: 0
    },
    files: [],
    failures
  };
  let halted = false;
  let fatalError = null;

  try {
    for (const file of files) {
      const fileReport = {
        fileName: path.basename(file),
        selectedTests: 0,
        summary: { passed: 0, failed: 0, skipped: 0 },
        cases: [],
        fileError: null
      };
      report.files.push(fileReport);
      let tests = [];

      try {
        const source = await readFile(file, 'utf8');
        tests = extractTestsFromSource(source, path.basename(file)).slice(0, opts.limitTests);
      } catch (err) {
        fileReport.fileError = err.message;
        const skipReason = err.message.includes('No <name>Tests array');
        if (skipReason) {
          skipped += 1;
          fileReport.summary.skipped += 1;
          fileReport.cases.push({
            testName: '<file>',
            backend: opts.backends[0],
            variant: opts.variants[0],
            status: 'skip',
            reason: err.message
          });
          console.log(`\n[FILE] ${fileReport.fileName} (non-graph test)`);
          console.log(`  - SKIP ${err.message}`);
          continue;
        }
        failures.push(`${fileReport.fileName} :: FILE_PARSE :: ${err.message}`);
        failed += 1;
        fileReport.summary.failed += 1;
        console.log(`\n[FILE] ${fileReport.fileName} (parse error)`);
        console.log(`  - FAIL file parse: ${err.message}`);
        continue;
      }

      fileReport.selectedTests = tests.length;
      console.log(`\n[FILE] ${fileReport.fileName} (${tests.length} tests)`);

      for (const backend of opts.backends) {
        for (const variant of opts.variants) {
          console.log(`[RUN] backend=${backend} variant=${variant}`);
          for (let testIndex = 0; testIndex < tests.length; testIndex += 1) {
            const test = tests[testIndex];
            const testName = test?.name ?? `[unnamed-${testIndex}]`;
            const started = Date.now();
            if (!test || typeof test !== 'object' || !test.graph) {
              skipped += 1;
              fileReport.summary.skipped += 1;
              fileReport.cases.push({
                testName,
                backend,
                variant,
                status: 'skip',
                reason: 'invalid extracted test case',
                durationMs: 0
              });
              console.log(`  - SKIP ${testName}: invalid extracted test case`);
              continue;
            }
            const skiplistReason = skiplistReasonForTest(
              skiplistEntries,
              fileReport.fileName,
              backend,
              variant,
              testName
            );
            if (skiplistReason) {
              skipped += 1;
              fileReport.summary.skipped += 1;
              fileReport.cases.push({
                testName,
                backend,
                variant,
                status: 'skip',
                reason: skiplistReason,
                durationMs: 0
              });
              console.log(`  - SKIP ${testName}: ${skiplistReason}`);
              continue;
            }
            try {
              const res = await runSingleTest({ runner, test, backend, variant, opts, testName });
              if (res.status === 'pass') {
                passed += 1;
                fileReport.summary.passed += 1;
                fileReport.cases.push({
                  testName,
                  backend,
                  variant,
                  status: 'pass',
                  durationMs: Date.now() - started
                });
              } else {
                skipped += 1;
                fileReport.summary.skipped += 1;
                fileReport.cases.push({
                  testName,
                  backend,
                  variant,
                  status: 'skip',
                  reason: res.reason,
                  durationMs: Date.now() - started
                });
                console.log(`  - SKIP ${testName}: ${res.reason}`);
              }
            } catch (err) {
              failed += 1;
              fileReport.summary.failed += 1;
              const msg = `${path.basename(file)} :: ${backend}/${variant} :: ${testName} :: ${err.message}`;
              failures.push(msg);
              fileReport.cases.push({
                testName,
                backend,
                variant,
                status: 'fail',
                error: err.message,
                durationMs: Date.now() - started
              });
              console.log(`  - FAIL ${testName}`);
              if (isRunnerCrashError(err)) {
                await runner.close();
                runner = new RunnerClient({ runnerFeatures: opts.runnerFeatures ?? [] });
                console.log('  - INFO restarted runner after backend crash');
              }
              if (opts.stopOnFail) {
                halted = true;
                break;
              }
            }
          }
          if (halted) break;
        }
        if (halted) break;
      }
      if (halted) break;
    }
  } catch (err) {
    fatalError = err;
    failed += 1;
    failures.push(`FATAL :: ${err.message}`);
  } finally {
    await runner.close();
  }

  report.meta.endedAt = new Date().toISOString();
  report.summary.passed = passed;
  report.summary.failed = failed;
  report.summary.skipped = skipped;
  report.summary.total = passed + failed + skipped;
  const denomWithoutSkips = passed + failed;
  const passRate = report.summary.total > 0 ? (passed / report.summary.total) * 100 : 0;
  const passRateExcludingSkips = denomWithoutSkips > 0 ? (passed / denomWithoutSkips) * 100 : 0;
  report.summary.passRatePct = Number(passRate.toFixed(1));
  report.summary.passRateExcludingSkipsPct = Number(passRateExcludingSkips.toFixed(1));

  console.log('\n=== SUMMARY ===');
  console.log(`passed=${passed} failed=${failed} skipped=${skipped} passRate=${report.summary.passRatePct}%`);
  if (failures.length > 0) {
    const max = opts.failureSummaryMax;
    const limit = !Number.isFinite(max) || max <= 0 ? failures.length : max;
    const lines = failures.slice(0, limit);
    const suffix =
      lines.length < failures.length ? ` (showing ${lines.length} of ${failures.length})` : '';
    console.log(`\nFailures${suffix}:`);
    for (const line of lines) {
      console.log(`- ${line}`);
    }
  }
  if (halted) {
    console.log('\nRun halted early due to --stop-on-fail.');
  }
  if (fatalError) {
    console.log(`\nFatal error: ${fatalError.message}`);
  }

  if (opts.reportJson) {
    await writeReportFile(opts.reportJson, `${JSON.stringify(report, null, 2)}\n`);
    console.log(`JSON report written: ${opts.reportJson}`);
  }
  if (opts.reportHtml) {
    await writeReportFile(opts.reportHtml, renderConformanceHtmlReport(report));
    console.log(`HTML report written: ${opts.reportHtml}`);
  }

  process.exitCode = (failed > 0 || fatalError) && !opts.exitZero ? 1 : 0;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
