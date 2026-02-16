#!/usr/bin/env node
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

import { RunnerClient } from '../bridge/runner-client.js';
import { buildGraphJson } from '../graph/build-graph-json.js';
import { executeGraphResources } from '../shim/webnn-shim.js';
import { extractTestsFromSource } from './extract-tests.js';
import { renderConformanceHtmlReport } from './render-report-html.js';
import { assertOutputClose } from './tolerance.js';

function parseArgs(argv) {
  const opts = {
    wptDir: process.env.WPT_DIR ?? path.join(process.cwd(), '.cache', 'wpt'),
    op: null,
    file: null,
    limitTests: Number.POSITIVE_INFINITY,
    limitFiles: Number.POSITIVE_INFINITY,
    variants: ['cpu'],
    stopOnFail: false,
    reportJson: null,
    reportHtml: null
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--wpt-dir') opts.wptDir = argv[++i];
    else if (arg === '--op') opts.op = argv[++i];
    else if (arg === '--file') opts.file = argv[++i];
    else if (arg === '--limit-tests') opts.limitTests = Number(argv[++i]);
    else if (arg === '--limit-files') opts.limitFiles = Number(argv[++i]);
    else if (arg === '--variants') opts.variants = argv[++i].split(',').map((s) => s.trim()).filter(Boolean);
    else if (arg === '--stop-on-fail') opts.stopOnFail = true;
    else if (arg === '--report-json') opts.reportJson = argv[++i];
    else if (arg === '--report-html') opts.reportHtml = argv[++i];
    else if (arg === '--help') {
      console.log('Usage: node src/wpt/run-conformance.js [--wpt-dir PATH] [--op NAME] [--file FILE] [--limit-tests N] [--limit-files N] [--variants cpu,gpu,npu] [--stop-on-fail] [--report-json PATH] [--report-html PATH]');
      process.exit(0);
    }
  }

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

function shouldSkipTest(test) {
  const inputs = Object.values(test.graph?.inputs ?? {});
  const outputs = Object.values(test.graph?.expectedOutputs ?? {});
  const tensors = [...inputs, ...outputs];

  for (const t of tensors) {
    const dt = t?.descriptor?.dataType;
    if (!SUPPORTED_DTYPES.has(dt)) {
      return `unsupported dataType: ${dt}`;
    }
  }
  return null;
}

function normalizeOpName(opName) {
  return opName.replace(/([a-z0-9])([A-Z])/g, '$1_$2').toLowerCase();
}

function contextOptionsForVariant(variant) {
  return { deviceType: variant };
}

async function runSingleTest({ runner, test, variant }) {
  const graph = test.graph;
  const skipReason = shouldSkipTest(test);
  if (skipReason) {
    return { status: 'skip', reason: skipReason };
  }

  const outputs = await executeGraphResources(runner, graph, contextOptionsForVariant(variant));
  const normalizedGraph = buildGraphJson(graph);
  const lastOp = normalizedGraph.nodes[normalizedGraph.nodes.length - 1]?.op ?? 'unknown';

  for (const [name, expected] of Object.entries(graph.expectedOutputs ?? {})) {
    const actual = outputs[name];
    if (!actual) {
      throw new Error(`missing output: ${name}`);
    }
    assertOutputClose({
      operatorName: normalizeOpName(lastOp),
      outputName: name,
      expected,
      actual
    });
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
    variants: opts.variants,
    stopOnFail: opts.stopOnFail,
    reportJson: opts.reportJson,
    reportHtml: opts.reportHtml
  };
}

async function writeReportFile(filePath, content) {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, content, 'utf8');
}

async function main() {
  const opts = parseArgs(process.argv);

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

  const runner = new RunnerClient();

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
      cwd: process.cwd()
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

  try {
    for (const file of files) {
      const source = await readFile(file, 'utf8');
      const tests = extractTestsFromSource(source, file).slice(0, opts.limitTests);
      const fileReport = {
        filePath: file,
        fileName: path.basename(file),
        selectedTests: tests.length,
        summary: { passed: 0, failed: 0, skipped: 0 },
        cases: []
      };
      report.files.push(fileReport);
      console.log(`\n[FILE] ${fileReport.fileName} (${tests.length} tests)`);

      for (const variant of opts.variants) {
        console.log(`[VARIANT] ${variant} -> cpu-backed runner`);
        for (const test of tests) {
          const started = Date.now();
          try {
            const res = await runSingleTest({ runner, test, variant });
            if (res.status === 'pass') {
              passed += 1;
              fileReport.summary.passed += 1;
              fileReport.cases.push({
                testName: test.name,
                variant,
                status: 'pass',
                durationMs: Date.now() - started
              });
            } else {
              skipped += 1;
              fileReport.summary.skipped += 1;
              fileReport.cases.push({
                testName: test.name,
                variant,
                status: 'skip',
                reason: res.reason,
                durationMs: Date.now() - started
              });
              console.log(`  - SKIP ${test.name}: ${res.reason}`);
            }
          } catch (err) {
            failed += 1;
            fileReport.summary.failed += 1;
            const msg = `${path.basename(file)} :: ${variant} :: ${test.name} :: ${err.message}`;
            failures.push(msg);
            fileReport.cases.push({
              testName: test.name,
              variant,
              status: 'fail',
              error: err.message,
              durationMs: Date.now() - started
            });
            console.log(`  - FAIL ${test.name}`);
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
  } finally {
    await runner.close();
  }

  report.meta.endedAt = new Date().toISOString();
  report.summary.passed = passed;
  report.summary.failed = failed;
  report.summary.skipped = skipped;

  console.log('\n=== SUMMARY ===');
  console.log(`passed=${passed} failed=${failed} skipped=${skipped}`);
  if (failures.length > 0) {
    console.log('\nFirst failures:');
    for (const line of failures.slice(0, 20)) {
      console.log(`- ${line}`);
    }
  }
  if (halted) {
    console.log('\nRun halted early due to --stop-on-fail.');
  }

  if (opts.reportJson) {
    await writeReportFile(opts.reportJson, `${JSON.stringify(report, null, 2)}\n`);
    console.log(`JSON report written: ${opts.reportJson}`);
  }
  if (opts.reportHtml) {
    await writeReportFile(opts.reportHtml, renderConformanceHtmlReport(report));
    console.log(`HTML report written: ${opts.reportHtml}`);
  }

  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
