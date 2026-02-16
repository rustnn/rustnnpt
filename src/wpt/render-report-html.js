function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function percent(passed, total) {
  if (total === 0) return '0.00';
  return ((passed / total) * 100).toFixed(2);
}

function durationMs(startedAt, endedAt) {
  const start = Date.parse(startedAt);
  const end = Date.parse(endedAt);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return 'n/a';
  return `${((end - start) / 1000).toFixed(2)}s`;
}

function wptSourceUrl(fileName) {
  return `https://github.com/web-platform-tests/wpt/blob/master/webnn/conformance_tests/${encodeURIComponent(fileName)}`;
}

export function renderConformanceHtmlReport(report) {
  const passed = report.summary.passed ?? 0;
  const failed = report.summary.failed ?? 0;
  const skipped = report.summary.skipped ?? 0;
  const total = passed + failed + skipped;
  const generatedAt = report.meta.endedAt ?? new Date().toISOString();
  const rustnnCommit = report.meta?.rustnn?.commit ?? null;
  const rustnnCommitUrl = report.meta?.rustnn?.commitUrl ?? null;
  const runDateText = new Date(generatedAt).toUTCString();
  const rustnnLine = rustnnCommit
    ? (rustnnCommitUrl
      ? `<p class="hero-meta">RustNN commit: <a href="${escapeHtml(rustnnCommitUrl)}">${escapeHtml(rustnnCommit.slice(0, 12))}</a></p>`
      : `<p class="hero-meta">RustNN commit: ${escapeHtml(rustnnCommit.slice(0, 12))}</p>`)
    : '<p class="hero-meta">RustNN commit: unknown</p>';

  const fileSections = report.files.map((file) => {
    const cases = file.cases ?? [];
    const failedCases = cases.filter((c) => c.status === 'fail');
    const skippedCases = cases.filter((c) => c.status === 'skip');
    const sourceUrl = wptSourceUrl(file.fileName);
    const statusLabel = file.summary.failed > 0 ? 'failing' : 'passing';
    const parseErrorBanner = file.fileError
      ? `<p class="file-error">File parse error: ${escapeHtml(file.fileError)}</p>`
      : '';

    const failedSection = failedCases.length > 0
      ? `
        <h4>Failures</h4>
        <table>
          <thead>
            <tr><th>Test</th><th>Variant</th><th>Error</th></tr>
          </thead>
          <tbody>${failedCases.map((c) => `<tr><td>${escapeHtml(c.testName)}</td><td>${escapeHtml(c.variant)}</td><td>${escapeHtml(c.error ?? '')}</td></tr>`).join('\n')}</tbody>
        </table>
      `
      : '';

    const skippedSection = skippedCases.length > 0
      ? `
        <h4>Skipped</h4>
        <table>
          <thead>
            <tr><th>Test</th><th>Variant</th><th>Reason</th></tr>
          </thead>
          <tbody>${skippedCases.map((c) => `<tr><td><a href="${escapeHtml(sourceUrl)}">${escapeHtml(c.testName)}</a></td><td>${escapeHtml(c.variant)}</td><td>${escapeHtml(c.reason ?? '')}</td></tr>`).join('\n')}</tbody>
        </table>
      `
      : '';

    return `
      <section class="file ${statusLabel}">
        <div class="file-head">
          <h3><a href="${escapeHtml(sourceUrl)}">${escapeHtml(file.fileName)}</a></h3>
          <div class="head-metrics">
            <span class="metric passed">${file.summary.passed} passed</span>
            ${file.summary.failed > 0 ? `<span class="metric failed">${file.summary.failed} failed</span>` : ''}
            ${file.summary.skipped > 0 ? `<span class="metric skipped">${file.summary.skipped} skipped</span>` : ''}
          </div>
        </div>
        ${parseErrorBanner}
        ${failedSection}
        ${skippedSection}
      </section>
    `;
  }).join('\n');

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>RustNNPT Conformance Report</title>
    <style>
      :root {
        --bg: #f4f7fb;
        --panel: #ffffff;
        --text: #102a43;
        --muted: #627d98;
        --ok: #1b7f4d;
        --bad: #c23b23;
        --skip: #a06b00;
        --line: #d9e2ec;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Avenir Next", "Segoe UI", sans-serif;
        color: var(--text);
        background: radial-gradient(circle at top right, #d9f5ec, transparent 35%),
                    radial-gradient(circle at 20% 10%, #dbeafe, transparent 45%),
                    var(--bg);
      }
      .wrap {
        max-width: 1100px;
        margin: 0 auto;
        padding: 24px 16px 40px;
      }
      .hero {
        background: linear-gradient(130deg, #0b3c5d, #186f8d);
        color: #fff;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 12px 30px rgba(16, 42, 67, 0.2);
      }
      .hero h1 { margin: 0 0 8px; font-size: 28px; }
      .hero p { margin: 0; opacity: 0.95; }
      .hero-meta { margin-top: 8px !important; font-size: 14px; opacity: 0.95; }
      .hero a { color: #d7f7ff; text-decoration: underline; }
      .cards {
        margin-top: 16px;
        display: grid;
        gap: 12px;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      }
      .card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 14px;
      }
      .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }
      .value { margin-top: 6px; font-size: 26px; font-weight: 700; }
      .ok { color: var(--ok); }
      .bad { color: var(--bad); }
      .skip { color: var(--skip); }
      .files {
        margin-top: 20px;
        display: grid;
        gap: 14px;
      }
      .file {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 16px;
      }
      .file-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
      }
      h3 { margin: 0; }
      h3 a {
        color: inherit;
        text-decoration: none;
      }
      h3 a:hover { text-decoration: underline; }
      .head-metrics {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 8px;
        flex-wrap: wrap;
      }
      .metric {
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 700;
        white-space: nowrap;
      }
      .metric.passed { background: #d9f7e8; color: var(--ok); }
      .metric.failed { background: #ffe2dc; color: var(--bad); }
      .metric.skipped { background: #fff0d1; color: var(--skip); }
      .file-error {
        margin: 10px 0 12px;
        padding: 8px 10px;
        border-radius: 8px;
        border: 1px solid #f5c2c7;
        background: #fff0f1;
        color: #9f1239;
        font-size: 13px;
      }
      h4 {
        margin: 12px 0 8px;
        font-size: 14px;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }
      th, td {
        border: 1px solid var(--line);
        text-align: left;
        padding: 7px 8px;
        vertical-align: top;
      }
      th {
        background: #f8fbff;
      }
    </style>
  </head>
  <body>
    <main class="wrap">
      <section class="hero">
        <h1>RustNNPT WebNN Conformance</h1>
        <p>Date: ${escapeHtml(runDateText)} | Duration ${escapeHtml(durationMs(report.meta.startedAt, report.meta.endedAt))}</p>
        ${rustnnLine}
      </section>

      <section class="cards">
        <article class="card"><div class="label">Total</div><div class="value">${total}</div></article>
        <article class="card"><div class="label">Pass</div><div class="value ok">${passed}</div></article>
        <article class="card"><div class="label">Fail</div><div class="value bad">${failed}</div></article>
        <article class="card"><div class="label">Skip</div><div class="value skip">${skipped}</div></article>
        <article class="card"><div class="label">Pass Rate</div><div class="value">${percent(passed, total)}%</div></article>
        <article class="card"><div class="label">Files</div><div class="value">${report.files.length}</div></article>
      </section>

      <section class="files">
        ${fileSections || '<section class="file"><p>No files were executed.</p></section>'}
      </section>
    </main>
  </body>
</html>
`;
}
