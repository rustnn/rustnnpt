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
    const statusLabel = file.summary.failed > 0 ? 'failing' : 'passing';
    const parseErrorBanner = file.fileError
      ? `<p class="file-error">File parse error: ${escapeHtml(file.fileError)}</p>`
      : '';

    const failedRows = failedCases.length === 0
      ? '<tr><td colspan="4">No failures in this file.</td></tr>'
      : failedCases.map((c) => `<tr><td>${escapeHtml(c.testName)}</td><td>${escapeHtml(c.variant)}</td><td>${escapeHtml(c.error ?? '')}</td><td>${c.durationMs ?? ''}</td></tr>`).join('\n');

    const skippedRows = skippedCases.length === 0
      ? '<tr><td colspan="4">No skipped tests in this file.</td></tr>'
      : skippedCases.map((c) => `<tr><td>${escapeHtml(c.testName)}</td><td>${escapeHtml(c.variant)}</td><td>${escapeHtml(c.reason ?? '')}</td><td>${c.durationMs ?? ''}</td></tr>`).join('\n');

    return `
      <section class="file ${statusLabel}">
        <div class="file-head">
          <h3>${escapeHtml(file.fileName)}</h3>
          <div class="pill ${statusLabel}">${statusLabel}</div>
        </div>
        <p class="meta">${escapeHtml(file.filePath)}</p>
        <div class="file-summary">
          <span>passed: ${file.summary.passed}</span>
          <span>failed: ${file.summary.failed}</span>
          <span>skipped: ${file.summary.skipped}</span>
        </div>
        ${parseErrorBanner}
        <h4>Failures</h4>
        <table>
          <thead>
            <tr><th>Test</th><th>Variant</th><th>Error</th><th>Duration</th></tr>
          </thead>
          <tbody>${failedRows}</tbody>
        </table>
        <h4>Skipped</h4>
        <table>
          <thead>
            <tr><th>Test</th><th>Variant</th><th>Reason</th><th>Duration</th></tr>
          </thead>
          <tbody>${skippedRows}</tbody>
        </table>
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
      .pill {
        border-radius: 999px;
        padding: 4px 9px;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
      }
      .pill.passing { background: #d9f7e8; color: var(--ok); }
      .pill.failing { background: #ffe2dc; color: var(--bad); }
      .meta {
        margin: 8px 0;
        color: var(--muted);
        font-size: 13px;
        word-break: break-all;
      }
      .file-summary {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 12px;
      }
      .file-error {
        margin: 0 0 12px;
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
