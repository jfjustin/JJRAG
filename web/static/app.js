/* JJRAG front end — vanilla JS, no dependencies, no external requests.
 * Everything talks to this same origin; the CSP forbids anything else. */

const API = {
  async json(path, options = {}) {
    const response = await fetch(path, {
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      ...options,
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok && response.status !== 422) {
      throw new Error(body.detail || `${response.status} ${response.statusText}`);
    }
    return { ok: response.ok, status: response.status, body };
  },
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

/* ── helpers ─────────────────────────────────────────────── */
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

let toastTimer = null;
function toast(message, kind = '') {
  const element = $('#toast');
  element.textContent = message;
  element.className = `toast ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.add('hidden'), 5200);
}

/* ── tabs ────────────────────────────────────────────────── */
$$('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    $$('.tab').forEach((t) => t.classList.remove('active'));
    $$('.panel').forEach((p) => p.classList.remove('active'));
    tab.classList.add('active');
    $(`#panel-${tab.dataset.panel}`).classList.add('active');
    if (tab.dataset.panel === 'documents') loadDocuments();
    if (tab.dataset.panel === 'pipeline') loadRuns();
    if (tab.dataset.panel === 'compliance') loadCompliance();
  });
});

/* ── status strip ────────────────────────────────────────── */
async function loadHealth() {
  try {
    const { body } = await API.json('/api/health');
    const modelPill = $('#pill-model');
    modelPill.textContent = body.local_model_available
      ? `model: ${body.local_models[0] || 'local'}`
      : 'model: not running';
    modelPill.className = `pill ${body.local_model_available ? 'pill-good' : 'pill-bad'}`;
    modelPill.title = body.local_model_available
      ? `Local models installed: ${body.local_models.join(', ')}`
      : 'No local model server reachable. Start Ollama and pull a model.';

    const indexPill = $('#pill-index');
    indexPill.textContent = body.index_version
      ? `index v${body.index_version} · ${body.indexed_chunks} chunks`
      : 'no index yet';
    indexPill.className = `pill ${body.index_version ? '' : 'pill-warn'}`;
  } catch (error) {
    $('#pill-model').textContent = 'service unreachable';
    $('#pill-model').className = 'pill pill-bad';
  }
}

/* ── ask ─────────────────────────────────────────────────── */
function renderCitationMarkers(text) {
  return escapeHtml(text).replace(
    /\[(\d+(?:\s*,\s*\d+)*)\]/g,
    (match) => `<span class="cite">${match}</span>`
  );
}

function renderSources(sources) {
  const card = $('#sources-card');
  const list = $('#sources-list');
  if (!sources || sources.length === 0) {
    card.classList.add('hidden');
    list.innerHTML = '';
    return;
  }
  list.innerHTML = sources.map((source) => {
    const scores = [
      source.dense_score != null ? `semantic ${source.dense_score.toFixed(3)}` : null,
      source.lexical_score != null ? `keyword ${source.lexical_score.toFixed(2)}` : null,
    ].filter(Boolean).join(' · ');
    return `
      <div class="source">
        <div class="source-head">
          <span><strong>[${source.rank}] ${escapeHtml(source.filename)}</strong>
            ${source.segment_label ? ` — ${escapeHtml(source.segment_label)}` : ''}</span>
          <span>${escapeHtml(scores)}</span>
        </div>
        <div class="source-text">${escapeHtml(source.text)}</div>
      </div>`;
  }).join('');
  card.classList.remove('hidden');
}

$('#ask-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = $('#question').value.trim();
  if (!question) return;

  const button = $('#ask-button');
  button.disabled = true;
  button.textContent = 'Thinking…';

  const answerCard = $('#answer-card');
  const answerText = $('#answer-text');
  answerCard.classList.remove('hidden');
  answerText.innerHTML = '<span class="cursor"></span>';
  $('#answer-meta').textContent = 'retrieving…';
  renderSources(null);

  let buffer = '';
  try {
    const response = await fetch('/api/query/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        top_k: parseInt($('#top-k').value, 10) || null,
      }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `${response.status} ${response.statusText}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = '';

    // Server-sent events: blocks separated by a blank line.
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });
      const blocks = pending.split('\n\n');
      pending = blocks.pop() || '';

      for (const block of blocks) {
        const eventMatch = block.match(/^event: (.+)$/m);
        const dataMatch = block.match(/^data: ([\s\S]+)$/m);
        if (!eventMatch || !dataMatch) continue;
        const name = eventMatch[1].trim();
        let payload;
        try { payload = JSON.parse(dataMatch[1]); } catch { continue; }

        if (name === 'sources') {
          renderSources(payload);
          $('#answer-meta').textContent =
            `${payload.length} passage(s) retrieved · generating locally`;
        } else if (name === 'token') {
          buffer += payload;
          answerText.innerHTML = renderCitationMarkers(buffer) + '<span class="cursor"></span>';
        } else if (name === 'done') {
          answerText.innerHTML = renderCitationMarkers(buffer);
          if (payload.citations) renderSources(payload.citations);
          const parts = [];
          if (payload.model) parts.push(payload.model);
          if (payload.latency_ms) parts.push(`${(payload.latency_ms / 1000).toFixed(1)}s`);
          if (payload.index_version) parts.push(`index v${payload.index_version}`);
          $('#answer-meta').textContent = parts.join(' · ');
        } else if (name === 'error') {
          throw new Error(payload);
        }
      }
    }
  } catch (error) {
    answerText.innerHTML = `<span class="value-bad">${escapeHtml(error.message)}</span>`;
    $('#answer-meta').textContent = '';
    toast(error.message, 'bad');
  } finally {
    button.disabled = false;
    button.textContent = 'Ask';
  }
});

/* ── upload ──────────────────────────────────────────────── */
const dropzone = $('#dropzone');
const fileInput = $('#file-input');
let pendingFiles = [];

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); fileInput.click(); }
});
['dragenter', 'dragover'].forEach((name) =>
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add('dragover');
  })
);
['dragleave', 'drop'].forEach((name) =>
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove('dragover');
  })
);
dropzone.addEventListener('drop', (event) => handleFiles(event.dataTransfer.files));
fileInput.addEventListener('change', () => handleFiles(fileInput.files));

async function handleFiles(fileList) {
  const files = Array.from(fileList || []);
  if (files.length === 0) return;

  const list = $('#upload-list');
  const form = new FormData();
  files.forEach((file) => form.append('files', file, file.name));

  list.innerHTML = files
    .map((f) => `<li><span>${escapeHtml(f.name)}</span><span>uploading…</span></li>`)
    .join('');

  try {
    const response = await fetch('/api/documents', { method: 'POST', body: form });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'upload failed');

    list.innerHTML = [
      ...body.accepted.map((f) =>
        `<li><span>${escapeHtml(f.filename)}</span>
             <span class="ok">accepted · ${formatBytes(f.size_bytes)}</span></li>`),
      ...body.rejected.map((f) =>
        `<li><span>${escapeHtml(f.filename)}</span>
             <span class="bad">rejected — ${escapeHtml(f.reason || '')}</span></li>`),
    ].join('');

    pendingFiles = body.accepted;
    $('#ingest-button').disabled = pendingFiles.length === 0;
    toast(body.message, body.rejected.length ? 'bad' : 'good');
  } catch (error) {
    list.innerHTML = `<li><span class="bad">${escapeHtml(error.message)}</span></li>`;
    toast(error.message, 'bad');
  }
  fileInput.value = '';
}

/* ── ingest ──────────────────────────────────────────────── */
$('#ingest-button').addEventListener('click', async () => {
  const button = $('#ingest-button');
  const progress = $('#ingest-progress');
  button.disabled = true;
  button.textContent = 'Processing…';
  progress.classList.remove('hidden');
  progress.innerHTML = '<div>running: scan → extract → transform → validate → embed → load</div>';

  try {
    const { ok, body } = await API.json('/api/ingest', {
      method: 'POST',
      body: JSON.stringify({ force: $('#force-reingest').checked }),
    });

    progress.innerHTML = (body.stages || []).map((stage) => {
      const metrics = Object.entries(stage.metrics || {})
        .slice(0, 6)
        .map(([key, value]) =>
          `${key}=${typeof value === 'object' ? JSON.stringify(value) : value}`)
        .join(' ');
      return `<div><span class="stage">${escapeHtml(stage.name)}</span>
        ${stage.status === 'succeeded' ? '✓' : '✗'}
        ${stage.records_in}→${stage.records_out}
        ${stage.duration_s != null ? `(${stage.duration_s.toFixed(2)}s)` : ''}
        ${escapeHtml(metrics)}</div>`;
    }).join('');

    if (ok) {
      toast(`Indexed ${body.chunks} chunks from ${body.documents} document(s) · index v${body.index_version}`, 'good');
      pendingFiles = [];
    } else {
      progress.innerHTML += `<div class="value-bad">${escapeHtml(body.error || 'validation failed')}</div>`;
      toast(body.error || 'Validation gate failed — see the Pipeline tab', 'bad');
    }
    renderRunDetail(body);
    loadDocuments();
    loadHealth();
    loadRuns();
  } catch (error) {
    toast(error.message, 'bad');
    progress.innerHTML += `<div class="value-bad">${escapeHtml(error.message)}</div>`;
  } finally {
    button.disabled = false;
    button.textContent = 'Process documents';
  }
});

/* ── documents ───────────────────────────────────────────── */
async function loadDocuments() {
  const container = $('#documents-list');
  try {
    const { body } = await API.json('/api/documents');
    if (!body.length) {
      container.innerHTML = '<p class="hint">Nothing indexed yet.</p>';
      return;
    }
    container.innerHTML = body.map((doc) => {
      const redactions = Object.entries(doc.redactions || {})
        .map(([kind, count]) => `${kind}×${count}`).join(', ');
      return `
        <div class="document">
          <div>
            <div class="document-name">${escapeHtml(doc.filename)}</div>
            <div class="document-meta">
              ${doc.chunk_count} chunks · ${formatBytes(doc.size_bytes)}
              ${redactions ? ` · redacted: ${escapeHtml(redactions)}` : ''}
              · <code>${escapeHtml(doc.doc_id)}</code>
            </div>
          </div>
          <button class="danger" data-doc="${escapeHtml(doc.doc_id)}">Erase</button>
        </div>`;
    }).join('');

    container.querySelectorAll('button[data-doc]').forEach((button) => {
      button.addEventListener('click', async () => {
        const docId = button.dataset.doc;
        if (!confirm('Erase this document and rebuild the index without it?')) return;
        button.disabled = true;
        try {
          const { body: result } = await API.json(`/api/documents/${docId}`, { method: 'DELETE' });
          toast(result.message || 'Erased', 'good');
          setTimeout(() => { loadDocuments(); loadHealth(); }, 900);
        } catch (error) {
          toast(error.message, 'bad');
          button.disabled = false;
        }
      });
    });
  } catch (error) {
    container.innerHTML = `<p class="value-bad">${escapeHtml(error.message)}</p>`;
  }
}
$('#refresh-documents').addEventListener('click', loadDocuments);

/* ── pipeline ────────────────────────────────────────────── */
function renderRunDetail(run) {
  if (!run || !run.run_id) return;
  const stages = (run.stages || []).map((stage) => {
    const badge = stage.status === 'succeeded'
      ? 'badge-ok' : stage.status === 'failed' ? 'badge-fail' : 'badge-warn';
    const metrics = Object.entries(stage.metrics || {})
      .map(([key, value]) =>
        `${key}=${typeof value === 'object' ? JSON.stringify(value) : value}`)
      .join('  ');
    return `
      <div class="stage-row">
        <span class="stage-name">${escapeHtml(stage.name)}</span>
        <span class="badge ${badge}">${escapeHtml(stage.status)}</span>
        <span class="stage-metrics">${stage.records_in}→${stage.records_out}
          ${stage.duration_s != null ? `· ${stage.duration_s.toFixed(2)}s` : ''}
          ${escapeHtml(metrics)}</span>
      </div>`;
  }).join('');

  const issues = (run.issues || []).slice(0, 40).map((issue) => `
    <div class="issue issue-${escapeHtml(issue.severity)}">
      <code>${escapeHtml(issue.rule)}</code> — ${escapeHtml(issue.message)}
    </div>`).join('');

  const redactions = Object.entries(run.redactions || {})
    .map(([kind, count]) => `${kind}×${count}`).join(', ') || 'none';

  $('#run-detail').innerHTML = `
    <p class="meta">
      <code>${escapeHtml(run.run_id)}</code> ·
      <span class="${run.status === 'succeeded' ? 'value-good' : 'value-bad'}">${escapeHtml(run.status)}</span>
      ${run.duration_s != null ? ` · ${run.duration_s.toFixed(1)}s` : ''}
      · ${run.documents} docs · ${run.chunks} chunks
      ${run.index_version ? ` · index v${run.index_version}` : ''}
      · redacted: ${escapeHtml(redactions)}
      ${run.files_rejected ? ` · <span class="value-bad">${run.files_rejected} file(s) rejected</span>` : ''}
    </p>
    ${run.error ? `<div class="issue issue-error">${escapeHtml(run.error)}</div>` : ''}
    <div class="stage-grid">${stages}</div>
    ${issues ? `<h3 style="font-size:13px;margin:16px 0 6px">Validation findings</h3>${issues}` : ''}`;
}

async function loadRuns() {
  try {
    const { body } = await API.json('/api/runs?limit=15');
    const list = $('#runs-list');
    if (!body.length) {
      list.innerHTML = '<p class="hint">No runs recorded.</p>';
      return;
    }
    list.innerHTML = body.map((run) => `
      <div class="stage-row" style="grid-template-columns:200px 96px 1fr">
        <span class="stage-name">${escapeHtml(run.run_id)}</span>
        <span class="badge ${run.status === 'succeeded' ? 'badge-ok' : 'badge-fail'}">${escapeHtml(run.status)}</span>
        <span class="stage-metrics">${run.documents} docs · ${run.chunks} chunks
          ${run.index_version ? `· v${run.index_version}` : ''} · ${escapeHtml((run.started_at || '').slice(0, 19))}</span>
      </div>`).join('');

    const { body: latest } = await API.json(`/api/runs/${body[0].run_id}`);
    renderRunDetail(latest);
  } catch (error) {
    $('#runs-list').innerHTML = `<p class="value-bad">${escapeHtml(error.message)}</p>`;
  }
}
$('#refresh-runs').addEventListener('click', loadRuns);

/* ── compliance ──────────────────────────────────────────── */
async function loadCompliance() {
  const container = $('#compliance-body');
  try {
    const { body } = await API.json('/api/compliance');
    const posture = body.posture || {};
    const rows = [
      ['Generation runs on', `${posture.generation_provider} · ${posture.generation_model}`],
      ['Model host', posture.generation_host],
      ['Embeddings', `${posture.embedding_backend} · ${posture.embedding_model}`],
      ['Third-party model APIs', posture.third_party_model_apis_enabled ? 'ENABLED' : 'none — not implemented'],
      ['Egress restricted to localhost', posture.egress_restricted_to_localhost ? 'yes' : 'NO'],
      ['Egress guard active in process', body.egress_guard_active ? 'yes' : 'NO'],
      ['PII redaction', posture.pii_redaction_enabled
        ? `on — ${(posture.pii_types_redacted || []).join(', ')}` : 'OFF'],
      ['Validation gates', posture.validation_gates_enabled ? 'enforced' : 'OFF'],
      ['Audit log', posture.audit_log_enabled ? 'on' : 'off'],
      ['Retention policy', posture.retention_days ? `${posture.retention_days} days` : 'keep until erased'],
    ];

    const good = new Set(['yes', 'on', 'enforced', 'none — not implemented']);
    const bad = new Set(['NO', 'OFF', 'ENABLED']);

    const blocked = (body.blocked_egress_attempts || [])
      .map((a) => `${escapeHtml(a.host)}${a.port ? `:${a.port}` : ''}`).join(', ');

    container.innerHTML = `
      <table class="kv">
        ${rows.map(([key, value]) => {
          const text = String(value ?? '');
          const cls = good.has(text) || text.startsWith('on —') ? 'value-good'
            : bad.has(text) ? 'value-bad' : '';
          return `<tr><td>${escapeHtml(key)}</td><td class="${cls}">${escapeHtml(text)}</td></tr>`;
        }).join('')}
        <tr><td>Blocked outbound attempts</td>
            <td class="${blocked ? 'value-bad' : 'value-good'}">${blocked || 'none'}</td></tr>
        <tr><td>Corpus</td><td>${(body.catalog || {}).documents || 0} documents ·
            ${(body.catalog || {}).chunks || 0} chunks ·
            ${(body.catalog || {}).sources_rejected || 0} file(s) rejected at the gate</td></tr>
      </table>`;
  } catch (error) {
    container.innerHTML = `<p class="value-bad">${escapeHtml(error.message)}</p>`;
  }
}

/* ── boot ────────────────────────────────────────────────── */
loadHealth();
setInterval(loadHealth, 30000);
