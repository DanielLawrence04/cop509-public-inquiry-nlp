// Policy Response Analyser — app shell, context, and state management

const { useState, useEffect, useRef, useCallback, createContext, useContext, useMemo } = React;
const Icon = window.Icon;

// API base is injected at build time via index.html (Vite %VITE_API_BASE%
// substitution) or can be overridden at runtime by setting window.__API_BASE__.
// Falls back to local dev backend when neither is configured.
const __injected = (typeof window !== 'undefined' && window.__API_BASE__) || '';
const API_BASE = (__injected && !__injected.includes('%VITE_API_BASE%'))
  ? __injected.replace(/\/+$/, '')
  : 'http://localhost:8000';

// ── API fetch helper ──────────────────────────────────────────
window.apiFetch = async function(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { const b = await res.json(); msg = b.detail || b.message || msg; } catch {}
    throw new Error(msg);
  }
  return res.json();
};

// ── Label normalisation ───────────────────────────────────────
const LABEL_MAP = {
  accepted: 'accepted',
  partially_accepted: 'partial',
  partial: 'partial',
  rejected: 'rejected',
  not_addressed: 'not_addressed',
};

// ── adaptRec — backend rec shape → design component shape ─────
window.adaptRec = function(rec) {
  const bm = rec.best_match;
  const sim = Number(rec.best_similarity ?? bm?.similarity ?? bm?.alignment_confidence ?? 0);
  const status = (!bm || sim === 0) ? 'none'
    : sim >= 0.5 ? 'matched'
    : sim >= 0.3 ? 'weak'
    : 'none';
  const alignment_confidence = bm?.alignment_confidence ?? bm?.similarity ?? sim;
  const classification = LABEL_MAP[rec.best_label] || null;

  // ── Overall confidence ────────────────────────────────────────
  // Combines alignment strength with qualitative penalties so the
  // displayed score reflects how trustworthy the full result is,
  // not just how well the text matched.
  const overall_confidence = (() => {
    if (!bm || !bm.matched_text) return 0;   // no match at all
    let score = alignment_confidence;
    const factors = [];

    // Penalty: match found by fallback strategies (less reliable)
    if (bm.match_method === 'chunk_fallback') {
      score -= 0.15; factors.push('chunk_fallback −0.15');
    } else if (bm.match_method === 'sequence_correction') {
      score -= 0.10; factors.push('sequence_correction −0.10');
    }

    // Penalty: very short matched text (likely a heading or snippet)
    const mlen = (bm.matched_text || '').trim().length;
    if (mlen < 30) {
      score -= 0.25; factors.push('very_short_response −0.25');
    } else if (mlen < 100) {
      score -= 0.10; factors.push('short_response −0.10');
    }

    // Penalty: classification says "not addressed" but response text
    // contains strong government-response language — likely a miss-classification
    if (classification === 'not_addressed' && bm.matched_text) {
      const lower = bm.matched_text.toLowerCase();
      const responseSignals = /\b(accepts?|agrees?|rejects?|does not agree|will |has already|is already|intends?|plans? to|noted|welcomed?)\b/;
      if (responseSignals.test(lower)) {
        score -= 0.10; factors.push('possible_misclassification −0.10');
      }
    }

    // Penalty: classification implies a clear stance but alignment is very weak
    if (['accepted', 'partial', 'rejected'].includes(classification) && sim < 0.3) {
      score -= 0.15; factors.push('low_alignment_for_stance −0.15');
    }

    return { value: Math.max(0, Math.min(1, score)), factors };
  })();

  return {
    ...rec,
    status,
    alignment_confidence,
    classification,
    // overall_confidence is the primary confidence shown in the UI
    overall_confidence: overall_confidence.value,
    confidence_factors: overall_confidence.factors,
    alternatives: (rec.matches || [])
      .filter(m => m !== bm)
      .slice(0, 3)
      .map(m => ({
        page_number: m.page_number,
        similarity: m.similarity ?? m.alignment_confidence ?? 0,
        matched_chunk_id: m.matched_chunk_id,
        matched_text: m.matched_text || m.response_text || '',
      })),
    shared_response: bm?.boundary_reason === 'multi_label_block',
    sequence_corrected: bm?.match_method === 'sequence_correction',
  };
};

// ── adaptEvaluation — task2 payload → design eval shape ───────
window.adaptEvaluation = function(task2Data, presetLabel) {
  const ev = task2Data?.evaluation;
  const recs = task2Data?.recommendations || [];
  const support = recs.length;
  const dist = { accepted: 0, partial: 0, rejected: 0, not_addressed: 0 };
  for (const rec of recs) {
    const k = LABEL_MAP[rec.best_label] || 'not_addressed';
    dist[k] = (dist[k] || 0) + 1;
  }
  if (!ev) return null;
  return {
    pair_label: presetLabel || 'Active pair',
    accuracy: ev.accuracy ?? 0,
    f1_macro: ev.f1_macro ?? 0,
    precision_macro: ev.precision_macro ?? 0,
    recall_macro: ev.recall_macro ?? 0,
    correct: Math.round((ev.accuracy ?? 0) * support),
    support,
    confusion_matrix: ev.confusion_matrix || [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
    per_class: {
      accepted:     ev.per_class?.accepted     || {},
      partial:      ev.per_class?.partially_accepted || ev.per_class?.partial || {},
      rejected:     ev.per_class?.rejected     || {},
      not_addressed:ev.per_class?.not_addressed || {},
    },
    predicted_distribution: dist,
    evaluation_status: task2Data?.evaluation_status || 'Evaluation available.',
  };
};

// ── computeErrorCategories ────────────────────────────────────
window.computeErrorCategories = function(adaptedRecs) {
  if (!adaptedRecs?.length) return [];
  const cats = [];
  const noMatch      = adaptedRecs.filter(r => r.status === 'none');
  const weakMatches  = adaptedRecs.filter(r => r.status === 'weak');
  const pending      = adaptedRecs.filter(r => !r.classification || r.classification === 'pending');
  const seqCorrected = adaptedRecs.filter(r => r.sequence_corrected);
  const shared       = adaptedRecs.filter(r => r.shared_response);
  if (noMatch.length > 0) cats.push({
    count: noMatch.length, tone: 'grey',
    title: 'No response found',
    detail: 'These recommendations have no matched government response passage above the alignment threshold.',
  });
  if (weakMatches.length > 0) cats.push({
    count: weakMatches.length, tone: 'warn',
    title: 'Weak alignment matches',
    detail: 'Similarity between 0.3–0.5. Review these manually to confirm the correct response passage.',
  });
  if (pending.length > 0) cats.push({
    count: pending.length, tone: 'grey',
    title: 'Classification pending',
    detail: 'Run the classify stage to assign accepted / partial / rejected / not_addressed labels.',
  });
  if (seqCorrected.length > 0) cats.push({
    count: seqCorrected.length, tone: 'warn',
    title: 'Sequence corrected',
    detail: 'Alignment used sequence correction — the boundary may span multiple paragraphs.',
  });
  if (shared.length > 0) cats.push({
    count: shared.length, tone: 'indigo',
    title: 'Shared response block',
    detail: 'These recommendations reference the same response passage — a multi-label block.',
  });
  return cats.slice(0, 5);
};

// ── AppContext ─────────────────────────────────────────────────
window.AppContext = createContext(null);

// ── Derive recommendation stage from pipeline status ──────────
function deriveRecStage(status) {
  if (!status?.stages) return 'none';
  const s = status.stages;
  if (s.classify?.status === 'done') return 'classify';
  if (s.align?.status    === 'done') return 'classify';
  if (s.extract?.status  === 'done') return 'align';
  if (s.load?.status     === 'done') return 'extract';
  return 'none';
}

// ── Tabs definition ───────────────────────────────────────────
const TABS = [
  { id: 'documents',       step: 1, icon: 'docs',   label: 'Documents',               hint: 'Load inquiry & response pair' },
  { id: 'search',          step: 2, icon: 'search', label: 'Search',                  hint: 'Hybrid passage retrieval' },
  { id: 'recommendations', step: 3, icon: 'list',   label: 'Recommendation Analysis', hint: 'Extract · match · classify' },
  { id: 'evaluation',      step: 4, icon: 'chart',  label: 'Evaluation Summary',      hint: 'Aggregate metrics' },
];

const useHashRoute = () => {
  const get = () => (location.hash || '#documents').replace(/^#/, '');
  const [route, setRoute] = useState(get());
  useEffect(() => {
    const onHash = () => setRoute(get());
    window.addEventListener('hashchange', onHash);
    if (!location.hash) location.replace('#documents');
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  return [route, (id) => { location.hash = '#' + id; }];
};

// ── App ───────────────────────────────────────────────────────
const App = () => {
  const [route, setRoute] = useHashRoute();
  const [dark, setDark] = useState(() => localStorage.getItem('pra-dark') === '1');
  const [logOpen, setLogOpen] = useState(false);
  const [presets, setPresets] = useState([]);
  const [activePresetId, setActivePresetId] = useState(null);
  const [status, setStatus] = useState({});
  const [task2Data, setTask2Data] = useState(null);
  // task2DataMap accumulates per-pair results so "All loaded pairs" can merge across them
  const [task2DataMap, setTask2DataMap] = useState({});
  const [logEvents, setLogEvents] = useState([]);
  const [initError, setInitError] = useState(null);
  // Tracks whether the currently-displayed Recommendation Analysis / Evaluation
  // Summary data came from the validated final coursework export rather than
  // a live pipeline run on the hosted backend.
  const [finalResultsLoaded, setFinalResultsLoaded] = useState(false);
  const pollRef = useRef(null);

  // Load the validated 246-row coursework export from the static asset shipped
  // with the Vercel frontend (frontend/public/final_recommendations_246.json).
  // This is intentionally backend-independent so the hosted demo opens
  // instantly even when the free Render dyno is asleep or unavailable.
  // Live recomputation via the backend still works and will overwrite the
  // loaded validated rows for whichever preset is re-run.
  const loadFinalResults = useCallback(async () => {
    try {
      const res = await fetch('/final_recommendations_246.json', { cache: 'no-cache' });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const payload = await res.json();
      const adapted = window.adaptFinalExport(payload);
      setTask2DataMap(prev => ({ ...prev, ...adapted.byPreset }));
      setLoadedPresetIds(prev => {
        const next = new Set(prev);
        Object.keys(adapted.byPreset).forEach(k => next.add(k));
        return next;
      });
      // Fall back to synthesised preset labels when the backend presets
      // endpoint hasn't (yet) responded — keeps pair labels readable.
      setPresets(prev => (prev && prev.length > 0) ? prev : adapted.syntheticPresets);
      setFinalResultsLoaded(true);
      return adapted;
    } catch (err) {
      // No log helper available yet here — surface via console only; the
      // banner won't show and the UI falls through to whatever the backend
      // provides instead.
      console.warn('Validated final export load failed:', err);
      throw err;
    }
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    localStorage.setItem('pra-dark', dark ? '1' : '0');
  }, [dark]);

  // Auto-load the validated final export on mount. Runs independently of the
  // backend so a cold/asleep Render dyno never blocks the demo.
  useEffect(() => {
    loadFinalResults().catch(() => {});
  }, [loadFinalResults]);

  // Fetch presets on mount (best-effort — backend may be cold/asleep)
  useEffect(() => {
    window.apiFetch('/api/pipeline/presets')
      .then(data => setPresets(Array.isArray(data) ? data : data.presets || []))
      .catch(() => {
        // Silent: validated-final-export auto-load above seeds synthetic
        // presets so the UI keeps working without a backend.
      });
  }, []);

  // Fetch initial pipeline status (best-effort). If a live run is complete on
  // the backend, prefer it for that preset; otherwise the static export
  // already populated task2DataMap above.
  useEffect(() => {
    window.apiFetch('/api/pipeline/status').then(s => {
      setStatus(s);
      if (s.preset_id) setActivePresetId(s.preset_id);
      if (s.preset_statuses) {
        const completed = Object.entries(s.preset_statuses)
          .filter(([, v]) => v === 'complete')
          .map(([k]) => k);
        if (completed.length > 0) setLoadedPresetIds(prev => {
          const next = new Set(prev);
          completed.forEach(k => next.add(k));
          return next;
        });
      }
      if (s.stages?.extract?.status === 'done') {
        window.apiFetch('/api/pipeline/task2/results').then(d => {
          setTask2Data(d);
          if (s.preset_id) setTask2DataMap(prev => ({ ...prev, [s.preset_id]: d }));
        }).catch(() => {});
      }
    }).catch(() => {
      // Backend unavailable — static validated export already loaded above.
    });
  }, []);

  const addLog = useCallback((stage, message, ms, statusStr = 'info') => {
    const ts = new Date().toLocaleTimeString('en-GB', { hour12: false })
      + '.' + String(Date.now() % 1000).padStart(3, '0');
    setLogEvents(prev => [...prev.slice(-80), { ts, stage, message, status: statusStr, ms }]);
  }, []);

  // Poll status while a stage is running
  const startPoll = useCallback(() => {
    if (pollRef.current) return;
    pollRef.current = setInterval(async () => {
      try {
        const s = await window.apiFetch('/api/pipeline/status');
        setStatus(s);
        const running = s.stages && Object.values(s.stages).some(st => st.status === 'running');
        if (!running) {
          clearInterval(pollRef.current);
          pollRef.current = null;
          // Fetch task2 results when pipeline quiesces
          if (s.stages?.extract?.status === 'done') {
            window.apiFetch('/api/pipeline/task2/results')
              .then(d => {
                setTask2Data(d);
                if (s.preset_id) setTask2DataMap(prev => ({ ...prev, [s.preset_id]: d }));
              })
              .catch(() => {});
          }
        }
      } catch {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    }, 2000);
  }, []);

  const runStage = useCallback(async (stage, body = {}) => {
    addLog(stage, `Starting ${stage}…`);
    await window.apiFetch(`/api/pipeline/run/${stage}`, {
      method: 'POST',
      body: JSON.stringify(body),
    });
    startPoll();
  }, [addLog, startPoll]);

  // Sequential wait helper — polls until not running, then calls next
  const waitThenRun = useCallback((nextStage, nextBody = {}) => {
    return new Promise(resolve => {
      const check = setInterval(async () => {
        try {
          const s = await window.apiFetch('/api/pipeline/status');
          setStatus(s);
          const running = s.stages && Object.values(s.stages).some(st => st.status === 'running');
          if (!running) {
            clearInterval(check);
            resolve(s);
          }
        } catch { clearInterval(check); resolve({}); }
      }, 2000);
    }).then(() => runStage(nextStage, nextBody));
  }, [runStage]);

  const activatePreset = useCallback(async (presetId) => {
    try {
      setTask2Data(null);
      setActivePresetId(presetId);
      addLog('pipeline', `Activating ${presetId}…`);
      // If already cached, just activate; otherwise load fresh
      const s = await window.apiFetch('/api/pipeline/status');
      const cached = s.preset_statuses?.[presetId];
      if (cached) {
        await window.apiFetch(`/api/pipeline/activate/${presetId}`, { method: 'POST', body: '{}' });
        const fresh = await window.apiFetch('/api/pipeline/status');
        setStatus(fresh);
        if (fresh.stages?.extract?.status === 'done') {
          window.apiFetch('/api/pipeline/task2/results').then(d => {
            setTask2Data(d);
            setTask2DataMap(prev => ({ ...prev, [presetId]: d }));
          }).catch(() => {});
        }
      } else {
        await runStage('load', { preset_id: presetId });
        // Auto-pipeline: extract → align → classify
        waitThenRun('extract')
          .then(() => waitThenRun('align', { top_k: 3, similarity_threshold: 0.05 }))
          .then(() => waitThenRun('classify'))
          .then(() => {
            window.apiFetch('/api/pipeline/task2/results').then(d => {
              setTask2Data(d);
              setTask2DataMap(prev => ({ ...prev, [presetId]: d }));
            }).catch(() => {});
          })
          .catch(err => addLog('pipeline', `Pipeline error: ${err.message}`, undefined, 'warn'));
      }
    } catch (err) {
      addLog('pipeline', `Activate error: ${err.message}`, undefined, 'warn');
    }
  }, [addLog, runStage, waitThenRun]);

  const resetPipeline = useCallback(async () => {
    try {
      await window.apiFetch('/api/pipeline/reset', { method: 'POST', body: '{}' });
      setStatus({});
      setTask2Data(null);
      setTask2DataMap({});
      setActivePresetId(null);
      setPairLoadingState({});
      setLoadedPresetIds(new Set());
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
      addLog('pipeline', 'Pipeline reset.');
    } catch (err) {
      addLog('pipeline', `Reset error: ${err.message}`, undefined, 'warn');
    }
  }, [addLog]);

  const clearLog = useCallback(() => setLogEvents([]), []);

  const [isBatchRunning, setIsBatchRunning] = useState(false);
  const [pairLoadingState, setPairLoadingState] = useState({});
  const [loadedPresetIds, setLoadedPresetIds] = useState(new Set());

  const setPairState = useCallback((id, val) => {
    setPairLoadingState(prev => {
      const next = { ...prev };
      if (val === undefined) delete next[id]; else next[id] = val;
      return next;
    });
  }, []);

  const MIN_OVERLAY_MS = 800;
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  // Poll until no stage is in 'running' state. Returns the latest status.
  const waitForIdle = useCallback(async () => {
    for (let i = 0; i < 1200; i++) {
      const s = await window.apiFetch('/api/pipeline/status');
      setStatus(s);
      const running = s.stages && Object.values(s.stages).some(st => st.status === 'running');
      if (!running) return s;
      await sleep(700);
    }
    throw new Error('Pipeline timed out waiting for idle');
  }, []);

  const runStageAndWait = useCallback(async (stage, body = {}) => {
    await window.apiFetch(`/api/pipeline/run/${stage}`, {
      method: 'POST',
      body: JSON.stringify(body || {}),
    });
    const s = await waitForIdle();
    if (s.stages?.[stage]?.status === 'error') {
      throw new Error(s.stages[stage].error || `${stage} stage failed`);
    }
    return s;
  }, [waitForIdle]);

  // ── runPairPipeline: full per-pair pipeline with per-card loading state ──
  // 1) sets pair loading immediately
  // 2) activates cached snapshot if present, otherwise runs load
  // 3) advances through extract → align → classify until 'complete'
  // 4) refreshes task2 results
  // 5) keeps overlay visible for at least MIN_OVERLAY_MS
  // 6) marks pair as 'error' on failure without breaking other pairs
  const runPairPipeline = useCallback(async (presetId) => {
    const t0 = Date.now();
    setPairState(presetId, 'loading');
    try {
      let s = await window.apiFetch('/api/pipeline/status');
      setStatus(s);
      let pst = s.preset_statuses?.[presetId];

      if (pst !== 'complete') {
        if (pst && pst !== 'error') {
          // Restore cached snapshot to make this preset the active pipeline
          await window.apiFetch(`/api/pipeline/activate/${presetId}`, { method: 'POST', body: '{}' });
          s = await window.apiFetch('/api/pipeline/status');
          setStatus(s);
        } else {
          // Fresh load
          addLog('pipeline', `Loading ${presetId}…`);
          s = await runStageAndWait('load', { preset_id: presetId });
        }

        // Make sure each downstream stage is done
        for (const stg of ['extract', 'align', 'classify']) {
          s = await window.apiFetch('/api/pipeline/status');
          setStatus(s);
          if (s.stages?.[stg]?.status !== 'done') {
            const body = stg === 'align' ? { top_k: 3, similarity_threshold: 0.05 } : {};
            addLog('pipeline', `Running ${stg} for ${presetId}…`);
            s = await runStageAndWait(stg, body);
          }
        }
      }

      setActivePresetId(presetId);
      setLoadedPresetIds(prev => new Set([...prev, presetId]));
      const final = await window.apiFetch('/api/pipeline/status');
      setStatus(final);
      if (final.stages?.extract?.status === 'done') {
        try {
          const t2 = await window.apiFetch('/api/pipeline/task2/results');
          setTask2Data(t2);
          setTask2DataMap(prev => ({ ...prev, [presetId]: t2 }));
        } catch {}
      }

      const elapsed = Date.now() - t0;
      if (elapsed < MIN_OVERLAY_MS) await sleep(MIN_OVERLAY_MS - elapsed);
      setPairState(presetId, undefined);
    } catch (err) {
      addLog('pipeline', `Pair ${presetId} error: ${err.message}`, undefined, 'warn');
      const elapsed = Date.now() - t0;
      if (elapsed < MIN_OVERLAY_MS) await sleep(MIN_OVERLAY_MS - elapsed);
      setPairState(presetId, 'error');
    }
  }, [addLog, runStageAndWait, setPairState]);

  const batchEvaluate = useCallback(async () => {
    if (isBatchRunning) return;
    setIsBatchRunning(true);
    addLog('batch', 'Starting batch evaluation of all preset pairs…');
    try {
      for (const preset of presets) {
        const s = await window.apiFetch('/api/pipeline/status');
        const ps = s.preset_statuses || {};
        if (ps[preset.id] === 'complete') {
          addLog('batch', `${preset.label}: already evaluated, skipping.`);
          continue;
        }
        addLog('batch', `${preset.label}: running full pipeline…`);
        await runPairPipeline(preset.id);
      }
      addLog('batch', 'Batch evaluation complete.');
    } catch (err) {
      addLog('batch', `Batch error: ${err.message}`, undefined, 'warn');
    } finally {
      setIsBatchRunning(false);
    }
  }, [isBatchRunning, presets, runPairPipeline, addLog]);

  const activePreset = presets.find(p => p.id === activePresetId);
  const isRunning = status.stages && Object.values(status.stages).some(s => s.status === 'running');

  // The header pill should reflect *user-initiated* live backend actions
  // (batch evaluate, per-pair load) rather than any stage status the backend
  // happens to report on first connect. Static preload of validated results
  // must never make this pill say "Running…".
  const localActivity = isBatchRunning
    || Object.values(pairLoadingState || {}).some(v => v === 'loading');

  const loadedCount = loadedPresetIds.size;
  const statusPillLabel = localActivity
    ? '⟳ Running…'
    : loadedCount === 0
      ? (finalResultsLoaded ? 'Final results loaded' : '—')
      : finalResultsLoaded && loadedCount === Object.keys(task2DataMap).length
        ? `Final results loaded · ${loadedCount} pairs`
        : loadedCount === 1
          ? (presets.find(p => loadedPresetIds.has(p.id))?.label || '—')
          : `${loadedCount} pairs loaded`;
  const recStage = deriveRecStage(status);

  const ctx = {
    presets, activePresetId, activePreset, status, task2Data, task2DataMap, logEvents,
    activatePreset, runStage, resetPipeline, addLog, recStage, isRunning,
    batchEvaluate, isBatchRunning,
    runPairPipeline, pairLoadingState,
    finalResultsLoaded, loadFinalResults,
  };

  const Screen = {
    documents:       window.DocumentsScreen,
    search:          window.SearchScreen,
    recommendations: window.RecommendationAnalysisScreen,
    evaluation:      window.EvaluationScreen,
  }[route] || window.DocumentsScreen;

  return (
    <window.AppContext.Provider value={ctx}>
      <div className="app">
        <AppHeader
          statusPillLabel={statusPillLabel}
          dark={dark}
          onToggleDark={() => setDark(d => !d)}
          onLogToggle={() => setLogOpen(o => !o)}
          logOpen={logOpen}
          onReset={resetPipeline}
          isRunning={isRunning}
        />
        <main className="page">
          <InPageWorkflowTabs tabs={TABS} active={route} onSelect={setRoute}/>
          <div className="screen-mount">
            {initError && (
              <div style={{
                padding:'12px 18px', background:'var(--err-bg)',
                border:'1px solid var(--err-bd)', borderRadius:'var(--r-md)',
                color:'var(--err-fg)', fontSize:13, marginBottom:14,
              }}>
                {initError}
              </div>
            )}
            <Screen/>
          </div>
        </main>
        <LogDrawer open={logOpen} onClose={() => setLogOpen(false)} events={logEvents} onClear={clearLog}/>
      </div>
    </window.AppContext.Provider>
  );
};

// ── AppHeader ─────────────────────────────────────────────────
const AppHeader = ({ statusPillLabel, dark, onToggleDark, onLogToggle, logOpen, onReset, isRunning }) => (
  <header className="app-header">
    <div className="brand">
      <div className="logo"><span>PA</span></div>
      <div className="brand-text">
        <div className="brand-name">Policy Response Analyser</div>
        <div className="brand-tag">Inspect inquiry recommendations and government responses · COP509 NLP</div>
      </div>
    </div>
    <div className="header-actions">
      <button className="status-pill" title="Backend status">
        <span className="status-dot"/>
        <span style={{color:'var(--muted)', marginRight:4}}>Backend ·</span>
        <span style={{fontWeight:600, color:'var(--ink)'}}>
          {statusPillLabel}
        </span>
      </button>
      <button
        className={'icon-btn lg' + (logOpen ? ' on' : '')}
        onClick={onLogToggle}
        title="Live pipeline log">
        <Icon name="terminal" size={14}/>
      </button>
      <button className="icon-btn lg" onClick={onToggleDark} title={dark ? 'Switch to light' : 'Switch to dark'}>
        <Icon name={dark ? 'sun' : 'moon'} size={14}/>
      </button>
      <button className="btn danger" onClick={onReset} title="Clear all loaded state">
        <Icon name="reset" size={12}/> Reset pipeline
      </button>
    </div>
  </header>
);

// ── InPageWorkflowTabs ────────────────────────────────────────
const InPageWorkflowTabs = ({ tabs, active, onSelect }) => (
  <nav className="wf-tabs" role="tablist">
    {tabs.map((t, i) => {
      const isActive = active === t.id;
      return (
        <button key={t.id} role="tab" aria-selected={isActive}
          onClick={() => onSelect(t.id)}
          className={'wf-tab' + (isActive ? ' active' : '')}>
          <div className="wf-step">
            <span className="wf-step-num">{String(t.step).padStart(2, '0')}</span>
            <span className="wf-step-icon"><Icon name={t.icon} size={14}/></span>
          </div>
          <div className="wf-text">
            <div className="wf-label">{t.label}</div>
            <div className="wf-hint">{t.hint}</div>
          </div>
          {i < tabs.length - 1 && <span className="wf-arrow"><Icon name="chevRight" size={14}/></span>}
        </button>
      );
    })}
  </nav>
);

// ── LogDrawer ─────────────────────────────────────────────────
const LogDrawer = ({ open, onClose, events, onClear }) => {
  const bodyRef = useRef(null);
  useEffect(() => {
    if (!open || !bodyRef.current) return;
    bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
  }, [events, open]);

  return (
    <aside className={'log-drawer' + (open ? ' open' : '')}>
      <div className="log-header">
        <div style={{display:'flex', alignItems:'center', gap:8}}>
          <span className="dot ok pulse"/>
          <span style={{fontSize:12, fontWeight:600, color:'var(--ink)'}}>Live pipeline log</span>
          <span style={{fontSize:11, color:'var(--muted)'}} className="mono">/api/pipeline/status</span>
        </div>
        <button className="icon-btn" onClick={onClose}><Icon name="x" size={12}/></button>
      </div>
      <div className="log-body" ref={bodyRef}>
        {events.length === 0 && (
          <div style={{color:'#565d6b', fontSize:11.5, padding:'8px 0', fontStyle:'italic'}}>
            Waiting for pipeline events…
          </div>
        )}
        {events.map((e, i) => (
          <div key={i} className={'log-row ' + (e.status || 'info')}>
            <span className="log-ts mono">{e.ts}</span>
            <span className={'log-stage stage-' + (e.stage || 'pipeline')}>{e.stage || 'pipeline'}</span>
            <span className="log-msg">{e.message}</span>
            {e.ms != null && <span className="log-ms mono">{e.ms}ms</span>}
          </div>
        ))}
      </div>
      <div className="log-footer">
        <span className="mono" style={{fontSize:10.5, color:'var(--muted)'}}>{events.length} events</span>
        <span style={{flex:1}}/>
        <button className="btn sm" onClick={onClear}><Icon name="reset" size={10}/> Clear</button>
      </div>
    </aside>
  );
};

ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
