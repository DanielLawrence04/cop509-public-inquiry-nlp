// Screen 1 — Documents
// Per-pair state machine: ready | loading | loaded | needs-eval | error.
// "Loaded" means the full pipeline (load → extract → align → classify) is done.
// "Partially loaded" is no longer surfaced to the user.

const { useState, useMemo, useContext } = React;
const Icon         = window.Icon;
const ScreenHeader = window.ScreenHeader;
const EmptyState   = window.EmptyState;

// Known coursework-pair recommendation counts. Used only as a sanity hint when
// the cached summary is missing — we never fake counts; if backend hasn't
// finished extraction we render '—'.
const EXPECTED_RECS = {
  behaviour_change: 33,
  post_office_horizon_inquiry: 19,
  space_economy: 40,
  uk_covid_19_inquiry_module_1: 10,
  infected_blood_inquiry: 58,
};

const DocumentsScreen = () => {
  const {
    presets, status, task2Data,
    batchEvaluate, isBatchRunning,
    runPairPipeline, pairLoadingState,
  } = useContext(window.AppContext);

  const presetStatuses  = status.preset_statuses  || {};
  const presetSummaries = status.preset_summaries || {};
  const isOcr           = status.load_ocr === true;

  // Derive per-pair state purely from runPairPipeline tracking + cached backend status.
  const pairState = (id) => {
    const local = pairLoadingState?.[id];
    if (local === 'loading') return 'loading';
    if (local === 'error')   return 'error';
    const pst = presetStatuses[id];
    if (pst === 'error')    return 'error';
    if (pst === 'complete') return 'loaded';
    if (pst === 'loaded')   return 'needs-eval';
    return 'ready';
  };

  const numLoaded = presets.filter(p => pairState(p.id) === 'loaded').length;

  const groups = useMemo(() => {
    const out = {};
    for (const p of presets) {
      const g = p.dataset_group || 'other';
      out[g] ||= { id: g, label: p.group_label || g, description: p.group_description || '', items: [] };
      out[g].items.push(p);
    }
    return Object.values(out);
  }, [presets]);

  const anyPairLoading = Object.values(pairLoadingState || {}).some(v => v === 'loading');
  const batchBusy      = isBatchRunning || anyPairLoading;
  const canBatch       = presets.length > 0 && !batchBusy;

  // Clicking a card runs the full pipeline for that pair (no-op if already loaded).
  const handleCardClick = (presetId) => {
    const st = pairState(presetId);
    if (st === 'loaded' || st === 'loading') return;
    runPairPipeline(presetId);
  };

  // Pairs to show in the "Loaded pairs" table — anything with backend state we know about.
  const tablePairs = presets.filter(p => {
    const st = pairState(p.id);
    return st === 'loaded' || st === 'needs-eval' || st === 'loading' || st === 'error';
  });

  return (
    <div className="screen fade-in">
      <ScreenHeader
        title="Document collection"
        subtitle="Load preset inquiry/response pairs into the system. All loaded pairs are available across Search, Recommendation Analysis and Evaluation."
        meta={
          <>
            <span className="kv mono">
              <span className="k">{presets.length}</span>
              <span className="muted">pairs available</span>
            </span>
            {numLoaded > 0 && (
              <span className="kv mono">
                <span className="k" style={{color:'var(--ok)'}}>{numLoaded}</span>
                <span className="muted">loaded</span>
              </span>
            )}
          </>
        }
      />

      <div className="card">
        <div className="card-header">
          <span className="card-title">Document pairs</span>
          <span className="card-sub">
            {presets.length} preset pairs
            {numLoaded > 0 ? ` · ${numLoaded} loaded` : ''}
          </span>
          <span className="grow"/>
          <button
            className="btn primary"
            disabled={!canBatch}
            title={batchBusy ? 'Pipeline is busy…' : 'Run the full pipeline for all preset pairs'}
            onClick={batchEvaluate}
          >
            {isBatchRunning
              ? <><span className="spinner" style={{width:11,height:11,border:'1.5px solid rgba(255,255,255,0.35)',borderTopColor:'#fff'}}/> Evaluating…</>
              : <><Icon name="sparkle" size={12}/> Batch evaluate all</>
            }
          </button>
        </div>

        <div style={{padding: "8px 18px 22px 18px"}}>
          {presets.length === 0 && (
            <div style={{padding:'24px 0', textAlign:'center', color:'var(--muted)', fontSize:13}}>
              Loading presets…
            </div>
          )}

          {groups.map(g => (
            <div key={g.id} style={{marginTop: 14}}>
              <div style={{display:"flex", alignItems:"flex-end", gap:14, marginBottom:10}}>
                <div>
                  <div style={{fontSize: 12.5, fontWeight: 600, color: "var(--ink)", letterSpacing: "-0.005em"}}>
                    {g.label}
                  </div>
                  {g.description && (
                    <div style={{fontSize: 11.5, color: "var(--muted)", marginTop: 2, maxWidth: 600}}>
                      {g.description}
                    </div>
                  )}
                </div>
                <span className="grow"/>
                <span className="kv mono" style={{fontSize:11}}>
                  <span className="muted">{g.items.length} pair{g.items.length !== 1 ? "s" : ""}</span>
                </span>
              </div>

              <div style={{display:"grid", gridTemplateColumns:"repeat(auto-fill, minmax(244px,1fr))", gap:10}}>
                {g.items.map(p => {
                  const st = pairState(p.id);
                  return (
                    <PresetCard
                      key={p.id}
                      preset={p}
                      state={st}
                      isOcr={st === 'loading' && isOcr}
                      onClick={() => handleCardClick(p.id)}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <span className="card-title">Loaded pairs</span>
          <span className="card-sub">Loaded document pairs are available for Search, Recommendation Analysis and Evaluation.</span>
        </div>
        {tablePairs.length > 0 ? (
          <LoadedPairsList
            presets={tablePairs}
            pairState={pairState}
            presetSummaries={presetSummaries}
          />
        ) : (
          <EmptyState icon="docs" title="No pairs loaded yet"
            message="Click a pair card above to load and evaluate it, or use Batch evaluate all to process every pair."/>
        )}
      </div>

      <div style={{display:"flex", alignItems:"center", gap:8, color:"var(--muted)", fontSize:12}}>
        <Icon name="info" size={12}/>
        <span>Load pairs here, then use Search, Recommendation Analysis and Evaluation Summary to inspect the results.</span>
      </div>
    </div>
  );
};

/* ── Preset card ─────────────────────────────────────────── */
const STATE_META = {
  ready:        { label: "Ready to load",   cls: "grey", icon: "play"    },
  loading:      { label: "Evaluating…",     cls: "ok",   icon: "play"    },
  loaded:       { label: "Loaded",          cls: "ok",   icon: "check"   },
  'needs-eval': { label: "Needs evaluation",cls: "warn", icon: "warning" },
  error:        { label: "Error · click to retry", cls: "bad", icon: "warning" },
};

const PresetCard = ({ preset, state, isOcr, onClick }) => {
  const meta       = STATE_META[state] || STATE_META.ready;
  const loading    = state === 'loading';
  const isLoaded   = state === 'loaded';
  const isError    = state === 'error';
  const isInert    = isLoaded || loading;
  const overlayMsg = isOcr ? "OCR detected — processing may take longer" : "Loading document pair";

  return (
    <button
      onClick={isInert ? undefined : onClick}
      className="preset-card"
      style={{
        display: "flex", flexDirection: "column", gap: 10,
        padding: 14,
        background: "var(--surface)",
        border: isLoaded ? "1px solid var(--ok)" : isError ? "1px solid var(--err)" : "1px solid var(--line)",
        borderRadius: "var(--r-md)",
        boxShadow: isLoaded ? "0 0 0 1px var(--ok)" : "none",
        cursor: loading ? "wait" : isInert ? "default" : "pointer",
        textAlign: "left", color: "inherit",
        transition: "border 180ms ease-out, box-shadow 180ms ease-out",
        position: "relative",
        overflow: "hidden",
      }}
      onMouseEnter={e => !loading && !isInert && (e.currentTarget.style.transform = "translateY(-1px)")}
      onMouseLeave={e => !loading && !isInert && (e.currentTarget.style.transform = "translateY(0)")}>

      {loading && (
        <div className="fade-in" style={{
          position: "absolute", inset: 0,
          display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center",
          gap: 10, zIndex: 2,
          borderRadius: "inherit",
        }}>
          <div style={{
            position: "absolute", inset: 0,
            background: "var(--surface)",
            opacity: 0.9,
            backdropFilter: "blur(3px)",
            WebkitBackdropFilter: "blur(3px)",
            borderRadius: "inherit",
          }}/>
          <span className="spinner" style={{
            width:22, height:22, borderWidth:2.5,
            position:"relative", zIndex:1,
          }}/>
          <span style={{
            fontSize:11, fontWeight:600, color:"var(--accent)",
            textAlign:"center", padding:"0 12px", lineHeight:1.4,
            position:"relative", zIndex:1,
          }}>
            {overlayMsg}
          </span>
        </div>
      )}

      <div style={{display:"flex", alignItems:"flex-start", gap: 8}}>
        <DocPairThumb/>
        <span className="grow"/>
        <div style={{display:"flex", gap:4, flexWrap:"wrap", justifyContent:"flex-end"}}>
          {isLoaded && (
            <span className="pill" style={{
              fontSize:9.5, padding:"1px 6px",
              background:"var(--ok-bg,rgba(34,197,94,0.12))",
              color:"var(--ok)", border:"1px solid var(--ok)",
            }}>Loaded</span>
          )}
          {isError && (
            <span className="pill" style={{
              fontSize:9.5, padding:"1px 6px",
              background:"rgba(239,68,68,0.1)",
              color:"var(--err)", border:"1px solid var(--err)",
            }}>Error</span>
          )}
          {preset.is_extra && !isLoaded && !isError && (
            <span className="pill solid-grey" style={{fontSize:9.5, padding:"1px 6px"}}>Extra</span>
          )}
        </div>
      </div>

      <div>
        <div style={{fontSize: 13, fontWeight: 600, letterSpacing:"-0.005em", lineHeight: 1.32, color: "var(--ink)"}}>
          {preset.label}
        </div>
        <div className="mono" style={{fontSize: 10.5, color:"var(--faint)", marginTop:3}}>
          {preset.slug || preset.id}
        </div>
      </div>

      <div style={{display:"flex", alignItems:"center", gap: 6, marginTop:"auto"}}>
        <span className="pill solid-grey" style={{fontSize:9.5, padding:"1px 6px", textTransform:"uppercase", letterSpacing:"0.05em"}}>
          report + response
        </span>
      </div>

      <div style={{
        display:"flex", alignItems:"center", gap: 6,
        fontSize: 11, fontWeight: 500,
        color: meta.cls === "ok"   ? "var(--ok)"
             : meta.cls === "bad"  ? "var(--err)"
             : meta.cls === "warn" ? "var(--warn, #d97706)"
             : "var(--muted)",
        paddingTop: 8,
        borderTop: "1px dashed var(--line)",
      }}>
        <Icon name={meta.icon} size={11}/>
        <span>{meta.label}</span>
      </div>
    </button>
  );
};

/* ── Stylised mini document thumbnail pair ───────────────── */
const DocPairThumb = () => (
  <div style={{display:"flex", gap:4}}>
    {[0,1].map(i => (
      <div key={i} style={{
        width: 38, height: 50, borderRadius: 3,
        background: "linear-gradient(180deg, #fff 0%, var(--surface-soft) 100%)",
        border: "1px solid var(--line)",
        boxShadow: "0 1px 2px rgba(15,18,30,0.04)",
        padding: "5px 6px",
        display: "flex", flexDirection: "column", gap: 2,
        position:"relative",
      }}>
        <div style={{height: 5, background: i === 0 ? "var(--accent)" : "var(--ink-3)", borderRadius:1, width:"60%", marginBottom:1}}/>
        {[80, 70, 90, 50, 70, 40].map((w, k) => (
          <div key={k} style={{height: 1.6, background: "var(--line-2)", borderRadius:1, width: w + "%"}}/>
        ))}
      </div>
    ))}
  </div>
);

/* ── Loaded pairs list ───────────────────────────────────── */
const LoadedPairsList = ({ presets, pairState, presetSummaries }) => (
  <div>
    {presets.map((p, i) => {
      const summary    = presetSummaries[p.id] || {};
      const st         = pairState(p.id);
      const hasChunks  = summary.policy_chunks != null || summary.response_chunks != null;
      const totalChunks = hasChunks ? (summary.policy_chunks ?? 0) + (summary.response_chunks ?? 0) : null;
      // Only show recs when classify completed so the value reflects a real,
      // usable extraction. Otherwise '—'.
      const recs = (st === 'loaded') ? (summary.recommendations ?? null) : null;
      return (
        <LoadedPairRow
          key={p.id}
          preset={p}
          state={st}
          chunks={totalChunks}
          recs={recs}
          isLast={i === presets.length - 1}
        />
      );
    })}
  </div>
);

const BADGE_META = {
  loaded:       { label: 'Loaded',          cls: 'matched',     icon: 'check'   },
  loading:      { label: 'Evaluating',      cls: 'matched',     icon: 'play'    },
  'needs-eval': { label: 'Needs evaluation',cls: 'pending',     icon: 'warning' },
  error:        { label: 'Error',           cls: 'unmatched',   icon: 'warning' },
};

const LoadedPairRow = ({ preset, state, chunks, recs, isLast }) => {
  const badge = BADGE_META[state] || BADGE_META['needs-eval'];
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 14,
      padding: "11px 18px",
      borderBottom: isLast ? "none" : "1px solid var(--line-soft)",
    }}>
      <DocPairThumb/>
      <div style={{flex:1, minWidth:0}}>
        <div style={{fontSize:13, fontWeight:600, color:"var(--ink)", letterSpacing:"-0.005em", lineHeight:1.3}}>
          {preset.label}
        </div>
        <div className="mono" style={{fontSize:10.5, color:"var(--muted)", marginTop:2}}>
          {preset.slug || preset.id}
        </div>
      </div>
      <span className="kv">
        <span className="k">Chunks</span>
        <span className="v mono tabnum">{chunks ?? '—'}</span>
      </span>
      <span className="kv">
        <span className="k">Recs</span>
        <span className="v mono tabnum">{recs ?? '—'}</span>
      </span>
      <span className={"status-badge " + badge.cls}>
        {state === 'loading'
          ? <span className="spinner" style={{width:9, height:9, borderWidth:1.5}}/>
          : <Icon name={badge.icon} size={10}/>}
        {' '}{badge.label}
      </span>
    </div>
  );
};

window.DocumentsScreen = DocumentsScreen;
