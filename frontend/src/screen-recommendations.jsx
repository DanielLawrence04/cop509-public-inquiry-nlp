// Screen 3 — Recommendation Analysis (unified workspace)

const { useState, useMemo, useContext, useEffect } = React;
const Icon               = window.Icon;
const ScreenHeader       = window.ScreenHeader;
const SectionLabel       = window.SectionLabel;
const EmptyState         = window.EmptyState;
const StageChip          = window.StageChip;
const ClassificationPill = window.ClassificationPill;

const RecommendationAnalysisScreen = () => {
  const { task2DataMap, recStage, activePreset, runStage, presets, isRunning } = useContext(window.AppContext);

  const [pairFilter, setPairFilter]   = useState('all');

  // Merge all loaded pairs' recs; inject pair_id and _uid so filters work cross-pair
  const allRecs = useMemo(() => {
    return Object.entries(task2DataMap).flatMap(([presetId, d]) =>
      (d?.recommendations || []).map(r => {
        const adapted = window.adaptRec({ ...r, pair_id: presetId });
        adapted._uid = `${presetId}::${r.rec_id}`;
        return adapted;
      })
    );
  }, [task2DataMap]);
  const [classFilter, setClassFilter]     = useState('all');
  const [confMin, setConfMin]             = useState(0);
  const [text, setText]               = useState('');
  const [page, setPage]               = useState(1);
  const [selectedId, setSelectedId]   = useState(null);

  const PAGE_SIZE = 15;

  // Auto-select first rec when data arrives; use _uid for cross-pair uniqueness
  useEffect(() => {
    if (allRecs.length > 0 && !selectedId) setSelectedId(allRecs[0]._uid);
  }, [allRecs]);

  const groups = useMemo(() => {
    const out = {};
    for (const p of presets) {
      const g = p.dataset_group || 'other';
      out[g] ||= { id: g, label: p.group_label || g, items: [] };
      out[g].items.push(p);
    }
    return Object.values(out);
  }, [presets]);

  const filtered = useMemo(() => {
    return allRecs.filter(r => {
      if (pairFilter !== 'all' && r.pair_id !== pairFilter) return false;
      if (classFilter !== 'all') {
        const cl = r.classification || 'not_addressed';
        if (cl !== classFilter) return false;
      }
      if ((r.overall_confidence ?? 0) < confMin) return false;
      if (text) {
        const hay = (r.text + ' ' + (r.best_match?.matched_text || '')).toLowerCase();
        if (!hay.includes(text.toLowerCase())) return false;
      }
      return true;
    });
  }, [allRecs, pairFilter, classFilter, confMin, text]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageRecs   = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  useEffect(() => {
    if (page > totalPages) setPage(totalPages);
  }, [filtered.length, totalPages]);

  const counts = useMemo(() => {
    const c = { matched: 0, accepted: 0, partial: 0, rejected: 0, not_addressed: 0, sum: 0, n: 0 };
    for (const r of filtered) {
      if (r.status === 'matched') c.matched++;
      const cl = r.classification || 'not_addressed';
      if (cl in c) c[cl]++;
      if (r.overall_confidence != null) { c.sum += r.overall_confidence; c.n++; }
    }
    c.mean = c.n ? c.sum / c.n : 0;
    return c;
  }, [filtered]);

  const selected = allRecs.find(r => r._uid === selectedId) || pageRecs[0] || null;

  // Stage chip + action button state
  const stageRunning = isRunning;
  const stageInfo = (() => {
    if (recStage === 'none') return {
      chip: ['lock', 'Load documents first'],
      btn: 'Run extraction', btnDisabled: true, intent: '',
    };
    if (recStage === 'extract') return {
      chip: stageRunning ? ['run', `Extracting…`] : ['lock', `${allRecs.length} recs · alignment pending`],
      btn: 'Run alignment', btnDisabled: stageRunning, intent: 'primary',
    };
    if (recStage === 'align') return {
      chip: stageRunning ? ['run', `Aligning…`] : ['done', `${allRecs.length} recs · classification pending`],
      btn: 'Run classification', btnDisabled: stageRunning, intent: 'primary',
    };
    // classify / done
    return {
      chip: stageRunning ? ['run', `Classifying…`] : ['done', `${allRecs.length} recs · ${counts.accepted} accepted · ${counts.partial} partial`],
      btn: 'Re-run alignment', btnDisabled: stageRunning, intent: '',
    };
  })();

  const handleStageBtn = () => {
    if (recStage === 'extract') runStage('align', { top_k: 3, similarity_threshold: 0.05 });
    else if (recStage === 'align') runStage('classify');
    else runStage('align', { top_k: 3, similarity_threshold: 0.05 });
  };

  const subtitle = allRecs.length === 0
    ? 'Load a document pair and run the pipeline to see recommendations.'
    : `${counts.accepted} accepted · ${counts.partial} partial · ${counts.rejected} rejected · ${counts.not_addressed} not addressed · showing ${filtered.length} of ${allRecs.length}`;

  return (
    <div className="screen fade-in">
      <ScreenHeader
        title="Recommendation analysis"
        subtitle={subtitle}
        meta={
          <div style={{display:"flex", alignItems:"center", gap:10}}>
            <StageChip variant={stageInfo.chip[0]}>{stageInfo.chip[1]}</StageChip>
            <button
              className={"btn " + (stageInfo.intent === "primary" ? "primary" : "")}
              disabled={stageInfo.btnDisabled}
              onClick={handleStageBtn}>
              <Icon name={stageInfo.intent === "primary" ? "play" : "refresh"} size={12}/>
              {stageInfo.btn}
            </button>
          </div>
        }
      />

      <div className="card">
        {/* Filter bar */}
        <div style={{
          padding: "12px 16px",
          borderBottom: "1px solid var(--line)",
          display:"grid", gridTemplateColumns:"1.4fr 0.9fr 1fr 1.3fr auto auto auto",
          gap: 10, alignItems:"end",
        }}>
          <div className="field-group">
            <span className="field-label">Document pair</span>
            <select className="select" value={pairFilter} onChange={e => setPairFilter(e.target.value)}>
              <option value="all">All loaded pairs</option>
              {groups.map(g => (
                <optgroup key={g.id} label={g.label}>
                  {g.items.map(p => (
                    <option key={p.id} value={p.id}>{p.label} ({p.id})</option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>
          <div className="field-group">
            <span className="field-label">Classification</span>
            <select className="select" value={classFilter} onChange={e => setClassFilter(e.target.value)}>
              <option value="all">All</option>
              <option value="accepted">Accepted</option>
              <option value="partial">Partial</option>
              <option value="rejected">Rejected</option>
              <option value="not_addressed">Not addressed</option>
            </select>
          </div>
          <div className="field-group">
            <span className="field-label">
              Confidence ≥ <span className="mono" style={{color:"var(--ink-2)"}}>{confMin.toFixed(2)}</span>
            </span>
            <div className="range-row" style={{height: 32}}>
              <input type="range" min="0" max="1" step="0.01" value={confMin}
                onChange={e => setConfMin(Number(e.target.value))} className="range"/>
            </div>
          </div>
          <div className="field-group">
            <span className="field-label">Search</span>
            <div className="search-input">
              <Icon name="search" size={12}/>
              <input className="input" value={text} onChange={e => setText(e.target.value)}
                placeholder="recommendation or response text…"/>
              {text && (
                <button
                  type="button"
                  className="search-clear"
                  onClick={() => setText('')}
                  title="Clear search"
                  aria-label="Clear search"
                >
                  <Icon name="x" size={11}/>
                </button>
              )}
            </div>
          </div>
          <button className="btn" style={{height: 32, alignSelf:"end"}} title="Export filtered rows as CSV"
            disabled={filtered.length === 0}
            onClick={() => {
              const esc = v => { const s = String(v ?? ''); return s.includes(',') || s.includes('"') || s.includes('\n') ? '"' + s.replace(/"/g, '""') + '"' : s; };
              const ts = new Date().toISOString().slice(0,19).replace(/:/g,'-');
              const header = ['id','document_pair','recommendation_page','recommendation_text','matched_response_page','matched_response_text','classification','confidence'];
              const rows = filtered.map(r => [
                r.item_label ?? r.rec_id,
                r.pair_id ?? r.document ?? '',
                r.page_number ?? '',
                r.text ?? '',
                r.best_match?.page_number ?? '',
                r.best_match?.matched_text ?? '',
                r.classification ?? 'not_addressed',
                (r.overall_confidence ?? 0).toFixed(3),
              ].map(esc).join(','));
              const csv = [header.join(','), ...rows].join('\r\n');
              const blob = new Blob([csv], {type:'text/csv'});
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a'); a.href=url; a.download=`recs-${ts}.csv`; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
            }}>
            <Icon name="download" size={11}/> CSV
          </button>
          <button className="btn" style={{height: 32, alignSelf:"end"}} title="Export filtered rows as JSON"
            disabled={filtered.length === 0}
            onClick={() => {
              const ts = new Date().toISOString().slice(0,19).replace(/:/g,'-');
              const payload = JSON.stringify({
                exported_at: new Date().toISOString(),
                total: filtered.length,
                recommendations: filtered.map(r => ({
                  id: r.item_label ?? r.rec_id,
                  document_pair: r.pair_id ?? r.document ?? null,
                  recommendation_page: r.page_number ?? null,
                  recommendation_text: r.text ?? null,
                  matched_response_page: r.best_match?.page_number ?? null,
                  matched_response_text: r.best_match?.matched_text ?? null,
                  classification: r.classification ?? 'not_addressed',
                  confidence: r.overall_confidence != null ? parseFloat(r.overall_confidence.toFixed(3)) : null,
                  debug: {
                    alignment_confidence: r.alignment_confidence ?? null,
                    alignment_method: r.best_match?.match_method ?? null,
                    lexical_similarity: r.best_match?.similarity ?? null,
                    confidence_factors: r.confidence_factors ?? [],
                    classification_confidence: r.classification_confidence ?? null,
                    classifier_method: r.classifier_method ?? r.classification_method ?? "rule_based",
                  },
                })),
              }, null, 2);
              const blob = new Blob([payload], {type:'application/json'});
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a'); a.href=url; a.download=`recs-${ts}.json`; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
            }}>
            <Icon name="download" size={11}/> JSON
          </button>
        </div>

        {/* Summary strip */}
        {allRecs.length > 0 && (
          <div style={{
            padding: "10px 18px", borderBottom:"1px solid var(--line)",
            background: "var(--surface-2)",
            display:"flex", flexWrap:"wrap", alignItems:"center", gap: 14,
            fontSize: 12, color:"var(--ink-3)",
          }}>
            <span>
              <b className="mono tabnum" style={{color:"var(--ink)"}}>{filtered.length}</b>
              <span style={{color:"var(--muted)"}}>/{allRecs.length}</span> recommendations
            </span>
            <span className="divider-v"/>
            <span style={{display:"inline-flex", alignItems:"center", gap:5}}>
              <span style={{width:7,height:7,borderRadius:999,background:"var(--ok)",display:"inline-block"}}/>
              Accepted <b className="mono tabnum" style={{color:"var(--ink)"}}>{counts.accepted}</b>
            </span>
            <span style={{display:"inline-flex", alignItems:"center", gap:5}}>
              <span style={{width:7,height:7,borderRadius:999,background:"var(--warn)",display:"inline-block"}}/>
              Partial <b className="mono tabnum" style={{color:"var(--ink)"}}>{counts.partial}</b>
            </span>
            <span style={{display:"inline-flex", alignItems:"center", gap:5}}>
              <span style={{width:7,height:7,borderRadius:999,background:"var(--err)",display:"inline-block"}}/>
              Rejected <b className="mono tabnum" style={{color:"var(--ink)"}}>{counts.rejected}</b>
            </span>
            <span style={{display:"inline-flex", alignItems:"center", gap:5}}>
              <span style={{width:7,height:7,borderRadius:999,background:"var(--muted)",display:"inline-block"}}/>
              Not addressed <b className="mono tabnum" style={{color:"var(--ink)"}}>{counts.not_addressed}</b>
            </span>
            <span className="divider-v"/>
            <span>mean conf <b className="mono tabnum" style={{color:"var(--ink)"}}>{counts.mean.toFixed(2)}</b></span>
            <span className="grow" style={{flex:1}}/>
            {recStage === 'extract' && (
              <span style={{color:"var(--warn-fg)"}}>⚠ Alignment not run yet — click <b>Run alignment</b> to compute matches.</span>
            )}
          </div>
        )}

        {/* Split: table + drill panel */}
        {allRecs.length === 0 ? (
          <EmptyState icon="docs" title="No recommendations yet"
            message="Select a document pair on the Documents screen to load and run the extraction pipeline."/>
        ) : (
          <div style={{display:"grid", gridTemplateColumns:"minmax(0, 1.4fr) minmax(0, 1fr)"}}>

            {/* TABLE */}
            <div style={{borderRight:"1px solid var(--line)", minWidth:0}}>
              <UnifiedRecommendationTable
                rows={pageRecs}
                selectedId={selectedId}
                onSelect={setSelectedId}
              />

              <div style={{
                padding:"10px 16px", borderTop:"1px solid var(--line)",
                display:"flex", alignItems:"center", gap:10,
                fontSize: 12, color:"var(--ink-3)", background:"var(--surface)",
              }}>
                <span>Page <b className="mono tabnum" style={{color:"var(--ink)"}}>{page}</b>
                  <span style={{color:"var(--muted)"}}> / {totalPages}</span>
                </span>
                <span className="divider-v"/>
                <span style={{color:"var(--muted)"}}>
                  rows <b className="mono tabnum" style={{color:"var(--ink-2)"}}>
                    {(page - 1) * PAGE_SIZE + 1}–{Math.min(page * PAGE_SIZE, filtered.length)}
                  </b> of <b className="mono tabnum" style={{color:"var(--ink-2)"}}>{filtered.length}</b>
                </span>
                <span className="grow" style={{flex:1}}/>
                <button className="btn sm" disabled={page === 1} onClick={() => setPage(p => Math.max(1, p - 1))}>
                  <Icon name="chevLeft" size={11}/> Prev
                </button>
                <button className="btn sm" disabled={page === totalPages} onClick={() => setPage(p => Math.min(totalPages, p + 1))}>
                  Next <Icon name="chevRight" size={11}/>
                </button>
              </div>
            </div>

            {/* DRILL */}
            <div style={{minWidth:0, background:"var(--surface-2)"}}>
              <DrillPanel rec={selected}/>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

/* ── UnifiedRecommendationTable ──────────────────────────── */
const UnifiedRecommendationTable = ({ rows, selectedId, onSelect }) => (
  <div className="utable-wrap">
    <table className="utable" style={{minWidth: 520}}>
      <colgroup><col style={{width:44}}/><col style={{width:120}}/><col style={{width:40}}/><col/><col/><col style={{width:130}}/><col style={{width:68}}/></colgroup>
      <thead>
        <tr>
          <th>ID</th>
          <th>Document</th>
          <th>PG</th>
          <th>Recommendation</th>
          <th>Matched Response</th>
          <th>Classification</th>
          <th style={{textAlign:'center'}}>Conf</th>
        </tr>
      </thead>
      <tbody>
        {rows.length === 0 && (
          <tr><td colSpan={7}>
            <div style={{padding: "40px 20px", textAlign:"center", color:"var(--muted)", fontSize: 13}}>
              No rows match the current filters.
            </div>
          </td></tr>
        )}
        {rows.map(r => (
          <tr key={r._uid} className={selectedId === r._uid ? "sel" : ""}
            onClick={() => onSelect(r._uid)}>
            <td>
              <span className="mono tabnum" style={{fontSize:11.5, fontWeight:600, color:"var(--ink-2)"}}>
                {r.item_label}
              </span>
            </td>
            <td>
              <div className="truncate" title={r.document} style={{fontSize:12, color:"var(--ink-2)"}}>
                {r.document}
              </div>
            </td>
            <td>
              <span className="mono tabnum" style={{fontSize:11.5, color:"var(--ink-3)"}}>
                {r.page_number}
              </span>
            </td>
            <td>
              <div className="clamp-3" style={{fontSize:12.5, lineHeight:1.5, color:"var(--ink)"}}>
                {r.text}
              </div>
            </td>
            <td>
              {r.best_match?.matched_text ? (
                <div>
                  <div className="clamp-2" style={{fontSize:12, lineHeight:1.5, color:"var(--ink-2)"}}>
                    {r.best_match.matched_text}
                  </div>
                  <div className="mono" style={{fontSize:10.5, color:"var(--muted)", marginTop:3}}>
                    p. {r.best_match.page_number}
                  </div>
                </div>
              ) : (
                <span style={{fontSize:12, color:"var(--muted)", fontStyle:"italic"}}>— no response —</span>
              )}
            </td>
            <td style={{padding:'8px 6px', verticalAlign:'middle', whiteSpace:'nowrap'}}>
              <ClassificationPill label={r.classification || 'not_addressed'} compact/>
            </td>
            <td style={{padding:'8px 6px', verticalAlign:'middle', textAlign:'center'}}>
              {(() => {
                const v = r.overall_confidence ?? 0;
                const c = v >= 0.8 ? 'var(--ok)' : v >= 0.6 ? 'var(--accent)' : v >= 0.4 ? 'var(--warn)' : 'var(--muted)';
                return <span className="mono tabnum" style={{fontSize:11, fontWeight:600, color:c}}>{v.toFixed(2)}</span>;
              })()}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

/* ── DrillPanel ──────────────────────────────────────────── */
const _NA = <span style={{color:"var(--muted)", fontStyle:"italic"}}>Not available</span>;
const _val = v => (v == null || v === '') ? _NA : v;

const DrillPanel = ({ rec }) => {
  if (!rec) return <EmptyState icon="docs" title="Select a recommendation"/>;

  const m = rec.best_match;
  const hasMatch = !!m && !!m.matched_text;

  return (
    <div className="fade-in" style={{padding: 14, height:"100%", overflowY:"auto", boxSizing:"border-box", display:"flex", flexDirection:"column", gap:12}}>

      {/* ── Primary card: header + recommendation + matched response ── */}
      <div className="card slide-in">

        {/* Card header — mirrors SourcePreviewPanel header */}
        <div style={{padding:"13px 18px", borderBottom:"1px solid var(--line)", background:"var(--surface-2)"}}>
          <div style={{display:"flex", alignItems:"center", gap:8, marginBottom:8, flexWrap:"wrap"}}>
            <span className="mono" style={{
              padding:"2px 8px", border:"1px solid var(--line)",
              borderRadius:4, fontSize:11, fontWeight:700, color:"var(--ink-2)",
              background:"var(--surface)", flexShrink:0,
            }}>{rec.item_label}</span>
            <ClassificationPill label={rec.classification || 'not_addressed'}/>
            <span className="grow" style={{flex:1}}/>
            <span style={{fontSize:11, color:"var(--ink-3)", flexShrink:0}}>
              conf <span className="mono tabnum" style={{fontWeight:700, color:"var(--ink-2)"}}>
                {(rec.overall_confidence ?? 0).toFixed(2)}
              </span>
            </span>
          </div>
          <div style={{display:"flex", gap:10, flexWrap:"wrap", alignItems:"center"}}>
            <span className="kv">
              <span className="k">source</span>
              <span className="v" style={{maxWidth:160, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap"}} title={rec.document}>{rec.document || '—'}</span>
            </span>
            <span className="dot-sep"/>
            <span className="kv"><span className="k">page</span><span className="v mono tabnum">{rec.page_number ?? '—'}</span></span>
            {rec.pair_id && (<>
              <span className="dot-sep"/>
              <span className="kv"><span className="k">pair</span><span className="v mono" style={{fontSize:10.5}}>{rec.pair_id}</span></span>
            </>)}
            {rec.shared_response && (
              <span className="pill indigo" style={{fontSize:10, marginLeft:4}}>
                <Icon name="link" size={10}/> Shared response
              </span>
            )}
            {rec.sequence_corrected && (
              <span className="pill warn" style={{fontSize:10, marginLeft:4}}>
                <Icon name="warning" size={10}/> Sequence corrected
              </span>
            )}
          </div>
        </div>

        {/* Recommendation — mini-pdf card */}
        <div style={{padding:"12px 14px 0"}}>
          <SectionLabel>Recommendation</SectionLabel>
        </div>
        <div className="mini-pdf" style={{margin:"6px 14px 14px"}}>
          <div className="pg-header">
            <span>{rec.document || '—'}</span>
            <span>p. {rec.page_number ?? '—'}</span>
          </div>
          <p style={{wordBreak:"break-word", overflowWrap:"anywhere"}}>{rec.text || '—'}</p>
          {rec.span_id && <div className="pg-num">[{rec.span_id}]</div>}
        </div>

        {/* Matched Response — mini-pdf card */}
        <div style={{padding:"0 14px 0", borderTop:"1px solid var(--line-soft)"}}>
          <div style={{paddingTop:12}}>
            <SectionLabel>Matched Response</SectionLabel>
          </div>
        </div>
        {hasMatch ? (
          <div className="mini-pdf" style={{margin:"6px 14px 14px"}}>
            <div className="pg-header">
              <span>{m.source || '—'}</span>
              <span>p. {m.page_number ?? '—'}</span>
            </div>
            <p style={{wordBreak:"break-word", overflowWrap:"anywhere"}}>{m.matched_text}</p>
            <div style={{display:"flex", gap:10, flexWrap:"wrap", alignItems:"center", marginTop:6}}>
              {m.matched_chunk_id != null && (
                <span className="kv" style={{fontSize:11}}>
                  <span className="k">chunk</span>
                  <code className="mono" style={{background:"var(--surface-soft)", padding:"1px 5px", borderRadius:3, fontSize:10.5}}>{m.matched_chunk_id}</code>
                </span>
              )}
              {m.boundary_reason && (<>
                {m.matched_chunk_id != null && <span className="dot-sep"/>}
                <span className="kv" style={{fontSize:11}}>
                  <span className="k">boundary</span>
                  <span className="v mono" style={{fontSize:10.5}}>{m.boundary_reason}</span>
                </span>
              </>)}
            </div>
            <div className="pg-num">— {m.page_number} —</div>
          </div>
        ) : (
          <div className="mini-pdf" style={{
            margin:"6px 14px 14px", textAlign:"center",
            color:"var(--muted)", fontStyle:"italic", fontSize:12.5,
          }}>
            No response passage crossed the alignment threshold.
          </div>
        )}
      </div>

      {/* ── Alignment & Classification card ── */}
      <div className="card">
        <div style={{
          padding:"10px 16px", borderBottom:"1px solid var(--line)",
          background:"var(--surface-2)",
          fontSize:12, fontWeight:600, color:"var(--ink-2)", letterSpacing:"-0.005em",
        }}>
          Alignment &amp; Classification
        </div>
        <div style={{padding:"8px 14px 10px"}}>
          <p style={{fontSize:11.5, color:"var(--ink-3)", margin:"0 0 10px", lineHeight:1.5}}>
            Overall confidence combines alignment strength and classification certainty, with penalties for fallback matches, very short passages, and low-alignment stances.
          </p>
          <div className="mini-pdf" style={{padding:"10px 14px"}}>
            <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:"6px 16px", fontSize:11.5}}>
              {[
                ["Overall confidence", rec.overall_confidence != null ? rec.overall_confidence.toFixed(3) : null, false],
                ["Alignment score",    m?.alignment_confidence != null ? m.alignment_confidence.toFixed(3) : null, false],
                ["Lexical similarity", m?.similarity != null ? m.similarity.toFixed(3) : null, false],
                ["Alignment method",   m?.match_method,          false],
                ["Match status",       rec.status,               false],
                ["Classification",     rec.classification,        false],
                ["Classifier conf",    rec.classification_confidence != null ? Number(rec.classification_confidence).toFixed(3) : null, false],
                ["Classifier method",  rec.classifier_method ?? rec.classification_method ?? null, false],
                ["Confidence factors", Array.isArray(rec.confidence_factors) ? (rec.confidence_factors.length ? rec.confidence_factors.join(', ') : 'None') : null, true],
                ["Rationale",          rec.classification_rationale ?? rec.rationale ?? null, true],
              ].map(([k, v, fullWidth]) => (
                <div key={k} style={{display:"flex", gap:6, alignItems:"baseline", ...(fullWidth ? {gridColumn:"1 / -1"} : {})}}>
                  <span style={{color:"var(--muted)", flexShrink:0, minWidth:118}}>{k}</span>
                  <span style={{color:"var(--ink-2)", fontWeight:500, wordBreak:"break-word", overflowWrap:"anywhere"}}>
                    {v != null && v !== '' ? <span className="mono" style={{fontSize:11}}>{v}</span> : _NA}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── Source evidence ── */}
      {(rec.quoted_recommendation_text || m?.heading_text) && (
        <div className="card">
          <details className="evidence" style={{border:0, borderRadius:0, background:"transparent"}}>
            <summary style={{padding:"10px 16px"}}>
              <Icon name="chevRight" size={11}/> Source evidence
            </summary>
            <div style={{padding:"0 14px 12px"}}>
              {rec.quoted_recommendation_text && (
                <div style={{marginBottom:8}}>
                  <div className="ev-label">Quoted recommendation</div>
                  <blockquote className="ev-q">"{rec.quoted_recommendation_text}"</blockquote>
                </div>
              )}
              {m?.heading_text && (
                <div>
                  <div className="ev-label">Heading text in response</div>
                  <blockquote className="ev-q">{m.heading_text}</blockquote>
                </div>
              )}
            </div>
          </details>
        </div>
      )}

    </div>
  );
};

window.RecommendationAnalysisScreen = RecommendationAnalysisScreen;
