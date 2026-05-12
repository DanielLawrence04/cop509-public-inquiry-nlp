// Screen 2 — Search (hybrid passage retrieval)

const { useState, useMemo, useContext, useCallback, useEffect, useRef } = React;
const Icon         = window.Icon;
const ScreenHeader = window.ScreenHeader;
const SectionLabel = window.SectionLabel;
const EmptyState   = window.EmptyState;

// ─────────────────────────────────────────────────────────────────────────────
// Text cleaning helpers  (display-only — raw text preserved in r.text for export)
// ─────────────────────────────────────────────────────────────────────────────

// Strip all HTML tags, returning plain text.
function plainText(html) {
  return (html || '').replace(/<[^>]+>/g, '');
}

// Remove lines that are obviously citation/URL noise:
//   • bare URLs (http://...)
//   • lines where punctuation density indicates a reference list entry
//   • very short lines of just reference numbers like [1] or ibid.
function removeNoisyLines(text) {
  return text
    .split('\n')
    .filter(line => {
      const t = line.trim();
      if (!t) return false;
      if (/^https?:\/\/\S+$/.test(t)) return false;
      if (/^\[\d{1,3}\]\s*$/.test(t)) return false;
      if (/^ibid\.?$/i.test(t)) return false;
      // Lines where URL-like chars dominate and the line is short → likely citation
      const urlish = (t.match(/[\/:.#%?&=]/g) || []).length;
      if (urlish / t.length > 0.28 && t.length < 160) return false;
      return true;
    })
    .join(' ')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

// Remove sentences whose first 55 characters already appeared (overlap dedup).
function deduplicateSentences(text) {
  const parts = text.split(/(?<=[.!?])\s+/);
  const seen  = new Set();
  return parts
    .filter(s => {
      const norm = s.toLowerCase().replace(/\s+/g, ' ').trim();
      if (norm.length < 25) return true; // keep short fragments
      const fp = norm.slice(0, 55);
      if (seen.has(fp)) return false;
      seen.add(fp);
      return true;
    })
    .join(' ');
}

// Find the window of maxWords words with the densest query-term coverage,
// then centre the excerpt there. Falls back gracefully when no terms match.
function centreOnMatch(text, matchedTerms, maxWords) {
  const words = text.split(/\s+/);
  if (words.length <= maxWords) return text;

  const terms = (matchedTerms || []).map(t => t.toLowerCase().replace(/[^a-z0-9]/g, ''));
  if (terms.length === 0) {
    // No terms — return the first maxWords words
    return words.slice(0, maxWords).join(' ') + (words.length > maxWords ? '…' : '');
  }

  // Score each word position: 1 if it matches any term, 0 otherwise
  const hits = words.map(w => {
    const wn = w.toLowerCase().replace(/[^a-z0-9]/g, '');
    return terms.some(t => wn === t || (wn.length >= 4 && t.length >= 4 && wn.startsWith(t.slice(0, 4)))) ? 1 : 0;
  });

  // Sliding-window sum to find the densest region
  let windowSum = hits.slice(0, maxWords).reduce((a, b) => a + b, 0);
  let bestSum   = windowSum;
  let bestStart = 0;
  for (let i = 1; i <= words.length - maxWords; i++) {
    windowSum += hits[i + maxWords - 1] - hits[i - 1];
    if (windowSum > bestSum) { bestSum = windowSum; bestStart = i; }
  }

  // Shift start slightly back so there's leading context before the first hit
  const firstHit = hits.slice(bestStart, bestStart + maxWords).indexOf(1);
  const shift    = Math.min(firstHit, Math.floor(maxWords * 0.25));
  const start    = Math.max(0, bestStart - shift);
  const end      = Math.min(words.length, start + maxWords);

  // Only add ellipsis when we've actually cut a meaningful amount (>4 words)
  const prefix = start > 4 ? '… ' : '';
  const suffix = end < words.length - 4 ? ' …' : '';
  return prefix + words.slice(start, end).join(' ') + suffix;
}

// Build a clean display preview from the raw chunk.
// Full original r.text is preserved in the expand view and export.
function makePreview(rawHtml, matchedTerms, maxWords = 65) {
  const plain   = plainText(rawHtml);
  const cleaned = removeNoisyLines(plain);
  const deduped = deduplicateSentences(cleaned);
  return centreOnMatch(deduped, matchedTerms, maxWords);
}

// ─────────────────────────────────────────────────────────────────────────────
// Highlight helpers
// ─────────────────────────────────────────────────────────────────────────────

const STOPWORDS = new Set([
  'the','and','how','what','should','a','an','of','in','to','is','are','be',
  'for','that','this','with','or','not','it','as','at','by','from','on',
  'was','were','been','have','has','had','which','who','its','if','but',
  'can','will','may','would','could','more','also','such','than','into',
  'their','they','we','our','us','do','did','does','been','about','when',
]);

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Pick useful highlight terms from matched_terms (preferred) or fall back to
// splitting the raw query string. Filters stopwords and very short tokens.
function deriveTerms(matchedTerms, query) {
  const raw = (matchedTerms && matchedTerms.length > 0)
    ? matchedTerms
    : (query || '').toLowerCase().split(/\s+/);
  return raw.filter(t => t.length >= 3 && !STOPWORDS.has(t.toLowerCase()));
}

// Inline highlight — for short snippets (collapsed preview).
// No paragraph/line-break conversion; just HTML-escape + highlight spans.
function applyHighlightsInline(plain, terms) {
  let s = (plain || '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  if (terms.length > 0) {
    s = s.replace(new RegExp(`(${terms.map(escapeRegex).join('|')})`, 'gi'),
      '<span class="hi">$1</span>');
  }
  return s;
}

// Block highlight — for full passages. Preserves paragraph/line spacing.
function applyHighlights(plain, terms) {
  // 1. HTML-escape
  let s = (plain || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  // 2. Wrap matched terms
  if (terms.length > 0) {
    const pattern = terms.map(escapeRegex).join('|');
    s = s.replace(new RegExp(`(${pattern})`, 'gi'), '<span class="hi">$1</span>');
  }
  // 3. Paragraph / line-break conversion
  s = s.replace(/\n\n+/g, '</p><p style="margin:6px 0 0 0">').replace(/\n/g, '<br>');
  return '<p style="margin:0">' + s + '</p>';
}

// Human-readable labels for known source filenames.
const SOURCE_LABELS = {
  'PostOfficeHorizon-I- Inquiry-Recomm.pdf':          'Post Office Horizon Inquiry Recommendations',
  'PostOfficeHorizon-IT-Inquiry-Response.pdf':         'Post Office Horizon Inquiry Response',
  'Behaviour-Change-Report-Recomm.pdf':                'Behaviour Change Report Recommendations',
  'Behaviour-Change-Response.pdf':                     'Behaviour Change Response',
  'UK-Covid-19-Inquiry-Module-1-Recomm.pdf':           'UK Covid-19 Inquiry Module 1 Recommendations',
  'UK-Covid-19_Inquiry_Module_1_Response.pdf':         'UK Covid-19 Inquiry Module 1 Response',
  'Volume_1-Blood-Inquiry-Recomm.pdf':                 'Infected Blood Inquiry Recommendations',
  'Volume_1-Blood-Inquiry-Response.pdf':               'Infected Blood Inquiry Response',
  'TheSpaceEconomyReport.pdf':                         'Space Economy Report Recommendations',
  'TheSpaceEconomyResponse.pdf':                       'Space Economy Response',
};

function sourceLabel(filename) {
  if (!filename) return '';
  if (SOURCE_LABELS[filename]) return SOURCE_LABELS[filename];
  // Generic clean-up for unknown filenames
  return filename.replace(/\.pdf$/i, '').replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
}

// Heading: prefer backend heading, fall back to readable source label + page.
function headingFor(result) {
  if (result.heading && result.heading.trim()) return result.heading.trim();
  return `${sourceLabel(result.source)} · p. ${result.page_number}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Download helpers
// ─────────────────────────────────────────────────────────────────────────────

function downloadFile(content, filename, mime) {
  const blob = new Blob([content], { type: mime });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

function isoStamp() {
  return new Date().toISOString().slice(0, 19).replace(/:/g, '-');
}

function searchResultsToJSON(results, query, retriever, meta) {
  return JSON.stringify({
    exported_at:   new Date().toISOString(),
    query, retriever,
    total_results: results.length,
    elapsed_ms:    meta?.elapsed_ms ?? null,
    results: results.map(r => ({
      rank:            r.rank,
      source:          r.source,
      pair_id:         r.pair_id ?? null,
      page_number:     r.page_number,
      chunk_id:        r.chunk_id ?? null,
      score:           r.score,
      confidence:      r.confidence ?? null,
      heading:         r.heading ?? null,
      heading_display: headingFor(r),
      text:            plainText(r.text || ''),          // full raw text
      display_preview: makePreview(r.text, r.matched_terms), // clean preview
      matched_terms:   r.matched_terms ?? [],
    })),
  }, null, 2);
}

function searchResultsToCSV(results, query, retriever) {
  const esc = v => {
    const s = String(v ?? '');
    return s.includes(',') || s.includes('"') || s.includes('\n')
      ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  const header = ['rank','query','retriever','source','pair_id','page','score',
                   'confidence','heading','text','display_preview'];
  const rows = results.map(r => [
    r.rank, query, retriever, r.source, r.pair_id ?? '',
    r.page_number, (r.score ?? 0).toFixed(6), r.confidence ?? '',
    headingFor(r),
    plainText(r.text || ''),
    makePreview(r.text, r.matched_terms),
  ].map(esc).join(','));
  return [header.join(','), ...rows].join('\r\n');
}

// ─────────────────────────────────────────────────────────────────────────────
// Loading hooks
// ─────────────────────────────────────────────────────────────────────────────

function useSearchProgress(loading) {
  const [progress, setProgress] = useState(0);
  const [done, setDone]         = useState(false);
  const rafRef   = useRef(null);
  const startRef = useRef(null);

  useEffect(() => {
    if (!loading) {
      setProgress(0); setDone(false);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      return;
    }
    setProgress(10);
    startRef.current = performance.now();
    const tick = now => {
      const p = 88 * (1 - Math.exp(-(now - startRef.current) / 14000)) + 10;
      setProgress(Math.min(p, 88));
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [loading]);

  useEffect(() => {
    if (!done) return;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    setProgress(100);
  }, [done]);

  return [progress, setDone];
}

function useElapsedTimer(active) {
  const [elapsed, setElapsed] = useState(0);
  const t0Ref = useRef(null);
  const ivRef = useRef(null);
  useEffect(() => {
    if (active) {
      t0Ref.current = Date.now(); setElapsed(0);
      ivRef.current = setInterval(() => setElapsed(Date.now() - t0Ref.current), 100);
    } else {
      if (ivRef.current) { clearInterval(ivRef.current); ivRef.current = null; }
    }
    return () => { if (ivRef.current) clearInterval(ivRef.current); };
  }, [active]);
  return elapsed;
}

// ─────────────────────────────────────────────────────────────────────────────

const RETRIEVER_LABELS = {
  tfidf:    'Running keyword search…',
  semantic: 'Running semantic search…',
  hybrid:   'Running hybrid search…',
};
const HYBRID_SLOW_NOTE = 'Hybrid search may take longer on first run while the model and index warm up.';

// ─────────────────────────────────────────────────────────────────────────────
// SearchScreen
// ─────────────────────────────────────────────────────────────────────────────

const SearchScreen = () => {
  const { presets, activePresetId, status } = useContext(window.AppContext);

  const [query, setQuery]         = useState('');
  const [scope, setScope]         = useState('all');
  const [topK, setTopK]           = useState(5);
  const [retriever, setRetriever] = useState('hybrid');
  const [selectedRank, setSelectedRank] = useState(null);
  const [results, setResults]     = useState(null);
  const [meta, setMeta]           = useState(null);
  const [loading, setLoading]     = useState(false);
  const [searchError, setSearchError]   = useState(null);
  const [completionMsg, setCompletionMsg]         = useState(null);
  const [completionVisible, setCompletionVisible] = useState(false);
  const [showSlowNote, setShowSlowNote] = useState(false);
  const completionTimerRef = useRef(null);
  const reqIdRef           = useRef(0);

  const [progress, setProgressDone] = useSearchProgress(loading);
  const elapsed = useElapsedTimer(loading);

  const semanticAvailable = status?.semantic_available ?? true;
  const presetStatuses    = status?.preset_statuses || {};

  // Presets that have any cached state (available for searching)
  const loadedPresets = useMemo(
    () => presets.filter(p => presetStatuses[p.id] && presetStatuses[p.id] !== 'error'),
    [presets, presetStatuses],
  );

  // Hybrid slow-note
  const slowNoteTimerRef = useRef(null);
  useEffect(() => {
    if (loading && retriever === 'hybrid') {
      slowNoteTimerRef.current = setTimeout(() => setShowSlowNote(true), 2000);
    } else {
      setShowSlowNote(false);
      if (slowNoteTimerRef.current) clearTimeout(slowNoteTimerRef.current);
    }
    return () => { if (slowNoteTimerRef.current) clearTimeout(slowNoteTimerRef.current); };
  }, [loading, retriever]);

  const groups = useMemo(() => {
    const out = {};
    for (const p of presets) {
      const g = p.dataset_group || 'other';
      out[g] ||= { id: g, label: p.group_label || g, items: [] };
      out[g].items.push(p);
    }
    return Object.values(out);
  }, [presets]);

  const scopeLabel = scope === 'all'
    ? 'All loaded pairs'
    : presets.find(p => p.id === scope)?.label || 'Active pair';

  const doSearch = useCallback(async () => {
    if (!query.trim() || loading) return;
    const myReqId = ++reqIdRef.current;
    const t0 = Date.now();
    setLoading(true);
    setSearchError(null);
    setCompletionMsg(null);
    setCompletionVisible(false);
    if (completionTimerRef.current) { clearTimeout(completionTimerRef.current); completionTimerRef.current = null; }
    setResults(null); setMeta(null); setSelectedRank(null);
    try {
      const body = { query: query.trim(), top_k: topK, retriever, scope: scope === 'all' ? 'all' : 'current' };
      if (scope !== 'all' && scope !== activePresetId) {
        await window.apiFetch(`/api/pipeline/activate/${scope}`, { method: 'POST', body: '{}' });
      }
      if (myReqId !== reqIdRef.current) return;
      const data = await window.apiFetch('/api/search/', { method: 'POST', body: JSON.stringify(body) });
      if (myReqId !== reqIdRef.current) return;
      const ranked = (data.results || []).map((r, i) => ({ ...r, rank: r.rank ?? i + 1 }));
      setProgressDone(true);
      await new Promise(r => setTimeout(r, 220));
      if (myReqId !== reqIdRef.current) return;
      setResults(ranked); setMeta(data); setSelectedRank(ranked[0]?.rank ?? null);
      const secs = ((Date.now() - t0) / 1000).toFixed(1);
      const msg  = `Found ${ranked.length} result${ranked.length !== 1 ? 's' : ''} in ${secs}s`;
      setCompletionMsg(msg); setCompletionVisible(true);
      if (completionTimerRef.current) clearTimeout(completionTimerRef.current);
      completionTimerRef.current = setTimeout(() => {
        setCompletionVisible(false);
        completionTimerRef.current = setTimeout(() => setCompletionMsg(null), 600);
      }, 3000);
    } catch (err) {
      if (myReqId !== reqIdRef.current) return;
      setSearchError(err.message);
    } finally {
      if (myReqId === reqIdRef.current) setLoading(false);
    }
  }, [query, topK, retriever, scope, activePresetId, loading, setProgressDone]);

  const handleKey = e => { if (e.key === 'Enter') doSearch(); };

  const displayResults = results ? results.slice(0, topK) : [];
  const selected = displayResults.find(r => r.rank === selectedRank) || displayResults[0] || null;
  const elapsedSec   = (elapsed / 1000).toFixed(1);
  const loadingLabel = RETRIEVER_LABELS[retriever] || 'Searching…';

  // Corpus info for scope=all: which pairs are loaded, any extras?
  const corpusInfo = useMemo(() => {
    if (scope !== 'all') return null;
    const coursework = loadedPresets.filter(p => p.dataset_group === 'coursework_given');
    const extra      = loadedPresets.filter(p => p.dataset_group !== 'coursework_given');
    return { coursework, extra, total: loadedPresets.length };
  }, [scope, loadedPresets]);

  return (
    <div className="screen fade-in">
      <ScreenHeader
        title="Hybrid passage retrieval"
        subtitle="Search the loaded chunk corpus across one pair or all loaded pairs. Hybrid combines TF-IDF lexical scoring with sentence-transformer semantic similarity."
        meta={
          <div style={{display:"flex", alignItems:"center", gap:8}}>
            <span className="pill solid-grey">{scopeLabel}</span>
            {loadedPresets.length > 0 && (
              <span className="pill solid-grey">{loadedPresets.length} pair{loadedPresets.length !== 1 ? 's' : ''} in corpus</span>
            )}
          </div>
        }
      />

      {/* ── Query bar card ── */}
      <div className="card">
        <div style={{padding:"16px 18px 14px 18px"}}>
          <div style={{display:"flex", gap:10, alignItems:"flex-end"}}>
            <div style={{flex:1, minWidth:0}}>
              <div className="search-input">
                <Icon name="search" size={14}/>
                <input
                  className="input lg"
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  onKeyDown={handleKey}
                  placeholder="e.g. investment targets, government response, recommendations accepted"
                  disabled={loading}
                />
                {query && (
                  <button
                    type="button"
                    className="search-clear"
                    onClick={() => setQuery('')}
                    disabled={loading}
                    title="Clear search"
                    aria-label="Clear search"
                  >
                    <Icon name="x" size={12}/>
                  </button>
                )}
              </div>
            </div>
            <button className="btn primary lg" onClick={doSearch} disabled={loading || !query.trim()}>
              {loading
                ? <span className="spinner" style={{width:12, height:12}}/>
                : <Icon name="search" size={13}/>}
              {loading ? 'Searching…' : 'Search'}
              {!loading && <span className="kbd">↵</span>}
            </button>
          </div>

          <div style={{display:"flex", gap:18, alignItems:"flex-end", marginTop:14, flexWrap:"wrap"}}>
            <div className="field-group" style={{flex:"1 1 240px"}}>
              <span className="field-label">Pair scope</span>
              <select className="select" value={scope} onChange={e => setScope(e.target.value)} disabled={loading}>
                <option value="all">All loaded pairs ({loadedPresets.length})</option>
                {groups.map(g => (
                  <optgroup key={g.id} label={g.label}>
                    {g.items.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
                  </optgroup>
                ))}
              </select>
            </div>

            <div className="field-group" style={{flex:"1 1 200px"}}>
              <span className="field-label">
                Top K · <span className="mono" style={{color:"var(--ink-2)"}}>{topK}</span>
              </span>
              <div className="range-row">
                <input type="range" min="1" max="20" value={topK}
                  onChange={e => setTopK(Number(e.target.value))} className="range" disabled={loading}/>
                <span className="range-val">{topK}</span>
              </div>
            </div>

            <div className="field-group" style={{flex: "0 1 auto"}}>
              <span className="field-label">Retriever</span>
              <div className="segmented">
                <button className={"seg" + (retriever === "tfidf" ? " on" : "")}
                  onClick={() => !loading && setRetriever("tfidf")} disabled={loading}>
                  TF-IDF <span className="muted" style={{fontSize:10.5, marginLeft:4}}>keyword</span>
                </button>
                <button className={"seg" + (retriever === "hybrid" ? " on" : "")}
                  onClick={() => !loading && setRetriever("hybrid")}
                  disabled={loading || !semanticAvailable}
                  title={!semanticAvailable ? "Hybrid needs sentence-transformers" : "α-blend of TF-IDF and semantic"}>
                  Hybrid <span style={{fontSize:9.5, marginLeft:4, color: retriever === "hybrid" ? "var(--accent-ink)" : "var(--faint)"}}>recommended</span>
                </button>
                <button className={"seg" + (retriever === "semantic" ? " on" : "")}
                  onClick={() => !loading && setRetriever("semantic")}
                  disabled={loading || !semanticAvailable}
                  title={!semanticAvailable ? "Semantic needs sentence-transformers" : "Bi-encoder embeddings"}>
                  Semantic <span className="muted" style={{fontSize:10.5, marginLeft:4}}>meaning-based</span>
                </button>
              </div>
            </div>
          </div>

          {/* Corpus scope info — shown when searching all loaded pairs */}
          {corpusInfo && corpusInfo.total > 0 && !loading && (
            <div style={{
              marginTop: 12, padding: "8px 12px",
              background: "var(--surface-2)", borderRadius: "var(--r)",
              border: "1px solid var(--line-soft)",
              fontSize: 11.5, color: "var(--muted)",
              display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8,
            }}>
              <Icon name="info" size={11}/>
              <span style={{color:"var(--ink-3)", fontWeight:500}}>Searching corpus:</span>
              {corpusInfo.coursework.map(p => (
                <span key={p.id} className="pill solid-grey" style={{fontSize:10, padding:"1px 7px"}}>{p.label}</span>
              ))}
              {corpusInfo.extra.length > 0 && (
                <>
                  <span style={{color:"var(--muted)"}}>+ extra:</span>
                  {corpusInfo.extra.map(p => (
                    <span key={p.id} className="pill" style={{
                      fontSize:10, padding:"1px 7px",
                      background:"rgba(245,158,11,0.1)", color:"#b45309",
                      border:"1px solid rgba(245,158,11,0.35)",
                    }}>{p.label}</span>
                  ))}
                  <span style={{fontSize:10.5, color:"var(--muted)", fontStyle:"italic"}}>
                    — extra/extension docs may appear in results
                  </span>
                </>
              )}
              {corpusInfo.total === 0 && (
                <span style={{color:"var(--warn,#d97706)"}}>No pairs loaded — load a pair on the Documents tab first.</span>
              )}
            </div>
          )}
        </div>

        {/* Meta strip */}
        {meta && !loading && (
          <div style={{
            padding:"10px 18px",
            background: "var(--surface-2)",
            borderTop: "1px solid var(--line)",
            fontSize: 12,
            display:"flex", flexWrap:"wrap", alignItems:"center",
            gap: 14, color:"var(--ink-3)",
          }}>
            <MetaStrip label="results"   value={displayResults.length}/>
            <span className="divider-v"/>
            <MetaStrip label="top score" value={(meta.top_score ?? 0).toFixed(4)} mono/>
            {meta.elapsed_ms != null && <><span className="divider-v"/>
              <MetaStrip label="ran in" value={meta.elapsed_ms + " ms"} mono/></>}
            {meta.chunks_searched != null && <><span className="divider-v"/>
              <MetaStrip label="corpus" value={meta.chunks_searched + " chunks"}/></>}
            {meta.query_type && <><span className="divider-v"/>
              <MetaStrip label="query" value={meta.query_type}/></>}
            {meta.alpha != null && <><span className="divider-v"/>
              <MetaStrip label="α blend" value={meta.alpha.toFixed(2)} mono/></>}
          </div>
        )}

        {/* Loading progress bar */}
        {loading && (
          <div className="fade-in" style={{
            padding:"14px 18px 16px 18px",
            borderTop: "1px solid var(--line)",
            background: "var(--surface-2)",
          }}>
            <div style={{
              height: 5, borderRadius: 999,
              background: "var(--surface-soft)",
              border: "1px solid var(--line)",
              overflow: "hidden", marginBottom: 10,
            }}>
              <div style={{
                height: "100%", width: progress + "%",
                background: "linear-gradient(90deg, var(--accent-2), var(--accent))",
                borderRadius: 999,
                transition: progress >= 100 ? "width 180ms ease-out" : "width 400ms linear",
              }}/>
            </div>
            <div style={{display:"flex", alignItems:"center", gap:10}}>
              <span className="spinner" style={{width:13, height:13, flexShrink:0}}/>
              <span style={{fontSize:12.5, fontWeight:600, color:"var(--ink-2)", flex:1}}>
                {loadingLabel}
                {' '}
                <span className="mono tabnum" style={{
                  fontSize:11.5, fontWeight:500, color:"var(--accent)",
                  background:"var(--accent-softer)", padding:"1px 6px", borderRadius:4,
                }}>{elapsedSec}s</span>
              </span>
            </div>
            {showSlowNote && (
              <div className="fade-in" style={{
                marginTop: 10, fontSize: 11.5, color: "var(--muted)", lineHeight: 1.5,
                display:"flex", alignItems:"flex-start", gap:6,
              }}>
                <Icon name="info" size={11}/>
                <span>{HYBRID_SLOW_NOTE}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Completion banner */}
      {completionMsg && !loading && (
        <div style={{
          padding:"9px 14px",
          background:"var(--ok-bg,rgba(34,197,94,0.08))",
          border:"1px solid var(--ok)",
          borderRadius:"var(--r-md)",
          color:"var(--ok)", fontSize:12,
          display:"flex", alignItems:"center", gap:8, fontWeight:500,
          opacity: completionVisible ? 1 : 0,
          transition: "opacity 600ms ease-out",
          pointerEvents: completionVisible ? "auto" : "none",
        }}>
          <Icon name="check" size={12}/>{completionMsg}
        </div>
      )}

      {/* Error banner */}
      {searchError && (
        <div style={{
          padding:"10px 14px",
          background:"var(--err-bg)", border:"1px solid var(--err-bd)",
          borderRadius:"var(--r-md)", color:"var(--err-fg)", fontSize:12,
          display:"flex", gap:10, alignItems:"center",
        }}>
          <Icon name="warning" size={13}/>
          <span>Search failed: {searchError}</span>
        </div>
      )}

      {/* ── Results + preview ── */}
      {displayResults.length > 0 && !loading && (
        <div style={{display:"grid", gridTemplateColumns:"minmax(0, 6fr) minmax(0, 4fr)", gap:18, alignItems:"start"}}>
          <div>
            <SectionLabel count={`${displayResults.length} results`}>Ranked passages</SectionLabel>
            <div style={{display:"flex", flexDirection:"column", gap:8}}>
              {displayResults.map(r => (
                <ResultCard
                  key={r.rank}
                  r={r}
                  query={query}
                  selected={r.rank === selectedRank}
                  topScore={meta?.top_score || 1}
                  onSelect={() => setSelectedRank(r.rank)}
                />
              ))}
            </div>
            <div style={{
              marginTop: 14, padding: "10px 14px",
              display:"flex", alignItems:"center", gap:10,
              border: "1px solid var(--line)", borderRadius: "var(--r)",
              background: "var(--surface)", fontSize:12,
            }}>
              <span style={{color:"var(--muted)"}}>
                {displayResults.length} results · scope: <b style={{color:"var(--ink)"}}>{scopeLabel}</b>
              </span>
              <span className="grow" style={{flex:1}}/>
              <button className="btn sm" disabled={displayResults.length === 0}
                onClick={() => downloadFile(
                  searchResultsToCSV(displayResults, query, retriever),
                  `search-results-${isoStamp()}.csv`, 'text/csv')}>
                <Icon name="download" size={11}/> Export CSV
              </button>
              <button className="btn sm" disabled={displayResults.length === 0}
                onClick={() => downloadFile(
                  searchResultsToJSON(displayResults, query, retriever, meta),
                  `search-results-${isoStamp()}.json`, 'application/json')}>
                <Icon name="download" size={11}/> Export JSON
              </button>
            </div>
          </div>

          <div style={{position:"sticky", top: 90}}>
            <SectionLabel>Source preview</SectionLabel>
            <SourcePreviewPanel result={selected} query={query}/>
          </div>
        </div>
      )}

      {!loading && results === null && !searchError && (
        <div style={{padding:'48px 0', textAlign:'center', color:'var(--muted)', fontSize:13}}>
          <Icon name="search" size={24}/>
          <div style={{marginTop:10}}>Enter a query above and press Search.</div>
          {loadedPresets.length === 0 && (
            <div style={{marginTop:6, fontSize:12, color:"var(--warn,#d97706)"}}>
              No document pairs are loaded yet — go to Documents to load a pair first.
            </div>
          )}
        </div>
      )}

      {!loading && results !== null && displayResults.length === 0 && (
        <EmptyState icon="search" title="No results"
          message="Try a different query or widen the pair scope."/>
      )}
    </div>
  );
};

const MetaStrip = ({ label, value, mono }) => (
  <span style={{display:"inline-flex", alignItems:"baseline", gap:5}}>
    <span style={{color:"var(--muted)", fontSize:11.5}}>{label}</span>
    <span className={mono ? "mono tabnum" : ""} style={{fontWeight:600, color:"var(--ink)", fontSize:12}}>{value}</span>
  </span>
);

// ─────────────────────────────────────────────────────────────────────────────
// ResultCard — clean preview by default, full passage expandable
// ─────────────────────────────────────────────────────────────────────────────

const ResultCard = ({ r, query, selected, topScore, onSelect }) => {
  const [expanded, setExpanded] = useState(false);
  const pct        = topScore > 0 ? (r.score / topScore) * 100 : 0;
  const isResponse = (r.source || '').toLowerCase().includes("response");
  const heading    = headingFor(r);
  const terms      = deriveTerms(r.matched_terms, query);

  // Clean plain-text preview, then inline-highlighted (no <p> wrapping)
  const previewPlain = makePreview(r.text, r.matched_terms); // 65 words default
  const previewHtml  = applyHighlightsInline(previewPlain, terms);

  // Full passage: strip any residual markup from backend text, then highlight
  const fullPlain = removeNoisyLines(plainText(r.text || ''));
  const fullHtml  = applyHighlights(fullPlain, terms);

  const hasFullText = fullPlain.length > 0;

  return (
    <div
      onClick={onSelect}
      style={{
        padding: "13px 16px",
        background: selected ? "var(--accent-softer)" : "var(--surface)",
        border: "1px solid " + (selected ? "var(--accent)" : "var(--line)"),
        borderRadius: "var(--r-md)",
        boxShadow: selected ? "var(--sh-accent)" : "none",
        cursor: "pointer",
        transition: "all 160ms ease-out",
      }}>

      {/* Header row */}
      <div style={{display:"flex", alignItems:"center", gap:8, marginBottom:6}}>
        <span style={{
          fontFamily:"var(--mono)", fontSize: 12, fontWeight: 700,
          color: selected ? "var(--accent)" : "var(--ink)", letterSpacing:"-0.01em",
        }}>#{r.rank}</span>
        <span className={"pill " + (isResponse ? "solid-grey" : "indigo")} style={{fontSize:10}}>
          {isResponse ? "response" : "policy"}
        </span>
        <span className="pill solid-grey mono" style={{fontSize:10.5}}>p. {r.page_number}</span>
        {r.confidence && (
          <span className={"pill " + (r.confidence === "high" ? "ok" : r.confidence === "medium" ? "warn" : "grey")}
            style={{fontSize: 9.5, padding:"1px 6px", textTransform:"uppercase", letterSpacing:"0.05em"}}>
            {r.confidence}
          </span>
        )}
        <span className="grow" style={{flex:1}}/>
        <span style={{fontSize:11, color:"var(--muted)", fontWeight:500, maxWidth:200, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap"}}>
          {sourceLabel(r.source)}
        </span>
      </div>

      {/* Heading */}
      <div style={{fontSize:12, fontWeight:600, color:"var(--ink-2)", marginBottom:6, lineHeight:1.3}}>
        {heading}
      </div>

      {/* Score bar */}
      <div style={{display:"flex", alignItems:"center", gap:10, marginBottom:8}}>
        <div style={{
          flex:1, height: 4, background: "var(--surface-soft)",
          borderRadius: 999, overflow: "hidden", border: "1px solid var(--line)",
        }}>
          <div style={{height:"100%", width: pct + "%", background: "linear-gradient(90deg, var(--accent-2), var(--accent))", borderRadius: 999}}/>
        </div>
        <span className="mono tabnum" style={{fontSize:11.5, fontWeight:600, color:"var(--ink-2)", minWidth:54, textAlign:"right"}}>
          {(r.score || 0).toFixed(4)}
        </span>
      </div>

      {/* Preview with subtle highlights (collapsed) */}
      {!expanded && (
        <div
          className="search-hl"
          style={{fontSize: 13, lineHeight: 1.55, color: "var(--ink-2)"}}
          dangerouslySetInnerHTML={{__html: previewHtml}}
        />
      )}

      {/* Full passage with highlights (expanded) */}
      {expanded && (
        <div
          className="search-hl"
          style={{fontSize: 13, lineHeight: 1.6, color: "var(--ink-2)"}}
          dangerouslySetInnerHTML={{__html: fullHtml}}
        />
      )}

      {/* Matched terms — show up to 5 pills, then "+N more" */}
      {r.matched_terms?.length > 0 && (
        <div style={{marginTop: 8, fontSize: 11.5, color: "var(--muted)", display:"flex", flexWrap:"wrap", gap:5, alignItems:"center"}}>
          <span>Matched:</span>
          {r.matched_terms.slice(0, 5).map((t, i) => (
            <span key={i} className="mono" style={{
              padding:"1px 6px", background:"var(--surface-soft)", border:"1px solid var(--line)",
              borderRadius: 3, fontSize: 10.5, color:"var(--ink-2)",
            }}>{t}</span>
          ))}
          {r.matched_terms.length > 5 && (
            <span style={{fontSize:10.5, color:"var(--faint)"}}>+{r.matched_terms.length - 5} more</span>
          )}
        </div>
      )}

      {/* Expand / collapse toggle — hidden when no text to show */}
      {hasFullText && (
      <div style={{marginTop:8}}>
        <button
          onClick={e => { e.stopPropagation(); setExpanded(x => !x); }}
          style={{
            background:"none", border:0, padding:"3px 0", color:"var(--muted)",
            cursor:"pointer", display:"inline-flex", alignItems:"center", gap:4,
            fontFamily:"inherit", fontSize: 11.5,
          }}>
          <span style={{transition:"transform 160ms", transform: expanded ? "rotate(90deg)" : "rotate(0)", display:"inline-flex"}}>
            <Icon name="chevRight" size={10}/>
          </span>
          {expanded ? 'Collapse passage' : 'View full passage'}
        </button>
      </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// SourcePreviewPanel  (reused on screen 3)
// ─────────────────────────────────────────────────────────────────────────────

const SourcePreviewPanel = ({ result, query }) => {
  if (!result) return <EmptyState icon="docs" title="No result selected"/>;
  const isResponse   = (result.source || '').toLowerCase().includes("response");
  const heading      = headingFor(result);
  const terms        = deriveTerms(result.matched_terms, query);
  const previewPlain = makePreview(result.text, result.matched_terms, 90); // right panel slightly longer
  const previewHtml  = applyHighlights(previewPlain, terms);
  const fullPlain    = removeNoisyLines(plainText(result.text || ''));
  const fullHtml     = applyHighlights(fullPlain, terms);

  return (
    <div className="card slide-in">
      <div style={{padding:"14px 18px", borderBottom:"1px solid var(--line)", background: "var(--surface-2)"}}>
        <div style={{display:"flex", alignItems:"center", gap:8, fontSize:13, fontWeight:600, color:"var(--ink)", letterSpacing:"-0.005em"}}>
          <Icon name="file" size={13}/>
          <span className="truncate" style={{flex:1}}>{result.source}</span>
        </div>
        <div style={{marginTop:5, fontSize:12, fontWeight:600, color:"var(--ink-2)"}}>
          {heading}
        </div>
        <div style={{display:"flex", gap:10, marginTop:10, flexWrap:"wrap", alignItems:"center"}}>
          <span className="kv"><span className="k">page</span><span className="v mono tabnum">{result.page_number}</span></span>
          <span className="dot-sep"/>
          {result.chunk_id && <>
            <span className="kv"><span className="k">chunk</span>
              <code className="mono" style={{background:"var(--surface-soft)", padding:"1px 6px", borderRadius:3, fontSize:11}}>{result.chunk_id}</code>
            </span>
            <span className="dot-sep"/>
          </>}
          <span className="kv"><span className="k">score</span><span className="v mono tabnum">{(result.score || 0).toFixed(4)}</span></span>
          {result.confidence && (
            <><span className="dot-sep"/>
              <span className={"pill " + (result.confidence === "high" ? "ok" : result.confidence === "medium" ? "warn" : "grey")}>
                {result.confidence}
              </span>
            </>
          )}
          <span className="grow" style={{flex:1}}/>
          <span className={"pill " + (isResponse ? "solid-grey" : "indigo")}>{isResponse ? "Response doc" : "Policy doc"}</span>
        </div>
      </div>

      <div className="mini-pdf" style={{margin: 14, padding: "16px 22px"}}>
        <div className="pg-header">
          <span>{result.source}</span>
          <span>page {result.page_number}</span>
        </div>
        <div className="pg-h" style={{marginBottom:8}}>{heading}</div>
        {/* Highlighted preview */}
        <div
          className="search-hl"
          style={{color:"var(--ink-2)", fontSize:13, lineHeight:1.6, marginBottom:10}}
          dangerouslySetInnerHTML={{__html: previewHtml}}
        />
        {fullPlain.length > previewPlain.length && (
          <details style={{fontSize:12, color:"var(--muted)"}}>
            <summary style={{cursor:"pointer", marginBottom:6, userSelect:"none"}}>Full passage with highlights</summary>
            <div
              className="search-hl"
              style={{marginTop:8, fontSize:13, lineHeight:1.6, color:"var(--ink-2)"}}
              dangerouslySetInnerHTML={{__html: fullHtml}}/>
            {result.context_before && <p style={{color:"var(--ink-3)", fontStyle:"italic"}}>{result.context_before}</p>}
            {result.context_after  && <p style={{color:"var(--ink-3)", fontStyle:"italic"}}>{result.context_after}</p>}
          </details>
        )}
        <div className="pg-num">— {result.page_number} —</div>
      </div>
    </div>
  );
};

window.SearchScreen = SearchScreen;
window.SourcePreviewPanel = SourcePreviewPanel;
