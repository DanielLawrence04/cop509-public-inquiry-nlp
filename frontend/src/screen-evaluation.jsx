// Screen 4 - Evaluation Summary

const { useMemo, useContext, useState } = React;
const Icon               = window.Icon;
const ScreenHeader       = window.ScreenHeader;
const ClassificationPill = window.ClassificationPill;

const EvaluationScreen = () => {
  const { task2Data, task2DataMap, activePresetId, presets } = useContext(window.AppContext);
  const utils = window.EvaluationSummaryUtils;
  const [pairFilter, setPairFilter] = useState('all');

  const loadedMap = useMemo(() => {
    if (task2DataMap && Object.keys(task2DataMap).length > 0) return task2DataMap;
    if (task2Data?.recommendations?.length) return { [activePresetId || 'current']: task2Data };
    return {};
  }, [task2DataMap, task2Data, activePresetId]);

  const allRows = useMemo(
    () => utils.flattenRecommendations(loadedMap, window.adaptRec),
    [loadedMap]
  );

  const classifiedRows = useMemo(
    () => utils.getClassifiedRows(allRows, pairFilter),
    [allRows, pairFilter]
  );

  const summary = useMemo(
    () => utils.buildPredictionSummary(classifiedRows),
    [classifiedRows]
  );

  const pairOptions = useMemo(() => {
    const loadedIds = Object.keys(loadedMap);
    return loadedIds.map(id => ({
      id,
      label: presets.find(p => p.id === id)?.label || id,
    }));
  }, [loadedMap, presets]);

  const selectedData = pairFilter !== 'all' ? loadedMap[pairFilter] : null;
  const benchmarkPair = pairFilter !== 'all'
    ? pairOptions.find(p => p.id === pairFilter)?.label
    : pairOptions.length === 1 ? pairOptions[0].label : 'Loaded results';
  const benchmarkData = pairFilter !== 'all'
    ? utils.buildBenchmarkData(selectedData, benchmarkPair)
    : pairOptions.length === 1
      ? utils.buildBenchmarkData(loadedMap[pairOptions[0]?.id], benchmarkPair)
      : null;
  const hasBenchmark = Boolean(benchmarkData);

  const groundTruthByUid = useMemo(() => {
    const out = {};
    for (const row of classifiedRows) {
      const truth = row.true_label || row.ground_truth_label || row.gold_label;
      if (truth) out[row._uid] = truth;
    }
    return out;
  }, [classifiedRows]);

  const exportCSV = () => {
    const csv = utils.predictionsToCSV(classifiedRows, groundTruthByUid);
    const ts = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `evaluation-predictions-${ts}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  if (classifiedRows.length === 0) {
    return (
      <div className="screen fade-in">
        <ScreenHeader
          title="Evaluation summary"
          subtitle="Prediction summary metrics for currently loaded classified recommendations."
        />
        <div className="card" style={{padding:"40px 24px", textAlign:"center"}}>
          <div style={{fontSize: 32, marginBottom: 10}}>-</div>
          <div style={{fontSize: 14, fontWeight: 600, color:"var(--ink)"}}>No classified recommendations are loaded yet.</div>
          <div style={{fontSize: 12.5, color:"var(--muted)", marginTop: 6, maxWidth: 460, margin: "6px auto 0"}}>
            Run Recommendation Analysis first.
          </div>
        </div>
      </div>
    );
  }

  const titleScope = pairFilter === 'all'
    ? `${pairOptions.length || 1} loaded pair${(pairOptions.length || 1) === 1 ? '' : 's'}`
    : (pairOptions.find(p => p.id === pairFilter)?.label || pairFilter);

  return (
    <div className="screen fade-in">
      <ScreenHeader
        title="Evaluation summary"
        subtitle={`Prediction summary for ${titleScope}.`}
        meta={
          <div style={{display:"flex", alignItems:"center", gap:8, flexWrap:"wrap"}}>
            <select className="select" style={{width:220}} value={pairFilter} onChange={e => setPairFilter(e.target.value)}>
              <option value="all">All loaded results</option>
              {pairOptions.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
            </select>
            <span className="pill solid-grey mono">{summary.total} classified</span>
            <button className="btn" onClick={exportCSV}><Icon name="download" size={12}/> Predictions (CSV)</button>
          </div>
        }
      />

      {!hasBenchmark && (
        <div className="card" style={{padding:"12px 16px", fontSize:12.5, color:"var(--ink-3)"}}>
          Manual ground-truth labels are not loaded, so this page is showing prediction-only summary metrics.
        </div>
      )}

      <div style={{display:"grid", gridTemplateColumns:"repeat(4, 1fr)", gap: 14}}>
        <KPICard label="Classified rows" value={summary.total} sublabel="current selection" format="int" tone="accent"/>
        <KPICard label="Overall confidence" value={summary.averageOverallConfidence} sublabel="average" format="dec"/>
        <KPICard label="Alignment confidence" value={summary.averageAlignmentConfidence} sublabel="average" format="dec"/>
        <KPICard label="Classifier confidence" value={summary.averageClassificationConfidence} sublabel="average" format="dec"/>
      </div>

      <DistributionCard summary={summary}/>

      <div style={{display:"grid", gridTemplateColumns:"minmax(0, 1fr) minmax(0, 1fr)", gap:14, alignItems:"start"}}>
        <CountTable title="Rows by document pair" subtitle="Current loaded state" counts={summary.pairCounts}/>
        <CountTable title="Rows by alignment method" subtitle="Matcher provenance" counts={summary.methodCounts}/>
      </div>

      <div style={{display:"grid", gridTemplateColumns:"repeat(4, 1fr)", gap: 14}}>
        <KPICard label="Low confidence" value={summary.lowConfidenceRows.length} sublabel={`overall < ${summary.lowConfidenceThreshold}`} format="int"/>
        <KPICard label="Confidence factors" value={summary.confidenceFactorRows.length} sublabel="rows with factors" format="int"/>
        <KPICard label="Rejected" value={summary.rejectedRows.length} sublabel="rows to inspect" format="int"/>
        <KPICard label="Not addressed" value={summary.notAddressedRows.length} sublabel="rows to inspect" format="int"/>
      </div>

      <IssueRowsCard
        sections={[
          ['Rejected rows', summary.rejectedRows],
          ['Not addressed rows', summary.notAddressedRows],
          ['Low-confidence rows', summary.lowConfidenceRows],
          ['Rows with confidence factors', summary.confidenceFactorRows],
          ['Fallback or sequence-corrected alignment', summary.fallbackOrSequenceRows],
        ]}
      />

      {hasBenchmark && (
        <>
          <div style={{display:"grid", gridTemplateColumns:"repeat(4, 1fr)", gap: 14}}>
            <KPICard label="Accuracy" value={benchmarkData.accuracy} sublabel={`${benchmarkData.correct}/${benchmarkData.support} correct`} format="pct" tone="accent"/>
            <KPICard label="F1 (macro)" value={benchmarkData.f1_macro} sublabel="avg per-label" format="dec"/>
            <KPICard label="Precision (macro)" value={benchmarkData.precision_macro} sublabel="avg per-label" format="dec"/>
            <KPICard label="Recall (macro)" value={benchmarkData.recall_macro} sublabel="valid matched ground truth" format="dec"/>
          </div>
          <div style={{display:"grid", gridTemplateColumns:"minmax(0, 5fr) minmax(0, 7fr)", gap: 14, alignItems:"start"}}>
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">Confusion matrix</h3>
                <span className="card-sub mono">true x predicted - n = {benchmarkData.support}</span>
              </div>
              <div style={{padding: "8px 16px 18px"}}>
                <ConfusionMatrix matrix={benchmarkData.confusion_matrix} labels={utils.LABELS}/>
              </div>
            </div>
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">Per-label metrics</h3>
                <span className="card-sub">Precision, recall and F1 by class</span>
              </div>
              <PerLabelMetricsTable per_class={benchmarkData.per_class} order={utils.LABELS}/>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

const classFill = (l) => ({
  accepted: "var(--ok)",
  partial: "var(--warn)",
  rejected: "var(--err)",
  not_addressed: "var(--muted-2)",
}[l] || "var(--accent)");

const KPICard = ({ label, value, sublabel, format, tone }) => {
  const display = format === "pct" ? ((value || 0) * 100).toFixed(1) + "%"
    : format === "int" ? String(value || 0)
    : (value || 0).toFixed(3);
  return (
    <div className="card kpi" style={{
      padding:"16px 18px 14px",
      borderColor: tone === "accent" ? "var(--accent-border)" : "var(--line)",
      background: tone === "accent"
        ? "linear-gradient(180deg, var(--accent-softer) 0%, var(--surface) 60%)"
        : "var(--surface)",
    }}>
      <div className="field-label" style={{margin:0}}>{label}</div>
      <div style={{
        marginTop: 4, fontSize: 32, fontWeight: 600, color: tone === "accent" ? "var(--accent)" : "var(--ink)",
        fontFeatureSettings:'"tnum"', fontVariantNumeric:"tabular-nums",
      }}>{display}</div>
      <div style={{fontSize: 11.5, color:"var(--muted)", marginTop: 2}}>{sublabel}</div>
    </div>
  );
};

const DistributionCard = ({ summary }) => (
  <div className="card">
    <div className="card-header">
      <h3 className="card-title">Predicted label distribution</h3>
      <span className="card-sub">Across {summary.total} classified recommendations</span>
    </div>
    <div style={{padding: "10px 16px 18px", display:"grid", gridTemplateColumns:"repeat(4, 1fr)", gap: 12}}>
      {window.EvaluationSummaryUtils.LABELS.map(label => {
        const n = summary.labelCounts[label] ?? 0;
        const pct = summary.labelPercentages[label] ?? 0;
        return (
          <div key={label} style={{padding:"14px 14px 12px", background:"var(--surface)", border:"1px solid var(--line)", borderRadius:"var(--r-md)"}}>
            <div style={{display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom: 8}}>
              <ClassificationPill label={label}/>
              <span className="mono tabnum" style={{fontSize:11, color:"var(--muted)"}}>{pct.toFixed(1)}%</span>
            </div>
            <div style={{display:"flex", alignItems:"baseline", gap:6, marginBottom: 8}}>
              <span style={{fontSize: 26, fontWeight: 600, color:"var(--ink)"}}>{n}</span>
              <span style={{fontSize: 11.5, color:"var(--muted)"}}>recs</span>
            </div>
            <div style={{height: 4, background:"var(--surface-soft)", borderRadius: 999, overflow:"hidden"}}>
              <div style={{height:"100%", width: pct + "%", background: classFill(label), borderRadius: 999}}/>
            </div>
          </div>
        );
      })}
    </div>
  </div>
);

const CountTable = ({ title, subtitle, counts }) => (
  <div className="card">
    <div className="card-header">
      <h3 className="card-title">{title}</h3>
      <span className="card-sub">{subtitle}</span>
    </div>
    <table className="utable plain">
      <thead><tr><th>Name</th><th className="num">Rows</th></tr></thead>
      <tbody>
        {Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
          <tr key={k}><td><span className="mono">{k}</span></td><td className="num"><span className="mono tabnum">{v}</span></td></tr>
        ))}
      </tbody>
    </table>
  </div>
);

const IssueRowsCard = ({ sections }) => (
  <div className="card">
    <div className="card-header">
      <h3 className="card-title">Rows to inspect</h3>
      <span className="card-sub">Prediction-only diagnostic groups</span>
    </div>
    <div style={{padding:"8px 16px 18px", display:"flex", flexDirection:"column", gap:10}}>
      {sections.map(([title, rows]) => (
        <details key={title} className="evidence" open={rows.length > 0}>
          <summary style={{cursor:"pointer", fontSize:13, fontWeight:600, color:"var(--ink)"}}>
            {title} <span className="mono tabnum" style={{color:"var(--muted)"}}>({rows.length})</span>
          </summary>
          {rows.length === 0 ? (
            <div style={{padding:"8px 0", color:"var(--muted)", fontSize:12}}>No rows in this group.</div>
          ) : (
            <table className="utable plain" style={{marginTop:8}}>
              <thead><tr><th>Pair</th><th>ID</th><th>Label</th><th className="num">Confidence</th><th>Method</th></tr></thead>
              <tbody>
                {rows.slice(0, 20).map(row => (
                  <tr key={row._uid}>
                    <td><span className="mono">{row.pair_id}</span></td>
                    <td><span className="mono">{row.item_label ?? row.rec_id}</span></td>
                    <td><ClassificationPill label={row.classification || row.best_label}/></td>
                    <td className="num"><span className="mono tabnum">{(row.overall_confidence ?? 0).toFixed(3)}</span></td>
                    <td><span className="method-tag">{window.EvaluationSummaryUtils.getAlignmentMethod(row)}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </details>
      ))}
    </div>
  </div>
);

const ConfusionMatrix = ({ matrix, labels }) => {
  const max = Math.max(1, ...matrix.flat());
  const totals = matrix.map(r => r.reduce((a,b) => a+b, 0));
  const cellBg = (v, isDiag) => {
    if (v === 0) return "var(--surface-2)";
    const t = v / max;
    return isDiag ? `oklch(0.96 0.04 175 / ${0.25 + t * 0.65})` : `oklch(0.96 0.025 30 / ${0.2 + t * 0.5})`;
  };

  return (
    <table className="cm-table">
      <thead>
        <tr>
          <th></th>
          {labels.map(l => <th key={l}><ClassificationPill label={l} compact/></th>)}
          <th><span style={{fontSize:10, color:"var(--muted)", fontWeight:600}}>total</span></th>
        </tr>
      </thead>
      <tbody>
        {matrix.map((row, i) => (
          <tr key={i}>
            <th><ClassificationPill label={labels[i]} compact/></th>
            {row.map((v, j) => (
              <td key={j} style={{background: cellBg(v, i === j), border: "1px solid var(--line-soft)"}}>
                <span className="mono tabnum" style={{fontSize:14, fontWeight:i === j ? 700 : 500}}>{v}</span>
              </td>
            ))}
            <td className="total-cell"><span className="mono tabnum" style={{fontSize:12, color:"var(--ink-3)"}}>{totals[i]}</span></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

const PerLabelMetricsTable = ({ per_class, order }) => (
  <table className="utable plain">
    <thead><tr><th>Label</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">Support</th></tr></thead>
    <tbody>
      {order.map(l => {
        const m = per_class[l] || {};
        return (
          <tr key={l}>
            <td><ClassificationPill label={l}/></td>
            <td className="num"><MetricCell value={m.precision}/></td>
            <td className="num"><MetricCell value={m.recall}/></td>
            <td className="num"><MetricCell value={m.f1}/></td>
            <td className="num"><span className="mono tabnum" style={{color:"var(--ink-3)"}}>{m.support ?? 0}</span></td>
          </tr>
        );
      })}
    </tbody>
  </table>
);

const MetricCell = ({ value }) => {
  const v = value ?? 0;
  return <span className="mono tabnum" style={{fontSize:12, fontWeight:600, color:"var(--ink-2)"}}>{v.toFixed(3)}</span>;
};

window.EvaluationScreen = EvaluationScreen;
