(function(root) {
  const LABELS = ['accepted', 'partial', 'rejected', 'not_addressed'];
  const LABEL_ALIASES = {
    accepted: 'accepted',
    partially_accepted: 'partial',
    partial: 'partial',
    rejected: 'rejected',
    not_addressed: 'not_addressed',
    'not addressed': 'not_addressed',
  };
  const LOW_CONFIDENCE_THRESHOLD = 0.5;

  function normaliseLabel(label) {
    return LABEL_ALIASES[String(label || '').trim().toLowerCase()] || null;
  }

  function toNumber(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function average(values) {
    const nums = values.map(toNumber).filter(v => v != null);
    return nums.length ? nums.reduce((s, v) => s + v, 0) / nums.length : 0;
  }

  function getAlignmentMethod(row) {
    return row.best_match?.match_method || row.alignment_method || row.match_method || 'unknown';
  }

  function hasFallbackOrSequence(row) {
    const method = getAlignmentMethod(row);
    return /fallback|sequence/i.test(method) || Boolean(row.sequence_corrected);
  }

  function flattenRecommendations(task2DataMap, adaptRec) {
    return Object.entries(task2DataMap || {}).flatMap(([pairId, data]) =>
      (data?.recommendations || []).map(raw => {
        const adapted = adaptRec ? adaptRec({ ...raw, pair_id: pairId }) : { ...raw, pair_id: pairId };
        return {
          ...adapted,
          pair_id: adapted.pair_id || pairId,
          _uid: `${pairId}::${adapted.rec_id ?? adapted.item_label ?? adapted.id ?? ''}`,
        };
      })
    );
  }

  function getClassifiedRows(rows, pairId) {
    return (rows || []).filter(row => {
      if (pairId && pairId !== 'all' && row.pair_id !== pairId) return false;
      return Boolean(normaliseLabel(row.classification || row.best_label));
    });
  }

  function buildPredictionSummary(rows) {
    const total = rows.length;
    const labelCounts = Object.fromEntries(LABELS.map(label => [label, 0]));
    const pairCounts = {};
    const methodCounts = {};
    const rejectedRows = [];
    const notAddressedRows = [];
    const lowConfidenceRows = [];
    const confidenceFactorRows = [];
    const fallbackOrSequenceRows = [];

    for (const row of rows) {
      const label = normaliseLabel(row.classification || row.best_label) || 'not_addressed';
      labelCounts[label] = (labelCounts[label] || 0) + 1;
      pairCounts[row.pair_id || row.document || 'unknown'] = (pairCounts[row.pair_id || row.document || 'unknown'] || 0) + 1;

      const method = getAlignmentMethod(row);
      methodCounts[method] = (methodCounts[method] || 0) + 1;

      const overallConfidence = toNumber(row.overall_confidence ?? row.confidence);
      if (overallConfidence != null && overallConfidence < LOW_CONFIDENCE_THRESHOLD) lowConfidenceRows.push(row);
      if (Array.isArray(row.confidence_factors) && row.confidence_factors.length > 0) confidenceFactorRows.push(row);
      if (hasFallbackOrSequence(row)) fallbackOrSequenceRows.push(row);
      if (label === 'rejected') rejectedRows.push(row);
      if (label === 'not_addressed') notAddressedRows.push(row);
    }

    return {
      total,
      labelCounts,
      labelPercentages: Object.fromEntries(LABELS.map(label => [label, total ? (labelCounts[label] / total) * 100 : 0])),
      pairCounts,
      methodCounts,
      averageOverallConfidence: average(rows.map(row => row.overall_confidence ?? row.confidence)),
      averageAlignmentConfidence: average(rows.map(row => row.alignment_confidence ?? row.best_match?.alignment_confidence ?? row.best_match?.similarity)),
      averageClassificationConfidence: average(rows.map(row => row.classification_confidence)),
      lowConfidenceRows,
      confidenceFactorRows,
      rejectedRows,
      notAddressedRows,
      fallbackOrSequenceRows,
      lowConfidenceThreshold: LOW_CONFIDENCE_THRESHOLD,
    };
  }

  function hasGroundTruth(data) {
    return Boolean(data?.evaluation);
  }

  function buildBenchmarkData(data, pairLabel) {
    const ev = data?.evaluation;
    if (!ev) return null;
    const recs = data?.recommendations || [];
    return {
      pair_label: pairLabel || 'Selected pair',
      accuracy: ev.accuracy ?? 0,
      f1_macro: ev.f1_macro ?? 0,
      precision_macro: ev.precision_macro ?? 0,
      recall_macro: ev.recall_macro ?? 0,
      correct: Math.round((ev.accuracy ?? 0) * recs.length),
      support: recs.length,
      confusion_matrix: ev.confusion_matrix || [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
      per_class: {
        accepted: ev.per_class?.accepted || {},
        partial: ev.per_class?.partially_accepted || ev.per_class?.partial || {},
        rejected: ev.per_class?.rejected || {},
        not_addressed: ev.per_class?.not_addressed || {},
      },
    };
  }

  function csvEscape(value) {
    const s = Array.isArray(value) ? value.join('; ') : String(value ?? '');
    return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }

  function predictionsToCSV(rows, groundTruthByUid) {
    const hasTruth = groundTruthByUid && Object.keys(groundTruthByUid).length > 0;
    const header = [
      'document_pair', 'id', 'recommendation_page', 'matched_response_page',
      'classification', 'confidence', 'alignment_method', 'alignment_confidence',
      'classification_confidence', 'classifier_method', 'confidence_factors',
      'recommendation_text', 'matched_response_text',
    ];
    if (hasTruth) header.push('true_label', 'correct');

    const lines = [header.join(',')];
    for (const row of rows) {
      const truth = hasTruth ? groundTruthByUid[row._uid] : null;
      const classification = normaliseLabel(row.classification || row.best_label) || '';
      const values = [
        row.pair_id || row.document || '',
        row.item_label ?? row.rec_id ?? row.id ?? '',
        row.page_number ?? '',
        row.best_match?.page_number ?? '',
        classification,
        row.overall_confidence ?? row.confidence ?? '',
        getAlignmentMethod(row),
        row.alignment_confidence ?? row.best_match?.alignment_confidence ?? row.best_match?.similarity ?? '',
        row.classification_confidence ?? '',
        row.classifier_method ?? row.classification_method ?? '',
        row.confidence_factors || [],
        row.text ?? row.recommendation ?? '',
        row.best_match?.matched_text ?? row.best_match?.response_text ?? '',
      ];
      if (hasTruth) {
        const trueLabel = normaliseLabel(truth);
        values.push(trueLabel || '', trueLabel ? String(trueLabel === classification) : '');
      }
      lines.push(values.map(csvEscape).join(','));
    }
    return lines.join('\r\n');
  }

  const api = {
    LABELS,
    LOW_CONFIDENCE_THRESHOLD,
    normaliseLabel,
    flattenRecommendations,
    getClassifiedRows,
    buildPredictionSummary,
    buildBenchmarkData,
    hasGroundTruth,
    predictionsToCSV,
    getAlignmentMethod,
    hasFallbackOrSequence,
  };

  root.EvaluationSummaryUtils = api;
  if (typeof module !== 'undefined') module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
