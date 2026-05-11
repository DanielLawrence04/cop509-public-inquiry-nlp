// shim.jsx — small adapters shared across screens.

// ── Static validated-final-export adapter ─────────────────────
// Converts the canonical 246-row coursework export into the per-preset Task2
// shape that the Recommendation Analysis and Evaluation Summary screens
// already consume.
//
// Returns: { byPreset: {presetId: Task2ResultsResponse-shaped}, summary, source }
window.adaptFinalExport = function(payload) {
  const rows = (payload && payload.recommendations) || [];

  // Friendly labels for synthesised preset entries used when the backend
  // /api/pipeline/presets endpoint is unavailable.
  const PRESET_LABELS = {
    behaviour_change: { label: 'Behaviour Change', group: 'coursework_given', group_label: 'Coursework given documents' },
    post_office: { label: 'Post Office Horizon Inquiry', group: 'coursework_given', group_label: 'Coursework given documents' },
    space_economy: { label: 'The Space Economy', group: 'coursework_given', group_label: 'Coursework given documents' },
    covid_inquiry: { label: 'UK Covid-19 Inquiry Module 1', group: 'coursework_given', group_label: 'Coursework given documents' },
    blood_inquiry: { label: 'Infected Blood Inquiry', group: 'coursework_given', group_label: 'Coursework given documents' },
    grenfell_phase2: { label: 'Grenfell Tower Inquiry — Phase 2', group: 'extra_found', group_label: 'Extra documents found for extension/testing' },
    covid_inquiry_module2: { label: 'UK Covid-19 Inquiry Module 2', group: 'extra_found', group_label: 'Extra documents found for extension/testing' },
    summer_2024_disorder: { label: 'Police response to the 2024 summer disorder', group: 'extra_found', group_label: 'Extra documents found for extension/testing' },
  };

  // Frontend LABEL_MAP accepts both 'partial' and 'partially_accepted', so
  // pass the JSON's classification through unchanged as best_label.
  const toTask2Match = (row) => {
    const matchedText = row.matched_response_text;
    if (!matchedText) return null;
    const debug = row.debug || {};
    const align = Number(debug.alignment_confidence ?? row.confidence ?? 0) || 0;
    const lex = Number(debug.lexical_similarity ?? align) || 0;
    let page = row.matched_response_page;
    if (typeof page === 'string' && /^\d+$/.test(page)) page = Number(page);
    return {
      matched_chunk_id: null,
      matched_text: String(matchedText),
      source: null,
      page_number: typeof page === 'number' ? page : null,
      similarity: lex,
      alignment_confidence: align,
      label: row.classification || null,
      label_display: null,
      no_match: false,
      match_method: String(debug.alignment_method || 'validated_final_export'),
      boundary_reason: null,
      quoted_recommendation_text: null,
      heading_text: null,
    };
  };

  const toTask2Rec = (row, idx, presetId) => {
    const debug = row.debug || {};
    const bestMatch = toTask2Match(row);
    const cls = row.classification || 'not_addressed';
    const align = Number(debug.alignment_confidence ?? row.confidence ?? 0) || 0;
    const lex = Number(debug.lexical_similarity ?? align) || 0;
    const clsConf = debug.classification_confidence;
    // Display source: friendly label paired with stable pair id so the
    // marker can see both in the Document column / detail panel without
    // losing the canonical id from outputs/final_recommendations_246.json.
    const friendly = (PRESET_LABELS[presetId] && PRESET_LABELS[presetId].label) || presetId;
    const documentLabel = presetId ? `${friendly} (${presetId})` : friendly;
    return {
      rec_id: idx,
      item_label: String(row.id ?? idx),
      text: String(row.recommendation_text || ''),
      document: documentLabel,
      page_number: row.recommendation_page ?? null,
      detector: 'validated_final_export',
      extraction_method: 'validated_final_export',
      confidence: Number(row.confidence || 0),
      ocr: false,
      span_id: null,
      matches: bestMatch ? [bestMatch] : [],
      best_match: bestMatch,
      best_label: cls,
      label_display: null,
      best_similarity: lex,
      alignment_confidence: align,
      classification_confidence: clsConf != null ? Number(clsConf) : null,
      classifier_method: String(debug.classifier_method || 'rule_based'),
      classification_rationale: 'Loaded from validated coursework final export.',
      extraction_source: 'validated_final_export',
      source_document_role: 'policy',
      extraction_note: null,
      source_paragraph: null,
      source_item_type: null,
    };
  };

  const rowsByPair = {};
  const classificationDistribution = {};
  for (const row of rows) {
    const pair = String(row.document_pair || '');
    (rowsByPair[pair] ||= []).push(row);
    const cls = String(row.classification || 'not_addressed');
    classificationDistribution[cls] = (classificationDistribution[cls] || 0) + 1;
  }

  const byPreset = {};
  const pairCounts = {};
  const syntheticPresets = [];
  for (const [presetId, presetRows] of Object.entries(rowsByPair)) {
    const recs = presetRows.map((row, idx) => toTask2Rec(row, idx, presetId));
    const meanAlign = recs.length ? recs.reduce((s, r) => s + r.alignment_confidence, 0) / recs.length : 0;
    const meanExtract = recs.length ? recs.reduce((s, r) => s + r.confidence, 0) / recs.length : 0;
    byPreset[presetId] = {
      preset_id: presetId,
      stages: { load: 'done', extract: 'done', align: 'done', classify: 'done' },
      summary: {
        recommendations: recs.length,
        alignments: recs.filter(r => r.best_match).length,
        classified: recs.length,
        mean_extraction_confidence: meanExtract,
        mean_alignment_confidence: meanAlign,
      },
      recommendations: recs,
      evaluation: null,
      evaluation_status: 'Validated final export: 246-row coursework evidence. '
        + 'Full manual ground-truth labels are unavailable, so accuracy/F1 metrics '
        + 'are not computed here — see Notebook 2 for the prediction-only evaluation.',
    };
    pairCounts[presetId] = recs.length;
    const meta = PRESET_LABELS[presetId] || { label: presetId, group: 'coursework_given', group_label: 'Loaded results' };
    syntheticPresets.push({
      id: presetId,
      label: meta.label,
      dataset_group: meta.group,
      group_label: meta.group_label,
      group_description: '',
      is_extra: meta.group === 'extra_found',
    });
  }

  return {
    source: 'validated_final_export',
    exported_at: payload && payload.exported_at,
    summary: {
      total: rows.length,
      pair_counts: pairCounts,
      classification_distribution: classificationDistribution,
    },
    byPreset,
    syntheticPresets,
  };
};

window.adaptFinalResultsPayload = function(payload) {
  if (payload && payload.by_preset) {
    const byPreset = payload.by_preset || {};
    const syntheticPresets = Object.keys(byPreset).map(presetId => ({
      id: presetId,
      label: presetId,
      dataset_group: 'coursework_given',
      group_label: 'Loaded results',
      group_description: '',
      is_extra: false,
    }));
    return {
      source: payload.source || 'validated_final_export',
      exported_at: payload.exported_at,
      summary: payload.summary || {},
      byPreset,
      syntheticPresets,
    };
  }
  return window.adaptFinalExport(payload);
};
