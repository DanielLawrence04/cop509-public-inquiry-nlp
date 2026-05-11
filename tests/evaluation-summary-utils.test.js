const assert = require('assert');
const utils = require('../frontend/src/evaluation-summary.js');

const sampleMap = {
  pair_a: {
    recommendations: [
      {
        rec_id: 1,
        item_label: '1',
        page_number: 10,
        text: 'Recommendation one',
        classification: 'accepted',
        overall_confidence: 0.9,
        alignment_confidence: 0.8,
        classification_confidence: 0.95,
        classifier_method: 'rule_based',
        best_match: { page_number: 11, matched_text: 'Accepted response', match_method: 'semantic', alignment_confidence: 0.8 },
      },
      {
        rec_id: 2,
        item_label: '2',
        page_number: 12,
        text: 'Recommendation two',
        classification: 'rejected',
        overall_confidence: 0.4,
        alignment_confidence: 0.45,
        classification_confidence: 0.7,
        confidence_factors: ['low_alignment_for_stance -0.15'],
        best_match: { page_number: 13, matched_text: 'Rejected response', match_method: 'chunk_fallback', alignment_confidence: 0.45 },
      },
    ],
  },
  pair_b: {
    recommendations: [
      {
        rec_id: 3,
        item_label: '3',
        page_number: 20,
        text: 'Recommendation three',
        classification: 'not_addressed',
        overall_confidence: 0.55,
        alignment_confidence: 0.6,
        classification_confidence: 0.65,
        best_match: { page_number: 21, matched_text: 'No response', match_method: 'sequence_correction', alignment_confidence: 0.6 },
      },
      {
        rec_id: 4,
        item_label: '4',
        page_number: 22,
        text: 'Recommendation four',
        classification: 'partial',
        overall_confidence: 0.7,
        alignment_confidence: 0.75,
        classification_confidence: 0.8,
        best_match: { page_number: 23, matched_text: 'Partial response', match_method: 'structure', alignment_confidence: 0.75 },
      },
    ],
    evaluation: {
      accuracy: 0.75,
      precision_macro: 0.7,
      recall_macro: 0.8,
      f1_macro: 0.72,
      confusion_matrix: [[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]],
      per_class: { accepted: { f1: 1, precision: 1, recall: 1, support: 1 } },
    },
  },
};

const rows = utils.flattenRecommendations(sampleMap);

{
  const classified = utils.getClassifiedRows(rows, 'all');
  const summary = utils.buildPredictionSummary(classified);
  assert.strictEqual(classified.length, 4, 'renders from current classified pipeline output without labels.json');
  assert.strictEqual(summary.total, 4, 'aggregate/current loaded summary works');
  assert.strictEqual(summary.labelCounts.accepted, 1);
  assert.strictEqual(summary.labelCounts.partial, 1);
  assert.strictEqual(summary.labelCounts.rejected, 1);
  assert.strictEqual(summary.labelCounts.not_addressed, 1);
  assert.deepStrictEqual(summary.pairCounts, { pair_a: 2, pair_b: 2 });
  assert.strictEqual(summary.lowConfidenceRows.length, 1);
  assert.strictEqual(summary.confidenceFactorRows.length, 1);
  assert.strictEqual(summary.fallbackOrSequenceRows.length, 2);
}

{
  const pairRows = utils.getClassifiedRows(rows, 'pair_b');
  const summary = utils.buildPredictionSummary(pairRows);
  assert.strictEqual(pairRows.length, 2, 'pair filtering works');
  assert.deepStrictEqual(summary.pairCounts, { pair_b: 2 });
  assert.strictEqual(summary.labelCounts.partial, 1);
}

{
  const empty = utils.getClassifiedRows([], 'all');
  assert.strictEqual(empty.length, 0, 'empty classified state shows Run Recommendation Analysis first in UI');
}

{
  const csv = utils.predictionsToCSV(utils.getClassifiedRows(rows, 'all'));
  assert(csv.startsWith('document_pair,id,recommendation_page,matched_response_page,classification'), 'CSV export header works');
  assert(csv.includes('pair_a,1,10,11,accepted'), 'CSV export rows work');
  assert(!csv.includes('true_label,correct'), 'missing labels.json no longer implies benchmark columns');
}

{
  const csv = utils.predictionsToCSV(utils.getClassifiedRows(rows, 'pair_a'), { 'pair_a::1': 'accepted' });
  assert(csv.includes('true_label,correct'), 'CSV includes ground-truth columns when labels are present');
  assert(csv.includes('accepted,true'), 'CSV marks correct predictions when labels are present');
}

{
  assert.strictEqual(utils.hasGroundTruth(sampleMap.pair_a), false, 'missing labels.json no longer causes an unavailable error');
  assert.strictEqual(utils.hasGroundTruth(sampleMap.pair_b), true, 'existing ground-truth evaluation still works if labels are present');
  const benchmark = utils.buildBenchmarkData(sampleMap.pair_b, 'Pair B');
  assert.strictEqual(benchmark.accuracy, 0.75);
  assert.strictEqual(benchmark.support, 2);
}

console.log('evaluation-summary-utils tests passed');
