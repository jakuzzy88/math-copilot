/**
 * Candidate pipeline tests.
 *
 * Sprint D: Comprehensive tests for the OCR result pipeline adapter.
 *
 * Test categories:
 *   1. Valid candidate wins
 *   2. Corrected candidate wins over invalid raw candidate
 *   3. Impossible strings rejected
 *   4. Lower confidence valid beats higher confidence invalid
 *   5. One equals sign required
 *   6. Unsupported symbols rejected
 *   7. Edge cases (empty input, single candidate, etc.)
 *   8. Scoring correctness
 *   9. Determinism
 */

import { processCandidates, evaluateCandidate } from '../pipeline/candidatePipeline';
import type { OcrCandidate, PipelineAcceptedResult, PipelineRejectedResult } from '../pipeline/types';

// ── Helper ──────────────────────────────────────────────────────────

function accepted(result: ReturnType<typeof processCandidates>): PipelineAcceptedResult {
  expect(result.accepted).toBe(true);
  return result as PipelineAcceptedResult;
}

function rejected(result: ReturnType<typeof processCandidates>): PipelineRejectedResult {
  expect(result.accepted).toBe(false);
  return result as PipelineRejectedResult;
}


// ═══════════════════════════════════════════════════════════════════════
// 1. Valid candidate wins
// ═══════════════════════════════════════════════════════════════════════

describe('Pipeline — valid candidate wins', () => {
  test('single valid candidate is accepted', () => {
    const result = accepted(processCandidates([
      { text: '3x+4=10', confidence: 0.82 },
    ]));
    expect(result.equation).toBe('3x+4=10');
    expect(result.solution).toBe('x=2');
    expect(result.explanationSteps.length).toBeGreaterThan(0);
    expect(result.sourceCandidate.text).toBe('3x+4=10');
  });

  test('highest confidence valid candidate wins from multiple valid', () => {
    const result = accepted(processCandidates([
      { text: '2x=8', confidence: 0.60 },
      { text: '3x+4=10', confidence: 0.90 },
      { text: 'x+5=9', confidence: 0.50 },
    ]));
    // 3x+4=10 has highest confidence and all are valid → it should win
    expect(result.equation).toBe('3x+4=10');
    expect(result.solution).toBe('x=2');
  });

  test('accepted result contains explanation steps', () => {
    const result = accepted(processCandidates([
      { text: '5x-2=18', confidence: 0.85 },
    ]));
    expect(result.explanationSteps.length).toBeGreaterThan(0);
    const lastStep = result.explanationSteps[result.explanationSteps.length - 1];
    expect(lastStep.isFinal).toBe(true);
  });

  test('accepts parenthesised equation', () => {
    const result = accepted(processCandidates([
      { text: '2(x+1)=10', confidence: 0.75 },
    ]));
    expect(result.equation).toBe('2(x+1)=10');
    expect(result.solution).toBe('x=4');
  });

  test('accepts division equation', () => {
    const result = accepted(processCandidates([
      { text: 'x/2=5', confidence: 0.80 },
    ]));
    expect(result.solution).toBe('x=10');
  });
});


// ═══════════════════════════════════════════════════════════════════════
// 2. Corrected candidate wins over invalid raw candidate
// ═══════════════════════════════════════════════════════════════════════

describe('Pipeline — corrected candidate wins', () => {
  test('O corrected to 0 makes candidate valid', () => {
    const result = accepted(processCandidates([
      { text: '3x+4=1O', confidence: 0.82 },
    ]));
    // '1O' → '10'
    expect(result.equation).toBe('3x+4=10');
    expect(result.solution).toBe('x=2');
    expect(result.corrections.length).toBeGreaterThan(0);
    expect(result.corrections).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ original: 'O', replacement: '0' }),
      ]),
    );
  });

  test('I corrected to 1 makes candidate valid', () => {
    const result = accepted(processCandidates([
      { text: '3x+4=I0', confidence: 0.78 },
    ]));
    expect(result.equation).toBe('3x+4=10');
    expect(result.solution).toBe('x=2');
  });

  test('corrected lower-confidence beats uncorrectable higher-confidence', () => {
    const result = accepted(processCandidates([
      { text: '3x++4=10', confidence: 0.95 },  // grammar invalid, uncorrectable
      { text: '3x+4=1O', confidence: 0.70 },    // correctable to 3x+4=10
    ]));
    expect(result.equation).toBe('3x+4=10');
    expect(result.solution).toBe('x=2');
    expect(result.sourceCandidate.text).toBe('3x+4=1O');
  });

  test('X normalised to x', () => {
    const result = accepted(processCandidates([
      { text: '3X+4=10', confidence: 0.80 },
    ]));
    expect(result.equation).toBe('3x+4=10');
  });

  test('spaces removed', () => {
    const result = accepted(processCandidates([
      { text: '3x + 4 = 10', confidence: 0.80 },
    ]));
    expect(result.equation).toBe('3x+4=10');
    expect(result.solution).toBe('x=2');
  });
});


// ═══════════════════════════════════════════════════════════════════════
// 3. Impossible strings rejected
// ═══════════════════════════════════════════════════════════════════════

describe('Pipeline — impossible strings rejected', () => {
  test('random garbage is rejected', () => {
    const result = rejected(processCandidates([
      { text: 'abcdef', confidence: 0.90 },
    ]));
    expect(result.rejection.code).toBe('ALL_GRAMMAR_INVALID');
  });

  test('empty candidates array is rejected', () => {
    const result = rejected(processCandidates([]));
    expect(result.rejection.code).toBe('NO_CANDIDATES');
  });

  test('empty string candidate is rejected', () => {
    const result = rejected(processCandidates([
      { text: '', confidence: 0.50 },
    ]));
    expect(result.rejection.code).toBe('ALL_GRAMMAR_INVALID');
  });

  test('all candidates invalid gives rejection with error details', () => {
    const result = rejected(processCandidates([
      { text: '3x++4=10', confidence: 0.80 },
      { text: '!!!', confidence: 0.60 },
    ]));
    expect(result.rejection.candidateErrors.length).toBe(2);
    expect(result.rejection.candidateErrors[0].errors.length).toBeGreaterThan(0);
  });
});


// ═══════════════════════════════════════════════════════════════════════
// 4. Lower confidence valid beats higher confidence invalid
// ═══════════════════════════════════════════════════════════════════════

describe('Pipeline — validity trumps confidence', () => {
  test('low confidence valid candidate beats high confidence invalid', () => {
    const result = accepted(processCandidates([
      { text: '3x+4=1O', confidence: 0.10 },  // correctable, solvable
      { text: '???=!!!', confidence: 0.99 },   // grammar invalid
    ]));
    expect(result.equation).toBe('3x+4=10');
    expect(result.solution).toBe('x=2');
    expect(result.sourceCandidate.confidence).toBe(0.10);
  });

  test('valid candidate with 0.05 confidence still accepted over invalid 0.99', () => {
    const result = accepted(processCandidates([
      { text: '#$%^&', confidence: 0.99 },
      { text: 'x+5=9', confidence: 0.05 },
    ]));
    expect(result.equation).toBe('x+5=9');
    expect(result.solution).toBe('x=4');
  });
});


// ═══════════════════════════════════════════════════════════════════════
// 5. One equals sign required
// ═══════════════════════════════════════════════════════════════════════

describe('Pipeline — equals sign validation', () => {
  test('rejects equation with no equals sign', () => {
    const result = rejected(processCandidates([
      { text: '3x+4', confidence: 0.80 },
    ]));
    expect(result.rejection.code).toBe('ALL_GRAMMAR_INVALID');
    expect(result.rejection.candidateErrors[0].errors).toEqual(
      expect.arrayContaining([expect.stringContaining('Missing equals')]),
    );
  });

  test('rejects equation with two equals signs', () => {
    const result = rejected(processCandidates([
      { text: '3x+4=10=5', confidence: 0.80 },
    ]));
    expect(result.rejection.code).toBe('ALL_GRAMMAR_INVALID');
    expect(result.rejection.candidateErrors[0].errors).toEqual(
      expect.arrayContaining([expect.stringContaining('equals signs')]),
    );
  });
});


// ═══════════════════════════════════════════════════════════════════════
// 6. Unsupported symbols rejected
// ═══════════════════════════════════════════════════════════════════════

describe('Pipeline — unsupported symbols', () => {
  test('rejects equation with !', () => {
    const result = rejected(processCandidates([
      { text: '3x+4=10!', confidence: 0.80 },
    ]));
    expect(result.rejection.candidateErrors[0].errors).toEqual(
      expect.arrayContaining([expect.stringContaining('Unsupported character')]),
    );
  });

  test('rejects equation with @', () => {
    const result = rejected(processCandidates([
      { text: '3x@4=10', confidence: 0.80 },
    ]));
    expect(result.rejection.code).toBe('ALL_GRAMMAR_INVALID');
  });

  test('rejects equation with ^', () => {
    const result = rejected(processCandidates([
      { text: 'x^2=4', confidence: 0.80 },
    ]));
    expect(result.rejection.code).toBe('ALL_GRAMMAR_INVALID');
  });
});


// ═══════════════════════════════════════════════════════════════════════
// 7. Edge cases
// ═══════════════════════════════════════════════════════════════════════

describe('Pipeline — edge cases', () => {
  test('the example from the spec produces correct output', () => {
    const result = accepted(processCandidates([
      { text: '3x+4=10', confidence: 0.82 },
      { text: '3x+4=1O', confidence: 0.10 },
      { text: '3x++4=10', confidence: 0.05 },
    ]));
    expect(result.accepted).toBe(true);
    expect(result.equation).toBe('3x+4=10');
    expect(result.solution).toBe('x=2');
    expect(result.explanationSteps.length).toBeGreaterThan(0);
  });

  test('non-integer solution is formatted correctly', () => {
    // 3x = 10 → x = 10/3 ≈ 3.333...
    const result = accepted(processCandidates([
      { text: '3x=10', confidence: 0.80 },
    ]));
    expect(result.solution).toMatch(/^x=3\.3+/);
  });

  test('negative solution works', () => {
    // 2x+10=0 → x = -5
    const result = accepted(processCandidates([
      { text: '2x+10=0', confidence: 0.80 },
    ]));
    expect(result.solution).toBe('x=-5');
  });

  test('score is a number between 0 and 1', () => {
    const result = accepted(processCandidates([
      { text: '3x+4=10', confidence: 0.82 },
    ]));
    expect(result.score).toBeGreaterThan(0);
    expect(result.score).toBeLessThanOrEqual(1);
  });
});


// ═══════════════════════════════════════════════════════════════════════
// 8. Scoring correctness
// ═══════════════════════════════════════════════════════════════════════

describe('Pipeline — scoring', () => {
  test('valid candidate has higher score than grammar-invalid candidate', () => {
    const valid = evaluateCandidate({ text: '3x+4=10', confidence: 0.50 });
    const invalid = evaluateCandidate({ text: '3x++4=10', confidence: 0.50 });
    expect(valid.score).toBeGreaterThan(invalid.score);
  });

  test('solvable candidate has higher score than parseable-but-unsolvable', () => {
    const solvable = evaluateCandidate({ text: '3x+4=10', confidence: 0.50 });
    // x=x+5 is parseable but has no solution
    const unsolvable = evaluateCandidate({ text: 'x=x+5', confidence: 0.50 });
    expect(solvable.score).toBeGreaterThan(unsolvable.score);
  });

  test('corrected candidate has slightly lower score due to correction penalty', () => {
    const clean = evaluateCandidate({ text: '3x+4=10', confidence: 0.80 });
    const corrected = evaluateCandidate({ text: '3x+4=1O', confidence: 0.80 });
    expect(clean.score).toBeGreaterThan(corrected.score);
    // But the corrected one should still be solvable
    expect(corrected.solvable).toBe(true);
  });

  test('higher confidence produces higher score (all else equal)', () => {
    const high = evaluateCandidate({ text: '3x+4=10', confidence: 0.90 });
    const low = evaluateCandidate({ text: '3x+4=10', confidence: 0.50 });
    expect(high.score).toBeGreaterThan(low.score);
  });
});


// ═══════════════════════════════════════════════════════════════════════
// 9. Determinism
// ═══════════════════════════════════════════════════════════════════════

describe('Pipeline — determinism', () => {
  test('same input produces same output', () => {
    const candidates: OcrCandidate[] = [
      { text: '3x+4=10', confidence: 0.82 },
      { text: '3x+4=1O', confidence: 0.10 },
      { text: '3x++4=10', confidence: 0.05 },
    ];
    const a = processCandidates(candidates);
    const b = processCandidates(candidates);
    expect(a).toEqual(b);
  });

  test('candidate order does not affect which candidate wins (best score wins)', () => {
    const c1: OcrCandidate = { text: '3x+4=10', confidence: 0.82 };
    const c2: OcrCandidate = { text: '2x=8', confidence: 0.50 };

    const result1 = accepted(processCandidates([c1, c2]));
    const result2 = accepted(processCandidates([c2, c1]));

    // Both should pick the same winner (c1 has higher confidence)
    expect(result1.equation).toBe(result2.equation);
    expect(result1.solution).toBe(result2.solution);
  });
});
