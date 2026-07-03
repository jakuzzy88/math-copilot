/**
 * Tests for CTC decoder.
 *
 * Sprint 5C: Mobile Static Image Inference Preparation.
 */

import {
  ctcGreedyDecode,
  ctcGreedyDecodeFromLogProbs,
  computeConfidence,
} from '../inference/ctcDecoder';
import { BLANK_IDX, VOCAB_SIZE } from '../inference/ctcVocabulary';

describe('ctcGreedyDecode', () => {
  it('returns empty string for empty input', () => {
    expect(ctcGreedyDecode([])).toBe('');
  });

  it('collapses consecutive repeated indices', () => {
    // '3' is index 4, repeated 3 times → should collapse to single '3'
    expect(ctcGreedyDecode([4, 4, 4])).toBe('3');
  });

  it('removes blank tokens', () => {
    // blank, '1' (idx 2), blank, '2' (idx 3), blank
    expect(ctcGreedyDecode([0, 2, 0, 3, 0])).toBe('12');
  });

  it('collapses repeats AND removes blanks', () => {
    // '3' '3' blank '3' → collapses to '3' blank '3' → removes blank → '33'
    expect(ctcGreedyDecode([4, 4, 0, 4])).toBe('33');
  });

  it('decodes a known sequence into "3x+4=10"', () => {
    // Character to index mapping:
    //   '3' → 4, 'x' → 11, '+' → 12, '4' → 5, '=' → 14, '1' → 2, '0' → 1
    //
    // Simulate CTC output with blanks and repeats:
    //   blank, 4, 4, blank, 11, blank, 12, 12, blank, 5, blank, 14, 14, blank, 2, blank, 1, blank
    const indices = [
      BLANK_IDX,     // blank
      4, 4,          // '3' repeated
      BLANK_IDX,     // blank
      11,            // 'x'
      BLANK_IDX,     // blank
      12, 12,        // '+' repeated
      BLANK_IDX,     // blank
      5,             // '4'
      BLANK_IDX,     // blank
      14, 14,        // '=' repeated
      BLANK_IDX,     // blank
      2,             // '1'
      BLANK_IDX,     // blank
      1,             // '0'
      BLANK_IDX,     // blank
    ];
    expect(ctcGreedyDecode(indices)).toBe('3x+4=10');
  });

  it('decodes direct indices without blanks or repeats', () => {
    // '2' → 3, 'x' → 11, '=' → 14, '6' → 7
    expect(ctcGreedyDecode([3, 11, 14, 7])).toBe('2x=6');
  });

  it('handles all-blank input', () => {
    expect(ctcGreedyDecode([0, 0, 0, 0])).toBe('');
  });

  it('handles space character (index 18)', () => {
    // '1' → 2, ' ' → 18, '+' → 12, ' ' → 18, '2' → 3
    expect(ctcGreedyDecode([2, 18, 12, 18, 3])).toBe('1 + 2');
  });

  it('throws RangeError for invalid index (too large)', () => {
    expect(() => ctcGreedyDecode([0, 99])).toThrow(RangeError);
    expect(() => ctcGreedyDecode([0, 99])).toThrow(/Invalid CTC index 99/);
  });

  it('throws RangeError for negative index', () => {
    expect(() => ctcGreedyDecode([-1])).toThrow(RangeError);
  });

  it('throws RangeError for non-integer index', () => {
    expect(() => ctcGreedyDecode([1.5])).toThrow(RangeError);
  });

  it('handles parentheses and division', () => {
    // '(' → 16, '1' → 2, '+' → 12, '2' → 3, ')' → 17, '/' → 15, '3' → 4
    expect(ctcGreedyDecode([16, 2, 12, 3, 17, 15, 4])).toBe('(1+2)/3');
  });
});

describe('ctcGreedyDecodeFromLogProbs', () => {
  it('returns empty string for empty input', () => {
    expect(ctcGreedyDecodeFromLogProbs([])).toBe('');
  });

  it('takes argmax and decodes correctly', () => {
    // Create log-prob rows where the max is at the desired index.
    const makeRow = (targetIdx: number): number[] => {
      const row = new Array(VOCAB_SIZE).fill(-10);
      row[targetIdx] = 0; // log(1) = 0, highest
      return row;
    };

    // '2' → 3, 'x' → 11, '=' → 14, '4' → 5
    const logProbs = [makeRow(3), makeRow(11), makeRow(14), makeRow(5)];
    expect(ctcGreedyDecodeFromLogProbs(logProbs)).toBe('2x=4');
  });
});

describe('computeConfidence', () => {
  it('returns 0 for empty input', () => {
    expect(computeConfidence([])).toBe(0);
  });

  it('returns ~1.0 when max log-prob is 0 (prob=1) at each step', () => {
    const row = new Array(VOCAB_SIZE).fill(-100);
    row[1] = 0; // exp(0) = 1
    const confidence = computeConfidence([row, row, row]);
    expect(confidence).toBeCloseTo(1.0, 5);
  });

  it('returns value between 0 and 1', () => {
    const row = new Array(VOCAB_SIZE).fill(-2);
    row[1] = -0.5; // exp(-0.5) ≈ 0.607
    const confidence = computeConfidence([row]);
    expect(confidence).toBeGreaterThan(0);
    expect(confidence).toBeLessThan(1);
  });
});
