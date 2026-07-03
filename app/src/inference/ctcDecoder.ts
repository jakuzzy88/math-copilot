/**
 * CTC greedy decoder for handwritten equation recognition.
 *
 * Sprint 5C: Mobile Static Image Inference Preparation.
 *
 * Mirrors the CTC greedy decoding logic from
 * `training/models/vocabulary.py:ctc_greedy_decode`.
 *
 * Supports two input forms:
 *   1. A flat array of class indices (already argmax-ed).
 *   2. A 2-D array of shape [T, C] containing log-probabilities.
 */

import { BLANK_IDX, VOCAB_SIZE, indexToChar } from './ctcVocabulary';

// ---------------------------------------------------------------------------
// CTC greedy decode
// ---------------------------------------------------------------------------

/**
 * Perform CTC greedy decoding from a 1-D array of class indices.
 *
 * Standard CTC collapse rule:
 *   1. Merge consecutive duplicate indices.
 *   2. Remove blank tokens (index 0).
 *   3. Map remaining indices to characters.
 *
 * @param indices - Array of integer class indices, one per time step.
 * @returns Decoded string.
 * @throws {RangeError} if any index is outside [0, VOCAB_SIZE-1].
 */
export function ctcGreedyDecode(indices: number[]): string {
  if (indices.length === 0) {
    return '';
  }

  // Validate and collapse consecutive duplicates.
  const collapsed: number[] = [];
  let prev = -1;

  for (const idx of indices) {
    // Validate index range.
    if (!Number.isInteger(idx) || idx < 0 || idx >= VOCAB_SIZE) {
      throw new RangeError(
        `Invalid CTC index ${idx}: must be an integer in [0, ${VOCAB_SIZE - 1}].`,
      );
    }

    if (idx !== prev) {
      collapsed.push(idx);
    }
    prev = idx;
  }

  // Remove blanks and map to characters.
  const chars: string[] = [];
  for (const idx of collapsed) {
    if (idx === BLANK_IDX) {
      continue;
    }
    chars.push(indexToChar(idx));
  }

  return chars.join('');
}

/**
 * Perform CTC greedy decoding from a 2-D log-probability matrix.
 *
 * Takes argmax over the class dimension for each time step, then
 * applies the standard CTC collapse.
 *
 * @param logProbs - 2-D array of shape [T][C], log-probabilities per class.
 * @returns Decoded string.
 * @throws {RangeError} if argmax indices are invalid.
 */
export function ctcGreedyDecodeFromLogProbs(logProbs: number[][]): string {
  if (logProbs.length === 0) {
    return '';
  }

  // Argmax over class dimension for each time step.
  const indices: number[] = logProbs.map((row) => {
    let maxIdx = 0;
    let maxVal = row[0];
    for (let i = 1; i < row.length; i++) {
      if (row[i] > maxVal) {
        maxVal = row[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  });

  return ctcGreedyDecode(indices);
}

/**
 * Compute the average maximum probability across time steps.
 *
 * Useful for computing a simple confidence score from log-probabilities.
 * Converts log-probs to probabilities (exp) before averaging.
 *
 * @param logProbs - 2-D array of shape [T][C].
 * @returns Average max probability in [0, 1].
 */
export function computeConfidence(logProbs: number[][]): number {
  if (logProbs.length === 0) {
    return 0;
  }

  let sumMaxProb = 0;
  for (const row of logProbs) {
    let maxLogProb = row[0];
    for (let i = 1; i < row.length; i++) {
      if (row[i] > maxLogProb) {
        maxLogProb = row[i];
      }
    }
    // Convert log-probability to probability.
    sumMaxProb += Math.exp(maxLogProb);
  }

  return sumMaxProb / logProbs.length;
}
