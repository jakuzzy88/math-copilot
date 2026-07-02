/**
 * OCR text normalizer.
 *
 * Sprint D: OCR Result Pipeline Adapter.
 *
 * Applies deterministic corrections to raw OCR output to fix common
 * misrecognition errors BEFORE grammar validation and parsing.
 *
 * Rules:
 *   1. Strip whitespace
 *   2. Normalise variable case: 'X' → 'x'
 *   3. Normalise multiplication sign: '×' (U+00D7) → '*'
 *   4. 'O' → '0' only in numeric-looking contexts
 *   5. 'I' / 'l' → '1' only in numeric-looking contexts
 *
 * All corrections are tracked for diagnostics and transparency.
 */

import type { OcrCorrection, NormalizationResult } from './types';

/**
 * Determine whether a character is a digit.
 */
function isDigit(ch: string): boolean {
  return ch >= '0' && ch <= '9';
}

/**
 * Determine whether a character is an operator or structural symbol
 * that typically borders a number (i.e. not a variable).
 */
function isNumericBorder(ch: string | undefined): boolean {
  if (ch === undefined) return true; // start/end of string
  return isDigit(ch) || ch === '=' || ch === '+' || ch === '-'
    || ch === '*' || ch === '/' || ch === '(' || ch === ')';
}

/**
 * Normalise raw OCR text into a clean equation string.
 *
 * This function is pure and deterministic: same input always produces
 * the same output and the same corrections list.
 */
export function normalizeOcrText(raw: string): NormalizationResult {
  const corrections: OcrCorrection[] = [];

  // --- Phase 1: Character-level replacements (before stripping spaces) ---

  // We work on an array so position tracking stays correct.
  const chars = Array.from(raw);
  const result: string[] = [];

  // Track original-position offset caused by space removal.
  let origPos = 0;

  for (let i = 0; i < chars.length; i++) {
    const ch = chars[i];
    const prev = i > 0 ? chars[i - 1] : undefined;
    const next = i < chars.length - 1 ? chars[i + 1] : undefined;

    // Rule 1: Strip whitespace
    if (ch === ' ' || ch === '\t') {
      if (ch !== ' ' || raw[i] !== ' ') {
        // Only record correction for non-trivial whitespace
        corrections.push({
          position: origPos,
          original: ch,
          replacement: '',
          rule: 'strip_whitespace',
        });
      }
      origPos++;
      continue;
    }

    // Rule 2: Normalise variable case 'X' → 'x'
    if (ch === 'X') {
      corrections.push({
        position: origPos,
        original: 'X',
        replacement: 'x',
        rule: 'uppercase_variable',
      });
      result.push('x');
      origPos++;
      continue;
    }

    // Rule 3: Normalise multiplication sign '×' (U+00D7) → '*'
    if (ch === '×' || ch === '·') {
      corrections.push({
        position: origPos,
        original: ch,
        replacement: '*',
        rule: 'normalize_multiplication',
      });
      result.push('*');
      origPos++;
      continue;
    }

    // Rule 4: 'O' → '0' in numeric-looking contexts
    // "Numeric-looking" = the character is bordered by digits, operators,
    // '=', or string boundaries — but NOT adjacent to 'x' (which would
    // suggest it's part of a variable context).
    if (ch === 'O') {
      const prevIsNumBorder = isNumericBorder(prev);
      const nextIsNumBorder = isNumericBorder(next);

      if (prevIsNumBorder && nextIsNumBorder) {
        corrections.push({
          position: origPos,
          original: 'O',
          replacement: '0',
          rule: 'O_to_zero',
        });
        result.push('0');
        origPos++;
        continue;
      }
    }

    // Rule 5: 'I' or 'l' (lowercase L) → '1' in numeric-looking contexts
    if (ch === 'I' || ch === 'l') {
      const prevIsNumBorder = isNumericBorder(prev);
      const nextIsNumBorder = isNumericBorder(next);

      if (prevIsNumBorder && nextIsNumBorder) {
        corrections.push({
          position: origPos,
          original: ch,
          replacement: '1',
          rule: ch === 'I' ? 'I_to_one' : 'l_to_one',
        });
        result.push('1');
        origPos++;
        continue;
      }
    }

    // No correction needed — pass through.
    result.push(ch);
    origPos++;
  }

  return {
    normalized: result.join(''),
    corrections,
  };
}
