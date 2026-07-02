/**
 * OCR normalizer tests.
 *
 * Sprint D: Verifies deterministic OCR text normalization.
 */

import { normalizeOcrText } from '../pipeline/ocrNormalizer';

describe('OCR Normalizer', () => {
  // ── Whitespace stripping ──────────────────────────────────────────

  test('strips spaces from equation', () => {
    const { normalized } = normalizeOcrText('3x + 4 = 10');
    expect(normalized).toBe('3x+4=10');
  });

  test('strips tabs', () => {
    const { normalized } = normalizeOcrText('3x\t+\t4=10');
    expect(normalized).toBe('3x+4=10');
  });

  // ── Variable case normalisation ───────────────────────────────────

  test('normalises uppercase X to x', () => {
    const result = normalizeOcrText('3X+4=10');
    expect(result.normalized).toBe('3x+4=10');
    expect(result.corrections).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ original: 'X', replacement: 'x', rule: 'uppercase_variable' }),
      ]),
    );
  });

  // ── Multiplication sign ───────────────────────────────────────────

  test('normalises × to *', () => {
    const result = normalizeOcrText('3×x=9');
    expect(result.normalized).toBe('3*x=9');
    expect(result.corrections).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ original: '×', rule: 'normalize_multiplication' }),
      ]),
    );
  });

  test('normalises · (middle dot) to *', () => {
    const result = normalizeOcrText('3·x=9');
    expect(result.normalized).toBe('3*x=9');
  });

  // ── O → 0 in numeric contexts ────────────────────────────────────

  test('corrects O to 0 between digits', () => {
    const result = normalizeOcrText('1O+x=10');
    expect(result.normalized).toBe('10+x=10');
    expect(result.corrections).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ original: 'O', replacement: '0', rule: 'O_to_zero' }),
      ]),
    );
  });

  test('corrects O to 0 at end after =', () => {
    // "3x+4=1O" → "3x+4=10"
    const result = normalizeOcrText('3x+4=1O');
    expect(result.normalized).toBe('3x+4=10');
  });

  test('corrects O to 0 at start before digit', () => {
    // "O5x=10" → "05x=10"
    const result = normalizeOcrText('O5x=10');
    expect(result.normalized).toBe('05x=10');
  });

  test('does NOT correct O adjacent to x (variable context)', () => {
    // "Ox" looks like it could be a variable-like context
    // 'O' is next to 'x', which is not a numeric border, so no correction
    const result = normalizeOcrText('Ox=5');
    expect(result.normalized).toBe('Ox=5');
  });

  // ── I / l → 1 in numeric contexts ────────────────────────────────

  test('corrects I to 1 between digits', () => {
    const result = normalizeOcrText('2x=I0');
    expect(result.normalized).toBe('2x=10');
    expect(result.corrections).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ original: 'I', replacement: '1', rule: 'I_to_one' }),
      ]),
    );
  });

  test('corrects lowercase l to 1 in numeric context', () => {
    const result = normalizeOcrText('3x=l2');
    expect(result.normalized).toBe('3x=12');
    expect(result.corrections).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ original: 'l', replacement: '1', rule: 'l_to_one' }),
      ]),
    );
  });

  test('does NOT correct l adjacent to x', () => {
    // "lx=5" — l is next to x, not a numeric border
    const result = normalizeOcrText('lx=5');
    expect(result.normalized).toBe('lx=5');
  });

  // ── Combined corrections ──────────────────────────────────────────

  test('applies multiple corrections', () => {
    // "3X+4=1O" → X→x, O→0
    const result = normalizeOcrText('3X+4=1O');
    expect(result.normalized).toBe('3x+4=10');
    expect(result.corrections.length).toBe(2);
  });

  // ── No corrections needed ─────────────────────────────────────────

  test('returns empty corrections for clean input', () => {
    const result = normalizeOcrText('3x+4=10');
    expect(result.normalized).toBe('3x+4=10');
    expect(result.corrections).toHaveLength(0);
  });

  // ── Determinism ───────────────────────────────────────────────────

  test('is deterministic — same input gives same output', () => {
    const a = normalizeOcrText('3X+4=1O');
    const b = normalizeOcrText('3X+4=1O');
    expect(a.normalized).toBe(b.normalized);
    expect(a.corrections).toEqual(b.corrections);
  });
});
