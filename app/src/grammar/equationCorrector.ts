/**
 * Equation corrector.
 *
 * Attempts to fix common recognition errors in symbol sequences
 * before parsing. For example:
 *   - Stray whitespace normalisation
 *   - Common OCR misreads (future: 'O' → '0', 'l' → '1')
 *
 * Status: Minimal implementation for Sprint 1.
 */

/**
 * Normalise an equation string for parsing.
 *
 * Currently handles:
 *   - Stripping whitespace
 *   - Normalising 'X' (uppercase) to 'x'
 *
 * Future sprints will add OCR-specific corrections.
 */
export function normaliseEquation(input: string): string {
  let result = input.trim();
  // Normalise variable case
  result = result.replace(/X/g, 'x');
  // Remove spaces
  result = result.replace(/\s+/g, '');
  return result;
}
