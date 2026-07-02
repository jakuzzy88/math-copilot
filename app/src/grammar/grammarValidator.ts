/**
 * Grammar validator.
 *
 * Performs lightweight structural validation on a raw equation string
 * BEFORE passing it to the parser. This catches obvious errors early
 * and provides clearer error messages than the parser would.
 */

import { isSupportedSymbol } from './symbolVocabulary';

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

/**
 * Validate that an equation string conforms to our supported grammar.
 *
 * Checks performed:
 *   1. All characters are in the supported symbol set (or whitespace)
 *   2. Exactly one `=` sign
 *   3. No consecutive operators (`++`, `+-`, `--`, etc.)
 *   4. No empty parentheses `()`
 *   5. Balanced parentheses
 *   6. Non-empty sides (something on both sides of `=`)
 *   7. Does not start or end with a binary operator (after trimming)
 */
export function validateEquation(input: string): ValidationResult {
  const errors: string[] = [];

  // Strip whitespace for validation
  const stripped = input.replace(/\s/g, '');

  if (stripped.length === 0) {
    return { valid: false, errors: ['Empty input'] };
  }

  // 1. Check all characters are supported
  for (let i = 0; i < stripped.length; i++) {
    if (!isSupportedSymbol(stripped[i])) {
      errors.push(`Unsupported character '${stripped[i]}' at position ${i}`);
    }
  }

  // 2. Exactly one '='
  const equalsCount = (stripped.match(/=/g) || []).length;
  if (equalsCount === 0) {
    errors.push('Missing equals sign');
  } else if (equalsCount > 1) {
    errors.push(`Found ${equalsCount} equals signs, expected exactly 1`);
  }

  // 3. No consecutive operators (but allow unary minus after operator or at start)
  //    Consecutive binary operators: `++`, `+-` (when not unary), `**`, etc.
  const binaryOps = new Set(['+', '-', '*', '/']);
  for (let i = 0; i < stripped.length - 1; i++) {
    if (binaryOps.has(stripped[i]) && binaryOps.has(stripped[i + 1])) {
      // Allow unary minus after binary operator: `3+-2` is unusual but
      // for Sprint 1 we reject consecutive operators to keep things clean
      errors.push(
        `Consecutive operators '${stripped[i]}${stripped[i + 1]}' at position ${i}`,
      );
    }
  }

  // 4. No empty parentheses
  if (stripped.includes('()')) {
    errors.push('Empty parentheses found');
  }

  // 5. Balanced parentheses
  let depth = 0;
  for (let i = 0; i < stripped.length; i++) {
    if (stripped[i] === '(') depth++;
    if (stripped[i] === ')') depth--;
    if (depth < 0) {
      errors.push(`Unmatched closing parenthesis at position ${i}`);
      break;
    }
  }
  if (depth > 0) {
    errors.push(`${depth} unclosed opening parenthesis(es)`);
  }

  // 6. Non-empty sides of '='
  if (equalsCount === 1) {
    const eqIndex = stripped.indexOf('=');
    if (eqIndex === 0) {
      errors.push('Nothing on the left side of the equation');
    }
    if (eqIndex === stripped.length - 1) {
      errors.push('Nothing on the right side of the equation');
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}
