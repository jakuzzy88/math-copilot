/**
 * Grammar validator tests.
 *
 * Verifies that the validator correctly accepts valid equations
 * and rejects malformed ones with clear error messages.
 */

import { validateEquation } from '../grammar/grammarValidator';

describe('Grammar Validator – valid equations', () => {
  const validEquations = [
    'x+5=9',
    '2x=8',
    '3x+4=10',
    '5x-2=18',
    'x/2=5',
    '3x/4=6',
    '2(x+1)=10',
  ];

  validEquations.forEach((eq) => {
    test(`accepts: ${eq}`, () => {
      const result = validateEquation(eq);
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });
  });
});

describe('Grammar Validator – invalid equations', () => {
  test('rejects consecutive operators: 2x++5=13', () => {
    const result = validateEquation('2x++5=13');
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes('Consecutive operators'))).toBe(true);
  });

  test('rejects missing equals: 2x+5', () => {
    const result = validateEquation('2x+5');
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes('Missing equals'))).toBe(true);
  });

  test('rejects double equals: 2x+5==13', () => {
    const result = validateEquation('2x+5==13');
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes('equals signs'))).toBe(true);
  });

  test('rejects empty parentheses: 2x+()=13', () => {
    const result = validateEquation('2x+()=13');
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes('Empty parentheses'))).toBe(true);
  });

  test('rejects empty input', () => {
    const result = validateEquation('');
    expect(result.valid).toBe(false);
  });

  test('rejects unsupported characters', () => {
    const result = validateEquation('2x+5=13!');
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes('Unsupported character'))).toBe(true);
  });

  test('rejects unbalanced parentheses', () => {
    const result = validateEquation('2(x+1=10');
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes('unclosed'))).toBe(true);
  });
});
