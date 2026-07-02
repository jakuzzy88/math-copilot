/**
 * Tokenizer tests.
 *
 * Verifies that all supported equation forms are correctly tokenized,
 * including implicit multiplication insertion.
 */

import { tokenize, Token, TokenType } from '../parser/tokenizer';

/** Extract just the types from a token array for easy assertion. */
function types(tokens: Token[]): TokenType[] {
  return tokens.map((t) => t.type);
}

/** Extract just the values from a token array. */
function values(tokens: Token[]): string[] {
  return tokens.map((t) => t.value);
}

describe('Tokenizer', () => {
  test('x+5=9', () => {
    const tokens = tokenize('x+5=9');
    expect(types(tokens)).toEqual([
      'VARIABLE', 'PLUS', 'NUMBER', 'EQUALS', 'NUMBER', 'EOF',
    ]);
    expect(values(tokens)).toEqual(['x', '+', '5', '=', '9', '']);
  });

  test('2x=8 – implicit multiplication', () => {
    const tokens = tokenize('2x=8');
    expect(types(tokens)).toEqual([
      'NUMBER', 'STAR', 'VARIABLE', 'EQUALS', 'NUMBER', 'EOF',
    ]);
    expect(values(tokens)).toEqual(['2', '*', 'x', '=', '8', '']);
  });

  test('3x+4=10', () => {
    const tokens = tokenize('3x+4=10');
    expect(types(tokens)).toEqual([
      'NUMBER', 'STAR', 'VARIABLE', 'PLUS', 'NUMBER', 'EQUALS', 'NUMBER', 'EOF',
    ]);
  });

  test('5x-2=18', () => {
    const tokens = tokenize('5x-2=18');
    expect(types(tokens)).toEqual([
      'NUMBER', 'STAR', 'VARIABLE', 'MINUS', 'NUMBER', 'EQUALS', 'NUMBER', 'EOF',
    ]);
  });

  test('x/2=5', () => {
    const tokens = tokenize('x/2=5');
    expect(types(tokens)).toEqual([
      'VARIABLE', 'SLASH', 'NUMBER', 'EQUALS', 'NUMBER', 'EOF',
    ]);
  });

  test('3x/4=6', () => {
    const tokens = tokenize('3x/4=6');
    expect(types(tokens)).toEqual([
      'NUMBER', 'STAR', 'VARIABLE', 'SLASH', 'NUMBER', 'EQUALS', 'NUMBER', 'EOF',
    ]);
  });

  test('2(x+1)=10 – implicit multiplication before paren', () => {
    const tokens = tokenize('2(x+1)=10');
    expect(types(tokens)).toEqual([
      'NUMBER', 'STAR', 'LPAREN', 'VARIABLE', 'PLUS', 'NUMBER', 'RPAREN',
      'EQUALS', 'NUMBER', 'EOF',
    ]);
  });

  test('handles whitespace', () => {
    const tokens = tokenize('x + 5 = 9');
    expect(types(tokens)).toEqual([
      'VARIABLE', 'PLUS', 'NUMBER', 'EQUALS', 'NUMBER', 'EOF',
    ]);
  });

  test('multi-digit numbers', () => {
    const tokens = tokenize('12x+345=678');
    const nums = tokens.filter((t) => t.type === 'NUMBER').map((t) => t.value);
    expect(nums).toEqual(['12', '345', '678']);
  });

  test('throws on unrecognised character', () => {
    expect(() => tokenize('2x+5=13!')).toThrow(/Unrecognised character/);
  });
});
