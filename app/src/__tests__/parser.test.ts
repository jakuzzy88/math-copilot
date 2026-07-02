/**
 * Parser tests.
 *
 * Verifies that valid equations parse into correct ASTs
 * and that invalid equations are properly rejected.
 */

import { parseEquation } from '../parser/parser';

describe('Parser – valid equations', () => {
  test('x+5=9', () => {
    const ast = parseEquation('x+5=9');
    expect(ast.type).toBe('Equation');
    // LHS should be BinaryOp(+, Variable(x), Number(5))
    expect(ast.left).toEqual({
      type: 'BinaryOp',
      operator: '+',
      left: { type: 'Variable', name: 'x' },
      right: { type: 'NumberLiteral', value: 5 },
    });
    // RHS should be Number(9)
    expect(ast.right).toEqual({ type: 'NumberLiteral', value: 9 });
  });

  test('2x=8 – implicit multiplication', () => {
    const ast = parseEquation('2x=8');
    expect(ast.left).toEqual({
      type: 'BinaryOp',
      operator: '*',
      left: { type: 'NumberLiteral', value: 2 },
      right: { type: 'Variable', name: 'x' },
    });
    expect(ast.right).toEqual({ type: 'NumberLiteral', value: 8 });
  });

  test('3x+4=10', () => {
    const ast = parseEquation('3x+4=10');
    expect(ast.type).toBe('Equation');
    // LHS: BinaryOp(+, BinaryOp(*, 3, x), 4)
    expect(ast.left.type).toBe('BinaryOp');
  });

  test('5x-2=18', () => {
    const ast = parseEquation('5x-2=18');
    expect(ast.type).toBe('Equation');
  });

  test('x/2=5', () => {
    const ast = parseEquation('x/2=5');
    expect(ast.left).toEqual({
      type: 'BinaryOp',
      operator: '/',
      left: { type: 'Variable', name: 'x' },
      right: { type: 'NumberLiteral', value: 2 },
    });
  });

  test('3x/4=6', () => {
    const ast = parseEquation('3x/4=6');
    expect(ast.type).toBe('Equation');
  });

  test('2(x+1)=10 – distribution', () => {
    const ast = parseEquation('2(x+1)=10');
    expect(ast.type).toBe('Equation');
    // LHS: BinaryOp(*, 2, BinaryOp(+, x, 1))
    expect(ast.left).toEqual({
      type: 'BinaryOp',
      operator: '*',
      left: { type: 'NumberLiteral', value: 2 },
      right: {
        type: 'BinaryOp',
        operator: '+',
        left: { type: 'Variable', name: 'x' },
        right: { type: 'NumberLiteral', value: 1 },
      },
    });
  });
});

describe('Parser – invalid equations', () => {
  test('2x++5=13 – consecutive operators', () => {
    expect(() => parseEquation('2x++5=13')).toThrow();
  });

  test('2x+5 – missing equals sign', () => {
    expect(() => parseEquation('2x+5')).toThrow();
  });

  test('2x+5==13 – double equals', () => {
    expect(() => parseEquation('2x+5==13')).toThrow();
  });

  test('2x+()=13 – empty parentheses', () => {
    expect(() => parseEquation('2x+()=13')).toThrow();
  });
});
