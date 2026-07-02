/**
 * AST simplification utilities.
 *
 * Walks an AST and collapses constant sub-expressions, e.g.:
 *   BinaryOp(+, 3, 4)  →  NumberLiteral(7)
 *
 * Also handles unary-minus folding and identity removal.
 */

import { ASTNode, num, binOp, unaryMinus as makeUnaryMinus } from '../parser/ast';

/**
 * Recursively simplify an AST node.
 *
 * Only performs constant folding and trivial identity removal.
 * Does NOT rearrange terms or solve equations.
 */
export function simplify(node: ASTNode): ASTNode {
  switch (node.type) {
    case 'NumberLiteral':
    case 'Variable':
      return node;

    case 'UnaryMinus': {
      const operand = simplify(node.operand);
      // Fold: -(N)  →  -N
      if (operand.type === 'NumberLiteral') {
        return num(-operand.value);
      }
      // Double negation: -(-x)  →  x
      if (operand.type === 'UnaryMinus') {
        return operand.operand;
      }
      return makeUnaryMinus(operand);
    }

    case 'BinaryOp': {
      const left = simplify(node.left);
      const right = simplify(node.right);

      // Constant folding: both sides are numbers
      if (left.type === 'NumberLiteral' && right.type === 'NumberLiteral') {
        return num(evaluateOp(node.operator, left.value, right.value));
      }

      // Identity removal for addition: x + 0 = x, 0 + x = x
      if (node.operator === '+') {
        if (right.type === 'NumberLiteral' && right.value === 0) return left;
        if (left.type === 'NumberLiteral' && left.value === 0) return right;
      }

      // Identity removal for subtraction: x - 0 = x
      if (node.operator === '-') {
        if (right.type === 'NumberLiteral' && right.value === 0) return left;
      }

      // Identity removal for multiplication: x * 1 = x, 1 * x = x
      if (node.operator === '*') {
        if (right.type === 'NumberLiteral' && right.value === 1) return left;
        if (left.type === 'NumberLiteral' && left.value === 1) return right;
        // x * 0 = 0, 0 * x = 0
        if (right.type === 'NumberLiteral' && right.value === 0) return num(0);
        if (left.type === 'NumberLiteral' && left.value === 0) return num(0);
      }

      // Identity removal for division: x / 1 = x
      if (node.operator === '/') {
        if (right.type === 'NumberLiteral' && right.value === 1) return left;
      }

      return binOp(node.operator, left, right);
    }
  }
}

/**
 * Evaluate a binary arithmetic operation on two numbers.
 * No eval() – just a switch.
 */
function evaluateOp(op: '+' | '-' | '*' | '/', a: number, b: number): number {
  switch (op) {
    case '+': return a + b;
    case '-': return a - b;
    case '*': return a * b;
    case '/':
      if (b === 0) throw new Error('Division by zero');
      return a / b;
  }
}
