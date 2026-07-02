/**
 * AST node types for linear equations.
 *
 * The tree is intentionally simple – Sprint 1 only needs linear equations
 * with a single variable `x`.
 */

// ── Leaf nodes ───────────────────────────────────────────────────────

/** A numeric literal, e.g. `42`. */
export interface NumberLiteral {
  type: 'NumberLiteral';
  value: number;
}

/** A variable reference – always `x` in Sprint 1. */
export interface Variable {
  type: 'Variable';
  name: string;
}

// ── Composite nodes ──────────────────────────────────────────────────

/** A binary operation: `left op right`. */
export interface BinaryOp {
  type: 'BinaryOp';
  operator: '+' | '-' | '*' | '/';
  left: ASTNode;
  right: ASTNode;
}

/** Unary minus applied to a sub-expression, e.g. `-x` or `-(x+1)`. */
export interface UnaryMinus {
  type: 'UnaryMinus';
  operand: ASTNode;
}

// ── Top-level ────────────────────────────────────────────────────────

/** An equation `lhs = rhs`. */
export interface Equation {
  type: 'Equation';
  left: ASTNode;
  right: ASTNode;
}

// ── Union type ───────────────────────────────────────────────────────

export type ASTNode = NumberLiteral | Variable | BinaryOp | UnaryMinus;

export type ASTRoot = Equation;

// ── Helper constructors ──────────────────────────────────────────────

export function num(value: number): NumberLiteral {
  return { type: 'NumberLiteral', value };
}

export function variable(name: string = 'x'): Variable {
  return { type: 'Variable', name };
}

export function binOp(
  operator: BinaryOp['operator'],
  left: ASTNode,
  right: ASTNode,
): BinaryOp {
  return { type: 'BinaryOp', operator, left, right };
}

export function unaryMinus(operand: ASTNode): UnaryMinus {
  return { type: 'UnaryMinus', operand };
}

export function equation(left: ASTNode, right: ASTNode): Equation {
  return { type: 'Equation', left, right };
}
