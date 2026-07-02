/**
 * Symbolic linear equation solver.
 *
 * Strategy:
 *   1. Flatten / expand the AST on both sides into a linear form: ax + b
 *   2. Collect coefficients: (leftA - rightA)x = (rightB - leftB)
 *   3. Divide to solve for x
 *
 * All steps are recorded in an ActionLog for the explanation engine.
 * Descriptions are pedagogically rich — they explain *why* each
 * operation is performed, not just *what* is done.
 */

import { ASTNode, ASTRoot } from '../parser/ast';
import { parseEquation } from '../parser/parser';
import { ActionLog } from './actionLog';

// ── Linear form: ax + b ──────────────────────────────────────────────

interface LinearForm {
  /** Coefficient of x */
  a: number;
  /** Constant term */
  b: number;
}

// ── Public API ───────────────────────────────────────────────────────

export interface SolveResult {
  /** The value of x */
  solution: number;
  /** Step-by-step action log */
  log: ActionLog;
}

/**
 * Solve a linear equation string for x.
 *
 * @param input – equation string, e.g. `"3x+4=10"`
 * @returns solution value and action log
 * @throws Error if the equation is not linear or has no/infinite solutions
 */
export function solveLinear(input: string): SolveResult {
  const ast = parseEquation(input);
  return solveFromAST(ast, input);
}

/**
 * Solve a linear equation from a pre-parsed AST.
 */
export function solveFromAST(ast: ASTRoot, originalInput: string): SolveResult {
  const log = new ActionLog();

  // Step 1: Extract linear coefficients from both sides
  const leftForm = extractLinearForm(ast.left);
  const rightForm = extractLinearForm(ast.right);

  // ── Describe the x-term and constant for pedagogical context ──
  const xTermStr = formatXTerm(leftForm.a, rightForm.a);
  const constAttached = describeConstantAttachment(leftForm.b);

  log.add(
    'SIMPLIFY',
    `We want to find the value of x. In ${originalInput}, the x-term is ${xTermStr}${constAttached}.`,
    originalInput,
    {
      goal: 'Understand the equation',
      targetTerm: xTermStr,
      reason: 'Identify the variable term and any constants attached to it',
    },
  );

  // Step 2: Move variable terms to the left, constants to the right
  // leftA*x + leftB = rightA*x + rightB
  // (leftA - rightA)*x = rightB - leftB
  const combinedA = leftForm.a - rightForm.a;
  const combinedB = rightForm.b - leftForm.b;

  if (leftForm.b !== 0 || rightForm.a !== 0) {
    // Only log the move step if there are actually terms to move
    if (rightForm.a !== 0) {
      const rCoeff = formatCoeff(rightForm.a);
      log.add(
        'MOVE_TERM',
        `Variable terms appear on both sides. To collect all x-terms on the left, subtract ${rCoeff}x from both sides to keep the equation balanced.`,
        `${formatCoeff(combinedA)}x + ${leftForm.b} = ${rightForm.b}`,
        {
          goal: `Collect all x-terms on the left side`,
          targetTerm: `${rCoeff}x`,
          inverseOperation: 'subtraction',
          reason: 'Moving variable terms to one side simplifies solving',
        },
      );
    }
    if (leftForm.b !== 0) {
      const absB = Math.abs(leftForm.b);
      const coeffXStr = formatCoeff(combinedA) + 'x';
      if (leftForm.b > 0) {
        log.add(
          'MOVE_TERM',
          `To isolate ${coeffXStr}, we need to remove +${leftForm.b} from the left side. To keep the equation balanced, we subtract ${leftForm.b} from both sides.`,
          `${formatCoeff(combinedA)}x = ${combinedB}`,
          {
            goal: `Isolate ${coeffXStr}`,
            targetTerm: `+${leftForm.b}`,
            inverseOperation: 'subtraction',
            reason: `Subtracting ${leftForm.b} undoes the addition of ${leftForm.b}`,
          },
        );
      } else {
        log.add(
          'MOVE_TERM',
          `To isolate ${coeffXStr}, we need to remove ${leftForm.b} from the left side. To keep the equation balanced, we add ${absB} to both sides.`,
          `${formatCoeff(combinedA)}x = ${combinedB}`,
          {
            goal: `Isolate ${coeffXStr}`,
            targetTerm: `${leftForm.b}`,
            inverseOperation: 'addition',
            reason: `Adding ${absB} undoes the subtraction of ${absB}`,
          },
        );
      }
    }
  }

  // Step 3: Check solvability
  if (combinedA === 0) {
    if (combinedB === 0) {
      throw new Error('Infinite solutions: the equation is always true');
    }
    throw new Error('No solution: the equation is contradictory');
  }

  // Step 4: Divide both sides by the coefficient of x
  const solution = combinedB / combinedA;

  if (combinedA !== 1) {
    log.add(
      'DIVIDE_BOTH',
      `Now x is multiplied by ${combinedA}. Division is the inverse of multiplication, so we divide both sides by ${combinedA} to leave x alone.`,
      `x = ${combinedB} / ${combinedA}`,
      {
        goal: 'Isolate x',
        targetTerm: `${combinedA}`,
        inverseOperation: 'division',
        reason: `Dividing by ${combinedA} undoes the multiplication by ${combinedA}`,
      },
    );
  }

  log.add(
    'SIMPLIFY',
    `${combinedB} divided by ${combinedA} is ${solution}.`,
    `x = ${solution}`,
    {
      goal: 'Simplify the result',
      reason: `Performing the arithmetic: ${combinedB} ÷ ${combinedA} = ${solution}`,
    },
  );

  log.add(
    'RESULT',
    `The value that makes the original equation true is x = ${solution}. Substituting ${solution} back into ${originalInput} confirms the solution.`,
    `x = ${solution}`,
    {
      goal: 'State the final answer',
      reason: `x = ${solution} satisfies the original equation ${originalInput}`,
    },
  );

  return { solution, log };
}

// ── Coefficient extraction ───────────────────────────────────────────

/**
 * Walk an AST expression and extract the linear form `ax + b`.
 *
 * This handles:
 *   - Number literals
 *   - Variable references
 *   - Addition, subtraction, multiplication, division
 *   - Unary minus
 *   - Implicit multiplication (already handled by tokenizer, but
 *     multiplication nodes `a * x` are handled here)
 *
 * @throws Error if the expression is not linear in x.
 */
function extractLinearForm(node: ASTNode): LinearForm {
  switch (node.type) {
    case 'NumberLiteral':
      return { a: 0, b: node.value };

    case 'Variable':
      return { a: 1, b: 0 };

    case 'UnaryMinus': {
      const inner = extractLinearForm(node.operand);
      return { a: -inner.a, b: -inner.b };
    }

    case 'BinaryOp': {
      const left = extractLinearForm(node.left);
      const right = extractLinearForm(node.right);

      switch (node.operator) {
        case '+':
          return { a: left.a + right.a, b: left.b + right.b };

        case '-':
          return { a: left.a - right.a, b: left.b - right.b };

        case '*':
          // For linear equations, at least one side must be constant
          // (no x*x allowed in Sprint 1)
          if (left.a === 0) {
            // left is a constant, multiply through
            return { a: left.b * right.a, b: left.b * right.b };
          }
          if (right.a === 0) {
            // right is a constant, multiply through
            return { a: right.b * left.a, b: right.b * left.b };
          }
          throw new Error(
            'Non-linear equation: cannot multiply two expressions containing x',
          );

        case '/':
          // Division: the denominator must be constant
          if (right.a !== 0) {
            throw new Error(
              'Non-linear equation: cannot divide by an expression containing x',
            );
          }
          if (right.b === 0) {
            throw new Error('Division by zero');
          }
          return { a: left.a / right.b, b: left.b / right.b };
      }
    }
  }
}

// ── Formatting helpers ───────────────────────────────────────────────

function formatCoeff(n: number): string {
  if (n === 1) return '';
  if (n === -1) return '-';
  return String(n);
}

/**
 * Describe the x-term in human-readable form for the first step.
 */
function formatXTerm(leftA: number, rightA: number): string {
  // Total coefficient on the left side (most common case)
  if (leftA === 1 && rightA === 0) return 'x';
  if (leftA === -1 && rightA === 0) return '-x';
  if (rightA === 0) return `${leftA}x`;
  // x terms on both sides — describe left side's x-term
  if (leftA === 1) return 'x';
  if (leftA === -1) return '-x';
  return `${leftA}x`;
}

/**
 * Describe what constant is attached to the x-term on the left side.
 */
function describeConstantAttachment(leftB: number): string {
  if (leftB === 0) return '';
  if (leftB > 0) return `, but it has +${leftB} added to it`;
  return `, but it has ${leftB} subtracted from it`;
}
