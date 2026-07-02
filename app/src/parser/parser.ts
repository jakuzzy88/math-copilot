/**
 * Recursive-descent parser for linear equations.
 *
 * Grammar (after tokenisation & implicit-multiplication insertion):
 *
 *   equation   ::= expression '=' expression EOF
 *   expression ::= term (('+' | '-') term)*
 *   term       ::= factor (('*' | '/') factor)*
 *   factor     ::= NUMBER | VARIABLE | '(' expression ')' | '-' factor
 *
 * The parser produces an AST (see ast.ts) or throws a descriptive error.
 */

import { Token, TokenType, tokenize } from './tokenizer';
import {
  ASTNode,
  ASTRoot,
  num,
  variable,
  binOp,
  unaryMinus,
  equation,
} from './ast';

// ── Parser state ─────────────────────────────────────────────────────

class Parser {
  private tokens: Token[];
  private pos: number = 0;

  constructor(tokens: Token[]) {
    this.tokens = tokens;
  }

  // ── Helpers ──────────────────────────────────────────────────────

  private current(): Token {
    return this.tokens[this.pos];
  }

  private peek(): TokenType {
    return this.current().type;
  }

  private eat(expected: TokenType): Token {
    const tok = this.current();
    if (tok.type !== expected) {
      throw new Error(
        `Expected ${expected} but found ${tok.type} ('${tok.value}') at position ${tok.position}`,
      );
    }
    this.pos++;
    return tok;
  }

  // ── Grammar rules ────────────────────────────────────────────────

  /**
   * equation ::= expression '=' expression EOF
   */
  parseEquation(): ASTRoot {
    const left = this.parseExpression();
    this.eat('EQUALS');

    // Guard against double-equals: `2x+5==13`
    if (this.peek() === 'EQUALS') {
      throw new Error(
        `Unexpected second '=' at position ${this.current().position}`,
      );
    }

    const right = this.parseExpression();
    this.eat('EOF');
    return equation(left, right);
  }

  /**
   * expression ::= term (('+' | '-') term)*
   */
  private parseExpression(): ASTNode {
    let node = this.parseTerm();

    while (this.peek() === 'PLUS' || this.peek() === 'MINUS') {
      const op = this.current().value as '+' | '-';
      this.pos++;
      const right = this.parseTerm();
      node = binOp(op, node, right);
    }

    return node;
  }

  /**
   * term ::= factor (('*' | '/') factor)*
   */
  private parseTerm(): ASTNode {
    let node = this.parseFactor();

    while (this.peek() === 'STAR' || this.peek() === 'SLASH') {
      const op = this.current().value as '*' | '/';
      this.pos++;
      const right = this.parseFactor();
      node = binOp(op, node, right);
    }

    return node;
  }

  /**
   * factor ::= NUMBER | VARIABLE | '(' expression ')' | '-' factor
   */
  private parseFactor(): ASTNode {
    const tok = this.current();

    // Number literal
    if (tok.type === 'NUMBER') {
      this.pos++;
      return num(parseInt(tok.value, 10));
    }

    // Variable
    if (tok.type === 'VARIABLE') {
      this.pos++;
      return variable(tok.value);
    }

    // Parenthesised expression
    if (tok.type === 'LPAREN') {
      this.pos++;

      // Guard against empty parentheses: `()`
      if (this.peek() === 'RPAREN') {
        throw new Error(
          `Empty parentheses at position ${tok.position}`,
        );
      }

      const node = this.parseExpression();
      this.eat('RPAREN');
      return node;
    }

    // Unary minus
    if (tok.type === 'MINUS') {
      this.pos++;
      const operand = this.parseFactor();
      return unaryMinus(operand);
    }

    throw new Error(
      `Unexpected token ${tok.type} ('${tok.value}') at position ${tok.position}`,
    );
  }
}

// ── Public API ───────────────────────────────────────────────────────

/**
 * Parse an equation string into an AST.
 *
 * @param input – raw equation string, e.g. `"3x+4=10"`
 * @returns the AST root (an Equation node)
 * @throws Error on syntax errors
 */
export function parseEquation(input: string): ASTRoot {
  const tokens = tokenize(input);
  const parser = new Parser(tokens);
  return parser.parseEquation();
}
