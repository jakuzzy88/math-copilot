/**
 * Tokenizer for linear equations.
 *
 * Converts a raw equation string like `2x+5=13` into a token stream.
 * Handles implicit multiplication:
 *   - `2x`      → NUMBER(2), STAR, VARIABLE(x)
 *   - `2(x+1)`  → NUMBER(2), STAR, LPAREN, ...
 */

// ── Token types ──────────────────────────────────────────────────────

export type TokenType =
  | 'NUMBER'
  | 'VARIABLE'
  | 'PLUS'
  | 'MINUS'
  | 'STAR'
  | 'SLASH'
  | 'EQUALS'
  | 'LPAREN'
  | 'RPAREN'
  | 'EOF';

export interface Token {
  type: TokenType;
  value: string;
  position: number;
}

// ── Character classification helpers ─────────────────────────────────

function isDigit(ch: string): boolean {
  return ch >= '0' && ch <= '9';
}

function isVariable(ch: string): boolean {
  return ch === 'x' || ch === 'X';
}

function isWhitespace(ch: string): boolean {
  return ch === ' ' || ch === '\t';
}

// ── Tokenizer ────────────────────────────────────────────────────────

/**
 * Tokenizes an equation string into a list of tokens.
 *
 * Implicit multiplication is inserted when:
 *   1. A number is immediately followed by a variable (`2x`)
 *   2. A number is immediately followed by `(` (`2(x+1)`)
 *   3. A `)` is immediately followed by `(` (`)(`)
 *   4. A variable is immediately followed by `(` (`x(...)`) – rare but valid
 *
 * @throws Error on unrecognised characters.
 */
export function tokenize(input: string): Token[] {
  const tokens: Token[] = [];
  let pos = 0;

  while (pos < input.length) {
    const ch = input[pos];

    // Skip whitespace
    if (isWhitespace(ch)) {
      pos++;
      continue;
    }

    // Multi-digit numbers (integers only in Sprint 1)
    if (isDigit(ch)) {
      const start = pos;
      while (pos < input.length && isDigit(input[pos])) {
        pos++;
      }
      const numberValue = input.slice(start, pos);
      tokens.push({ type: 'NUMBER', value: numberValue, position: start });

      // Implicit multiplication: number followed by variable or '('
      if (pos < input.length && (isVariable(input[pos]) || input[pos] === '(')) {
        tokens.push({ type: 'STAR', value: '*', position: pos });
      }
      continue;
    }

    // Variable
    if (isVariable(ch)) {
      tokens.push({ type: 'VARIABLE', value: 'x', position: pos });
      pos++;

      // Implicit multiplication: variable followed by '('
      if (pos < input.length && input[pos] === '(') {
        tokens.push({ type: 'STAR', value: '*', position: pos });
      }
      continue;
    }

    // Single-character operators and grouping
    switch (ch) {
      case '+':
        tokens.push({ type: 'PLUS', value: '+', position: pos });
        break;
      case '-':
        tokens.push({ type: 'MINUS', value: '-', position: pos });
        break;
      case '*':
        tokens.push({ type: 'STAR', value: '*', position: pos });
        break;
      case '/':
        tokens.push({ type: 'SLASH', value: '/', position: pos });
        break;
      case '=':
        tokens.push({ type: 'EQUALS', value: '=', position: pos });
        break;
      case '(':
        tokens.push({ type: 'LPAREN', value: '(', position: pos });
        break;
      case ')':
        tokens.push({ type: 'RPAREN', value: ')', position: pos });
        pos++;
        // Implicit multiplication: ')' followed by '(' or variable or number
        if (
          pos < input.length &&
          (input[pos] === '(' || isVariable(input[pos]) || isDigit(input[pos]))
        ) {
          tokens.push({ type: 'STAR', value: '*', position: pos });
        }
        continue; // skip pos++ below
      default:
        throw new Error(
          `Unrecognised character '${ch}' at position ${pos}`,
        );
    }

    pos++;
  }

  tokens.push({ type: 'EOF', value: '', position: pos });
  return tokens;
}
