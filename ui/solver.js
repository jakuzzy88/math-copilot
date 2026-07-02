/**
 * solver.js — Pure-JS port of the Math Core (tokenizer + parser + solver + explanation).
 *
 * Sprint B: Self-contained equation solving engine for the local UI.
 * This is a faithful port of the TypeScript modules in app/src/.
 */

// ═══════════════════════════════════════════════════════════════════════
// Tokenizer
// ═══════════════════════════════════════════════════════════════════════

function isDigit(ch) { return ch >= '0' && ch <= '9'; }
function isVariable(ch) { return ch === 'x' || ch === 'X'; }
function isWhitespace(ch) { return ch === ' ' || ch === '\t'; }

function tokenize(input) {
  const tokens = [];
  let pos = 0;

  while (pos < input.length) {
    const ch = input[pos];

    if (isWhitespace(ch)) { pos++; continue; }

    if (isDigit(ch)) {
      const start = pos;
      while (pos < input.length && isDigit(input[pos])) pos++;
      tokens.push({ type: 'NUMBER', value: input.slice(start, pos), position: start });
      if (pos < input.length && (isVariable(input[pos]) || input[pos] === '(')) {
        tokens.push({ type: 'STAR', value: '*', position: pos });
      }
      continue;
    }

    if (isVariable(ch)) {
      tokens.push({ type: 'VARIABLE', value: 'x', position: pos });
      pos++;
      if (pos < input.length && input[pos] === '(') {
        tokens.push({ type: 'STAR', value: '*', position: pos });
      }
      continue;
    }

    switch (ch) {
      case '+': tokens.push({ type: 'PLUS', value: '+', position: pos }); break;
      case '-': tokens.push({ type: 'MINUS', value: '-', position: pos }); break;
      case '*': tokens.push({ type: 'STAR', value: '*', position: pos }); break;
      case '/': tokens.push({ type: 'SLASH', value: '/', position: pos }); break;
      case '=': tokens.push({ type: 'EQUALS', value: '=', position: pos }); break;
      case '(':
        tokens.push({ type: 'LPAREN', value: '(', position: pos });
        break;
      case ')':
        tokens.push({ type: 'RPAREN', value: ')', position: pos });
        pos++;
        if (pos < input.length && (input[pos] === '(' || isVariable(input[pos]) || isDigit(input[pos]))) {
          tokens.push({ type: 'STAR', value: '*', position: pos });
        }
        continue;
      default:
        throw new Error(`Unrecognised character '${ch}' at position ${pos}`);
    }
    pos++;
  }

  tokens.push({ type: 'EOF', value: '', position: pos });
  return tokens;
}


// ═══════════════════════════════════════════════════════════════════════
// AST helpers
// ═══════════════════════════════════════════════════════════════════════

function num(value) { return { type: 'NumberLiteral', value }; }
function variable(name) { return { type: 'Variable', name: name || 'x' }; }
function binOp(op, left, right) { return { type: 'BinaryOp', operator: op, left, right }; }
function unaryMinus(operand) { return { type: 'UnaryMinus', operand }; }
function equation(left, right) { return { type: 'Equation', left, right }; }


// ═══════════════════════════════════════════════════════════════════════
// Parser
// ═══════════════════════════════════════════════════════════════════════

class Parser {
  constructor(tokens) {
    this.tokens = tokens;
    this.pos = 0;
  }

  current() { return this.tokens[this.pos]; }
  peek() { return this.current().type; }

  eat(expected) {
    const tok = this.current();
    if (tok.type !== expected) {
      throw new Error(`Expected ${expected} but found ${tok.type} ('${tok.value}') at position ${tok.position}`);
    }
    this.pos++;
    return tok;
  }

  parseEquation() {
    const left = this.parseExpression();
    this.eat('EQUALS');
    if (this.peek() === 'EQUALS') {
      throw new Error(`Unexpected second '=' at position ${this.current().position}`);
    }
    const right = this.parseExpression();
    this.eat('EOF');
    return equation(left, right);
  }

  parseExpression() {
    let node = this.parseTerm();
    while (this.peek() === 'PLUS' || this.peek() === 'MINUS') {
      const op = this.current().value;
      this.pos++;
      const right = this.parseTerm();
      node = binOp(op, node, right);
    }
    return node;
  }

  parseTerm() {
    let node = this.parseFactor();
    while (this.peek() === 'STAR' || this.peek() === 'SLASH') {
      const op = this.current().value;
      this.pos++;
      const right = this.parseFactor();
      node = binOp(op, node, right);
    }
    return node;
  }

  parseFactor() {
    const tok = this.current();
    if (tok.type === 'NUMBER') {
      this.pos++;
      return num(parseInt(tok.value, 10));
    }
    if (tok.type === 'VARIABLE') {
      this.pos++;
      return variable(tok.value);
    }
    if (tok.type === 'LPAREN') {
      this.pos++;
      if (this.peek() === 'RPAREN') throw new Error(`Empty parentheses at position ${tok.position}`);
      const node = this.parseExpression();
      this.eat('RPAREN');
      return node;
    }
    if (tok.type === 'MINUS') {
      this.pos++;
      const operand = this.parseFactor();
      return unaryMinus(operand);
    }
    throw new Error(`Unexpected token ${tok.type} ('${tok.value}') at position ${tok.position}`);
  }
}

function parseEquation(input) {
  const tokens = tokenize(input);
  const parser = new Parser(tokens);
  return parser.parseEquation();
}


// ═══════════════════════════════════════════════════════════════════════
// Linear Solver (with pedagogical descriptions)
// ═══════════════════════════════════════════════════════════════════════

function extractLinearForm(node) {
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
        case '+': return { a: left.a + right.a, b: left.b + right.b };
        case '-': return { a: left.a - right.a, b: left.b - right.b };
        case '*':
          if (left.a === 0) return { a: left.b * right.a, b: left.b * right.b };
          if (right.a === 0) return { a: right.b * left.a, b: right.b * left.b };
          throw new Error('Non-linear equation: cannot multiply two expressions containing x');
        case '/':
          if (right.a !== 0) throw new Error('Non-linear: cannot divide by expression containing x');
          if (right.b === 0) throw new Error('Division by zero');
          return { a: left.a / right.b, b: left.b / right.b };
      }
    }
  }
}

function formatCoeff(n) {
  if (n === 1) return '';
  if (n === -1) return '-';
  return String(n);
}

function formatXTerm(leftA, rightA) {
  if (leftA === 1 && rightA === 0) return 'x';
  if (leftA === -1 && rightA === 0) return '-x';
  if (rightA === 0) return leftA + 'x';
  if (leftA === 1) return 'x';
  if (leftA === -1) return '-x';
  return leftA + 'x';
}

function describeConstantAttachment(leftB) {
  if (leftB === 0) return '';
  if (leftB > 0) return ', but it has +' + leftB + ' added to it';
  return ', but it has ' + leftB + ' subtracted from it';
}

function solveLinear(input) {
  const ast = parseEquation(input);
  const log = [];

  const leftForm = extractLinearForm(ast.left);
  const rightForm = extractLinearForm(ast.right);

  const xTermStr = formatXTerm(leftForm.a, rightForm.a);
  const constAttached = describeConstantAttachment(leftForm.b);

  log.push({
    type: 'SIMPLIFY',
    description: 'We want to find the value of x. In ' + input + ', the x-term is ' + xTermStr + constAttached + '.',
    equationAfter: input,
    meta: { goal: 'Understand the equation', targetTerm: xTermStr, reason: 'Identify the variable term and any constants attached to it' },
  });

  const combinedA = leftForm.a - rightForm.a;
  const combinedB = rightForm.b - leftForm.b;

  if (leftForm.b !== 0 || rightForm.a !== 0) {
    if (rightForm.a !== 0) {
      const rCoeff = formatCoeff(rightForm.a);
      log.push({
        type: 'MOVE_TERM',
        description: 'Variable terms appear on both sides. To collect all x-terms on the left, subtract ' + rCoeff + 'x from both sides to keep the equation balanced.',
        equationAfter: formatCoeff(combinedA) + 'x + ' + leftForm.b + ' = ' + rightForm.b,
        meta: { goal: 'Collect all x-terms on the left side', targetTerm: rCoeff + 'x', inverseOperation: 'subtraction' },
      });
    }
    if (leftForm.b !== 0) {
      const absB = Math.abs(leftForm.b);
      const coeffXStr = formatCoeff(combinedA) + 'x';
      if (leftForm.b > 0) {
        log.push({
          type: 'MOVE_TERM',
          description: 'To isolate ' + coeffXStr + ', we need to remove +' + leftForm.b + ' from the left side. To keep the equation balanced, we subtract ' + leftForm.b + ' from both sides.',
          equationAfter: formatCoeff(combinedA) + 'x = ' + combinedB,
          meta: { goal: 'Isolate ' + coeffXStr, targetTerm: '+' + leftForm.b, inverseOperation: 'subtraction', reason: 'Subtracting ' + leftForm.b + ' undoes the addition of ' + leftForm.b },
        });
      } else {
        log.push({
          type: 'MOVE_TERM',
          description: 'To isolate ' + coeffXStr + ', we need to remove ' + leftForm.b + ' from the left side. To keep the equation balanced, we add ' + absB + ' to both sides.',
          equationAfter: formatCoeff(combinedA) + 'x = ' + combinedB,
          meta: { goal: 'Isolate ' + coeffXStr, targetTerm: '' + leftForm.b, inverseOperation: 'addition', reason: 'Adding ' + absB + ' undoes the subtraction of ' + absB },
        });
      }
    }
  }

  if (combinedA === 0) {
    if (combinedB === 0) throw new Error('Infinite solutions: the equation is always true');
    throw new Error('No solution: the equation is contradictory');
  }

  const solution = combinedB / combinedA;

  if (combinedA !== 1) {
    log.push({
      type: 'DIVIDE_BOTH',
      description: 'Now x is multiplied by ' + combinedA + '. Division is the inverse of multiplication, so we divide both sides by ' + combinedA + ' to leave x alone.',
      equationAfter: 'x = ' + combinedB + ' / ' + combinedA,
      meta: { goal: 'Isolate x', targetTerm: '' + combinedA, inverseOperation: 'division', reason: 'Dividing by ' + combinedA + ' undoes the multiplication by ' + combinedA },
    });
  }

  log.push({
    type: 'SIMPLIFY',
    description: combinedB + ' divided by ' + combinedA + ' is ' + solution + '.',
    equationAfter: 'x = ' + solution,
    meta: { goal: 'Simplify the result', reason: 'Performing the arithmetic: ' + combinedB + ' ÷ ' + combinedA + ' = ' + solution },
  });

  log.push({
    type: 'RESULT',
    description: 'The value that makes the original equation true is x = ' + solution + '. Substituting ' + solution + ' back into ' + input + ' confirms the solution.',
    equationAfter: 'x = ' + solution,
    meta: { goal: 'State the final answer', reason: 'x = ' + solution + ' satisfies the original equation ' + input },
  });

  return { solution, log, ast };
}


// ═══════════════════════════════════════════════════════════════════════
// Explanation Engine (pedagogical)
// ═══════════════════════════════════════════════════════════════════════

const TEMPLATES = {
  EXPAND:       { header: 'Expand',             title: 'Distribute and expand',  body: '{description}' },
  COMBINE_LIKE: { header: 'Combine like terms', title: 'Combine like terms',     body: '{description}' },
  MOVE_TERM:    { header: 'Remove a term',      title: '{goal}',                 body: '{description}' },
  DIVIDE_BOTH:  { header: 'Undo multiplication',title: '{goal}',                 body: '{description}' },
  MULTIPLY_BOTH:{ header: 'Undo division',      title: '{goal}',                 body: '{description}' },
  SIMPLIFY:     { header: 'Simplify',           title: '{goal}',                 body: '{description}' },
  RESULT:       { header: '✓ Final answer',     title: 'Final answer',           body: '{description}' },
};

function applyTemplateMeta(templateStr, action) {
  var result = templateStr.replace('{description}', action.description);
  if (action.meta) {
    result = result.replace('{goal}', action.meta.goal || '');
    result = result.replace('{targetTerm}', action.meta.targetTerm || '');
    result = result.replace('{inverseOperation}', action.meta.inverseOperation || '');
    result = result.replace('{reason}', action.meta.reason || '');
  } else {
    result = result.replace('{goal}', action.description);
    result = result.replace('{targetTerm}', '');
    result = result.replace('{inverseOperation}', '');
    result = result.replace('{reason}', '');
  }
  return result;
}

function generateExplanation(actionLog) {
  return actionLog.map(function(action, idx) {
    var tmpl = TEMPLATES[action.type] || { header: action.type, title: action.type, body: '{description}' };
    return {
      stepNumber: idx + 1,
      header: tmpl.header,
      title: applyTemplateMeta(tmpl.title, action),
      body: applyTemplateMeta(tmpl.body, action),
      equationState: action.equationAfter,
      isFinal: action.type === 'RESULT',
      meta: action.meta || null,
    };
  });
}


// ═══════════════════════════════════════════════════════════════════════
// AST Pretty Printer
// ═══════════════════════════════════════════════════════════════════════

function prettyPrintAST(node, indent) {
  indent = indent || 0;
  const pad = '  '.repeat(indent);
  switch (node.type) {
    case 'Equation':
      return `${pad}Equation\n${pad}├─ left:\n${prettyPrintAST(node.left, indent + 2)}\n${pad}└─ right:\n${prettyPrintAST(node.right, indent + 2)}`;
    case 'BinaryOp':
      return `${pad}BinaryOp [${node.operator}]\n${pad}├─ left:\n${prettyPrintAST(node.left, indent + 2)}\n${pad}└─ right:\n${prettyPrintAST(node.right, indent + 2)}`;
    case 'UnaryMinus':
      return `${pad}UnaryMinus\n${pad}└─ operand:\n${prettyPrintAST(node.operand, indent + 2)}`;
    case 'NumberLiteral':
      return `${pad}Number(${node.value})`;
    case 'Variable':
      return `${pad}Variable(${node.name || 'x'})`;
    default:
      return `${pad}Unknown(${node.type})`;
  }
}


// Export for app.js
window.MathSolver = {
  tokenize,
  parseEquation,
  solveLinear,
  generateExplanation,
  prettyPrintAST,
};
