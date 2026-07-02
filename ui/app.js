/**
 * app.js — UI controller for Math Copilot.
 *
 * Wires up the equation input, solve button, example chips,
 * tab switching, and renders results into the DOM.
 *
 * The Steps tab now renders pedagogically rich explanations
 * with step titles, explanation paragraphs, and before/after
 * equation states.
 */

(function () {
  'use strict';

  const { solveLinear, generateExplanation, prettyPrintAST, parseEquation } = window.MathSolver;

  // ── DOM refs ────────────────────────────────────────────────────────
  const inputEl      = document.getElementById('equation-input');
  const solveBtn     = document.getElementById('solve-btn');
  const errorMsg     = document.getElementById('error-msg');
  const resultsEl    = document.getElementById('results-section');
  const solutionVal  = document.getElementById('solution-value');
  const stepsEl      = document.getElementById('steps-container');
  const astOutput    = document.getElementById('ast-output');
  const logTbody     = document.getElementById('log-tbody');

  const tabs = document.querySelectorAll('.tab');
  const tabContents = document.querySelectorAll('.tab-content');
  const exampleBtns = document.querySelectorAll('.example-btn');

  // ── Step colour map ─────────────────────────────────────────────────
  const STEP_COLORS = {
    SIMPLIFY:      'var(--step-simplify)',
    MOVE_TERM:     'var(--step-move)',
    DIVIDE_BOTH:   'var(--step-divide)',
    MULTIPLY_BOTH: 'var(--step-divide)',
    EXPAND:        'var(--step-expand)',
    COMBINE_LIKE:  'var(--step-combine)',
    RESULT:        'var(--step-result)',
  };

  // ── Step icons ──────────────────────────────────────────────────────
  const STEP_ICONS = {
    SIMPLIFY:      '🔍',
    MOVE_TERM:     '↔️',
    DIVIDE_BOTH:   '➗',
    MULTIPLY_BOTH: '✖️',
    EXPAND:        '📐',
    COMBINE_LIKE:  '🔗',
    RESULT:        '✅',
  };

  // ── Tab switching ───────────────────────────────────────────────────
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      tabs.forEach(t => t.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById('content-' + target).classList.add('active');
    });
  });

  // ── Example buttons ─────────────────────────────────────────────────
  exampleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      inputEl.value = btn.dataset.eq;
      solve();
    });
  });

  // ── Solve on Enter ──────────────────────────────────────────────────
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') solve();
  });

  // ── Solve button ────────────────────────────────────────────────────
  solveBtn.addEventListener('click', solve);

  // ── Main solve function ─────────────────────────────────────────────
  function solve() {
    const input = inputEl.value.trim();
    hideError();

    if (!input) {
      showError('Please enter an equation.');
      return;
    }

    try {
      const result = solveLinear(input);
      const steps = generateExplanation(result.log);

      // Format solution
      const solNum = result.solution;
      const solStr = Number.isInteger(solNum) ? String(solNum) : solNum.toFixed(4).replace(/\.?0+$/, '');
      solutionVal.textContent = `x = ${solStr}`;

      // Render educational steps
      renderSteps(steps, result.log);

      // Render AST
      astOutput.textContent = prettyPrintAST(result.ast);

      // Render action log table
      renderActionLog(result.log);

      // Show results
      resultsEl.classList.remove('hidden');
      // Re-trigger animation
      resultsEl.style.animation = 'none';
      resultsEl.offsetHeight; // reflow
      resultsEl.style.animation = '';

    } catch (err) {
      showError(err.message);
      resultsEl.classList.add('hidden');
    }
  }

  // ── Render educational explanation steps ─────────────────────────────
  function renderSteps(steps, log) {
    stepsEl.innerHTML = '';

    steps.forEach((step, idx) => {
      const action = log[idx];
      const card = document.createElement('div');
      card.className = 'step-card' + (step.isFinal ? ' final' : '');
      card.style.setProperty('--step-color', STEP_COLORS[action.type] || 'var(--accent-primary)');
      card.style.animationDelay = `${idx * 0.10}s`;
      card.id = `step-${step.stepNumber}`;

      // Step number badge
      const numBadge = document.createElement('div');
      numBadge.className = 'step-number';
      numBadge.textContent = step.isFinal ? '✓' : step.stepNumber;

      // Step title (pedagogical)
      const titleEl = document.createElement('div');
      titleEl.className = 'step-title';
      const icon = STEP_ICONS[action.type] || '📌';
      titleEl.textContent = `${icon}  Step ${step.stepNumber} — ${step.title}`;

      // Step header (action type label)
      const header = document.createElement('div');
      header.className = 'step-header';
      header.textContent = step.header;

      // Step body (educational paragraph)
      const body = document.createElement('div');
      body.className = 'step-body';
      body.textContent = step.body;

      // Equation state before/after
      const eqState = document.createElement('div');
      eqState.className = 'step-equation';
      eqState.textContent = step.equationState;

      // Metadata hint (if available)
      if (step.meta && step.meta.reason && !step.isFinal) {
        const hint = document.createElement('div');
        hint.className = 'step-hint';
        hint.textContent = '💡 ' + step.meta.reason;
        card.appendChild(numBadge);
        card.appendChild(titleEl);
        card.appendChild(header);
        card.appendChild(body);
        card.appendChild(eqState);
        card.appendChild(hint);
      } else {
        card.appendChild(numBadge);
        card.appendChild(titleEl);
        card.appendChild(header);
        card.appendChild(body);
        card.appendChild(eqState);
      }

      stepsEl.appendChild(card);
    });
  }

  // ── Render action log table ─────────────────────────────────────────
  function renderActionLog(log) {
    logTbody.innerHTML = '';
    log.forEach((action, idx) => {
      const tr = document.createElement('tr');

      const tdNum = document.createElement('td');
      tdNum.textContent = idx + 1;

      const tdType = document.createElement('td');
      const typeSpan = document.createElement('span');
      typeSpan.className = 'log-action-type';
      typeSpan.textContent = action.type;
      tdType.appendChild(typeSpan);

      const tdDesc = document.createElement('td');
      tdDesc.textContent = action.description;

      const tdEq = document.createElement('td');
      tdEq.className = 'log-equation';
      tdEq.textContent = action.equationAfter;

      tr.appendChild(tdNum);
      tr.appendChild(tdType);
      tr.appendChild(tdDesc);
      tr.appendChild(tdEq);
      logTbody.appendChild(tr);
    });
  }

  // ── Error helpers ───────────────────────────────────────────────────
  function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.classList.remove('hidden');
  }

  function hideError() {
    errorMsg.classList.add('hidden');
  }

  // Focus input on load
  inputEl.focus();
})();
