/**
 * Explanation engine tests — pedagogical content quality.
 *
 * Verifies that the explanation engine produces deterministic,
 * structured, and educationally useful output from solver action logs.
 *
 * Key requirements:
 *   - Explanations mention isolating x or the x-term
 *   - Constant removal explains balance, not only "both sides"
 *   - Division explains inverse operation
 *   - Final answer explains that this value satisfies the original equation
 *   - All MVP equation forms solve and produce explanations
 *   - No old weak-only text appears as a full explanation
 */

import { solveLinear } from '../solver/linearSolver';
import {
  generateExplanation,
  formatExplanationText,
} from '../explanation/explanationEngine';

// ── Helper: get all step bodies as a single string for searching ──
function allBodies(input: string): string {
  const result = solveLinear(input);
  const explanation = generateExplanation(result.log);
  return explanation.steps.map((s) => s.body).join('\n');
}

function allTitles(input: string): string {
  const result = solveLinear(input);
  const explanation = generateExplanation(result.log);
  return explanation.steps.map((s) => s.title).join('\n');
}

// ═══════════════════════════════════════════════════════════════════════
// Core determinism and structure tests
// ═══════════════════════════════════════════════════════════════════════

describe('Explanation Engine — Structure', () => {
  test('creates deterministic text from action log', () => {
    const result = solveLinear('3x+4=10');
    const explanation = generateExplanation(result.log);

    const text1 = formatExplanationText(explanation);
    const text2 = formatExplanationText(generateExplanation(result.log));
    expect(text1).toBe(text2);
  });

  test('final solution text is included in steps', () => {
    const result = solveLinear('3x+4=10');
    const explanation = generateExplanation(result.log);

    expect(explanation.finalAnswer).toBe('x = 2');

    const lastStep = explanation.steps[explanation.steps.length - 1];
    expect(lastStep.isFinal).toBe(true);
    expect(lastStep.equationState).toBe('x = 2');
  });

  test('all steps have sequential step numbers', () => {
    const result = solveLinear('2(x+1)=10');
    const explanation = generateExplanation(result.log);

    explanation.steps.forEach((step, index) => {
      expect(step.stepNumber).toBe(index + 1);
    });
  });

  test('steps have non-empty headers, titles, and bodies', () => {
    const result = solveLinear('5x-2=18');
    const explanation = generateExplanation(result.log);

    explanation.steps.forEach((step) => {
      expect(step.header.length).toBeGreaterThan(0);
      expect(step.title.length).toBeGreaterThan(0);
      expect(step.body.length).toBeGreaterThan(0);
    });
  });

  test('formatted text contains step markers with titles', () => {
    const result = solveLinear('x+5=9');
    const explanation = generateExplanation(result.log);
    const text = formatExplanationText(explanation);

    expect(text).toContain('Step 1');
    expect(text).toContain('x = 4');
    // Should have title format: "Step N — Title"
    expect(text).toMatch(/Step 1 — /);
  });

  test('explanation for x/2=5 includes correct final answer', () => {
    const result = solveLinear('x/2=5');
    const explanation = generateExplanation(result.log);

    expect(explanation.finalAnswer).toBe('x = 10');
  });

  test('steps include pedagogical metadata when available', () => {
    const result = solveLinear('3x+2=17');
    const explanation = generateExplanation(result.log);

    const stepsWithMeta = explanation.steps.filter((s) => s.meta);
    expect(stepsWithMeta.length).toBeGreaterThan(0);

    // Check that metadata has expected fields
    const firstMeta = stepsWithMeta[0].meta!;
    expect(firstMeta.goal).toBeDefined();
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Pedagogical content quality tests
// ═══════════════════════════════════════════════════════════════════════

describe('Explanation Engine — Pedagogical Quality', () => {
  test('explanations mention isolating x or the x-term', () => {
    const bodies = allBodies('3x+2=17');
    expect(bodies).toMatch(/isolate/i);
  });

  test('constant removal explains balance, not only "both sides"', () => {
    const bodies = allBodies('3x+2=17');
    // Should mention both "balance" / "balanced" AND "both sides"
    expect(bodies).toMatch(/balanced?/i);
    expect(bodies).toMatch(/both sides/i);
    // The MOVE_TERM step should NOT be just "Subtract 2 from both sides"
    const result = solveLinear('3x+2=17');
    const explanation = generateExplanation(result.log);
    const moveSteps = explanation.steps.filter((s) => s.header === 'Remove a term');
    moveSteps.forEach((step) => {
      // The body should contain more than just "... from both sides"
      expect(step.body.length).toBeGreaterThan(30);
      expect(step.body).toMatch(/isolate|remove|left side/i);
    });
  });

  test('division explains inverse operation', () => {
    const bodies = allBodies('3x+2=17');
    expect(bodies).toMatch(/inverse/i);
    expect(bodies).toMatch(/division|divide/i);
  });

  test('final answer explains that this value satisfies the original equation', () => {
    const bodies = allBodies('3x+2=17');
    // Should mention that the value makes the equation true / satisfies it
    expect(bodies).toMatch(/makes the original equation true|satisfies|confirms/i);
  });

  test('first step mentions wanting to find x', () => {
    const bodies = allBodies('3x+2=17');
    expect(bodies).toMatch(/find the value of x/i);
  });

  test('subtraction step for positive constant explains removing the constant', () => {
    const result = solveLinear('5x+3=18');
    const explanation = generateExplanation(result.log);
    const moveStep = explanation.steps.find(
      (s) => s.meta?.inverseOperation === 'subtraction'
    );
    expect(moveStep).toBeDefined();
    expect(moveStep!.body).toMatch(/remove \+3/i);
    expect(moveStep!.body).toMatch(/subtract 3/i);
    expect(moveStep!.body).toMatch(/balanced/i);
  });

  test('addition step for negative constant explains undoing subtraction', () => {
    const result = solveLinear('5x-2=18');
    const explanation = generateExplanation(result.log);
    const moveStep = explanation.steps.find(
      (s) => s.meta?.inverseOperation === 'addition'
    );
    expect(moveStep).toBeDefined();
    expect(moveStep!.body).toMatch(/remove -2|remove \-2/i);
    expect(moveStep!.body).toMatch(/add 2/i);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// No old weak-only text
// ═══════════════════════════════════════════════════════════════════════

describe('Explanation Engine — No Weak Explanations', () => {
  const weakTexts = [
    'Subtract 2 from both sides',
    'Divide both sides by 3',
    'Add 2 to both sides',
    'Subtract 4 from both sides',
    'Divide both sides by 5',
    'Multiply both sides by 2',
  ];

  const testEquations = [
    '3x+2=17',
    '5x-2=18',
    'x/2=5',
    '2(x+1)=10',
    '2(x-3)=10',
    'x+5=9',
    'x-3=7',
    '4x=20',
    '3x/4=6',
  ];

  testEquations.forEach((eq) => {
    test(`${eq}: no step body is just a weak mechanical explanation`, () => {
      const result = solveLinear(eq);
      const explanation = generateExplanation(result.log);

      explanation.steps.forEach((step) => {
        weakTexts.forEach((weak) => {
          // The body should NOT be exactly the weak text (case insensitive match)
          // It's fine if the weak phrase appears as part of a longer explanation.
          expect(step.body.trim().toLowerCase()).not.toBe(weak.toLowerCase());
        });
      });
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════
// MVP equation forms — solve + produce explanations
// ═══════════════════════════════════════════════════════════════════════

describe('Explanation Engine — All MVP Equation Forms', () => {
  const mvpCases: Array<{ input: string; expected: number; desc: string }> = [
    { input: 'x+5=9',     expected: 4,  desc: 'x+a=b' },
    { input: 'x-3=7',     expected: 10, desc: 'x-a=b' },
    { input: '4x=20',     expected: 5,  desc: 'ax=b' },
    { input: '3x+2=17',   expected: 5,  desc: 'ax+b=c' },
    { input: '5x-2=18',   expected: 4,  desc: 'ax-b=c' },
    { input: 'x/2=5',     expected: 10, desc: 'x/a=b' },
    { input: '3x/4=6',    expected: 8,  desc: 'ax/b=c' },
    { input: '2(x+1)=10', expected: 4,  desc: 'a(x+b)=c' },
    { input: '2(x-3)=10', expected: 8,  desc: 'a(x-b)=c' },
  ];

  mvpCases.forEach(({ input, expected, desc }) => {
    test(`${desc}: ${input} → x=${expected} with explanations`, () => {
      const result = solveLinear(input);
      expect(result.solution).toBe(expected);

      const explanation = generateExplanation(result.log);
      expect(explanation.steps.length).toBeGreaterThanOrEqual(3);
      expect(explanation.finalAnswer).toBe(`x = ${expected}`);

      // Every step should have non-empty content
      explanation.steps.forEach((step) => {
        expect(step.body.length).toBeGreaterThan(0);
        expect(step.title.length).toBeGreaterThan(0);
        expect(step.equationState.length).toBeGreaterThan(0);
      });

      // Last step should be final
      const last = explanation.steps[explanation.steps.length - 1];
      expect(last.isFinal).toBe(true);
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Solver correctness unchanged (regression)
// ═══════════════════════════════════════════════════════════════════════

describe('Explanation Engine — Solver Correctness Unchanged', () => {
  test('3x+4=10 → x=2', () => {
    expect(solveLinear('3x+4=10').solution).toBe(2);
  });

  test('x/2=5 → x=10', () => {
    expect(solveLinear('x/2=5').solution).toBe(10);
  });

  test('2(x+1)=10 → x=4', () => {
    expect(solveLinear('2(x+1)=10').solution).toBe(4);
  });

  test('5x-2=18 → x=4', () => {
    expect(solveLinear('5x-2=18').solution).toBe(4);
  });

  test('no-solution equation throws', () => {
    expect(() => solveLinear('x=x+5')).toThrow(/No solution/);
  });

  test('action log ends with RESULT', () => {
    const result = solveLinear('3x+4=10');
    const actions = result.log.getActions();
    expect(actions[actions.length - 1].type).toBe('RESULT');
  });

  test('action log result equationAfter is correct', () => {
    const result = solveLinear('3x+4=10');
    const actions = result.log.getActions();
    const resultAction = actions.find((a) => a.type === 'RESULT');
    expect(resultAction).toBeDefined();
    expect(resultAction!.equationAfter).toBe('x = 2');
  });
});
