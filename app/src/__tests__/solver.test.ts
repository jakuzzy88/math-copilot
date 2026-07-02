/**
 * Solver tests.
 *
 * Verifies that the symbolic linear solver produces correct solutions
 * for all supported equation forms.
 */

import { solveLinear } from '../solver/linearSolver';

describe('Linear Solver', () => {
  test('3x+4=10 → x=2', () => {
    const result = solveLinear('3x+4=10');
    expect(result.solution).toBe(2);
  });

  test('x/2=5 → x=10', () => {
    const result = solveLinear('x/2=5');
    expect(result.solution).toBe(10);
  });

  test('2(x+1)=10 → x=4', () => {
    const result = solveLinear('2(x+1)=10');
    expect(result.solution).toBe(4);
  });

  test('5x-2=18 → x=4', () => {
    const result = solveLinear('5x-2=18');
    expect(result.solution).toBe(4);
  });

  test('2x=8 → x=4', () => {
    const result = solveLinear('2x=8');
    expect(result.solution).toBe(4);
  });

  test('x+5=9 → x=4', () => {
    const result = solveLinear('x+5=9');
    expect(result.solution).toBe(4);
  });

  test('3x/4=6 → x=8', () => {
    const result = solveLinear('3x/4=6');
    expect(result.solution).toBe(8);
  });

  test('produces a non-empty action log', () => {
    const result = solveLinear('3x+4=10');
    const actions = result.log.getActions();
    expect(actions.length).toBeGreaterThan(0);
    // The last action should be a RESULT
    expect(actions[actions.length - 1].type).toBe('RESULT');
  });

  test('action log contains solution text', () => {
    const result = solveLinear('3x+4=10');
    const actions = result.log.getActions();
    const resultAction = actions.find((a) => a.type === 'RESULT');
    expect(resultAction).toBeDefined();
    expect(resultAction!.equationAfter).toBe('x = 2');
  });

  test('no-solution equation throws', () => {
    // 0x = 5 → no solution (would need 2x - 2x = 5, but we can construct
    // an equation like x - x = 5 via "x=x+5" which is 0x = 5 after rearranging)
    expect(() => solveLinear('x=x+5')).toThrow(/No solution/);
  });
});
