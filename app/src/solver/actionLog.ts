/**
 * Action log for the solver.
 *
 * Each step the solver takes is recorded as an Action so the
 * explanation engine can produce human-readable educational output.
 */

export type ActionType =
  | 'EXPAND'         // e.g. "Distribute 2 across (x+1)"
  | 'COMBINE_LIKE'   // e.g. "Combine like terms on the left"
  | 'MOVE_TERM'      // e.g. "Subtract 4 from both sides"
  | 'DIVIDE_BOTH'    // e.g. "Divide both sides by 3"
  | 'MULTIPLY_BOTH'  // e.g. "Multiply both sides by 2"
  | 'SIMPLIFY'       // e.g. "Simplify: 6/3 = 2"
  | 'RESULT';        // e.g. "x = 2"

/**
 * Optional pedagogical metadata attached to solver actions.
 *
 * These fields let the explanation engine produce richer,
 * more educational output without changing the solver's core logic.
 */
export interface ExplanationMeta {
  /** What this step is trying to achieve, e.g. "Isolate 3x" */
  goal?: string;
  /** The term being targeted, e.g. "+2" or "3x" */
  targetTerm?: string;
  /** The inverse operation being applied, e.g. "subtraction" */
  inverseOperation?: string;
  /** Human-readable reason for the step */
  reason?: string;
}

export interface Action {
  type: ActionType;
  description: string;
  /** Equation state after this step, as a readable string. */
  equationAfter: string;
  /** Optional pedagogical metadata for richer explanations. */
  meta?: ExplanationMeta;
}

/**
 * Mutable log that accumulates solver actions during a solve pass.
 */
export class ActionLog {
  private actions: Action[] = [];

  add(type: ActionType, description: string, equationAfter: string, meta?: ExplanationMeta): void {
    this.actions.push({ type, description, equationAfter, ...(meta ? { meta } : {}) });
  }

  getActions(): ReadonlyArray<Action> {
    return this.actions;
  }

  clear(): void {
    this.actions = [];
  }
}
