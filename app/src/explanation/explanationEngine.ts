/**
 * Explanation engine.
 *
 * Converts a solver ActionLog into a list of human-readable,
 * pedagogically rich step-by-step explanation steps suitable
 * for UI display.
 *
 * Each step now includes:
 *   - a descriptive title (not just a technical label)
 *   - an educational body explaining *why* and *how*
 *   - optional metadata for programmatic access
 */

import { ActionLog, Action, ExplanationMeta } from '../solver/actionLog';
import { EXPLANATION_TEMPLATES, ExplanationTemplate } from './explanationTemplates';

// ── Public types ─────────────────────────────────────────────────────

export interface ExplanationStep {
  /** Step number (1-based) */
  stepNumber: number;
  /** Short header, e.g. "Remove a term" */
  header: string;
  /** Pedagogical step title, e.g. "Isolate 3x" */
  title: string;
  /** Detailed explanation text */
  body: string;
  /** Equation state after this step */
  equationState: string;
  /** Whether this is the final result step */
  isFinal: boolean;
  /** Optional pedagogical metadata */
  meta?: ExplanationMeta;
}

export interface Explanation {
  steps: ExplanationStep[];
  /** The final solution as a readable string, e.g. "x = 2" */
  finalAnswer: string;
}

// ── Engine ───────────────────────────────────────────────────────────

/**
 * Generate a complete explanation from a solver action log.
 *
 * @param log – the action log produced by the solver
 * @returns structured explanation with numbered steps
 */
export function generateExplanation(log: ActionLog): Explanation {
  const actions = log.getActions();

  if (actions.length === 0) {
    throw new Error('Cannot generate explanation from empty action log');
  }

  const steps: ExplanationStep[] = actions.map((action, index) => {
    const template = EXPLANATION_TEMPLATES[action.type];
    return {
      stepNumber: index + 1,
      header: template.header,
      title: applyTemplateMeta(template.title, action),
      body: applyTemplateMeta(template.body, action),
      equationState: action.equationAfter,
      isFinal: action.type === 'RESULT',
      ...(action.meta ? { meta: action.meta } : {}),
    };
  });

  // The last action should be the RESULT
  const lastAction = actions[actions.length - 1];
  const finalAnswer = lastAction.equationAfter;

  return { steps, finalAnswer };
}

/**
 * Format the full explanation as a plain-text string.
 * Useful for testing and console output.
 */
export function formatExplanationText(explanation: Explanation): string {
  const lines = explanation.steps.map((step) => {
    const titleLine = `Step ${step.stepNumber} — ${step.title}`;
    const bodyLine = `  "${step.body}"`;
    const eqLine = `  → ${step.equationState}`;
    return `${titleLine}\n${bodyLine}\n${eqLine}`;
  });

  return lines.join('\n\n');
}

// ── Template helpers ─────────────────────────────────────────────────

/**
 * Apply template substitution with both description and metadata fields.
 */
function applyTemplateMeta(templateStr: string, action: Action): string {
  let result = templateStr.replace('{description}', action.description);

  if (action.meta) {
    result = result.replace('{goal}', action.meta.goal ?? '');
    result = result.replace('{targetTerm}', action.meta.targetTerm ?? '');
    result = result.replace('{inverseOperation}', action.meta.inverseOperation ?? '');
    result = result.replace('{reason}', action.meta.reason ?? '');
  } else {
    // Fallback: remove unfilled placeholders
    result = result.replace('{goal}', action.description);
    result = result.replace('{targetTerm}', '');
    result = result.replace('{inverseOperation}', '');
    result = result.replace('{reason}', '');
  }

  return result;
}
