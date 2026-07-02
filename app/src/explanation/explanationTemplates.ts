/**
 * Explanation templates.
 *
 * Maps action types to human-readable template strings.
 * Templates use `{placeholder}` syntax for substitution.
 *
 * Each template now has a pedagogical `title` (short, learner-friendly),
 * in addition to the header and body for detailed explanations.
 */

import { ActionType } from '../solver/actionLog';

export interface ExplanationTemplate {
  /** Short header for UI display */
  header: string;
  /** Pedagogical step title, e.g. "Understand the equation" */
  title: string;
  /** Detailed explanation body */
  body: string;
}

/**
 * Template map keyed by ActionType.
 *
 * Each template has a header (for step summaries), a pedagogical title,
 * and body (for detail view).  The body uses `{description}` for the
 * solver-generated text and `{goal}` / `{reason}` for metadata.
 */
export const EXPLANATION_TEMPLATES: Record<ActionType, ExplanationTemplate> = {
  EXPAND: {
    header: 'Expand',
    title: 'Distribute and expand',
    body: '{description}',
  },
  COMBINE_LIKE: {
    header: 'Combine like terms',
    title: 'Combine like terms',
    body: '{description}',
  },
  MOVE_TERM: {
    header: 'Remove a term',
    title: '{goal}',
    body: '{description}',
  },
  DIVIDE_BOTH: {
    header: 'Undo multiplication',
    title: '{goal}',
    body: '{description}',
  },
  MULTIPLY_BOTH: {
    header: 'Undo division',
    title: '{goal}',
    body: '{description}',
  },
  SIMPLIFY: {
    header: 'Simplify',
    title: '{goal}',
    body: '{description}',
  },
  RESULT: {
    header: '✓ Final answer',
    title: 'Final answer',
    body: '{description}',
  },
};
