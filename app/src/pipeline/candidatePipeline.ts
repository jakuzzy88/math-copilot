/**
 * Candidate pipeline — deterministic OCR result adapter.
 *
 * Sprint D: OCR Result Pipeline Adapter.
 *
 * Receives an array of OCR candidates, normalises each, validates,
 * parses, solves, scores, and returns the best accepted equation
 * or a structured rejection.
 *
 * Pipeline stages per candidate:
 *   1. Normalise OCR text (fix common misreads)
 *   2. Validate grammar (structural checks)
 *   3. Parse equation into AST
 *   4. Solve for x
 *   5. Generate explanation steps
 *   6. Compute composite score
 *
 * The candidate with the highest composite score wins.
 * If no candidate passes all stages, a structured rejection is returned.
 *
 * The pipeline does NOT duplicate solver/parser/explanation logic —
 * it imports and delegates to the existing modules.
 */

import type {
  OcrCandidate,
  CandidateEvaluation,
  PipelineResult,
  PipelineAcceptedResult,
  PipelineRejectedResult,
  RejectionReason,
} from './types';
import { normalizeOcrText } from './ocrNormalizer';
import { validateEquation } from '../grammar/grammarValidator';
import { parseEquation } from '../parser/parser';
import { solveFromAST } from '../solver/linearSolver';
import {
  generateExplanation,
  type ExplanationStep,
} from '../explanation/explanationEngine';


// ── Scoring ─────────────────────────────────────────────────────────

/**
 * Compute a composite score for a candidate evaluation.
 *
 * Score components (all deterministic):
 *   - Base: model confidence [0, 1]  (weight: 0.4)
 *   - Grammar validity bonus:         +0.15 if valid
 *   - Parse success bonus:             +0.15 if parseable
 *   - Solve success bonus:             +0.20 if solvable
 *   - Correction penalty:              -0.02 per correction applied
 *
 * Maximum possible score: 0.4 * 1.0 + 0.15 + 0.15 + 0.20 = 0.90
 * A fully correct, high-confidence candidate will score ~0.88.
 */
function computeScore(eval_: {
  confidence: number;
  grammarValid: boolean;
  parseable: boolean;
  solvable: boolean;
  correctionCount: number;
}): number {
  let score = eval_.confidence * 0.4;

  if (eval_.grammarValid) score += 0.15;
  if (eval_.parseable) score += 0.15;
  if (eval_.solvable) score += 0.20;

  // Penalty for corrections: each correction reduces trustworthiness.
  score -= eval_.correctionCount * 0.02;

  // Clamp to [0, 1].
  return Math.max(0, Math.min(1, score));
}


// ── Per-candidate evaluation ────────────────────────────────────────

/**
 * Evaluate a single OCR candidate through all pipeline stages.
 *
 * This function never throws — all errors are captured in the
 * evaluation result.
 */
function evaluateCandidate(candidate: OcrCandidate): CandidateEvaluation {
  // Stage 1: Normalise.
  const normalization = normalizeOcrText(candidate.text);
  const normalizedText = normalization.normalized;

  // Stage 2: Grammar validation.
  const grammarResult = validateEquation(normalizedText);

  // Early-out if grammar is invalid — don't waste time parsing.
  if (!grammarResult.valid) {
    return {
      candidate,
      normalization,
      grammarValid: false,
      grammarErrors: grammarResult.errors,
      parseable: false,
      parseError: null,
      solvable: false,
      solveError: null,
      solution: NaN,
      score: computeScore({
        confidence: candidate.confidence,
        grammarValid: false,
        parseable: false,
        solvable: false,
        correctionCount: normalization.corrections.length,
      }),
      explanationSteps: [],
    };
  }

  // Stage 3: Parse.
  let ast;
  try {
    ast = parseEquation(normalizedText);
  } catch (err: unknown) {
    const parseError = err instanceof Error ? err.message : String(err);
    return {
      candidate,
      normalization,
      grammarValid: true,
      grammarErrors: [],
      parseable: false,
      parseError,
      solvable: false,
      solveError: null,
      solution: NaN,
      score: computeScore({
        confidence: candidate.confidence,
        grammarValid: true,
        parseable: false,
        solvable: false,
        correctionCount: normalization.corrections.length,
      }),
      explanationSteps: [],
    };
  }

  // Stage 4: Solve.
  let solution: number;
  let explanationSteps: ExplanationStep[];
  try {
    const solveResult = solveFromAST(ast, normalizedText);
    solution = solveResult.solution;

    // Stage 5: Generate explanation.
    const explanation = generateExplanation(solveResult.log);
    explanationSteps = explanation.steps;
  } catch (err: unknown) {
    const solveError = err instanceof Error ? err.message : String(err);
    return {
      candidate,
      normalization,
      grammarValid: true,
      grammarErrors: [],
      parseable: true,
      parseError: null,
      solvable: false,
      solveError,
      solution: NaN,
      score: computeScore({
        confidence: candidate.confidence,
        grammarValid: true,
        parseable: true,
        solvable: false,
        correctionCount: normalization.corrections.length,
      }),
      explanationSteps: [],
    };
  }

  // All stages passed.
  return {
    candidate,
    normalization,
    grammarValid: true,
    grammarErrors: [],
    parseable: true,
    parseError: null,
    solvable: true,
    solveError: null,
    solution,
    score: computeScore({
      confidence: candidate.confidence,
      grammarValid: true,
      parseable: true,
      solvable: true,
      correctionCount: normalization.corrections.length,
    }),
    explanationSteps,
  };
}


// ── Pipeline entry point ────────────────────────────────────────────

/**
 * Process an array of OCR candidates and return the best accepted
 * equation or a structured rejection.
 *
 * The pipeline is fully deterministic: same input always produces
 * the same output. No randomness, no side effects.
 *
 * @param candidates – Array of OCR candidates (may be empty).
 * @returns Accepted result with solution, or rejected result with reason.
 */
export function processCandidates(candidates: OcrCandidate[]): PipelineResult {
  // Edge case: no candidates.
  if (candidates.length === 0) {
    return {
      accepted: false,
      rejection: {
        code: 'NO_CANDIDATES',
        message: 'No OCR candidates provided.',
        candidateErrors: [],
      },
    };
  }

  // Evaluate all candidates.
  const evaluations = candidates.map(evaluateCandidate);

  // Find the best solvable candidate (highest score among solvable ones).
  const solvable = evaluations
    .filter((e) => e.solvable)
    .sort((a, b) => b.score - a.score);

  if (solvable.length > 0) {
    const best = solvable[0];
    const solNum = best.solution;
    const solStr = Number.isInteger(solNum)
      ? String(solNum)
      : solNum.toFixed(6).replace(/\.?0+$/, '');

    const result: PipelineAcceptedResult = {
      accepted: true,
      equation: best.normalization.normalized,
      solution: `x=${solStr}`,
      explanationSteps: best.explanationSteps,
      sourceCandidate: best.candidate,
      corrections: best.normalization.corrections,
      score: best.score,
    };
    return result;
  }

  // No solvable candidate — determine the most specific rejection reason.
  const rejection = buildRejection(evaluations);
  return { accepted: false, rejection };
}


/**
 * Build a structured rejection reason from a set of failed evaluations.
 */
function buildRejection(evaluations: CandidateEvaluation[]): RejectionReason {
  const allGrammarInvalid = evaluations.every((e) => !e.grammarValid);
  const allParseFailed = evaluations.every((e) => !e.parseable);
  const allSolveFailed = evaluations.every((e) => !e.solvable);

  const candidateErrors = evaluations.slice(0, 5).map((e) => {
    const errors: string[] = [];
    if (!e.grammarValid) errors.push(...e.grammarErrors);
    if (e.parseError) errors.push(`Parse error: ${e.parseError}`);
    if (e.solveError) errors.push(`Solve error: ${e.solveError}`);
    return {
      text: e.candidate.text,
      normalizedText: e.normalization.normalized,
      errors,
    };
  });

  if (allGrammarInvalid) {
    return {
      code: 'ALL_GRAMMAR_INVALID',
      message: 'All candidates failed grammar validation.',
      candidateErrors,
    };
  }

  if (allParseFailed) {
    return {
      code: 'ALL_PARSE_FAILED',
      message: 'All candidates failed parsing.',
      candidateErrors,
    };
  }

  if (allSolveFailed) {
    return {
      code: 'ALL_SOLVE_FAILED',
      message: 'All candidates failed solving.',
      candidateErrors,
    };
  }

  // Fallback (shouldn't happen if solvable filter above is correct).
  return {
    code: 'ALL_SOLVE_FAILED',
    message: 'No candidate could be solved.',
    candidateErrors,
  };
}


// ── Exported for testing ────────────────────────────────────────────

export { evaluateCandidate, computeScore };
