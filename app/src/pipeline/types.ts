/**
 * Type definitions for the OCR result pipeline.
 *
 * Sprint D: OCR Result Pipeline Adapter.
 *
 * These types define the contract between the model/OCR layer and the
 * deterministic equation-solving pipeline.
 */

import type { ExplanationStep } from '../explanation/explanationEngine';

// ── Pipeline input ──────────────────────────────────────────────────

/** A single OCR candidate produced by the recognition model. */
export interface OcrCandidate {
  /** Raw text output from the recogniser. */
  text: string;
  /** Model confidence score in [0, 1]. */
  confidence: number;
}

// ── Correction tracking ─────────────────────────────────────────────

/** Describes a single normalisation correction applied to the raw text. */
export interface OcrCorrection {
  /** Character position in the original string. */
  position: number;
  /** Original character(s). */
  original: string;
  /** Replacement character(s). */
  replacement: string;
  /** Human-readable rule that triggered the correction. */
  rule: string;
}

/** Result of normalising a raw OCR string. */
export interface NormalizationResult {
  /** Normalised equation string. */
  normalized: string;
  /** List of corrections applied (empty if none). */
  corrections: OcrCorrection[];
}

// ── Per-candidate evaluation ────────────────────────────────────────

/** Internal evaluation result for a single candidate. */
export interface CandidateEvaluation {
  /** Original candidate. */
  candidate: OcrCandidate;
  /** Normalised text and corrections applied. */
  normalization: NormalizationResult;
  /** Whether the normalised text passes grammar validation. */
  grammarValid: boolean;
  /** Grammar validation error messages (empty if valid). */
  grammarErrors: string[];
  /** Whether the parser succeeded on the normalised text. */
  parseable: boolean;
  /** Parser error message, if parsing failed. */
  parseError: string | null;
  /** Whether the solver produced a solution. */
  solvable: boolean;
  /** Solver error message, if solving failed. */
  solveError: string | null;
  /** Solution value (NaN if not solvable). */
  solution: number;
  /** Composite score used for ranking (higher is better). */
  score: number;
  /** Explanation steps (empty if not solvable). */
  explanationSteps: ExplanationStep[];
}

// ── Pipeline output ─────────────────────────────────────────────────

/** Rejection reason when no candidate is accepted. */
export interface RejectionReason {
  /** Short classification of the rejection cause. */
  code:
    | 'NO_CANDIDATES'
    | 'ALL_GRAMMAR_INVALID'
    | 'ALL_PARSE_FAILED'
    | 'ALL_SOLVE_FAILED';
  /** Human-readable explanation. */
  message: string;
  /** Per-candidate failure details (first few). */
  candidateErrors: Array<{
    text: string;
    normalizedText: string;
    errors: string[];
  }>;
}

/** Successful pipeline result. */
export interface PipelineAcceptedResult {
  accepted: true;
  /** The normalised equation string that was solved. */
  equation: string;
  /** Solution as a string, e.g. "x=2". */
  solution: string;
  /** Step-by-step explanation. */
  explanationSteps: ExplanationStep[];
  /** The original candidate that produced the accepted result. */
  sourceCandidate: OcrCandidate;
  /** Corrections applied to the raw OCR text. */
  corrections: OcrCorrection[];
  /** Composite score of the winning candidate. */
  score: number;
}

/** Rejected pipeline result. */
export interface PipelineRejectedResult {
  accepted: false;
  /** Structured rejection reason. */
  rejection: RejectionReason;
}

/** Union type for the pipeline output. */
export type PipelineResult = PipelineAcceptedResult | PipelineRejectedResult;
