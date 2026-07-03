/**
 * Recognize-and-solve pipeline integration helper.
 *
 * Sprint 5C: Mobile Static Image Inference Preparation.
 *
 * Connects the inference layer (EquationRecognitionSession) to the
 * deterministic equation-solving pipeline (processCandidates).
 *
 * Flow:
 *   recognizer.recognize(input)
 *     → StaticRecognizerOutput { rawText, candidates }
 *       → processCandidates(candidates)
 *         → PipelineResult (accepted or rejected)
 */

import type { PipelineResult } from '../pipeline/types';
import { processCandidates } from '../pipeline/candidatePipeline';
import type {
  EquationRecognitionSession,
  StaticRecognizerInput,
} from './staticImageRecognizer';

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/** Combined result from recognition + solving. */
export interface RecognizeAndSolveResult {
  /** Raw decoded text from the recognizer. */
  rawText: string;
  /** Pipeline result (accepted with solution, or rejected with reason). */
  pipeline: PipelineResult;
}

// ---------------------------------------------------------------------------
// Pipeline integration
// ---------------------------------------------------------------------------

/**
 * Run the full recognize → solve pipeline.
 *
 * 1. Calls `recognizer.recognize(input)` to get raw text + candidates.
 * 2. Feeds candidates into `processCandidates()` for solving.
 * 3. Returns combined result.
 *
 * @param recognizer - An EquationRecognitionSession (real or fake).
 * @param input - Static recognizer input (grayscale pixels + dimensions).
 * @returns Combined recognition and solving result.
 */
export async function recognizeAndSolve(
  recognizer: EquationRecognitionSession,
  input: StaticRecognizerInput,
): Promise<RecognizeAndSolveResult> {
  const recognition = await recognizer.recognize(input);
  const pipeline = processCandidates(recognition.candidates);

  return {
    rawText: recognition.rawText,
    pipeline,
  };
}
