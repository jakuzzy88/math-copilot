/**
 * Recognition UI state adapter.
 *
 * Sprint 6A: UI Overlay Integration.
 *
 * Converts the internal StableRecognitionResult + LiveRecognitionDiagnostics
 * into a UI-friendly state object that overlay components can consume directly.
 *
 * Design goals:
 *   - Decouple UI rendering from internal pipeline types.
 *   - Provide a single flat state object with all display-ready fields.
 *   - Include human-readable formatting for confidence, diagnostics, and errors.
 *   - Support five visual modes: idle, scanning, stable, uncertain, error.
 */

import type { StableRecognitionResult } from '../pipeline/stabilityAggregator';
import type { LiveRecognitionDiagnostics } from '../inference/liveRecognitionController';
import type { ExplanationStep } from '../explanation/explanationEngine';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Visual mode of the recognition overlay. */
export type RecognitionMode = 'idle' | 'scanning' | 'stable' | 'uncertain' | 'error';

/** UI-friendly recognition state. */
export interface RecognitionUiState {
  /** Current visual mode for the overlay. */
  mode: RecognitionMode;
  /** The recognized equation string, or null if not stable. */
  recognizedEquation: string | null;
  /** The solution string (e.g. "x=2"), or null. */
  solution: string | null;
  /** Title of the current/first explanation step, or null. */
  currentStepTitle: string | null;
  /** Body text of the current/first explanation step, or null. */
  currentStepText: string | null;
  /** Confidence value in [0, 1], or 0 if unavailable. */
  confidence: number;
  /** Human-readable status message for the user. */
  statusMessage: string;
  /** Formatted diagnostics summary string. */
  diagnosticsSummary: string;
  /** Last error message, or null. */
  lastError: string | null;
}

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------

/**
 * Create an idle UI state — shown when the controller has not started.
 */
export function createIdleRecognitionUiState(): RecognitionUiState {
  return {
    mode: 'idle',
    recognizedEquation: null,
    solution: null,
    currentStepTitle: null,
    currentStepText: null,
    confidence: 0,
    statusMessage: 'Point camera at a handwritten equation to begin.',
    diagnosticsSummary: '',
    lastError: null,
  };
}

/**
 * Map a stable recognition result + diagnostics into a UI state.
 *
 * Determines the mode based on stability and confidence:
 *   - `stable`: result.stable === true AND confidence >= 0.65
 *   - `uncertain`: result.stable === true but confidence < 0.65,
 *                  OR result.stable === false but frames are being processed
 *   - `scanning`: frames are being processed but no stable result yet
 */
export function mapStableResultToUiState(
  result: StableRecognitionResult,
  diagnostics: LiveRecognitionDiagnostics,
  explanationSteps?: ExplanationStep[],
): RecognitionUiState {
  const hasProcessedFrames = diagnostics.framesProcessed > 0;

  // Determine mode.
  let mode: RecognitionMode;
  if (result.stable) {
    mode = result.confidence >= 0.65 ? 'stable' : 'uncertain';
  } else if (hasProcessedFrames) {
    mode = diagnostics.lastRawText !== null ? 'uncertain' : 'scanning';
  } else {
    mode = 'scanning';
  }

  // Extract first/current explanation step.
  const firstStep = explanationSteps && explanationSteps.length > 0
    ? explanationSteps[0]
    : null;

  // Build status message.
  let statusMessage: string;
  switch (mode) {
    case 'stable':
      statusMessage = `Equation recognized: ${result.equation}`;
      break;
    case 'uncertain':
      statusMessage = result.stable
        ? `Equation recognized with low confidence (${formatConfidence(result.confidence)}).`
        : `Scanning… ${result.reason}`;
      break;
    case 'scanning':
      statusMessage = 'Scanning for equation…';
      break;
    default:
      statusMessage = result.reason;
  }

  return {
    mode,
    recognizedEquation: result.equation,
    solution: result.solution,
    currentStepTitle: firstStep?.title ?? null,
    currentStepText: firstStep?.body ?? null,
    confidence: result.confidence,
    statusMessage,
    diagnosticsSummary: formatDiagnosticsSummary(diagnostics),
    lastError: null,
  };
}

/**
 * Map an error into an error-mode UI state.
 */
export function mapErrorToUiState(
  error: Error | string,
  diagnostics: LiveRecognitionDiagnostics,
): RecognitionUiState {
  const errorMessage = typeof error === 'string' ? error : error.message;

  return {
    mode: 'error',
    recognizedEquation: diagnostics.lastStableEquation,
    solution: null,
    currentStepTitle: null,
    currentStepText: null,
    confidence: 0,
    statusMessage: `Error: ${errorMessage}`,
    diagnosticsSummary: formatDiagnosticsSummary(diagnostics),
    lastError: errorMessage,
  };
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/**
 * Format a confidence value as a human-readable percentage string.
 *
 * @param value - Confidence in [0, 1].
 * @returns Formatted string, e.g. "87.5%".
 */
export function formatConfidence(value: number): string {
  if (value < 0 || value > 1) {
    return 'N/A';
  }
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * Format a LiveRecognitionDiagnostics object into a compact summary string.
 *
 * Example output:
 *   "Processed: 42 | Skipped: 3 | Rejected: 5 | Stable: 12 | Avg: 15.2ms"
 */
export function formatDiagnosticsSummary(
  diagnostics: LiveRecognitionDiagnostics,
): string {
  const parts: string[] = [
    `Processed: ${diagnostics.framesProcessed}`,
    `Skipped: ${diagnostics.framesSkippedBusy}`,
    `Rejected: ${diagnostics.framesRejectedByPipeline}`,
    `Stable: ${diagnostics.stableResultsEmitted}`,
    `Avg: ${diagnostics.averageTotalMs.toFixed(1)}ms`,
  ];

  if (diagnostics.framesFailedPreprocessing > 0) {
    parts.push(`PreprocessFail: ${diagnostics.framesFailedPreprocessing}`);
  }
  if (diagnostics.framesFailedRecognition > 0) {
    parts.push(`RecognitionFail: ${diagnostics.framesFailedRecognition}`);
  }

  return parts.join(' | ');
}
