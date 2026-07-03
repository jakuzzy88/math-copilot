/**
 * Solution card component.
 *
 * Sprint 6A: UI Overlay Integration.
 *
 * Displays the recognized equation, solution, confidence level,
 * and the first explanation step in a card overlay on top of the
 * camera preview.
 *
 * Written as a pure component description for testability.
 * Can be adapted to real React Native <View> / <Text> primitives.
 */

import type { RecognitionMode } from '../recognitionUiState';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

/** Props for the SolutionCard component. */
export interface SolutionCardProps {
  /** Current recognition mode. */
  mode: RecognitionMode;
  /** The recognized equation string. */
  equation: string | null;
  /** The solution string (e.g. "x=2"). */
  solution: string | null;
  /** Confidence in [0, 1]. */
  confidence: number;
  /** Title of the current explanation step. */
  stepTitle: string | null;
  /** Body text of the current explanation step. */
  stepText: string | null;
  /** Status message to display. */
  statusMessage: string;
}

// ---------------------------------------------------------------------------
// Render data
// ---------------------------------------------------------------------------

/** Computed display data for the solution card. */
export interface SolutionCardRenderData {
  type: 'SolutionCard';
  visible: boolean;
  equationText: string;
  solutionText: string;
  confidenceText: string;
  confidenceLevel: 'high' | 'medium' | 'low' | 'none';
  stepTitle: string;
  stepText: string;
  statusText: string;
  backgroundColor: string;
  accessibilityLabel: string;
}

/**
 * Compute the confidence level for visual styling.
 */
export function getConfidenceLevel(
  confidence: number,
): 'high' | 'medium' | 'low' | 'none' {
  if (confidence >= 0.85) return 'high';
  if (confidence >= 0.65) return 'medium';
  if (confidence > 0) return 'low';
  return 'none';
}

/**
 * Compute background color based on mode and confidence.
 */
export function getSolutionCardBackground(
  mode: RecognitionMode,
  confidence: number,
): string {
  if (mode === 'error') return 'rgba(244, 67, 54, 0.9)'; // Red
  if (mode === 'stable') {
    return confidence >= 0.85
      ? 'rgba(0, 200, 83, 0.92)' // Green — high confidence
      : 'rgba(33, 150, 243, 0.92)'; // Blue — medium confidence
  }
  if (mode === 'uncertain') return 'rgba(255, 152, 0, 0.9)'; // Orange
  return 'rgba(30, 30, 30, 0.85)'; // Dark — scanning/idle
}

/**
 * Render the SolutionCard as a data description.
 *
 * Returns a serializable render description that can be consumed
 * by a React Native renderer or validated in tests.
 */
export function renderSolutionCard(props: SolutionCardProps): SolutionCardRenderData {
  const {
    mode,
    equation,
    solution,
    confidence,
    stepTitle,
    stepText,
    statusMessage,
  } = props;

  const visible = mode !== 'idle';
  const confidenceLevel = getConfidenceLevel(confidence);
  const confidencePercent = confidence > 0
    ? `${(confidence * 100).toFixed(1)}%`
    : '';

  return {
    type: 'SolutionCard',
    visible,
    equationText: equation ?? '',
    solutionText: solution ?? '',
    confidenceText: confidencePercent,
    confidenceLevel,
    stepTitle: stepTitle ?? '',
    stepText: stepText ?? '',
    statusText: statusMessage,
    backgroundColor: getSolutionCardBackground(mode, confidence),
    accessibilityLabel: equation
      ? `Equation: ${equation}. Solution: ${solution ?? 'pending'}`
      : `Status: ${statusMessage}`,
  };
}
