/**
 * Recognition overlay composition component.
 *
 * Sprint 6A: UI Overlay Integration.
 *
 * Composes all overlay sub-components into a single overlay layer:
 *   - GuideBoxOverlay — equation positioning guide
 *   - SolutionCard — equation, solution, explanation step
 *   - DiagnosticsPanel — developer debug info
 *
 * This component acts as the layout coordinator. It receives the
 * full RecognitionUiState and distributes the relevant props to
 * each sub-component.
 */

import type { RecognitionUiState } from '../recognitionUiState';
import type { LiveRecognitionDiagnostics } from '../../inference/liveRecognitionController';
import {
  renderGuideBoxOverlay,
  type GuideBoxOverlayProps,
} from './GuideBoxOverlay';
import {
  renderSolutionCard,
  type SolutionCardProps,
} from './SolutionCard';
import {
  renderDiagnosticsPanel,
  type DiagnosticsPanelProps,
} from './DiagnosticsPanel';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

/** Props for the RecognitionOverlay composition. */
export interface RecognitionOverlayProps {
  /** The current UI state. */
  uiState: RecognitionUiState;
  /** Raw diagnostics for the debug panel. */
  diagnostics: LiveRecognitionDiagnostics;
  /** Whether the diagnostics panel is expanded. */
  diagnosticsExpanded: boolean;
  /** Callback to toggle the diagnostics panel. */
  onToggleDiagnostics?: () => void;
}

// ---------------------------------------------------------------------------
// Render data
// ---------------------------------------------------------------------------

/** Composed render data for the full overlay. */
export interface RecognitionOverlayRenderData {
  type: 'RecognitionOverlay';
  guideBox: ReturnType<typeof renderGuideBoxOverlay>;
  solutionCard: ReturnType<typeof renderSolutionCard>;
  diagnosticsPanel: ReturnType<typeof renderDiagnosticsPanel>;
}

/**
 * Render the full recognition overlay as composed render data.
 *
 * This assembles all sub-components into a single data structure
 * that can be consumed by a React Native renderer.
 */
export function renderRecognitionOverlay(
  props: RecognitionOverlayProps,
): RecognitionOverlayRenderData {
  const { uiState, diagnostics, diagnosticsExpanded, onToggleDiagnostics } = props;

  // Guide box — active when scanning or uncertain.
  const guideBoxProps: GuideBoxOverlayProps = {
    isActive: uiState.mode === 'scanning' || uiState.mode === 'uncertain',
  };

  // Solution card — shows equation, solution, step.
  const solutionCardProps: SolutionCardProps = {
    mode: uiState.mode,
    equation: uiState.recognizedEquation,
    solution: uiState.solution,
    confidence: uiState.confidence,
    stepTitle: uiState.currentStepTitle,
    stepText: uiState.currentStepText,
    statusMessage: uiState.statusMessage,
  };

  // Diagnostics panel.
  const diagnosticsPanelProps: DiagnosticsPanelProps = {
    isExpanded: diagnosticsExpanded,
    diagnostics,
    onToggle: onToggleDiagnostics,
  };

  return {
    type: 'RecognitionOverlay',
    guideBox: renderGuideBoxOverlay(guideBoxProps),
    solutionCard: renderSolutionCard(solutionCardProps),
    diagnosticsPanel: renderDiagnosticsPanel(diagnosticsPanelProps),
  };
}
