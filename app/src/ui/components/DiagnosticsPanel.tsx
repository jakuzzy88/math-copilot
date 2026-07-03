/**
 * Diagnostics panel component.
 *
 * Sprint 6A: UI Overlay Integration.
 *
 * Renders a collapsible/toggleable debug panel that displays live
 * recognition diagnostics. Intended for developer debugging only —
 * not shown to end users in production.
 *
 * Displays:
 *   - Frames processed / seen / skipped
 *   - Pipeline rejection count
 *   - Average processing times
 *   - Last raw OCR text
 *   - Last stable equation
 *   - Error counts
 */

import type { LiveRecognitionDiagnostics } from '../../inference/liveRecognitionController';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

/** Props for the DiagnosticsPanel component. */
export interface DiagnosticsPanelProps {
  /** Whether the panel is currently expanded/visible. */
  isExpanded: boolean;
  /** The diagnostics snapshot to display. */
  diagnostics: LiveRecognitionDiagnostics;
  /** Optional callback when the toggle button is pressed. */
  onToggle?: () => void;
}

// ---------------------------------------------------------------------------
// Render data
// ---------------------------------------------------------------------------

/** A single diagnostics row for display. */
export interface DiagnosticsRow {
  label: string;
  value: string;
  isWarning?: boolean;
}

/** Computed render data for the diagnostics panel. */
export interface DiagnosticsPanelRenderData {
  type: 'DiagnosticsPanel';
  isExpanded: boolean;
  rows: DiagnosticsRow[];
  toggleLabel: string;
  accessibilityLabel: string;
}

/**
 * Build the rows for the diagnostics panel.
 */
export function buildDiagnosticsRows(
  diagnostics: LiveRecognitionDiagnostics,
): DiagnosticsRow[] {
  const rows: DiagnosticsRow[] = [
    {
      label: 'Frames Seen',
      value: String(diagnostics.framesSeen),
    },
    {
      label: 'Frames Processed',
      value: String(diagnostics.framesProcessed),
    },
    {
      label: 'Frames Skipped (Busy)',
      value: String(diagnostics.framesSkippedBusy),
      isWarning: diagnostics.framesSkippedBusy > 0,
    },
    {
      label: 'Pipeline Rejections',
      value: String(diagnostics.framesRejectedByPipeline),
      isWarning: diagnostics.framesRejectedByPipeline > 0,
    },
    {
      label: 'Preprocessing Failures',
      value: String(diagnostics.framesFailedPreprocessing),
      isWarning: diagnostics.framesFailedPreprocessing > 0,
    },
    {
      label: 'Recognition Failures',
      value: String(diagnostics.framesFailedRecognition),
      isWarning: diagnostics.framesFailedRecognition > 0,
    },
    {
      label: 'Stable Results Emitted',
      value: String(diagnostics.stableResultsEmitted),
    },
    {
      label: 'Last Raw Text',
      value: diagnostics.lastRawText ?? '—',
    },
    {
      label: 'Last Stable Equation',
      value: diagnostics.lastStableEquation ?? '—',
    },
    {
      label: 'Avg Preprocess',
      value: `${diagnostics.averagePreprocessMs.toFixed(1)}ms`,
    },
    {
      label: 'Avg Recognition',
      value: `${diagnostics.averageRecognitionMs.toFixed(1)}ms`,
    },
    {
      label: 'Avg Total',
      value: `${diagnostics.averageTotalMs.toFixed(1)}ms`,
    },
  ];

  return rows;
}

/**
 * Render the DiagnosticsPanel as a data description.
 */
export function renderDiagnosticsPanel(
  props: DiagnosticsPanelProps,
): DiagnosticsPanelRenderData {
  const { isExpanded, diagnostics } = props;

  return {
    type: 'DiagnosticsPanel',
    isExpanded,
    rows: isExpanded ? buildDiagnosticsRows(diagnostics) : [],
    toggleLabel: isExpanded ? 'Hide Diagnostics' : 'Show Diagnostics',
    accessibilityLabel: isExpanded
      ? 'Diagnostics panel — expanded'
      : 'Diagnostics panel — collapsed',
  };
}
