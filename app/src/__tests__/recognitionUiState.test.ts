/**
 * Recognition UI state adapter tests.
 *
 * Sprint 6A: UI Overlay Integration.
 *
 * Tests the conversion of StableRecognitionResult + LiveRecognitionDiagnostics
 * into UI-friendly RecognitionUiState objects.
 */

import {
  createIdleRecognitionUiState,
  mapStableResultToUiState,
  mapErrorToUiState,
  formatConfidence,
  formatDiagnosticsSummary,
  type RecognitionUiState,
} from '../ui/recognitionUiState';
import type { StableRecognitionResult } from '../pipeline/stabilityAggregator';
import type { LiveRecognitionDiagnostics } from '../inference/liveRecognitionController';
import type { ExplanationStep } from '../explanation/explanationEngine';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createDiagnostics(
  overrides: Partial<LiveRecognitionDiagnostics> = {},
): LiveRecognitionDiagnostics {
  return {
    framesSeen: 0,
    framesProcessed: 0,
    framesSkippedBusy: 0,
    framesFailedPreprocessing: 0,
    framesFailedRecognition: 0,
    framesRejectedByPipeline: 0,
    stableResultsEmitted: 0,
    lastRawText: null,
    lastStableEquation: null,
    averagePreprocessMs: 0,
    averageRecognitionMs: 0,
    averageTotalMs: 0,
    ...overrides,
  };
}

function createStableResult(
  overrides: Partial<StableRecognitionResult> = {},
): StableRecognitionResult {
  return {
    stable: false,
    equation: null,
    solution: null,
    confidence: 0,
    reason: 'No frames processed yet.',
    history: {
      totalFrames: 0,
      acceptedFrames: 0,
      rejectedFrames: 0,
      distinctEquations: 0,
      consecutiveRejections: 0,
    },
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('recognitionUiState', () => {
  // ── Idle state ──────────────────────────────────────────────────────

  describe('createIdleRecognitionUiState', () => {
    it('returns idle mode with correct defaults', () => {
      const state = createIdleRecognitionUiState();

      expect(state.mode).toBe('idle');
      expect(state.recognizedEquation).toBeNull();
      expect(state.solution).toBeNull();
      expect(state.currentStepTitle).toBeNull();
      expect(state.currentStepText).toBeNull();
      expect(state.confidence).toBe(0);
      expect(state.statusMessage).toContain('Point camera');
      expect(state.diagnosticsSummary).toBe('');
      expect(state.lastError).toBeNull();
    });

    it('formats idle state correctly for display', () => {
      const state = createIdleRecognitionUiState();

      // All display fields should be safe for rendering.
      expect(typeof state.statusMessage).toBe('string');
      expect(state.statusMessage.length).toBeGreaterThan(0);
    });
  });

  // ── Stable result mapping ──────────────────────────────────────────

  describe('mapStableResultToUiState', () => {
    it('maps stable result with high confidence to stable mode', () => {
      const result = createStableResult({
        stable: true,
        equation: '3x+4=10',
        solution: 'x=2',
        confidence: 0.91,
        reason: 'Stable equation.',
      });
      const diag = createDiagnostics({
        framesProcessed: 10,
        lastRawText: '3x+4=10',
      });

      const state = mapStableResultToUiState(result, diag);

      expect(state.mode).toBe('stable');
      expect(state.recognizedEquation).toBe('3x+4=10');
      expect(state.solution).toBe('x=2');
      expect(state.confidence).toBe(0.91);
      expect(state.statusMessage).toContain('3x+4=10');
      expect(state.lastError).toBeNull();
    });

    it('maps stable result with low confidence to uncertain mode', () => {
      const result = createStableResult({
        stable: true,
        equation: 'x+3=5',
        solution: 'x=2',
        confidence: 0.55,
        reason: 'Low confidence stable.',
      });
      const diag = createDiagnostics({ framesProcessed: 5 });

      const state = mapStableResultToUiState(result, diag);

      expect(state.mode).toBe('uncertain');
      expect(state.recognizedEquation).toBe('x+3=5');
      expect(state.statusMessage).toContain('low confidence');
    });

    it('maps unstable result with processed frames to uncertain mode', () => {
      const result = createStableResult({
        stable: false,
        reason: 'Not enough agreement.',
      });
      const diag = createDiagnostics({
        framesProcessed: 3,
        lastRawText: '3x+4=10',
      });

      const state = mapStableResultToUiState(result, diag);

      expect(state.mode).toBe('uncertain');
      expect(state.recognizedEquation).toBeNull();
    });

    it('maps unstable result with no raw text to scanning mode', () => {
      const result = createStableResult({
        stable: false,
        reason: 'No frames received yet.',
      });
      const diag = createDiagnostics({
        framesProcessed: 2,
        lastRawText: null,
      });

      const state = mapStableResultToUiState(result, diag);

      expect(state.mode).toBe('scanning');
      expect(state.statusMessage).toContain('Scanning');
    });

    it('maps unstable result with no processed frames to scanning mode', () => {
      const result = createStableResult({
        stable: false,
        reason: 'No frames received.',
      });
      const diag = createDiagnostics({ framesProcessed: 0 });

      const state = mapStableResultToUiState(result, diag);

      expect(state.mode).toBe('scanning');
    });

    it('includes explanation step when provided', () => {
      const result = createStableResult({
        stable: true,
        equation: '3x+4=10',
        solution: 'x=2',
        confidence: 0.9,
      });
      const diag = createDiagnostics({ framesProcessed: 5 });
      const steps: ExplanationStep[] = [
        {
          stepNumber: 1,
          header: 'Remove a term',
          title: 'Subtract 4 from both sides',
          body: 'To isolate the variable term 3x, subtract 4.',
          equationState: '3x=6',
          isFinal: false,
        },
      ];

      const state = mapStableResultToUiState(result, diag, steps);

      expect(state.currentStepTitle).toBe('Subtract 4 from both sides');
      expect(state.currentStepText).toContain('isolate');
    });

    it('handles empty explanation steps gracefully', () => {
      const result = createStableResult({
        stable: true,
        equation: 'x=5',
        solution: 'x=5',
        confidence: 0.95,
      });
      const diag = createDiagnostics({ framesProcessed: 5 });

      const state = mapStableResultToUiState(result, diag, []);

      expect(state.currentStepTitle).toBeNull();
      expect(state.currentStepText).toBeNull();
    });

    it('includes diagnostics summary in the state', () => {
      const result = createStableResult({ stable: false });
      const diag = createDiagnostics({
        framesProcessed: 42,
        framesSkippedBusy: 3,
        framesRejectedByPipeline: 5,
        stableResultsEmitted: 12,
        averageTotalMs: 15.2,
      });

      const state = mapStableResultToUiState(result, diag);

      expect(state.diagnosticsSummary).toContain('42');
      expect(state.diagnosticsSummary).toContain('3');
      expect(state.diagnosticsSummary).toContain('5');
      expect(state.diagnosticsSummary).toContain('12');
    });
  });

  // ── Error state mapping ────────────────────────────────────────────

  describe('mapErrorToUiState', () => {
    it('maps Error object to error mode', () => {
      const error = new Error('ONNX session crashed');
      const diag = createDiagnostics({
        framesProcessed: 5,
        lastStableEquation: 'x+1=2',
      });

      const state = mapErrorToUiState(error, diag);

      expect(state.mode).toBe('error');
      expect(state.lastError).toBe('ONNX session crashed');
      expect(state.statusMessage).toContain('ONNX session crashed');
      expect(state.recognizedEquation).toBe('x+1=2');
      expect(state.confidence).toBe(0);
    });

    it('maps string error to error mode', () => {
      const diag = createDiagnostics();

      const state = mapErrorToUiState('Camera unavailable', diag);

      expect(state.mode).toBe('error');
      expect(state.lastError).toBe('Camera unavailable');
      expect(state.statusMessage).toContain('Camera unavailable');
    });

    it('includes readable message in status', () => {
      const error = new Error('Frame capture timeout after 5000ms');
      const diag = createDiagnostics();

      const state = mapErrorToUiState(error, diag);

      expect(state.statusMessage).toMatch(/Error:.*Frame capture timeout/);
    });
  });

  // ── Confidence formatting ─────────────────────────────────────────

  describe('formatConfidence', () => {
    it('formats 0.875 as "87.5%"', () => {
      expect(formatConfidence(0.875)).toBe('87.5%');
    });

    it('formats 1.0 as "100.0%"', () => {
      expect(formatConfidence(1.0)).toBe('100.0%');
    });

    it('formats 0 as "0.0%"', () => {
      expect(formatConfidence(0)).toBe('0.0%');
    });

    it('formats 0.5 as "50.0%"', () => {
      expect(formatConfidence(0.5)).toBe('50.0%');
    });

    it('returns "N/A" for negative values', () => {
      expect(formatConfidence(-0.1)).toBe('N/A');
    });

    it('returns "N/A" for values > 1', () => {
      expect(formatConfidence(1.5)).toBe('N/A');
    });
  });

  // ── Diagnostics formatting ────────────────────────────────────────

  describe('formatDiagnosticsSummary', () => {
    it('includes processed, skipped, rejected, stable, and avg', () => {
      const diag = createDiagnostics({
        framesProcessed: 42,
        framesSkippedBusy: 3,
        framesRejectedByPipeline: 5,
        stableResultsEmitted: 12,
        averageTotalMs: 15.234,
      });

      const summary = formatDiagnosticsSummary(diag);

      expect(summary).toContain('Processed: 42');
      expect(summary).toContain('Skipped: 3');
      expect(summary).toContain('Rejected: 5');
      expect(summary).toContain('Stable: 12');
      expect(summary).toContain('Avg: 15.2ms');
    });

    it('includes failure counters when non-zero', () => {
      const diag = createDiagnostics({
        framesFailedPreprocessing: 2,
        framesFailedRecognition: 1,
      });

      const summary = formatDiagnosticsSummary(diag);

      expect(summary).toContain('PreprocessFail: 2');
      expect(summary).toContain('RecognitionFail: 1');
    });

    it('omits failure counters when zero', () => {
      const diag = createDiagnostics();

      const summary = formatDiagnosticsSummary(diag);

      expect(summary).not.toContain('PreprocessFail');
      expect(summary).not.toContain('RecognitionFail');
    });

    it('handles all-zero diagnostics', () => {
      const diag = createDiagnostics();
      const summary = formatDiagnosticsSummary(diag);

      expect(summary).toContain('Processed: 0');
      expect(summary).toContain('Avg: 0.0ms');
    });
  });
});
