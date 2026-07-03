/**
 * Tests for MathCameraScreen adapter logic and demo mode integration.
 *
 * Sprint 6B: React Native Camera Shell.
 *
 * These tests validate the screen controller integration, demo mode
 * behavior, and UI state mapping without requiring a real React Native
 * runtime or camera hardware.
 *
 * Tests use the existing LiveRecognitionScreen controller
 * and DemoRecognizer directly (no React rendering).
 */

import {
  LiveRecognitionScreen,
  DemoRecognizer,
  type LiveRecognitionScreenState,
} from '../ui/LiveRecognitionScreen';
import type { RecognitionUiState } from '../ui/recognitionUiState';
import { computeGuideBoxStyle } from '../ui/components/GuideBoxOverlay';
import {
  getSolutionCardBackground,
  getConfidenceLevel,
} from '../ui/components/SolutionCard';
import { buildDiagnosticsRows } from '../ui/components/DiagnosticsPanel';
import { VisionCameraFrameProvider } from '../screens/realModeStubs';

// ---------------------------------------------------------------------------
// Demo mode controller integration
// ---------------------------------------------------------------------------

describe('MathCameraScreen — demo mode controller', () => {
  let screen: LiveRecognitionScreen;

  beforeEach(() => {
    screen = new LiveRecognitionScreen({ demoMode: true });
  });

  afterEach(async () => {
    screen.stop();
    await screen.dispose();
  });

  test('initialises in idle mode', () => {
    const state = screen.getUiState();
    expect(state.mode).toBe('idle');
    expect(state.recognizedEquation).toBeNull();
    expect(state.solution).toBeNull();
  });

  test('transitions through scanning to stable after processing frames', async () => {
    const states: RecognitionUiState[] = [];
    screen.addListener((s) => states.push(s.uiState));

    // Process enough frames for stability (minAgreement = 3).
    for (let i = 0; i < 5; i++) {
      await screen.processOneFrame();
    }

    // Should have received state updates.
    expect(states.length).toBeGreaterThan(0);

    // The last state should be stable with the demo equation.
    const lastState = states[states.length - 1];
    expect(lastState.mode).toBe('stable');
    expect(lastState.recognizedEquation).toBe('3x+4=10');
    expect(lastState.solution).toBe('x=2');
  });

  test('emits diagnostics with frame counts', async () => {
    for (let i = 0; i < 4; i++) {
      await screen.processOneFrame();
    }

    const state = screen.getState();
    expect(state.diagnostics.framesProcessed).toBe(4);
    expect(state.diagnostics.framesSeen).toBe(4);
  });

  test('listener receives updates on each frame', async () => {
    const updates: LiveRecognitionScreenState[] = [];
    screen.addListener((s) => updates.push(s));

    await screen.processOneFrame();
    await screen.processOneFrame();

    expect(updates.length).toBe(2);
  });

  test('unsubscribe stops listener calls', async () => {
    const updates: LiveRecognitionScreenState[] = [];
    const unsub = screen.addListener((s) => updates.push(s));

    await screen.processOneFrame();
    expect(updates.length).toBe(1);

    unsub();
    await screen.processOneFrame();
    expect(updates.length).toBe(1); // No new update.
  });
});

// ---------------------------------------------------------------------------
// DemoRecognizer
// ---------------------------------------------------------------------------

describe('DemoRecognizer', () => {
  test('returns predefined responses in cycle', async () => {
    const recognizer = new DemoRecognizer();
    const r1 = await recognizer.recognize({
      grayscalePixels: new Uint8Array(128 * 512),
      width: 512,
      height: 128,
    });
    expect(r1.rawText).toBe('3x+4=10');
    expect(r1.candidates.length).toBe(1);
    expect(r1.candidates[0].confidence).toBeCloseTo(0.92, 2);

    const r2 = await recognizer.recognize({
      grayscalePixels: new Uint8Array(128 * 512),
      width: 512,
      height: 128,
    });
    expect(r2.candidates[0].confidence).toBeCloseTo(0.88, 2);
  });

  test('cycles through all responses', async () => {
    const recognizer = new DemoRecognizer();
    const input = {
      grayscalePixels: new Uint8Array(128 * 512),
      width: 512,
      height: 128,
    };

    // 4 default responses.
    for (let i = 0; i < 4; i++) {
      await recognizer.recognize(input);
    }

    // Should cycle back.
    const r5 = await recognizer.recognize(input);
    expect(r5.candidates[0].confidence).toBeCloseTo(0.92, 2);
  });

  test('dispose is a no-op', async () => {
    const recognizer = new DemoRecognizer();
    await expect(recognizer.dispose()).resolves.toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Guide box style computation (used by MathCameraScreen)
// ---------------------------------------------------------------------------

describe('Guide box style — MathCameraScreen usage', () => {
  test('active state produces green border', () => {
    const style = computeGuideBoxStyle({ isActive: true });
    expect(style.borderColor).toBe('#00E676');
    expect(style.borderWidth).toBe(2.5);
    expect(style.opacity).toBe(1);
    expect(style.aspectRatio).toBe(4);
  });

  test('inactive state produces white border with reduced opacity', () => {
    const style = computeGuideBoxStyle({ isActive: false });
    expect(style.borderColor).toBe('rgba(255,255,255,0.4)');
    expect(style.borderWidth).toBe(1.5);
    expect(style.opacity).toBe(0.6);
  });
});

// ---------------------------------------------------------------------------
// Solution card background (used by MathCameraScreen)
// ---------------------------------------------------------------------------

describe('Solution card background — MathCameraScreen usage', () => {
  test('stable high confidence is green', () => {
    const bg = getSolutionCardBackground('stable', 0.9);
    expect(bg).toContain('0, 200, 83');
  });

  test('stable medium confidence is blue', () => {
    const bg = getSolutionCardBackground('stable', 0.7);
    expect(bg).toContain('33, 150, 243');
  });

  test('error mode is red', () => {
    const bg = getSolutionCardBackground('error', 0);
    expect(bg).toContain('244, 67, 54');
  });

  test('scanning mode is dark', () => {
    const bg = getSolutionCardBackground('scanning', 0);
    expect(bg).toContain('30, 30, 30');
  });
});

// ---------------------------------------------------------------------------
// Confidence level (used by MathCameraScreen)
// ---------------------------------------------------------------------------

describe('Confidence level — MathCameraScreen usage', () => {
  test.each([
    [0.9, 'high'],
    [0.85, 'high'],
    [0.7, 'medium'],
    [0.65, 'medium'],
    [0.3, 'low'],
    [0, 'none'],
  ] as const)('confidence %f → %s', (confidence, expected) => {
    expect(getConfidenceLevel(confidence)).toBe(expected);
  });
});

// ---------------------------------------------------------------------------
// Diagnostics rows (used by MathCameraScreen)
// ---------------------------------------------------------------------------

describe('Diagnostics rows — MathCameraScreen usage', () => {
  test('builds complete row set from diagnostics', () => {
    const rows = buildDiagnosticsRows({
      framesSeen: 100,
      framesProcessed: 95,
      framesSkippedBusy: 5,
      framesFailedPreprocessing: 0,
      framesFailedRecognition: 0,
      framesRejectedByPipeline: 3,
      stableResultsEmitted: 10,
      lastRawText: '3x+4=10',
      lastStableEquation: '3x+4=10',
      averagePreprocessMs: 12.5,
      averageRecognitionMs: 25.0,
      averageTotalMs: 37.5,
    });

    expect(rows.length).toBe(12);
    expect(rows[0].label).toBe('Frames Seen');
    expect(rows[0].value).toBe('100');

    // Skipped busy should be warning.
    const skippedRow = rows.find((r) => r.label === 'Frames Skipped (Busy)');
    expect(skippedRow?.isWarning).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// VisionCameraFrameProvider stub
// ---------------------------------------------------------------------------

describe('VisionCameraFrameProvider stub', () => {
  test('captureFrame returns null before any frame is set', async () => {
    const provider = new VisionCameraFrameProvider();
    const frame = await provider.captureFrame();
    expect(frame).toBeNull();
    await provider.dispose();
  });

  test('updateFrame throws (not yet implemented)', () => {
    const provider = new VisionCameraFrameProvider();
    expect(() => provider.updateFrame({})).toThrow('not yet implemented');
  });

  test('captureFrame returns null after dispose', async () => {
    const provider = new VisionCameraFrameProvider();
    await provider.dispose();
    const frame = await provider.captureFrame();
    expect(frame).toBeNull();
  });
});
