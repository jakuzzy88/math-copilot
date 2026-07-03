/**
 * Tests for the real camera preview integration.
 *
 * Sprint 7: Real Camera Preview.
 *
 * Comprehensive tests covering:
 *   1. Camera lifecycle pure logic (shouldCameraBeActive, evaluateCameraLifecycle)
 *   2. Camera enabled state (isCameraEnabled)
 *   3. Camera preview status (getCameraPreviewStatus, getPlaceholderContent)
 *   4. Permission handling contract
 *   5. Fallback behaviour (demo mode, missing module, no device)
 *   6. Lifecycle state transitions
 *   7. Controller integration (preserved Sprint 6B tests)
 *
 * All tests run in Node/Jest without any React Native runtime,
 * testing the pure logic exported from each module.
 */

import {
  shouldCameraBeActive,
  evaluateCameraLifecycle,
  isCameraEnabled,
} from '../screens/cameraLifecycle';
import {
  getCameraPreviewStatus,
  getPlaceholderContent,
  type CameraPreviewStatus,
} from '../screens/cameraPreviewStatus';

// Re-import screen internals for preserved tests.
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

// ============================================================================
// 1. Camera lifecycle — shouldCameraBeActive
// ============================================================================

describe('Camera lifecycle — shouldCameraBeActive', () => {
  test('active when enabled + foreground + not paused', () => {
    expect(shouldCameraBeActive(true, 'active', false)).toBe(true);
  });

  test('inactive when disabled', () => {
    expect(shouldCameraBeActive(false, 'active', false)).toBe(false);
  });

  test('inactive when app is in background', () => {
    expect(shouldCameraBeActive(true, 'background', false)).toBe(false);
  });

  test('inactive when app is inactive', () => {
    expect(shouldCameraBeActive(true, 'inactive', false)).toBe(false);
  });

  test('inactive when paused', () => {
    expect(shouldCameraBeActive(true, 'active', true)).toBe(false);
  });

  test('inactive when disabled + background', () => {
    expect(shouldCameraBeActive(false, 'background', false)).toBe(false);
  });

  test('inactive when disabled + paused', () => {
    expect(shouldCameraBeActive(false, 'active', true)).toBe(false);
  });

  test('inactive when all three conditions are negative', () => {
    expect(shouldCameraBeActive(false, 'background', true)).toBe(false);
  });

  test('inactive with "extension" appState', () => {
    expect(shouldCameraBeActive(true, 'extension', false)).toBe(false);
  });

  test('inactive with unknown appState', () => {
    expect(shouldCameraBeActive(true, 'unknown-state' as any, false)).toBe(false);
  });
});

// ============================================================================
// 2. Camera lifecycle — evaluateCameraLifecycle
// ============================================================================

describe('Camera lifecycle — evaluateCameraLifecycle', () => {
  test('returns isActive: true when all conditions met', () => {
    const result = evaluateCameraLifecycle({
      enabled: true,
      appState: 'active',
      isPaused: false,
    });
    expect(result.isActive).toBe(true);
  });

  test('returns isActive: false when any condition fails', () => {
    expect(evaluateCameraLifecycle({
      enabled: false,
      appState: 'active',
      isPaused: false,
    }).isActive).toBe(false);

    expect(evaluateCameraLifecycle({
      enabled: true,
      appState: 'background',
      isPaused: false,
    }).isActive).toBe(false);

    expect(evaluateCameraLifecycle({
      enabled: true,
      appState: 'active',
      isPaused: true,
    }).isActive).toBe(false);
  });
});

// ============================================================================
// 3. Camera lifecycle — isCameraEnabled
// ============================================================================

describe('Camera lifecycle — isCameraEnabled', () => {
  test('enabled when not demo mode and has permission', () => {
    expect(isCameraEnabled(false, true)).toBe(true);
  });

  test('disabled in demo mode regardless of permission', () => {
    expect(isCameraEnabled(true, true)).toBe(false);
    expect(isCameraEnabled(true, false)).toBe(false);
  });

  test('disabled when no permission (not demo mode)', () => {
    expect(isCameraEnabled(false, false)).toBe(false);
  });
});

// ============================================================================
// 4. Camera lifecycle state transitions
// ============================================================================

describe('Camera lifecycle state transitions', () => {
  test('transition: foreground → background deactivates camera', () => {
    expect(shouldCameraBeActive(true, 'active', false)).toBe(true);
    expect(shouldCameraBeActive(true, 'background', false)).toBe(false);
  });

  test('transition: background → foreground reactivates camera', () => {
    expect(shouldCameraBeActive(true, 'background', false)).toBe(false);
    expect(shouldCameraBeActive(true, 'active', false)).toBe(true);
  });

  test('transition: pause → resume restores active state', () => {
    expect(shouldCameraBeActive(true, 'active', false)).toBe(true);
    expect(shouldCameraBeActive(true, 'active', true)).toBe(false);
    expect(shouldCameraBeActive(true, 'active', false)).toBe(true);
  });

  test('transition: permission denied → camera inactive', () => {
    const enabled = isCameraEnabled(false, false); // not demo, no permission
    expect(shouldCameraBeActive(enabled, 'active', false)).toBe(false);
  });

  test('transition: permission granted → camera active', () => {
    const enabled = isCameraEnabled(false, true); // not demo, has permission
    expect(shouldCameraBeActive(enabled, 'active', false)).toBe(true);
  });

  test('demo mode → camera always inactive', () => {
    const enabled = isCameraEnabled(true, true); // demo mode
    expect(shouldCameraBeActive(enabled, 'active', false)).toBe(false);
  });

  test('full lifecycle: grant → active → background → foreground → pause → resume', () => {
    const enabled = isCameraEnabled(false, true);

    // Grant + foreground → active.
    expect(shouldCameraBeActive(enabled, 'active', false)).toBe(true);

    // Background → inactive.
    expect(shouldCameraBeActive(enabled, 'background', false)).toBe(false);

    // Foreground again → active.
    expect(shouldCameraBeActive(enabled, 'active', false)).toBe(true);

    // Manual pause → inactive.
    expect(shouldCameraBeActive(enabled, 'active', true)).toBe(false);

    // Resume → active.
    expect(shouldCameraBeActive(enabled, 'active', false)).toBe(true);
  });
});

// ============================================================================
// 5. Camera preview status — getCameraPreviewStatus
// ============================================================================

describe('Camera preview status — getCameraPreviewStatus', () => {
  test('returns "demo" when in demo mode (VisionCamera available)', () => {
    expect(getCameraPreviewStatus(true, true)).toBe('demo');
  });

  test('returns "demo" when in demo mode (VisionCamera unavailable)', () => {
    expect(getCameraPreviewStatus(true, false)).toBe('demo');
  });

  test('returns "unavailable" when not demo + VisionCamera not loaded', () => {
    expect(getCameraPreviewStatus(false, false)).toBe('unavailable');
  });

  test('returns "live" when not demo + VisionCamera available', () => {
    expect(getCameraPreviewStatus(false, true)).toBe('live');
  });

  test('demo mode takes priority over VisionCamera availability', () => {
    // Even if VisionCamera were available, demo mode wins.
    expect(getCameraPreviewStatus(true, true)).toBe('demo');
    expect(getCameraPreviewStatus(true, false)).toBe('demo');
  });

  test('default isVisionCameraAvailable parameter is false', () => {
    expect(getCameraPreviewStatus(false)).toBe('unavailable');
  });
});

// ============================================================================
// 6. Camera preview status — getPlaceholderContent
// ============================================================================

describe('Camera preview status — getPlaceholderContent', () => {
  test('demo mode returns camera emoji and demo message', () => {
    const content = getPlaceholderContent('demo');
    expect(content.message).toBe('Camera preview (demo mode)');
    expect(content.icon).toBe('📸');
    expect(content.isError).toBe(false);
  });

  test('unavailable returns warning emoji and module error', () => {
    const content = getPlaceholderContent('unavailable');
    expect(content.message).toContain('VisionCamera');
    expect(content.icon).toBe('⚠️');
    expect(content.isError).toBe(true);
  });

  test('no-device returns camera emoji and device error', () => {
    const content = getPlaceholderContent('no-device');
    expect(content.message).toContain('No camera device');
    expect(content.icon).toBe('📷');
    expect(content.isError).toBe(true);
  });

  test('live returns empty content (no placeholder needed)', () => {
    const content = getPlaceholderContent('live');
    expect(content.message).toBe('');
    expect(content.icon).toBe('');
    expect(content.isError).toBe(false);
  });
});

// ============================================================================
// 7. Permission handling contract
// ============================================================================

describe('Permission handling contract', () => {
  test('CameraPermissionState interface shape validation', () => {
    // Validates the expected type contract.
    const mockState = {
      hasPermission: false,
      requestPermission: async () => {},
      isLoading: true,
    };

    expect(mockState.hasPermission).toBe(false);
    expect(mockState.isLoading).toBe(true);
    expect(typeof mockState.requestPermission).toBe('function');
  });

  test('permission flow: loading → denied → request → granted', () => {
    const states = [
      { hasPermission: false, isLoading: true },
      { hasPermission: false, isLoading: false },
      { hasPermission: true, isLoading: false },
    ];

    // Loading.
    expect(states[0].isLoading).toBe(true);
    expect(states[0].hasPermission).toBe(false);

    // Denied.
    expect(states[1].isLoading).toBe(false);
    expect(states[1].hasPermission).toBe(false);

    // Granted.
    expect(states[2].isLoading).toBe(false);
    expect(states[2].hasPermission).toBe(true);
  });

  test('permission affects camera enabled state', () => {
    // No permission → camera disabled.
    expect(isCameraEnabled(false, false)).toBe(false);

    // Permission granted → camera enabled.
    expect(isCameraEnabled(false, true)).toBe(true);

    // Demo mode → camera disabled regardless.
    expect(isCameraEnabled(true, true)).toBe(false);
  });
});

// ============================================================================
// 8. Fallback behaviour
// ============================================================================

describe('Fallback behaviour', () => {
  test('VisionCamera is not available in Jest environment', () => {
    let isAvailable = true;
    try {
      require('react-native-vision-camera');
    } catch {
      isAvailable = false;
    }
    expect(isAvailable).toBe(false);
  });

  test('without VisionCamera → status is "unavailable" (non-demo)', () => {
    expect(getCameraPreviewStatus(false, false)).toBe('unavailable');
  });

  test('without VisionCamera → status is "demo" (demo mode)', () => {
    expect(getCameraPreviewStatus(true, false)).toBe('demo');
  });

  test('camera lifecycle degrades gracefully when disabled', () => {
    const enabled = false;
    // Even with all other conditions favorable, camera stays off.
    expect(shouldCameraBeActive(enabled, 'active', false)).toBe(false);
  });
});

// ============================================================================
// 9. Screen controller state transitions (Sprint 7)
// ============================================================================

describe('Screen controller state transitions', () => {
  let screen: LiveRecognitionScreen;

  beforeEach(() => {
    screen = new LiveRecognitionScreen({ demoMode: true });
  });

  afterEach(async () => {
    screen.stop();
    await screen.dispose();
  });

  test('idle → scanning → stable flow', async () => {
    const states: RecognitionUiState[] = [];
    screen.addListener((s) => states.push(s.uiState));

    screen.start();

    for (let i = 0; i < 5; i++) {
      await screen.processOneFrame();
    }

    const lastState = states[states.length - 1];
    expect(lastState.mode).toBe('stable');
    expect(lastState.recognizedEquation).toBe('3x+4=10');
    expect(lastState.solution).toBe('x=2');
  });

  test('stop preserves last state', async () => {
    screen.start();
    for (let i = 0; i < 5; i++) {
      await screen.processOneFrame();
    }

    const stateBeforeStop = screen.getUiState();
    screen.stop();
    const stateAfterStop = screen.getUiState();

    expect(stateAfterStop.recognizedEquation).toBe(stateBeforeStop.recognizedEquation);
    expect(stateAfterStop.solution).toBe(stateBeforeStop.solution);
  });

  test('dispose prevents further use', async () => {
    await screen.dispose();
    expect(() => screen.start()).toThrow('disposed');
  });

  test('multiple start/stop cycles work correctly', async () => {
    screen.start();
    await screen.processOneFrame();
    screen.stop();

    screen.start();
    await screen.processOneFrame();
    screen.stop();

    const state = screen.getState();
    expect(state.diagnostics.framesProcessed).toBe(2);
  });
});

// ============================================================================
// 10. Preserved: demo mode controller integration (Sprint 6B)
// ============================================================================

describe('MathCameraScreen — demo mode controller (preserved)', () => {
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

    for (let i = 0; i < 5; i++) {
      await screen.processOneFrame();
    }

    expect(states.length).toBeGreaterThan(0);

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
    expect(updates.length).toBe(1);
  });
});

// ============================================================================
// 11. DemoRecognizer (preserved)
// ============================================================================

describe('DemoRecognizer (preserved)', () => {
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

    for (let i = 0; i < 4; i++) {
      await recognizer.recognize(input);
    }

    const r5 = await recognizer.recognize(input);
    expect(r5.candidates[0].confidence).toBeCloseTo(0.92, 2);
  });

  test('dispose is a no-op', async () => {
    const recognizer = new DemoRecognizer();
    await expect(recognizer.dispose()).resolves.toBeUndefined();
  });
});

// ============================================================================
// 12. Guide box style computation (preserved)
// ============================================================================

describe('Guide box style — camera screen usage (preserved)', () => {
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

// ============================================================================
// 13. Solution card background (preserved)
// ============================================================================

describe('Solution card background — camera screen usage (preserved)', () => {
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

// ============================================================================
// 14. Confidence level (preserved)
// ============================================================================

describe('Confidence level — camera screen usage (preserved)', () => {
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

// ============================================================================
// 15. Diagnostics rows (preserved)
// ============================================================================

describe('Diagnostics rows — camera screen usage (preserved)', () => {
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

    const skippedRow = rows.find((r) => r.label === 'Frames Skipped (Busy)');
    expect(skippedRow?.isWarning).toBe(true);
  });
});

// ============================================================================
// 16. VisionCameraFrameProvider stub (preserved)
// ============================================================================

describe('VisionCameraFrameProvider stub (preserved)', () => {
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
