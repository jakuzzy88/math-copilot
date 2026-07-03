/**
 * LiveRecognitionScreen controller tests.
 *
 * Sprint 6A: UI Overlay Integration.
 *
 * Tests the screen controller in demo/static mode without
 * requiring real camera or ONNX runtime.
 */

import { LiveRecognitionScreen, DemoRecognizer } from '../ui/LiveRecognitionScreen';
import type { LiveRecognitionScreenState } from '../ui/LiveRecognitionScreen';
import {
  LiveRecognitionController,
  type LiveRecognitionOptions,
} from '../inference/liveRecognitionController';
import {
  StaticFrameProvider,
  createSyntheticRgbaFrame,
} from '../inference/cameraFrameProvider';
import type {
  EquationRecognitionSession,
  StaticRecognizerInput,
  StaticRecognizerOutput,
} from '../inference/staticImageRecognizer';
import { StabilityAggregator } from '../pipeline/stabilityAggregator';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeFrames(n = 8) {
  return Array.from({ length: n }, () => createSyntheticRgbaFrame(64, 16));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('LiveRecognitionScreen', () => {
  // ── Lifecycle ──────────────────────────────────────────────────────

  describe('lifecycle', () => {
    it('starts in idle mode', () => {
      const screen = new LiveRecognitionScreen({ demoMode: true });
      const state = screen.getState();

      expect(state.uiState.mode).toBe('idle');
      expect(state.isRunning).toBe(false);
    });

    it('transitions to scanning on start', () => {
      const screen = new LiveRecognitionScreen({ demoMode: true });
      screen.start();

      expect(screen.isRunning()).toBe(true);
      const state = screen.getState();
      expect(state.uiState.mode).toBe('scanning');

      screen.stop();
    });

    it('stops cleanly', () => {
      const screen = new LiveRecognitionScreen({ demoMode: true });
      screen.start();
      screen.stop();

      expect(screen.isRunning()).toBe(false);
    });

    it('disposes without error', async () => {
      const screen = new LiveRecognitionScreen({ demoMode: true });
      screen.start();
      await screen.dispose();
      expect(screen.isRunning()).toBe(false);
    });

    it('throws on start after dispose', async () => {
      const screen = new LiveRecognitionScreen({ demoMode: true });
      await screen.dispose();
      expect(() => screen.start()).toThrow('disposed');
    });
  });

  // ── Demo mode stable result ────────────────────────────────────────

  describe('demo mode — stable result flow', () => {
    it('emits stable result after processing enough frames', async () => {
      const recognizer = new DemoRecognizer();
      const provider = new StaticFrameProvider(makeFrames());

      const screen = new LiveRecognitionScreen({
        frameProvider: provider,
        recognizer,
        intervalMs: 1000,
      });

      // Process frames manually (bypass interval loop).
      for (let i = 0; i < 4; i++) {
        await screen.processOneFrame();
      }

      const state = screen.getState();

      // After 4 frames of "3x+4=10", should reach stable.
      expect(state.uiState.recognizedEquation).toBe('3x+4=10');
      expect(state.uiState.solution).toBe('x=2');
      expect(state.uiState.mode).toBe('stable');
      expect(state.uiState.confidence).toBeGreaterThan(0.5);

      screen.stop();
    });

    it('updates diagnostics during processing', async () => {
      const screen = new LiveRecognitionScreen({ demoMode: true });

      await screen.processOneFrame();
      const state = screen.getState();

      expect(state.diagnostics.framesSeen).toBe(1);
      expect(state.diagnostics.framesProcessed).toBe(1);

      screen.stop();
    });
  });

  // ── Listener management ────────────────────────────────────────────

  describe('listeners', () => {
    it('notifies listeners on state change', async () => {
      const screen = new LiveRecognitionScreen({ demoMode: true });
      const states: LiveRecognitionScreenState[] = [];

      screen.addListener((s) => states.push(s));
      await screen.processOneFrame();

      expect(states.length).toBeGreaterThanOrEqual(1);
      expect(states[0].diagnostics.framesProcessed).toBe(1);

      screen.stop();
    });

    it('unsubscribe removes listener', async () => {
      const screen = new LiveRecognitionScreen({ demoMode: true });
      const states: LiveRecognitionScreenState[] = [];

      const unsub = screen.addListener((s) => states.push(s));
      unsub();

      await screen.processOneFrame();

      // Listener was removed, so states should be empty.
      expect(states.length).toBe(0);

      screen.stop();
    });
  });

  // ── Diagnostics toggle ────────────────────────────────────────────

  describe('diagnostics toggle', () => {
    it('toggles diagnostics expanded state', () => {
      const screen = new LiveRecognitionScreen({ demoMode: true });

      expect(screen.getState().diagnosticsExpanded).toBe(false);
      screen.toggleDiagnostics();
      expect(screen.getState().diagnosticsExpanded).toBe(true);
      screen.toggleDiagnostics();
      expect(screen.getState().diagnosticsExpanded).toBe(false);
    });
  });

  // ── Error handling ────────────────────────────────────────────────

  describe('error handling', () => {
    it('transitions to error mode on recognizer error', async () => {
      const failingRecognizer: EquationRecognitionSession = {
        recognize: jest.fn().mockRejectedValue(new Error('ONNX crash')),
        dispose: jest.fn().mockResolvedValue(undefined),
      };
      const provider = new StaticFrameProvider(makeFrames());

      const screen = new LiveRecognitionScreen({
        frameProvider: provider,
        recognizer: failingRecognizer,
        intervalMs: 1000,
      });

      await screen.processOneFrame();
      const state = screen.getState();

      expect(state.uiState.mode).toBe('error');
      expect(state.uiState.lastError).toBe('ONNX crash');

      screen.stop();
    });
  });

  // ── Overlay rendering ─────────────────────────────────────────────

  describe('renderOverlay', () => {
    it('returns composed overlay data', () => {
      const screen = new LiveRecognitionScreen({ demoMode: true });
      const overlay = screen.renderOverlay();

      expect(overlay.type).toBe('RecognitionOverlay');
      expect(overlay.guideBox.type).toBe('GuideBoxOverlay');
      expect(overlay.solutionCard.type).toBe('SolutionCard');
      expect(overlay.diagnosticsPanel.type).toBe('DiagnosticsPanel');
    });
  });

  // ── DemoRecognizer ─────────────────────────────────────────────────

  describe('DemoRecognizer', () => {
    it('returns predefined responses', async () => {
      const rec = new DemoRecognizer();
      const input = { grayscalePixels: new Uint8Array(128 * 512), width: 512, height: 128 };

      const result = await rec.recognize(input);
      expect(result.rawText).toBe('3x+4=10');
      expect(result.candidates[0].confidence).toBeGreaterThan(0.5);
    });

    it('cycles through responses', async () => {
      const rec = new DemoRecognizer();
      const input = { grayscalePixels: new Uint8Array(128 * 512), width: 512, height: 128 };

      // Consume all default responses + wrap around.
      for (let i = 0; i < 5; i++) {
        const result = await rec.recognize(input);
        expect(result.rawText).toBe('3x+4=10');
      }
    });

    it('disposes without error', async () => {
      const rec = new DemoRecognizer();
      await rec.dispose(); // no-op, should not throw
    });
  });

  // ── External controller injection ──────────────────────────────────

  describe('external controller', () => {
    it('accepts a pre-built controller', async () => {
      const provider = new StaticFrameProvider(makeFrames());
      const recognizer = new DemoRecognizer();
      const controller = new LiveRecognitionController({
        frameProvider: provider,
        recognizer,
        intervalMs: 1000,
      });

      const screen = new LiveRecognitionScreen({ controller });

      expect(screen.getController()).toBe(controller);

      await screen.dispose();
    });
  });
});
