/**
 * Live recognition controller tests.
 *
 * Sprint 5F: Live OCR Loop with StabilityAggregator Integration.
 *
 * Uses fake frame providers and fake recognizers to test the full
 * live loop architecture without requiring ONNX Runtime.
 */

import {
  LiveRecognitionController,
  type LiveRecognitionOptions,
  type FrameProcessingResult,
  type LiveRecognitionDiagnostics,
} from '../inference/liveRecognitionController';
import type {
  CameraFrameProvider,
  CameraFrame,
} from '../inference/cameraFrameProvider';
import { createSyntheticRgbaFrame } from '../inference/cameraFrameProvider';
import type {
  EquationRecognitionSession,
  StaticRecognizerInput,
  StaticRecognizerOutput,
} from '../inference/staticImageRecognizer';
import type { StableRecognitionResult } from '../pipeline/stabilityAggregator';
import { StabilityAggregator } from '../pipeline/stabilityAggregator';

// ---------------------------------------------------------------------------
// Fake recognizer
// ---------------------------------------------------------------------------

/**
 * Fake recognizer that returns configurable text and confidence.
 * Supports sequencing different responses and simulating errors.
 */
class FakeRecognizer implements EquationRecognitionSession {
  disposed = false;
  recognizeCalls = 0;
  private responses: Array<StaticRecognizerOutput | Error>;
  private responseIndex = 0;

  constructor(responses: Array<StaticRecognizerOutput | Error> = []) {
    this.responses = responses;
  }

  /**
   * Create a recognizer that always returns the same equation text.
   */
  static returning(text: string, confidence = 0.9): FakeRecognizer {
    return new FakeRecognizer([
      {
        rawText: text,
        candidates: [{ text, confidence }],
      },
    ]);
  }

  /**
   * Create a recognizer that returns different responses in sequence.
   */
  static sequence(
    ...items: Array<{ text: string; confidence?: number } | Error>
  ): FakeRecognizer {
    const responses = items.map((item) => {
      if (item instanceof Error) return item;
      return {
        rawText: item.text,
        candidates: [{ text: item.text, confidence: item.confidence ?? 0.9 }],
      };
    });
    return new FakeRecognizer(responses);
  }

  async recognize(_input: StaticRecognizerInput): Promise<StaticRecognizerOutput> {
    this.recognizeCalls++;
    if (this.responses.length === 0) {
      return { rawText: '', candidates: [{ text: '', confidence: 0 }] };
    }
    const response = this.responses[this.responseIndex % this.responses.length];
    this.responseIndex++;
    if (response instanceof Error) {
      throw response;
    }
    return response;
  }

  async dispose(): Promise<void> {
    this.disposed = true;
  }
}

// ---------------------------------------------------------------------------
// Fake frame provider
// ---------------------------------------------------------------------------

class FakeFrameProvider implements CameraFrameProvider {
  disposed = false;
  private ready = true;
  captureCalls = 0;
  private frames: Array<CameraFrame | null | Error>;
  private frameIndex = 0;

  constructor(frames?: Array<CameraFrame | null | Error>) {
    this.frames = frames ?? [createSyntheticRgbaFrame(64, 16)];
  }

  async captureFrame(): Promise<CameraFrame | null> {
    this.captureCalls++;
    if (this.disposed) return null;
    const frame = this.frames[this.frameIndex % this.frames.length];
    this.frameIndex++;
    if (frame instanceof Error) throw frame;
    return frame;
  }

  isReady(): boolean {
    return this.ready && !this.disposed;
  }

  async dispose(): Promise<void> {
    this.disposed = true;
    this.ready = false;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeOptions(
  overrides: Partial<LiveRecognitionOptions> = {},
): LiveRecognitionOptions {
  return {
    frameProvider: new FakeFrameProvider(),
    recognizer: FakeRecognizer.returning('3x+4=10'),
    intervalMs: 1000, // Long interval — tests call processOneFrame manually.
    ...overrides,
  };
}

/** Process N frames sequentially. */
async function processN(
  controller: LiveRecognitionController,
  n: number,
): Promise<FrameProcessingResult[]> {
  const results: FrameProcessingResult[] = [];
  for (let i = 0; i < n; i++) {
    results.push(await controller.processOneFrame());
  }
  return results;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('LiveRecognitionController', () => {
  // ── Start / Stop ──────────────────────────────────────────────────

  describe('start and stop', () => {
    it('starts and reports isRunning as true', () => {
      const controller = new LiveRecognitionController(makeOptions());
      expect(controller.isRunning()).toBe(false);
      controller.start();
      expect(controller.isRunning()).toBe(true);
      controller.stop();
      expect(controller.isRunning()).toBe(false);
    });

    it('start is idempotent — calling twice does not error', () => {
      const controller = new LiveRecognitionController(makeOptions());
      controller.start();
      controller.start(); // should not throw
      expect(controller.isRunning()).toBe(true);
      controller.stop();
    });

    it('stop is idempotent — calling stop without start does not error', () => {
      const controller = new LiveRecognitionController(makeOptions());
      controller.stop(); // should not throw
      expect(controller.isRunning()).toBe(false);
    });

    it('throws if start is called after dispose', async () => {
      const controller = new LiveRecognitionController(makeOptions());
      await controller.dispose();
      expect(() => controller.start()).toThrow('disposed');
    });
  });

  // ── Single frame processing ───────────────────────────────────────

  describe('processOneFrame — accepted result', () => {
    it('processes a frame and returns a result', async () => {
      const controller = new LiveRecognitionController(
        makeOptions({
          recognizer: FakeRecognizer.returning('3x+4=10'),
        }),
      );

      const result = await controller.processOneFrame();

      expect(result.skipped).toBe(false);
      expect(result.rawText).toBe('3x+4=10');
      expect(result.pipelineResult).toBeDefined();
      expect(result.pipelineResult!.accepted).toBe(true);
      if (result.pipelineResult!.accepted) {
        expect(result.pipelineResult!.equation).toBe('3x+4=10');
        expect(result.pipelineResult!.solution).toBe('x=2');
      }

      controller.stop();
    });

    it('updates diagnostics after processing', async () => {
      const controller = new LiveRecognitionController(
        makeOptions({
          recognizer: FakeRecognizer.returning('x+5=9'),
        }),
      );

      await controller.processOneFrame();
      const diag = controller.getDiagnostics();

      expect(diag.framesSeen).toBe(1);
      expect(diag.framesProcessed).toBe(1);
      expect(diag.framesSkippedBusy).toBe(0);
      expect(diag.lastRawText).toBe('x+5=9');
      expect(diag.averagePreprocessMs).toBeGreaterThanOrEqual(0);
      expect(diag.averageRecognitionMs).toBeGreaterThanOrEqual(0);
      expect(diag.averageTotalMs).toBeGreaterThanOrEqual(0);

      controller.stop();
    });
  });

  // ── Multi-frame stability ─────────────────────────────────────────

  describe('stability aggregation', () => {
    it('emits stable result after 3 identical accepted frames', async () => {
      const stableResults: StableRecognitionResult[] = [];
      const controller = new LiveRecognitionController(
        makeOptions({
          recognizer: FakeRecognizer.returning('3x+4=10'),
          stabilityAggregator: new StabilityAggregator({
            windowSize: 8,
            minAgreement: 3,
            confidenceThreshold: 0.5,
          }),
          onStableResult: (r) => stableResults.push(r),
        }),
      );

      const results = await processN(controller, 3);

      // After 3 frames with the same equation, should be stable.
      const lastResult = results[results.length - 1];
      expect(lastResult.stableResult.stable).toBe(true);
      expect(lastResult.stableResult.equation).toBe('3x+4=10');
      expect(lastResult.stableResult.solution).toBe('x=2');

      // onStableResult should have been called.
      expect(stableResults.length).toBeGreaterThanOrEqual(1);

      const diag = controller.getDiagnostics();
      expect(diag.stableResultsEmitted).toBeGreaterThanOrEqual(1);
      expect(diag.lastStableEquation).toBe('3x+4=10');

      controller.stop();
    });

    it('unstable result when frames disagree', async () => {
      const recognizer = FakeRecognizer.sequence(
        { text: '3x+4=10' },
        { text: 'x+5=9' },
        { text: '2x=8' },
      );
      const controller = new LiveRecognitionController(
        makeOptions({
          recognizer,
          stabilityAggregator: new StabilityAggregator({
            minAgreement: 3,
            confidenceThreshold: 0.5,
          }),
        }),
      );

      const results = await processN(controller, 3);
      const lastResult = results[results.length - 1];

      // Different equations each time — should not be stable.
      expect(lastResult.stableResult.stable).toBe(false);

      controller.stop();
    });
  });

  // ── Rejected frames ───────────────────────────────────────────────

  describe('rejected frames', () => {
    it('tracks pipeline rejections in diagnostics', async () => {
      // Empty text → pipeline will reject (no candidates / grammar invalid).
      const controller = new LiveRecognitionController(
        makeOptions({
          recognizer: FakeRecognizer.returning(''),
        }),
      );

      await controller.processOneFrame();
      const diag = controller.getDiagnostics();

      expect(diag.framesProcessed).toBe(1);
      expect(diag.framesRejectedByPipeline).toBe(1);

      controller.stop();
    });

    it('handles invalid equation text (pipeline rejects)', async () => {
      const controller = new LiveRecognitionController(
        makeOptions({
          recognizer: FakeRecognizer.returning('+++invalid==='),
        }),
      );

      const result = await controller.processOneFrame();

      expect(result.pipelineResult).toBeDefined();
      expect(result.pipelineResult!.accepted).toBe(false);

      controller.stop();
    });
  });

  // ── Busy-frame skipping ───────────────────────────────────────────

  describe('busy-frame protection', () => {
    it('skips frame when already processing', async () => {
      // Create a recognizer that takes time to resolve.
      let resolveRecognize!: (value: StaticRecognizerOutput) => void;
      const slowRecognizer: EquationRecognitionSession = {
        recognize: jest.fn().mockImplementation(
          () =>
            new Promise<StaticRecognizerOutput>((resolve) => {
              resolveRecognize = resolve;
            }),
        ),
        dispose: jest.fn().mockResolvedValue(undefined),
      };

      const controller = new LiveRecognitionController(
        makeOptions({ recognizer: slowRecognizer }),
      );

      // Start first frame processing (will hang on recognize).
      const firstPromise = controller.processOneFrame();

      // Try to process a second frame while the first is busy.
      const secondResult = await controller.processOneFrame();

      expect(secondResult.skipped).toBe(true);

      // Resolve the first frame.
      resolveRecognize({
        rawText: '3x+4=10',
        candidates: [{ text: '3x+4=10', confidence: 0.9 }],
      });
      const firstResult = await firstPromise;
      expect(firstResult.skipped).toBe(false);

      const diag = controller.getDiagnostics();
      expect(diag.framesSeen).toBe(2);
      expect(diag.framesSkippedBusy).toBe(1);
      expect(diag.framesProcessed).toBe(1);

      controller.stop();
    });
  });

  // ── Error handling ────────────────────────────────────────────────

  describe('error handling', () => {
    it('handles frame provider error without crashing', async () => {
      const errors: Array<{ error: Error; context: string }> = [];
      const provider = new FakeFrameProvider([
        new Error('Camera disconnected'),
      ]);

      const controller = new LiveRecognitionController(
        makeOptions({
          frameProvider: provider,
          onError: (error, context) => errors.push({ error, context }),
        }),
      );

      const result = await controller.processOneFrame();

      expect(result.skipped).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.error!.message).toBe('Camera disconnected');
      expect(errors.length).toBe(1);
      expect(errors[0].context).toBe('captureFrame');

      const diag = controller.getDiagnostics();
      expect(diag.framesFailedPreprocessing).toBe(1);
      expect(diag.framesProcessed).toBe(0);

      controller.stop();
    });

    it('handles null frame from provider', async () => {
      const errors: Array<{ error: Error; context: string }> = [];
      const provider = new FakeFrameProvider([null]);

      const controller = new LiveRecognitionController(
        makeOptions({
          frameProvider: provider,
          onError: (error, context) => errors.push({ error, context }),
        }),
      );

      const result = await controller.processOneFrame();

      expect(result.error).toBeDefined();
      expect(errors.length).toBe(1);

      const diag = controller.getDiagnostics();
      expect(diag.framesFailedPreprocessing).toBe(1);

      controller.stop();
    });

    it('handles recognizer error without crashing', async () => {
      const errors: Array<{ error: Error; context: string }> = [];
      const recognizer = FakeRecognizer.sequence(
        new Error('ONNX session crashed'),
      );

      const controller = new LiveRecognitionController(
        makeOptions({
          recognizer,
          onError: (error, context) => errors.push({ error, context }),
        }),
      );

      const result = await controller.processOneFrame();

      expect(result.error).toBeDefined();
      expect(result.error!.message).toBe('ONNX session crashed');
      expect(errors.length).toBe(1);
      expect(errors[0].context).toBe('recognizeAndSolve');

      const diag = controller.getDiagnostics();
      expect(diag.framesFailedRecognition).toBe(1);
      expect(diag.framesProcessed).toBe(0);

      controller.stop();
    });

    it('continues processing after error', async () => {
      // First call: error. Second call: success.
      const recognizer = FakeRecognizer.sequence(
        new Error('Temporary ONNX error'),
        { text: '3x+4=10' },
      );

      const controller = new LiveRecognitionController(
        makeOptions({ recognizer }),
      );

      const result1 = await controller.processOneFrame();
      expect(result1.error).toBeDefined();

      const result2 = await controller.processOneFrame();
      expect(result2.error).toBeUndefined();
      expect(result2.rawText).toBe('3x+4=10');

      const diag = controller.getDiagnostics();
      expect(diag.framesFailedRecognition).toBe(1);
      expect(diag.framesProcessed).toBe(1);
      expect(diag.framesSeen).toBe(2);

      controller.stop();
    });
  });

  // ── Callbacks ─────────────────────────────────────────────────────

  describe('callbacks', () => {
    it('calls onFrameResult for each processed frame', async () => {
      const frameResults: FrameProcessingResult[] = [];
      const controller = new LiveRecognitionController(
        makeOptions({
          recognizer: FakeRecognizer.returning('x+5=9'),
          onFrameResult: (r) => frameResults.push(r),
        }),
      );

      await processN(controller, 3);

      expect(frameResults.length).toBe(3);
      expect(frameResults[0].rawText).toBe('x+5=9');

      controller.stop();
    });

    it('calls onStableResult when stable equation appears', async () => {
      const stableResults: StableRecognitionResult[] = [];
      const controller = new LiveRecognitionController(
        makeOptions({
          recognizer: FakeRecognizer.returning('2x=8'),
          stabilityAggregator: new StabilityAggregator({
            minAgreement: 2,
            confidenceThreshold: 0.5,
          }),
          onStableResult: (r) => stableResults.push(r),
        }),
      );

      await processN(controller, 3);

      // Should have at least one stable callback (may be called each time after stability).
      expect(stableResults.length).toBeGreaterThanOrEqual(1);
      expect(stableResults[0].stable).toBe(true);
      expect(stableResults[0].equation).toBe('2x=8');
      expect(stableResults[0].solution).toBe('x=4');

      controller.stop();
    });

    it('does not call onStableResult for unstable frames', async () => {
      const stableResults: StableRecognitionResult[] = [];
      const recognizer = FakeRecognizer.sequence(
        { text: '3x+4=10' },
        { text: 'x+5=9' },
      );

      const controller = new LiveRecognitionController(
        makeOptions({
          recognizer,
          stabilityAggregator: new StabilityAggregator({
            minAgreement: 3,
            confidenceThreshold: 0.5,
          }),
          onStableResult: (r) => stableResults.push(r),
        }),
      );

      await processN(controller, 2);

      // Only 2 frames with different equations, minAgreement=3 → no stable result.
      expect(stableResults.length).toBe(0);

      controller.stop();
    });

    it('calls onError when errors occur', async () => {
      const errors: Array<{ error: Error; context: string }> = [];
      const controller = new LiveRecognitionController(
        makeOptions({
          recognizer: FakeRecognizer.sequence(
            new Error('Recognition failed'),
          ),
          onError: (error, context) => errors.push({ error, context }),
        }),
      );

      await controller.processOneFrame();

      expect(errors.length).toBe(1);
      expect(errors[0].error.message).toBe('Recognition failed');

      controller.stop();
    });
  });

  // ── Diagnostics ───────────────────────────────────────────────────

  describe('diagnostics', () => {
    it('returns empty diagnostics initially', () => {
      const controller = new LiveRecognitionController(makeOptions());
      const diag = controller.getDiagnostics();

      expect(diag.framesSeen).toBe(0);
      expect(diag.framesProcessed).toBe(0);
      expect(diag.framesSkippedBusy).toBe(0);
      expect(diag.framesFailedPreprocessing).toBe(0);
      expect(diag.framesFailedRecognition).toBe(0);
      expect(diag.framesRejectedByPipeline).toBe(0);
      expect(diag.stableResultsEmitted).toBe(0);
      expect(diag.lastRawText).toBeNull();
      expect(diag.lastStableEquation).toBeNull();
      expect(diag.averagePreprocessMs).toBe(0);
      expect(diag.averageRecognitionMs).toBe(0);
      expect(diag.averageTotalMs).toBe(0);
    });

    it('accumulates timing averages over multiple frames', async () => {
      const controller = new LiveRecognitionController(
        makeOptions({
          recognizer: FakeRecognizer.returning('x+1=2'),
        }),
      );

      await processN(controller, 5);
      const diag = controller.getDiagnostics();

      expect(diag.framesProcessed).toBe(5);
      expect(diag.averagePreprocessMs).toBeGreaterThanOrEqual(0);
      expect(diag.averageRecognitionMs).toBeGreaterThanOrEqual(0);
      expect(diag.averageTotalMs).toBeGreaterThanOrEqual(0);

      controller.stop();
    });

    it('getDiagnostics returns a copy (not a reference)', async () => {
      const controller = new LiveRecognitionController(makeOptions());

      const diag1 = controller.getDiagnostics();
      await controller.processOneFrame();
      const diag2 = controller.getDiagnostics();

      // diag1 should not have been mutated.
      expect(diag1.framesSeen).toBe(0);
      expect(diag2.framesSeen).toBe(1);

      controller.stop();
    });
  });

  // ── Reset ─────────────────────────────────────────────────────────

  describe('reset', () => {
    it('clears diagnostics and aggregator state', async () => {
      const controller = new LiveRecognitionController(
        makeOptions({
          recognizer: FakeRecognizer.returning('3x+4=10'),
        }),
      );

      await processN(controller, 5);
      const diagBefore = controller.getDiagnostics();
      expect(diagBefore.framesProcessed).toBe(5);

      controller.reset();

      const diagAfter = controller.getDiagnostics();
      expect(diagAfter.framesSeen).toBe(0);
      expect(diagAfter.framesProcessed).toBe(0);
      expect(diagAfter.lastRawText).toBeNull();
      expect(diagAfter.lastStableEquation).toBeNull();
      expect(diagAfter.averageTotalMs).toBe(0);

      controller.stop();
    });

    it('allows processing after reset', async () => {
      const controller = new LiveRecognitionController(
        makeOptions({
          recognizer: FakeRecognizer.returning('x+1=2'),
        }),
      );

      await processN(controller, 3);
      controller.reset();

      const result = await controller.processOneFrame();
      expect(result.skipped).toBe(false);
      expect(result.rawText).toBe('x+1=2');

      const diag = controller.getDiagnostics();
      expect(diag.framesSeen).toBe(1);
      expect(diag.framesProcessed).toBe(1);

      controller.stop();
    });
  });

  // ── Dispose ───────────────────────────────────────────────────────

  describe('dispose', () => {
    it('disposes the recognizer', async () => {
      const recognizer = FakeRecognizer.returning('x+1=2');
      const controller = new LiveRecognitionController(
        makeOptions({ recognizer }),
      );

      await controller.dispose();

      expect(recognizer.disposed).toBe(true);
    });

    it('disposes the frame provider', async () => {
      const provider = new FakeFrameProvider();
      const controller = new LiveRecognitionController(
        makeOptions({ frameProvider: provider }),
      );

      await controller.dispose();

      expect(provider.disposed).toBe(true);
    });

    it('stops the loop on dispose', async () => {
      const controller = new LiveRecognitionController(makeOptions());
      controller.start();
      expect(controller.isRunning()).toBe(true);

      await controller.dispose();
      expect(controller.isRunning()).toBe(false);
    });

    it('is idempotent — calling dispose twice does not throw', async () => {
      const controller = new LiveRecognitionController(makeOptions());
      await controller.dispose();
      await controller.dispose(); // should not throw
    });
  });

  // ── Integration: mixed scenario ───────────────────────────────────

  describe('integration', () => {
    it('handles mixed success/failure/rejection sequence', async () => {
      const recognizer = FakeRecognizer.sequence(
        { text: '3x+4=10' },        // accepted
        new Error('Transient error'), // error
        { text: '3x+4=10' },         // accepted
        { text: '+++' },              // rejected by pipeline
        { text: '3x+4=10' },         // accepted
      );

      const errors: Error[] = [];
      const controller = new LiveRecognitionController(
        makeOptions({
          recognizer,
          onError: (e) => errors.push(e),
          stabilityAggregator: new StabilityAggregator({
            minAgreement: 3,
            confidenceThreshold: 0.5,
          }),
        }),
      );

      await processN(controller, 5);

      const diag = controller.getDiagnostics();
      expect(diag.framesSeen).toBe(5);
      expect(diag.framesProcessed).toBe(4); // 1 error
      expect(diag.framesFailedRecognition).toBe(1);
      expect(diag.framesRejectedByPipeline).toBe(1); // '+++'
      expect(errors.length).toBe(1);
      // 3 accepted frames of '3x+4=10' → should be stable.
      expect(diag.lastStableEquation).toBe('3x+4=10');

      controller.stop();
    });
  });
});
