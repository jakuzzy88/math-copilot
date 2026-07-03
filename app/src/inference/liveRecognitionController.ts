/**
 * Live recognition controller — frame loop with stability aggregation.
 *
 * Sprint 5F: Live OCR Loop with StabilityAggregator Integration.
 *
 * Wires the frame preprocessing pipeline into a live-style recognition loop:
 *
 *   CameraFrameProvider
 *     → processFrame()
 *       → recognizeAndSolve()
 *         → StabilityAggregator
 *           → stable UI-ready result
 *
 * Key design decisions:
 *   - **Busy-frame protection**: if a frame is still being processed when
 *     the next interval fires, the new frame is skipped (not queued).
 *     This prevents backlog accumulation on slow devices.
 *   - **Error isolation**: provider, preprocessing, and recognizer errors
 *     are caught individually and recorded in diagnostics without crashing
 *     the controller.
 *   - **Configurable interval**: default 250 ms (4 fps). Can be tuned per
 *     device capability.
 *   - **Callbacks**: optional `onStableResult`, `onFrameResult`, and
 *     `onError` hooks for UI integration.
 */

import type { CameraFrameProvider } from './cameraFrameProvider';
import type { EquationRecognitionSession } from './staticImageRecognizer';
import { processFrame } from './framePipeline';
import { recognizeAndSolve } from './recognizeAndSolve';
import type { RecognizeAndSolveResult } from './recognizeAndSolve';
import {
  StabilityAggregator,
  type StableRecognitionResult,
} from '../pipeline/stabilityAggregator';
import type { PipelineResult } from '../pipeline/types';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Live recognition diagnostics snapshot. */
export interface LiveRecognitionDiagnostics {
  /** Total frames seen (including skipped). */
  framesSeen: number;
  /** Frames that were fully processed. */
  framesProcessed: number;
  /** Frames skipped because the previous frame was still processing. */
  framesSkippedBusy: number;
  /** Frames that failed during preprocessing (grayscale/crop/resize). */
  framesFailedPreprocessing: number;
  /** Frames that failed during recognition (ONNX error). */
  framesFailedRecognition: number;
  /** Frames where the pipeline rejected the OCR result. */
  framesRejectedByPipeline: number;
  /** Number of stable results emitted to the callback. */
  stableResultsEmitted: number;
  /** Raw text from the last successful recognition. */
  lastRawText: string | null;
  /** Equation from the last stable result. */
  lastStableEquation: string | null;
  /** Running average preprocessing time in ms. */
  averagePreprocessMs: number;
  /** Running average recognition time in ms. */
  averageRecognitionMs: number;
  /** Running average total frame processing time in ms. */
  averageTotalMs: number;
}

/** Result from a single frame processing call. */
export interface FrameProcessingResult {
  /** Whether the frame was skipped because another frame is processing. */
  skipped: boolean;
  /** The stable recognition result (may be unchanged if skipped). */
  stableResult: StableRecognitionResult;
  /** The raw pipeline result, if the frame was processed. */
  pipelineResult?: PipelineResult;
  /** The raw text from recognition, if the frame was processed. */
  rawText?: string;
  /** Error that occurred during processing, if any. */
  error?: Error;
}

/** Controller configuration options. */
export interface LiveRecognitionOptions {
  /** Camera frame provider. */
  frameProvider: CameraFrameProvider;
  /** Equation recognizer (real or mock). */
  recognizer: EquationRecognitionSession;
  /** Optional stability aggregator (a default is created if omitted). */
  stabilityAggregator?: StabilityAggregator;
  /** Frame capture interval in milliseconds. Default: 250. */
  intervalMs?: number;
  /** Called when a new stable result is emitted. */
  onStableResult?: (result: StableRecognitionResult) => void;
  /** Called after every processed frame (not called for skipped frames). */
  onFrameResult?: (result: FrameProcessingResult) => void;
  /** Called when an error occurs during frame processing. */
  onError?: (error: Error, context: string) => void;
}

// ---------------------------------------------------------------------------
// Internal timing accumulator
// ---------------------------------------------------------------------------

interface TimingAccumulator {
  totalPreprocessMs: number;
  totalRecognitionMs: number;
  totalMs: number;
  count: number;
}

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------

/**
 * Live recognition controller.
 *
 * Manages the frame capture → preprocess → recognize → stabilize loop.
 *
 * Usage:
 * ```typescript
 * const controller = new LiveRecognitionController({
 *   frameProvider,
 *   recognizer,
 *   onStableResult: (result) => updateUI(result),
 * });
 *
 * controller.start();
 * // ... later ...
 * controller.stop();
 * await controller.dispose();
 * ```
 */
export class LiveRecognitionController {
  private readonly frameProvider: CameraFrameProvider;
  private readonly recognizer: EquationRecognitionSession;
  private readonly aggregator: StabilityAggregator;
  private readonly intervalMs: number;
  private readonly onStableResult?: (result: StableRecognitionResult) => void;
  private readonly onFrameResult?: (result: FrameProcessingResult) => void;
  private readonly onError?: (error: Error, context: string) => void;

  private running = false;
  private disposed = false;
  private busy = false;
  private intervalHandle: ReturnType<typeof setInterval> | null = null;

  private diagnostics: LiveRecognitionDiagnostics = createEmptyDiagnostics();
  private timing: TimingAccumulator = { totalPreprocessMs: 0, totalRecognitionMs: 0, totalMs: 0, count: 0 };
  private lastStableResult: StableRecognitionResult = createInitialStableResult();

  constructor(options: LiveRecognitionOptions) {
    this.frameProvider = options.frameProvider;
    this.recognizer = options.recognizer;
    this.aggregator = options.stabilityAggregator ?? new StabilityAggregator();
    this.intervalMs = options.intervalMs ?? 250;
    this.onStableResult = options.onStableResult;
    this.onFrameResult = options.onFrameResult;
    this.onError = options.onError;
  }

  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------

  /**
   * Start the live recognition loop.
   *
   * Frames are captured at the configured interval. If the controller
   * is already running, this is a no-op.
   */
  start(): void {
    if (this.disposed) {
      throw new Error('LiveRecognitionController has been disposed.');
    }
    if (this.running) {
      return;
    }

    this.running = true;
    this.intervalHandle = setInterval(() => {
      // Fire-and-forget — errors are caught inside processOneFrame.
      void this.processOneFrame();
    }, this.intervalMs);
  }

  /**
   * Stop the live recognition loop.
   *
   * Does not dispose resources — the controller can be restarted.
   */
  stop(): void {
    this.running = false;
    if (this.intervalHandle !== null) {
      clearInterval(this.intervalHandle);
      this.intervalHandle = null;
    }
  }

  /** Whether the loop is currently running. */
  isRunning(): boolean {
    return this.running;
  }

  /**
   * Process a single frame through the full pipeline.
   *
   * Can be called manually outside the interval loop (e.g. for testing).
   * If another frame is already being processed, the call returns
   * immediately with a skipped result.
   */
  async processOneFrame(): Promise<FrameProcessingResult> {
    this.diagnostics.framesSeen++;

    // Busy-frame protection.
    if (this.busy) {
      this.diagnostics.framesSkippedBusy++;
      return {
        skipped: true,
        stableResult: this.lastStableResult,
      };
    }

    this.busy = true;
    const frameStart = performance.now();

    try {
      return await this.processFrameInternal(frameStart);
    } finally {
      this.busy = false;
    }
  }

  /** Get a snapshot of the current diagnostics. */
  getDiagnostics(): LiveRecognitionDiagnostics {
    return { ...this.diagnostics };
  }

  /**
   * Reset the controller state.
   *
   * Clears the aggregator history and all diagnostics.
   * Does not stop the loop or dispose resources.
   */
  reset(): void {
    this.aggregator.reset();
    this.diagnostics = createEmptyDiagnostics();
    this.timing = { totalPreprocessMs: 0, totalRecognitionMs: 0, totalMs: 0, count: 0 };
    this.lastStableResult = createInitialStableResult();
  }

  /**
   * Dispose the controller and release resources.
   *
   * Stops the loop, disposes the frame provider and recognizer.
   * The controller cannot be used after disposal.
   */
  async dispose(): Promise<void> {
    if (this.disposed) {
      return;
    }
    this.stop();
    this.disposed = true;

    await this.frameProvider.dispose();
    await this.recognizer.dispose();
  }

  // -----------------------------------------------------------------------
  // Internal pipeline
  // -----------------------------------------------------------------------

  private async processFrameInternal(
    frameStart: number,
  ): Promise<FrameProcessingResult> {
    // 1. Capture frame.
    let frame;
    try {
      frame = await this.frameProvider.captureFrame();
    } catch (err) {
      const error = toError(err, 'Frame capture failed');
      this.diagnostics.framesFailedPreprocessing++;
      this.onError?.(error, 'captureFrame');
      return {
        skipped: false,
        stableResult: this.lastStableResult,
        error,
      };
    }

    if (frame === null) {
      const error = new Error('Frame provider returned null (camera unavailable).');
      this.diagnostics.framesFailedPreprocessing++;
      this.onError?.(error, 'captureFrame');
      return {
        skipped: false,
        stableResult: this.lastStableResult,
        error,
      };
    }

    // 2. Preprocess frame → StaticRecognizerInput.
    let preprocessed;
    const preprocessStart = performance.now();
    try {
      preprocessed = processFrame(frame);
    } catch (err) {
      const error = toError(err, 'Frame preprocessing failed');
      this.diagnostics.framesFailedPreprocessing++;
      this.onError?.(error, 'processFrame');
      return {
        skipped: false,
        stableResult: this.lastStableResult,
        error,
      };
    }
    const preprocessMs = performance.now() - preprocessStart;

    // 3. Recognize + solve.
    let recResult: RecognizeAndSolveResult;
    const recognizeStart = performance.now();
    try {
      recResult = await recognizeAndSolve(this.recognizer, preprocessed.input);
    } catch (err) {
      const error = toError(err, 'Recognition failed');
      this.diagnostics.framesFailedRecognition++;
      this.onError?.(error, 'recognizeAndSolve');
      return {
        skipped: false,
        stableResult: this.lastStableResult,
        error,
      };
    }
    const recognizeMs = performance.now() - recognizeStart;

    // 4. Update diagnostics.
    this.diagnostics.framesProcessed++;
    this.diagnostics.lastRawText = recResult.rawText;

    const totalMs = performance.now() - frameStart;
    this.updateTiming(preprocessMs, recognizeMs, totalMs);

    // Track pipeline rejections.
    if (!recResult.pipeline.accepted) {
      this.diagnostics.framesRejectedByPipeline++;
    }

    // 5. Feed into stability aggregator.
    const stableResult = this.aggregator.addFrame(recResult.pipeline);
    this.lastStableResult = stableResult;

    // 6. Track stable result emissions.
    if (stableResult.stable) {
      this.diagnostics.lastStableEquation = stableResult.equation;

      // Only call onStableResult if a new stable result has been emitted.
      this.diagnostics.stableResultsEmitted++;
      this.onStableResult?.(stableResult);
    }

    // 7. Notify frame callback.
    const frameResult: FrameProcessingResult = {
      skipped: false,
      stableResult,
      pipelineResult: recResult.pipeline,
      rawText: recResult.rawText,
    };
    this.onFrameResult?.(frameResult);

    return frameResult;
  }

  // -----------------------------------------------------------------------
  // Timing helpers
  // -----------------------------------------------------------------------

  private updateTiming(
    preprocessMs: number,
    recognitionMs: number,
    totalMs: number,
  ): void {
    this.timing.totalPreprocessMs += preprocessMs;
    this.timing.totalRecognitionMs += recognitionMs;
    this.timing.totalMs += totalMs;
    this.timing.count++;

    this.diagnostics.averagePreprocessMs =
      this.timing.totalPreprocessMs / this.timing.count;
    this.diagnostics.averageRecognitionMs =
      this.timing.totalRecognitionMs / this.timing.count;
    this.diagnostics.averageTotalMs =
      this.timing.totalMs / this.timing.count;
  }
}

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------

function createEmptyDiagnostics(): LiveRecognitionDiagnostics {
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
  };
}

function createInitialStableResult(): StableRecognitionResult {
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
  };
}

function toError(err: unknown, context: string): Error {
  if (err instanceof Error) {
    return err;
  }
  return new Error(`${context}: ${String(err)}`);
}
