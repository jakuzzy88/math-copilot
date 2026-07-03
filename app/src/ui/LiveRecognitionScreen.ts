/**
 * Live recognition screen / controller integration.
 *
 * Sprint 6A: UI Overlay Integration.
 *
 * Connects the LiveRecognitionController to the UI overlay layer:
 *   - Creates or accepts a LiveRecognitionController.
 *   - Starts/stops the recognition loop.
 *   - Subscribes to stable, frame, and error callbacks.
 *   - Maintains a RecognitionUiState for the overlay.
 *   - Supports demo/static mode for testing without real camera/ONNX.
 *
 * This module does not depend on React Native rendering —
 * it exports a pure TypeScript "screen controller" that manages
 * the UI state. A real React Native screen component would consume
 * this controller and render the overlay components.
 */

import {
  LiveRecognitionController,
  type LiveRecognitionOptions,
  type FrameProcessingResult,
  type LiveRecognitionDiagnostics,
} from '../inference/liveRecognitionController';
import type { StableRecognitionResult } from '../pipeline/stabilityAggregator';
import { StabilityAggregator } from '../pipeline/stabilityAggregator';
import type {
  CameraFrameProvider,
  CameraFrame,
} from '../inference/cameraFrameProvider';
import {
  StaticFrameProvider,
  createSyntheticRgbaFrame,
} from '../inference/cameraFrameProvider';
import type {
  EquationRecognitionSession,
  StaticRecognizerInput,
  StaticRecognizerOutput,
} from '../inference/staticImageRecognizer';
import type { PipelineAcceptedResult } from '../pipeline/types';
import type { ExplanationStep } from '../explanation/explanationEngine';
import {
  type RecognitionUiState,
  createIdleRecognitionUiState,
  mapStableResultToUiState,
  mapErrorToUiState,
} from './recognitionUiState';
import {
  renderRecognitionOverlay,
  type RecognitionOverlayRenderData,
} from './components/RecognitionOverlay';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Screen state snapshot. */
export interface LiveRecognitionScreenState {
  /** Current UI state for the overlay. */
  uiState: RecognitionUiState;
  /** Whether the controller is running. */
  isRunning: boolean;
  /** Whether the diagnostics panel is expanded. */
  diagnosticsExpanded: boolean;
  /** Raw diagnostics (for the panel). */
  diagnostics: LiveRecognitionDiagnostics;
  /** Last explanation steps from a stable accepted result. */
  lastExplanationSteps: ExplanationStep[];
}

/** Listener callback for state changes. */
export type StateChangeListener = (state: LiveRecognitionScreenState) => void;

/** Options for creating the screen controller. */
export interface LiveRecognitionScreenOptions {
  /** Pre-built controller (takes priority over frameProvider/recognizer). */
  controller?: LiveRecognitionController;
  /** Camera frame provider (used if no controller is provided). */
  frameProvider?: CameraFrameProvider;
  /** Equation recognizer (used if no controller is provided). */
  recognizer?: EquationRecognitionSession;
  /** Frame capture interval in ms. Default: 250. */
  intervalMs?: number;
  /** Whether to start in demo mode. */
  demoMode?: boolean;
}

// ---------------------------------------------------------------------------
// Demo / Fake recognizer
// ---------------------------------------------------------------------------

/**
 * Fake recognizer for demo/static mode.
 *
 * Returns a predefined sequence of equations for UI testing.
 */
export class DemoRecognizer implements EquationRecognitionSession {
  private index = 0;

  constructor(
    private readonly responses: StaticRecognizerOutput[] = DemoRecognizer.defaultResponses(),
  ) {}

  static defaultResponses(): StaticRecognizerOutput[] {
    return [
      {
        rawText: '3x+4=10',
        candidates: [{ text: '3x+4=10', confidence: 0.92 }],
      },
      {
        rawText: '3x+4=10',
        candidates: [{ text: '3x+4=10', confidence: 0.88 }],
      },
      {
        rawText: '3x+4=10',
        candidates: [{ text: '3x+4=10', confidence: 0.91 }],
      },
      {
        rawText: '3x+4=10',
        candidates: [{ text: '3x+4=10', confidence: 0.89 }],
      },
    ];
  }

  async recognize(_input: StaticRecognizerInput): Promise<StaticRecognizerOutput> {
    const response = this.responses[this.index % this.responses.length];
    this.index++;
    return response;
  }

  async dispose(): Promise<void> {
    // No resources to release.
  }
}

// ---------------------------------------------------------------------------
// Screen Controller
// ---------------------------------------------------------------------------

/**
 * LiveRecognitionScreen controller.
 *
 * Manages the lifecycle of a LiveRecognitionController, maintains
 * UI state, and notifies listeners on state changes.
 *
 * Usage:
 * ```typescript
 * const screen = new LiveRecognitionScreen({ demoMode: true });
 * screen.addListener((state) => renderOverlay(state));
 * screen.start();
 * // ... later ...
 * screen.stop();
 * await screen.dispose();
 * ```
 */
export class LiveRecognitionScreen {
  private controller: LiveRecognitionController;
  private uiState: RecognitionUiState;
  private diagnosticsExpanded = false;
  private lastExplanationSteps: ExplanationStep[] = [];
  private listeners: StateChangeListener[] = [];
  private disposed = false;

  constructor(options: LiveRecognitionScreenOptions = {}) {
    if (options.controller) {
      this.controller = options.controller;
    } else {
      const frameProvider = options.frameProvider ?? this.createDemoFrameProvider();
      const recognizer = options.recognizer ?? new DemoRecognizer();
      const intervalMs = options.intervalMs ?? 250;

      const controllerOptions: LiveRecognitionOptions = {
        frameProvider,
        recognizer,
        intervalMs,
        stabilityAggregator: new StabilityAggregator({
          minAgreement: 3,
          confidenceThreshold: 0.5,
        }),
        onStableResult: (result) => this.handleStableResult(result),
        onFrameResult: (result) => this.handleFrameResult(result),
        onError: (error, context) => this.handleError(error, context),
      };

      this.controller = new LiveRecognitionController(controllerOptions);
    }

    this.uiState = createIdleRecognitionUiState();
  }

  // -----------------------------------------------------------------------
  // Lifecycle
  // -----------------------------------------------------------------------

  /** Start the live recognition loop. */
  start(): void {
    if (this.disposed) {
      throw new Error('LiveRecognitionScreen has been disposed.');
    }
    this.controller.start();
    this.updateUiState({
      ...this.uiState,
      mode: 'scanning',
      statusMessage: 'Scanning for equation…',
    });
    this.notifyListeners();
  }

  /** Stop the recognition loop without disposing. */
  stop(): void {
    this.controller.stop();
    this.notifyListeners();
  }

  /** Dispose the controller and release all resources. */
  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    await this.controller.dispose();
    this.listeners = [];
  }

  /** Whether the recognition loop is currently running. */
  isRunning(): boolean {
    return this.controller.isRunning();
  }

  // -----------------------------------------------------------------------
  // State access
  // -----------------------------------------------------------------------

  /** Get the current screen state snapshot. */
  getState(): LiveRecognitionScreenState {
    return {
      uiState: { ...this.uiState },
      isRunning: this.controller.isRunning(),
      diagnosticsExpanded: this.diagnosticsExpanded,
      diagnostics: this.controller.getDiagnostics(),
      lastExplanationSteps: [...this.lastExplanationSteps],
    };
  }

  /** Get the current UI state. */
  getUiState(): RecognitionUiState {
    return { ...this.uiState };
  }

  /** Toggle the diagnostics panel visibility. */
  toggleDiagnostics(): void {
    this.diagnosticsExpanded = !this.diagnosticsExpanded;
    this.notifyListeners();
  }

  /** Render the full overlay composition. */
  renderOverlay(): RecognitionOverlayRenderData {
    return renderRecognitionOverlay({
      uiState: this.uiState,
      diagnostics: this.controller.getDiagnostics(),
      diagnosticsExpanded: this.diagnosticsExpanded,
      onToggleDiagnostics: () => this.toggleDiagnostics(),
    });
  }

  // -----------------------------------------------------------------------
  // Manual frame processing (for testing / demo)
  // -----------------------------------------------------------------------

  /**
   * Process a single frame manually.
   *
   * Useful for step-by-step testing or demo mode where you don't
   * want the interval loop running.
   */
  async processOneFrame(): Promise<void> {
    await this.controller.processOneFrame();
  }

  /** Get the underlying controller (for advanced use / testing). */
  getController(): LiveRecognitionController {
    return this.controller;
  }

  // -----------------------------------------------------------------------
  // Listener management
  // -----------------------------------------------------------------------

  /** Subscribe to state changes. Returns an unsubscribe function. */
  addListener(listener: StateChangeListener): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== listener);
    };
  }

  // -----------------------------------------------------------------------
  // Internal callbacks
  // -----------------------------------------------------------------------

  private handleStableResult(result: StableRecognitionResult): void {
    const diagnostics = this.controller.getDiagnostics();
    this.uiState = mapStableResultToUiState(
      result,
      diagnostics,
      this.lastExplanationSteps,
    );
    this.notifyListeners();
  }

  private handleFrameResult(result: FrameProcessingResult): void {
    if (result.skipped) return;

    // Extract explanation steps from accepted pipeline results.
    if (result.pipelineResult?.accepted) {
      const accepted = result.pipelineResult as PipelineAcceptedResult;
      this.lastExplanationSteps = accepted.explanationSteps;
    }

    const diagnostics = this.controller.getDiagnostics();
    this.uiState = mapStableResultToUiState(
      result.stableResult,
      diagnostics,
      this.lastExplanationSteps,
    );
    this.notifyListeners();
  }

  private handleError(error: Error, _context: string): void {
    const diagnostics = this.controller.getDiagnostics();
    this.uiState = mapErrorToUiState(error, diagnostics);
    this.notifyListeners();
  }

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------

  private updateUiState(partial: Partial<RecognitionUiState>): void {
    this.uiState = { ...this.uiState, ...partial };
  }

  private notifyListeners(): void {
    const state = this.getState();
    for (const listener of this.listeners) {
      listener(state);
    }
  }

  private createDemoFrameProvider(): CameraFrameProvider {
    // Create a series of synthetic frames for demo mode.
    const frames: CameraFrame[] = [];
    for (let i = 0; i < 8; i++) {
      frames.push(createSyntheticRgbaFrame(64, 16));
    }
    return new StaticFrameProvider(frames);
  }
}
