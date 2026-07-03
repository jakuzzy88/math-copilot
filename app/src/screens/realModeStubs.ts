/**
 * Real camera frame provider interface stubs.
 *
 * Sprint 6B: React Native Camera Shell.
 *
 * These stubs define the integration path for connecting
 * react-native-vision-camera to the existing LiveRecognitionController.
 *
 * TODO (Sprint 7+):
 *   - Implement VisionCameraFrameProvider using Frame Processor plugins.
 *   - Extract RGBA pixel data from camera frames.
 *   - Wire into LiveRecognitionController.frameProvider.
 */

import type { CameraFrame, CameraFrameProvider } from '../inference/cameraFrameProvider';

// ---------------------------------------------------------------------------
// Real camera frame extraction (stub)
// ---------------------------------------------------------------------------

/**
 * TODO: Implement real camera frame extraction from react-native-vision-camera.
 *
 * This provider would:
 *   1. Register a Frame Processor with react-native-vision-camera.
 *   2. On each frame callback, extract RGBA pixel data.
 *   3. Wrap the data in a CameraFrame and make it available via captureFrame().
 *
 * Challenges:
 *   - Frame Processor runs on the native thread; pixel data must be copied
 *     to the JS thread efficiently (SharedArrayBuffer or base64 bridge).
 *   - Frame format varies by device (NV21, YUV_420_888, BGRA, etc.).
 *   - A native Frame Processor plugin may be required for performant
 *     pixel-level access.
 *
 * Interim approach:
 *   Use the StaticFrameProvider or DemoRecognizer for testing until
 *   native frame extraction is implemented.
 */
export class VisionCameraFrameProvider implements CameraFrameProvider {
  private lastFrame: CameraFrame | null = null;
  private _disposed = false;

  /**
   * Called from a Frame Processor callback to update the latest frame.
   *
   * TODO: Implement actual frame extraction.
   */
  updateFrame(_nativeFrame: unknown): void {
    // TODO: Extract RGBA pixel data from the native frame object.
    // This requires a native Frame Processor plugin or bridge module.
    throw new Error(
      'VisionCameraFrameProvider.updateFrame() is not yet implemented. ' +
      'Real camera frame extraction requires a native module.'
    );
  }

  isReady(): boolean {
    return !this._disposed;
  }

  async captureFrame(): Promise<CameraFrame | null> {
    if (this._disposed) {
      return null;
    }
    return this.lastFrame;
  }

  async dispose(): Promise<void> {
    this._disposed = true;
    this.lastFrame = null;
  }
}

// ---------------------------------------------------------------------------
// Real-mode pipeline integration stubs
// ---------------------------------------------------------------------------

/**
 * TODO: Wire the real pipeline when ONNX Runtime is available on device.
 *
 * Steps:
 *   1. Import { OnnxEquationRecognizer } from '../inference/staticImageRecognizer'
 *   2. const recognizer = await OnnxEquationRecognizer.create(modelPath);
 *   3. const frameProvider = new VisionCameraFrameProvider();
 *   4. const controller = new LiveRecognitionController({
 *        frameProvider,
 *        recognizer,
 *        intervalMs: 250,
 *        onStableResult: (result) => updateUI(result),
 *      });
 *   5. Register a Frame Processor that calls frameProvider.updateFrame(frame).
 *   6. controller.start();
 */

/**
 * TODO: framePipeline.processFrame() integration.
 *
 * The existing processFrame() already handles:
 *   - RGBA → grayscale conversion
 *   - Auto or explicit 4:1 crop
 *   - Bilinear resize to 128×512
 *   - Normalisation to [0, 1] Float32Array
 *
 * No changes needed — just wire the VisionCameraFrameProvider output
 * into the LiveRecognitionController.
 */

/**
 * TODO: StabilityAggregator is already integrated into LiveRecognitionController.
 * No additional work needed for stability.
 */

/**
 * TODO: recognizeAndSolve() is already integrated into LiveRecognitionController.
 * No additional work needed for the solve pipeline.
 */
