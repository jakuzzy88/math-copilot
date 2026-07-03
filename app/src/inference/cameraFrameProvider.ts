/**
 * Camera frame provider abstraction.
 *
 * Sprint 5E: Camera Frame Capture + Crop + Resize Pipeline.
 *
 * Defines a platform-agnostic interface for capturing camera frames.
 * Concrete implementations will be provided by:
 *   - `react-native-camera` or `expo-camera` (device)
 *   - Synthetic frame generators (tests)
 *
 * The frame data is always a flat Uint8Array of RGBA pixel data
 * (width × height × 4 bytes), which is the standard format returned
 * by mobile camera APIs after conversion.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Pixel format of the frame data. */
export type PixelFormat = 'rgba' | 'rgb' | 'grayscale';

/** A single captured camera frame. */
export interface CameraFrame {
  /** Raw pixel data. */
  data: Uint8Array;
  /** Image width in pixels. */
  width: number;
  /** Image height in pixels. */
  height: number;
  /** Pixel format. */
  format: PixelFormat;
  /** Capture timestamp in milliseconds (performance.now or Date.now). */
  timestampMs: number;
}

/**
 * Camera frame provider interface.
 *
 * Implementations capture frames from different sources:
 * - Device camera (react-native-camera, expo-camera)
 * - Static test images
 * - Video file replay
 */
export interface CameraFrameProvider {
  /** Capture a single frame. Returns null if capture is unavailable. */
  captureFrame(): Promise<CameraFrame | null>;
  /** Whether the provider is ready to capture. */
  isReady(): boolean;
  /** Release camera resources. */
  dispose(): Promise<void>;
}

// ---------------------------------------------------------------------------
// Static frame provider (for testing)
// ---------------------------------------------------------------------------

/**
 * A frame provider that returns a fixed pre-loaded frame.
 *
 * Useful for unit tests and integration tests where no real camera
 * is available.
 */
export class StaticFrameProvider implements CameraFrameProvider {
  private disposed = false;
  private frameIndex = 0;

  /**
   * @param frames - Array of pre-loaded frames to cycle through.
   */
  constructor(private readonly frames: CameraFrame[]) {
    if (frames.length === 0) {
      throw new Error('StaticFrameProvider requires at least one frame.');
    }
  }

  async captureFrame(): Promise<CameraFrame | null> {
    if (this.disposed || this.frames.length === 0) {
      return null;
    }
    const frame = this.frames[this.frameIndex % this.frames.length];
    this.frameIndex++;
    return frame;
  }

  isReady(): boolean {
    return !this.disposed;
  }

  async dispose(): Promise<void> {
    this.disposed = true;
  }
}

// ---------------------------------------------------------------------------
// Synthetic frame factories
// ---------------------------------------------------------------------------

/**
 * Create a synthetic RGBA camera frame (e.g. 640×480).
 *
 * Fills with a uniform colour. Useful for testing the crop+resize pipeline.
 *
 * @param width - Frame width (default: 640).
 * @param height - Frame height (default: 480).
 * @param r - Red channel value [0,255] (default: 200).
 * @param g - Green channel value [0,255] (default: 200).
 * @param b - Blue channel value [0,255] (default: 200).
 * @param a - Alpha channel value [0,255] (default: 255).
 */
export function createSyntheticRgbaFrame(
  width = 640,
  height = 480,
  r = 200,
  g = 200,
  b = 200,
  a = 255,
): CameraFrame {
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const base = i * 4;
    data[base] = r;
    data[base + 1] = g;
    data[base + 2] = b;
    data[base + 3] = a;
  }
  return {
    data,
    width,
    height,
    format: 'rgba',
    timestampMs: Date.now(),
  };
}

/**
 * Create a synthetic RGBA frame with a horizontal gradient.
 *
 * Left edge = black, right edge = white.  Useful for verifying
 * that resize interpolation and grayscale conversion are correct.
 */
export function createGradientRgbaFrame(
  width = 640,
  height = 480,
): CameraFrame {
  const data = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const val = Math.round((x / (width - 1)) * 255);
      const base = (y * width + x) * 4;
      data[base] = val;
      data[base + 1] = val;
      data[base + 2] = val;
      data[base + 3] = 255;
    }
  }
  return {
    data,
    width,
    height,
    format: 'rgba',
    timestampMs: Date.now(),
  };
}

/**
 * Create a synthetic RGBA frame with a dark rectangle (simulating text)
 * on a light background.
 *
 * Mimics the look of a handwritten equation on white paper captured
 * by a camera.
 */
export function createTextLikeRgbaFrame(
  width = 640,
  height = 480,
): CameraFrame {
  const data = new Uint8Array(width * height * 4);
  // Fill with light background (paper colour).
  for (let i = 0; i < width * height; i++) {
    const base = i * 4;
    data[base] = 240;
    data[base + 1] = 235;
    data[base + 2] = 230;
    data[base + 3] = 255;
  }

  // Draw a dark horizontal band in the middle (simulating text line).
  const bandTop = Math.floor(height * 0.35);
  const bandBottom = Math.floor(height * 0.65);
  const bandLeft = Math.floor(width * 0.1);
  const bandRight = Math.floor(width * 0.9);

  for (let y = bandTop; y < bandBottom; y++) {
    for (let x = bandLeft; x < bandRight; x++) {
      const base = (y * width + x) * 4;
      data[base] = 30;
      data[base + 1] = 25;
      data[base + 2] = 20;
      data[base + 3] = 255;
    }
  }

  return {
    data,
    width,
    height,
    format: 'rgba',
    timestampMs: Date.now(),
  };
}
