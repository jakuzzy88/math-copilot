/**
 * Safe ONNX Runtime provider.
 *
 * Sprint 5D: ONNX Runtime Mobile Static Inference.
 *
 * This module provides a safe runtime abstraction over `onnxruntime-react-native`.
 * In a React Native environment with native bindings the real ONNX Runtime API
 * is returned.  In Node.js / Jest (where native bindings are unavailable) the
 * module returns `null` so that tests can mock inference without crashing.
 *
 * Usage:
 *   import { getOnnxRuntime, isOnnxAvailable } from './onnxRuntimeProvider';
 *   const ort = getOnnxRuntime();
 *   if (ort === null) { throw new Error('ONNX Runtime unavailable'); }
 */

// ---------------------------------------------------------------------------
// Types (re-exported subset of onnxruntime-react-native)
// ---------------------------------------------------------------------------

/**
 * Minimal type surface for ONNX Runtime.
 *
 * We define these ourselves so that the rest of the codebase does not
 * need a hard import of `onnxruntime-react-native` (which would fail
 * in Node).
 */
export interface OnnxTensor {
  readonly type: string;
  readonly data: Float32Array | Int32Array | Uint8Array;
  readonly dims: readonly number[];
}

export interface OnnxInferenceSession {
  run(
    feeds: Record<string, OnnxTensor>,
    options?: unknown,
  ): Promise<Record<string, OnnxTensor>>;
  release(): Promise<void>;
}

export interface OnnxRuntimeApi {
  InferenceSession: {
    create(
      modelPath: string,
      options?: unknown,
    ): Promise<OnnxInferenceSession>;
  };
  Tensor: new (
    type: string,
    data: Float32Array | Int32Array | Uint8Array | number[],
    dims: number[],
  ) => OnnxTensor;
}

// ---------------------------------------------------------------------------
// Lazy singleton
// ---------------------------------------------------------------------------

let cachedOrt: OnnxRuntimeApi | null | undefined;

/**
 * Attempt to load the ONNX Runtime API.
 *
 * Returns `null` if the native module is not available (e.g. in Jest / Node).
 * The result is cached after the first call.
 */
export function getOnnxRuntime(): OnnxRuntimeApi | null {
  if (cachedOrt !== undefined) {
    return cachedOrt;
  }

  try {
    // Dynamic require so that bundlers (Metro) resolve it at build time
    // but Node.js fails gracefully at runtime.
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const ort = require('onnxruntime-react-native') as OnnxRuntimeApi;
    cachedOrt = ort;
    return ort;
  } catch {
    cachedOrt = null;
    return null;
  }
}

/**
 * Whether the ONNX Runtime native module is available.
 */
export function isOnnxAvailable(): boolean {
  return getOnnxRuntime() !== null;
}

/**
 * Reset the cached runtime reference.
 *
 * **Test-only.** Allows tests to inject a mock after clearing the cache.
 */
export function _resetOnnxRuntimeCache(): void {
  cachedOrt = undefined;
}

/**
 * Inject a custom ONNX Runtime API for testing.
 *
 * **Test-only.** Sets the cached runtime to a mock/stub implementation.
 */
export function _setOnnxRuntime(mock: OnnxRuntimeApi | null): void {
  cachedOrt = mock;
}
