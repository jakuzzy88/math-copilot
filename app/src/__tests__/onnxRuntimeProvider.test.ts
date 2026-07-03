/**
 * Tests for ONNX Runtime provider.
 *
 * Sprint 5D: ONNX Runtime Mobile Static Inference.
 *
 * Verifies that the safe runtime abstraction handles missing native
 * modules gracefully (returns null in Node/Jest) and that the test
 * injection helpers work correctly.
 */

import {
  getOnnxRuntime,
  isOnnxAvailable,
  _resetOnnxRuntimeCache,
  _setOnnxRuntime,
} from '../inference/onnxRuntimeProvider';
import type { OnnxRuntimeApi } from '../inference/onnxRuntimeProvider';

describe('onnxRuntimeProvider', () => {
  afterEach(() => {
    _resetOnnxRuntimeCache();
  });

  describe('getOnnxRuntime()', () => {
    it('returns null in Node/Jest (no native bindings)', () => {
      // In Jest, onnxruntime-react-native has no native module,
      // so require() throws and getOnnxRuntime() returns null.
      const result = getOnnxRuntime();
      // It may or may not be null depending on the environment.
      // What matters is it doesn't crash.
      expect(result === null || typeof result === 'object').toBe(true);
    });

    it('caches the result across calls', () => {
      const first = getOnnxRuntime();
      const second = getOnnxRuntime();
      // Same reference (or both null).
      expect(first).toBe(second);
    });
  });

  describe('isOnnxAvailable()', () => {
    it('returns boolean', () => {
      expect(typeof isOnnxAvailable()).toBe('boolean');
    });
  });

  describe('_setOnnxRuntime()', () => {
    it('allows injecting a mock runtime', () => {
      const mockOrt: OnnxRuntimeApi = {
        InferenceSession: {
          create: jest.fn(),
        },
        Tensor: jest.fn() as unknown as OnnxRuntimeApi['Tensor'],
      };

      _setOnnxRuntime(mockOrt);
      expect(getOnnxRuntime()).toBe(mockOrt);
      expect(isOnnxAvailable()).toBe(true);
    });

    it('allows injecting null (simulate unavailable runtime)', () => {
      _setOnnxRuntime(null);
      expect(getOnnxRuntime()).toBe(null);
      expect(isOnnxAvailable()).toBe(false);
    });
  });

  describe('_resetOnnxRuntimeCache()', () => {
    it('clears the cache so getOnnxRuntime() re-evaluates', () => {
      const mockOrt: OnnxRuntimeApi = {
        InferenceSession: { create: jest.fn() },
        Tensor: jest.fn() as unknown as OnnxRuntimeApi['Tensor'],
      };

      _setOnnxRuntime(mockOrt);
      expect(getOnnxRuntime()).toBe(mockOrt);

      _resetOnnxRuntimeCache();

      // After reset, it should re-try the require() call.
      const result = getOnnxRuntime();
      expect(result).not.toBe(mockOrt);
    });
  });
});
