/**
 * Tests for recognizeAndSolve pipeline integration.
 *
 * Sprint 5C: Mobile Static Image Inference Preparation.
 *
 * Uses fake recognizers to test the full flow without ONNX Runtime.
 */

import { recognizeAndSolve } from '../inference/recognizeAndSolve';
import type {
  EquationRecognitionSession,
  StaticRecognizerInput,
  StaticRecognizerOutput,
} from '../inference/staticImageRecognizer';
import { MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH } from '../inference/modelIO';

// ---------------------------------------------------------------------------
// Fake recognizer helper
// ---------------------------------------------------------------------------

/**
 * Create a fake recognizer that returns a predetermined text and confidence.
 */
function createFakeRecognizer(
  text: string,
  confidence: number,
): EquationRecognitionSession {
  return {
    async recognize(_input: StaticRecognizerInput): Promise<StaticRecognizerOutput> {
      return {
        rawText: text,
        candidates: [{ text, confidence }],
      };
    },
    async dispose(): Promise<void> {
      // no-op
    },
  };
}

/** Dummy input for tests (correct dimensions, all-black). */
const dummyInput: StaticRecognizerInput = {
  grayscalePixels: new Uint8Array(MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH),
  width: MODEL_INPUT_WIDTH,
  height: MODEL_INPUT_HEIGHT,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('recognizeAndSolve', () => {
  it('accepts "3x+4=10" and returns x=2', async () => {
    const recognizer = createFakeRecognizer('3x+4=10', 0.95);
    const result = await recognizeAndSolve(recognizer, dummyInput);

    expect(result.rawText).toBe('3x+4=10');
    expect(result.pipeline.accepted).toBe(true);

    if (result.pipeline.accepted) {
      expect(result.pipeline.equation).toBe('3x+4=10');
      expect(result.pipeline.solution).toBe('x=2');
      expect(result.pipeline.explanationSteps.length).toBeGreaterThan(0);
      expect(result.pipeline.score).toBeGreaterThan(0);
    }
  });

  it('accepts "x+5=12" and returns x=7', async () => {
    const recognizer = createFakeRecognizer('x+5=12', 0.90);
    const result = await recognizeAndSolve(recognizer, dummyInput);

    expect(result.pipeline.accepted).toBe(true);
    if (result.pipeline.accepted) {
      expect(result.pipeline.solution).toBe('x=7');
    }
  });

  it('rejects invalid equation text', async () => {
    const recognizer = createFakeRecognizer('hello world', 0.80);
    const result = await recognizeAndSolve(recognizer, dummyInput);

    expect(result.rawText).toBe('hello world');
    expect(result.pipeline.accepted).toBe(false);

    if (!result.pipeline.accepted) {
      expect(result.pipeline.rejection).toBeDefined();
      expect(result.pipeline.rejection.code).toBeDefined();
      expect(result.pipeline.rejection.message).toBeDefined();
    }
  });

  it('rejects empty string', async () => {
    const recognizer = createFakeRecognizer('', 0.50);
    const result = await recognizeAndSolve(recognizer, dummyInput);

    expect(result.pipeline.accepted).toBe(false);
  });

  it('rejects nonsense symbols', async () => {
    const recognizer = createFakeRecognizer('+++===', 0.60);
    const result = await recognizeAndSolve(recognizer, dummyInput);

    expect(result.pipeline.accepted).toBe(false);
  });

  it('returns rawText from the recognizer regardless of pipeline outcome', async () => {
    const recognizer = createFakeRecognizer('garbage123', 0.70);
    const result = await recognizeAndSolve(recognizer, dummyInput);

    // rawText is always from the recognizer, even if pipeline rejects.
    expect(result.rawText).toBe('garbage123');
  });
});
