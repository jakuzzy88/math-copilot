/**
 * CTC vocabulary for handwritten equation recognition.
 *
 * Sprint 5C: Mobile Static Image Inference Preparation.
 *
 * This vocabulary MUST match `training/models/vocabulary.py` exactly:
 *   blank  0 1 2 3 4 5 6 7 8 9 x + - = / ( ) <space>
 *
 * The CTC blank token is always index 0.
 */

// ---------------------------------------------------------------------------
// Vocabulary definition
// ---------------------------------------------------------------------------

/** Printable characters in vocabulary order (indices 1..18). */
export const CHARACTERS = '0123456789x+-=/() ';

/** CTC blank token string representation. */
export const BLANK_TOKEN = '<blank>';

/** CTC blank token index (always 0). */
export const BLANK_IDX = 0;

/** Total vocabulary size: blank + printable characters. */
export const VOCAB_SIZE = 1 + CHARACTERS.length; // 19

// ---------------------------------------------------------------------------
// Mappings
// ---------------------------------------------------------------------------

/** Map from index → character (0 → '<blank>', 1 → '0', ..., 18 → ' '). */
const idxToChar: Map<number, string> = new Map();
idxToChar.set(BLANK_IDX, BLANK_TOKEN);

/** Map from character → index ('0' → 1, ..., ' ' → 18). */
const charToIdx: Map<string, number> = new Map();

for (let i = 0; i < CHARACTERS.length; i++) {
  const idx = i + 1;
  const ch = CHARACTERS[i];
  idxToChar.set(idx, ch);
  charToIdx.set(ch, idx);
}

// ---------------------------------------------------------------------------
// Lookup functions
// ---------------------------------------------------------------------------

/**
 * Return the character for a given index.
 *
 * Index 0 returns the blank token string.
 *
 * @throws {RangeError} if index is out of vocabulary range [0, VOCAB_SIZE-1].
 */
export function indexToChar(idx: number): string {
  const ch = idxToChar.get(idx);
  if (ch === undefined) {
    throw new RangeError(
      `Index ${idx} out of vocabulary range [0, ${VOCAB_SIZE - 1}].`,
    );
  }
  return ch;
}

/**
 * Return the integer index for a single character.
 *
 * @throws {RangeError} for unsupported characters.
 */
export function charToIndex(ch: string): number {
  const idx = charToIdx.get(ch);
  if (idx === undefined) {
    throw new RangeError(
      `Unsupported character '${ch}'. Supported: '${CHARACTERS}'`,
    );
  }
  return idx;
}

/**
 * Return the full vocabulary as an ordered array.
 *
 * Index 0 is the blank token, indices 1..18 are printable characters.
 */
export function getVocabulary(): string[] {
  const vocab: string[] = [];
  for (let i = 0; i < VOCAB_SIZE; i++) {
    vocab.push(idxToChar.get(i)!);
  }
  return vocab;
}
