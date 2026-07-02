/**
 * Symbol vocabulary – the set of symbols supported by the recogniser.
 *
 * This is the app-side mirror of `shared/symbols.json`.
 * We keep a TypeScript version for type-safe access and validation.
 */

export const SUPPORTED_DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] as const;
export const SUPPORTED_VARIABLES = ['x'] as const;
export const SUPPORTED_OPERATORS = ['+', '-', '*', '/'] as const;
export const SUPPORTED_GROUPING = ['(', ')'] as const;
export const SUPPORTED_RELATIONAL = ['='] as const;

/** Complete set of symbols the recogniser may output. */
export const ALL_SUPPORTED_SYMBOLS: ReadonlySet<string> = new Set([
  ...SUPPORTED_DIGITS,
  ...SUPPORTED_VARIABLES,
  ...SUPPORTED_OPERATORS,
  ...SUPPORTED_GROUPING,
  ...SUPPORTED_RELATIONAL,
]);

/**
 * Check whether a character is in the supported symbol set.
 */
export function isSupportedSymbol(ch: string): boolean {
  return ALL_SUPPORTED_SYMBOLS.has(ch);
}
