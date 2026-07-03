/**
 * cameraLifecycle — Pure camera lifecycle logic.
 *
 * Sprint 7: Real Camera Preview.
 *
 * Contains the pure, testable logic for determining camera activation state.
 * This module has NO React or React Native dependencies — it can be tested
 * in a Node/Jest environment without any native module mocking.
 *
 * The React hook (useCameraLifecycle) consumes these functions and wires
 * them to AppState + useState/useEffect.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Possible app state values (mirrors React Native's AppStateStatus).
 *
 * Defined here as a plain string union to avoid importing from react-native,
 * keeping this module fully testable in Node.
 */
export type AppStateLike = 'active' | 'inactive' | 'background' | 'extension' | (string & {});

/**
 * All inputs that determine whether the camera should be active.
 */
export interface CameraLifecycleInputs {
  /** Whether the camera feature is enabled (permission granted + not demo). */
  enabled: boolean;
  /** Current AppState value. */
  appState: AppStateLike;
  /** Whether the camera is explicitly paused by the user/screen. */
  isPaused: boolean;
}

/**
 * Camera lifecycle state — the result of evaluating all inputs.
 */
export interface CameraLifecycleResult {
  /** Whether the camera should be actively streaming. */
  isActive: boolean;
}

// ---------------------------------------------------------------------------
// Pure logic
// ---------------------------------------------------------------------------

/**
 * Determine whether the camera should be active given the inputs.
 *
 * The camera is active ONLY when ALL conditions are met:
 *   1. `enabled` is true (permission granted, not demo mode).
 *   2. `appState` is 'active' (app is in the foreground).
 *   3. `isPaused` is false (not explicitly paused by screen logic).
 *
 * @returns true if the camera should be streaming frames.
 */
export function shouldCameraBeActive(
  enabled: boolean,
  appState: AppStateLike,
  isPaused: boolean,
): boolean {
  const isAppForeground = appState === 'active';
  return enabled && isAppForeground && !isPaused;
}

/**
 * Evaluate camera lifecycle from a structured input.
 *
 * @param inputs - All conditions affecting camera activation.
 * @returns CameraLifecycleResult with the computed isActive state.
 */
export function evaluateCameraLifecycle(
  inputs: CameraLifecycleInputs,
): CameraLifecycleResult {
  return {
    isActive: shouldCameraBeActive(
      inputs.enabled,
      inputs.appState,
      inputs.isPaused,
    ),
  };
}

/**
 * Determine the camera enabled state from permission and mode.
 *
 * @param isDemoMode - Whether the app is in demo mode.
 * @param hasPermission - Whether camera permission is granted.
 * @returns true if the camera should be considered enabled.
 */
export function isCameraEnabled(
  isDemoMode: boolean,
  hasPermission: boolean,
): boolean {
  return !isDemoMode && hasPermission;
}
