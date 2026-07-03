/**
 * useCameraLifecycle — Camera lifecycle management hook.
 *
 * Sprint 7: Real Camera Preview.
 *
 * Manages the camera active state based on:
 *   1. Component mount/unmount.
 *   2. AppState transitions (active → background → inactive).
 *   3. Explicit pause/resume from the screen.
 *
 * The camera is only active when:
 *   - The component is mounted.
 *   - The app is in the foreground (AppState === 'active').
 *   - The screen is not explicitly paused.
 *
 * This ensures no camera resources leak when the app is backgrounded
 * or the screen is unmounted.
 *
 * Pure logic is in cameraLifecycle.ts (testable without React).
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { AppState, type AppStateStatus } from 'react-native';
import { shouldCameraBeActive } from './cameraLifecycle';

// Re-export pure logic for convenience.
export { shouldCameraBeActive } from './cameraLifecycle';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface CameraLifecycleState {
  /** Whether the camera should be actively streaming. */
  isActive: boolean;
  /** Whether the camera is explicitly paused by user/screen logic. */
  isPaused: boolean;
  /** Current AppState value. */
  appState: AppStateStatus;
  /** Pause the camera (e.g. when navigating away within the app). */
  pause: () => void;
  /** Resume the camera after a pause. */
  resume: () => void;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Hook to manage camera activation based on app lifecycle.
 *
 * @param enabled - Whether the camera feature is enabled at all
 *                  (e.g. false in DEMO_MODE or when permission is denied).
 */
export function useCameraLifecycle(enabled: boolean): CameraLifecycleState {
  const [isPaused, setIsPaused] = useState(false);
  const [appState, setAppState] = useState<AppStateStatus>(
    AppState.currentState ?? 'active',
  );
  const mountedRef = useRef(true);

  // Track AppState changes.
  useEffect(() => {
    const subscription = AppState.addEventListener('change', (nextAppState) => {
      if (mountedRef.current) {
        setAppState(nextAppState);
      }
    });

    return () => {
      subscription.remove();
    };
  }, []);

  // Track mount/unmount.
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const pause = useCallback(() => {
    setIsPaused(true);
  }, []);

  const resume = useCallback(() => {
    setIsPaused(false);
  }, []);

  // Camera is active when: enabled + foreground + not paused + mounted.
  const isActive = shouldCameraBeActive(enabled, appState, isPaused);

  return {
    isActive,
    isPaused,
    appState,
    pause,
    resume,
  };
}
