/**
 * App.tsx — Root React Native application entry.
 *
 * Sprint 6B: React Native Camera Shell.
 *
 * Renders the MathCameraScreen as the single-screen application.
 * Navigation can be added in future sprints if needed.
 */

import React from 'react';
import { MathCameraScreen } from './screens/MathCameraScreen';

export default function App(): React.JSX.Element {
  return <MathCameraScreen />;
}
