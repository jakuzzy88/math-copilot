/**
 * Guide box overlay component.
 *
 * Sprint 6A: UI Overlay Integration.
 *
 * Renders a rectangular guide box on top of the camera preview to help
 * users position their handwritten equation within the recognition area.
 *
 * The guide box uses a 4:1 aspect ratio (width:height) matching the
 * model's 512×128 input format. Corners are highlighted to give visual
 * feedback without obscuring the equation.
 *
 * This component is written as a pure React Native-compatible component.
 * It can be used in a real React Native app or tested in isolation.
 */

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

/** Props for the GuideBoxOverlay component. */
export interface GuideBoxOverlayProps {
  /** Whether the guide box is actively scanning. */
  isActive: boolean;
  /** Optional width of the guide box as percentage of container (0–100). Default: 85. */
  widthPercent?: number;
  /** Optional color for the guide box border. Default: '#00E676' (green). */
  borderColor?: string;
  /** Optional color when inactive. Default: 'rgba(255,255,255,0.4)'. */
  inactiveColor?: string;
}

// ---------------------------------------------------------------------------
// Component (pure functional — no JSX runtime required for testing)
// ---------------------------------------------------------------------------

/**
 * Compute the guide box style properties.
 *
 * Returns a style object that can be applied to a View element.
 * This is extracted as a pure function for easy testing without JSX.
 */
export function computeGuideBoxStyle(props: GuideBoxOverlayProps): {
  borderColor: string;
  borderWidth: number;
  borderRadius: number;
  widthPercent: number;
  aspectRatio: number;
  opacity: number;
} {
  const { isActive, widthPercent = 85, borderColor, inactiveColor } = props;

  const activeBorderColor = borderColor ?? '#00E676';
  const inactive = inactiveColor ?? 'rgba(255,255,255,0.4)';

  return {
    borderColor: isActive ? activeBorderColor : inactive,
    borderWidth: isActive ? 2.5 : 1.5,
    borderRadius: 12,
    widthPercent,
    aspectRatio: 4, // 4:1 width:height, matching 512:128
    opacity: isActive ? 1 : 0.6,
  };
}

/**
 * Render data for the GuideBoxOverlay.
 *
 * In a full React Native environment, this would return a <View> element.
 * For now, it returns a serializable render description that can be
 * consumed by a real renderer or validated in tests.
 */
export function renderGuideBoxOverlay(props: GuideBoxOverlayProps): {
  type: 'GuideBoxOverlay';
  style: ReturnType<typeof computeGuideBoxStyle>;
  accessibilityLabel: string;
} {
  return {
    type: 'GuideBoxOverlay',
    style: computeGuideBoxStyle(props),
    accessibilityLabel: props.isActive
      ? 'Equation guide box — scanning'
      : 'Equation guide box — idle',
  };
}
