import { useState, useEffect, useCallback, useRef, type RefObject } from "react";
import {
  getLockScrollY,
  handleScrollClamp,
  handleStageClick,
  onWheelEvent,
  onKeyEvent,
  onTouchEvent,
  type LockState,
} from "./landingLock";

interface UseLandingScrollParams {
  conduitRef: RefObject<HTMLElement | null>;
  inspectorRef: RefObject<HTMLElement | null>;
  leftPathRef: RefObject<SVGPathElement | null>;
  centerPathRef: RefObject<SVGPathElement | null>;
  rightPathRef: RefObject<SVGPathElement | null>;
}

function syncPaths(paths: (SVGPathElement | null)[], lengths: number[], clamped: number): void {
  paths.forEach((pathEl, idx) => {
    if (pathEl) {
      pathEl.style.strokeDasharray = `${lengths[idx]}`;
      pathEl.style.strokeDashoffset = `${lengths[idx] * (1 - clamped)}`;
    }
  });
}

function computeConduitProgress(scrollY: number): number {
  if (scrollY <= 10) return 0;
  return Math.max(0, Math.min(1, (scrollY - 10) / 250));
}

function updateScrollFrame(
  paths: (SVGPathElement | null)[],
  lengths: number[],
  setProgress: (val: number) => void,
  setGlowY: (val: number) => void,
): void {
  const scrollY = window.scrollY || document.documentElement.scrollTop;
  const progress = computeConduitProgress(scrollY);
  setProgress(progress);
  syncPaths(paths, lengths, progress);

  const docH = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
  setGlowY(20 + (scrollY / docH) * 55);
}

function attachLandingListeners(
  onScroll: () => void,
  onWheel: (e: WheelEvent) => void,
  onKey: (e: KeyboardEvent) => void,
  onTouchStart: (e: TouchEvent) => void,
  onTouchMove: (e: TouchEvent) => void,
): () => void {
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("wheel", onWheel, { passive: false });
  window.addEventListener("keydown", onKey);
  window.addEventListener("touchstart", onTouchStart, { passive: true });
  window.addEventListener("touchmove", onTouchMove, { passive: false });

  return () => {
    window.removeEventListener("scroll", onScroll);
    window.removeEventListener("wheel", onWheel);
    window.removeEventListener("keydown", onKey);
    window.removeEventListener("touchstart", onTouchStart);
    window.removeEventListener("touchmove", onTouchMove);
  };
}

function initScrollListener(
  inspectorRef: RefObject<HTMLElement | null>,
  paths: (SVGPathElement | null)[],
  lengths: number[],
  setProgress: (val: number) => void,
  setGlowY: (val: number) => void,
  lockState: LockState,
  touchStartYRef: React.MutableRefObject<number>,
) {
  let ticking = false;

  const onScroll = () => {
    const scrollY = window.scrollY || document.documentElement.scrollTop;
    const lockY = getLockScrollY(inspectorRef.current);
    handleScrollClamp(scrollY, lockY, lockState);

    if (!ticking) {
      requestAnimationFrame(() => {
        updateScrollFrame(paths, lengths, setProgress, setGlowY);
        ticking = false;
      });
      ticking = true;
    }
  };

  updateScrollFrame(paths, lengths, setProgress, setGlowY);

  return attachLandingListeners(
    onScroll,
    (e) => onWheelEvent(e, () => getLockScrollY(inspectorRef.current), lockState),
    (e) => onKeyEvent(e, () => getLockScrollY(inspectorRef.current), lockState),
    (e) => { touchStartYRef.current = e.touches[0]?.clientY || 0; },
    (e) => onTouchEvent(e, touchStartYRef.current, () => getLockScrollY(inspectorRef.current), lockState),
  );
}

export function useLandingScroll({
  inspectorRef,
  leftPathRef,
  centerPathRef,
  rightPathRef,
}: UseLandingScrollParams) {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [activeLayer, setActiveLayer] = useState(0);
  const [completedStages, setCompletedStages] = useState(-1);
  const [hasCompletedAllStages, setHasCompletedAllStages] = useState(false);
  const [ambientGlowY, setAmbientGlowY] = useState(25);

  const activeLayerRef = useRef(0);
  const completedStagesRef = useRef(-1);
  const isPinnedRef = useRef(false);
  const hasCompletedAllRef = useRef(false);
  const stageStartTimeRef = useRef(0);
  const wheelDeltaAccumRef = useRef(0);
  const touchStartYRef = useRef(0);

  const lockState: LockState = {
    isPinnedRef,
    hasCompletedAllRef,
    activeLayerRef,
    completedStagesRef,
    stageStartTimeRef,
    wheelDeltaAccumRef,
    setActiveLayer,
    setCompletedStages,
    setHasCompletedAllStages,
  };

  const handleSelectStage = useCallback((idx: number, layerId: string) => {
    handleStageClick(idx, layerId, inspectorRef.current, lockState);
  }, [inspectorRef]);

  useEffect(() => {
    const paths = [leftPathRef.current, centerPathRef.current, rightPathRef.current];
    const lengths = paths.map((p, idx) => (p?.getTotalLength ? p.getTotalLength() : idx === 1 ? 141 : 400));
    return initScrollListener(
      inspectorRef,
      paths,
      lengths,
      setScrollProgress,
      setAmbientGlowY,
      lockState,
      touchStartYRef,
    );
  }, [inspectorRef, leftPathRef, centerPathRef, rightPathRef]);

  const inspectorProgress = (activeLayer + 1) / 4;
  return {
    scrollProgress,
    inspectorProgress,
    activeLayer,
    ambientGlowY,
    handleSelectStage,
    completedStages,
    hasCompletedAllStages,
  };
}
