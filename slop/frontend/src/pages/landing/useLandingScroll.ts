import { useState, useEffect, useCallback, useRef, type RefObject } from "react";
import { track } from "../../utils/analytics";

interface UseLandingScrollParams {
  conduitRef: RefObject<HTMLElement | null>;
  inspectorRef: RefObject<HTMLElement | null>;
  leftPathRef: RefObject<SVGPathElement | null>;
  centerPathRef: RefObject<SVGPathElement | null>;
  rightPathRef: RefObject<SVGPathElement | null>;
}

interface LockRefs {
  inspectorEl: HTMLElement | null;
  activeLayer: number;
  lastChangeTime: number;
  stage4UnlockedTime: number;
  advance: () => void;
  retreat: () => void;
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

function isInspectorPinned(bounds: DOMRect, windowHeight: number): boolean {
  return bounds.top <= 76 && bounds.bottom > windowHeight * 0.6;
}

function stepForward(
  e: Event,
  pinTop: number,
  activeLayer: number,
  lastChange: number,
  stage4Time: number,
  advance: () => void,
): void {
  const now = Date.now();
  if (activeLayer < 3) {
    e.preventDefault();
    window.scrollTo({ top: pinTop, behavior: "instant" });
    if (now - lastChange >= 320) advance();
    return;
  }
  if (now - stage4Time < 350) {
    e.preventDefault();
    window.scrollTo({ top: pinTop, behavior: "instant" });
  }
}

function stepBackward(
  e: Event,
  pinTop: number,
  activeLayer: number,
  lastChange: number,
  retreat: () => void,
): void {
  if (activeLayer <= 0) return;
  e.preventDefault();
  window.scrollTo({ top: pinTop, behavior: "instant" });
  if (Date.now() - lastChange >= 320) retreat();
}

function processWheel(e: WheelEvent, refs: LockRefs): void {
  if (!refs.inspectorEl) return;
  const bounds = refs.inspectorEl.getBoundingClientRect();
  if (!isInspectorPinned(bounds, window.innerHeight)) return;

  const pinTop = window.scrollY + bounds.top - 68;
  if (e.deltaY > 0) {
    stepForward(e, pinTop, refs.activeLayer, refs.lastChangeTime, refs.stage4UnlockedTime, refs.advance);
  } else if (e.deltaY < 0) {
    stepBackward(e, pinTop, refs.activeLayer, refs.lastChangeTime, refs.retreat);
  }
}

function processTouch(e: TouchEvent, startY: number, refs: LockRefs): void {
  if (!refs.inspectorEl || !e.touches[0]) return;
  const bounds = refs.inspectorEl.getBoundingClientRect();
  if (!isInspectorPinned(bounds, window.innerHeight)) return;

  const deltaY = startY - e.touches[0].clientY;
  if (Math.abs(deltaY) < 30) return;
  const pinTop = window.scrollY + bounds.top - 68;

  if (deltaY > 0) {
    stepForward(e, pinTop, refs.activeLayer, refs.lastChangeTime, refs.stage4UnlockedTime, refs.advance);
  } else {
    stepBackward(e, pinTop, refs.activeLayer, refs.lastChangeTime, refs.retreat);
  }
}

function processKey(
  e: KeyboardEvent,
  el: HTMLElement | null,
  layer: number,
  setStage: (idx: number) => void,
): void {
  if (!el || !isInspectorPinned(el.getBoundingClientRect(), window.innerHeight)) return;
  if ((e.key === "ArrowDown" || e.key === "PageDown") && layer < 3) {
    e.preventDefault();
    setStage(layer + 1);
  } else if ((e.key === "ArrowUp" || e.key === "PageUp") && layer > 0) {
    e.preventDefault();
    setStage(layer - 1);
  }
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

export function useLandingScroll({
  inspectorRef,
  leftPathRef,
  centerPathRef,
  rightPathRef,
}: UseLandingScrollParams) {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [activeLayer, setActiveLayer] = useState(0);
  const [ambientGlowY, setAmbientGlowY] = useState(25);

  const activeLayerRef = useRef(0);
  const lastChangeTimeRef = useRef(0);
  const stage4UnlockedTimeRef = useRef(0);
  const touchStartYRef = useRef(0);

  const setStage = useCallback((nextIdx: number) => {
    activeLayerRef.current = nextIdx;
    lastChangeTimeRef.current = Date.now();
    if (nextIdx === 3) stage4UnlockedTimeRef.current = Date.now();
    setActiveLayer(nextIdx);
  }, []);

  const handleSelectStage = useCallback((idx: number, layerId: string) => {
    setStage(idx);
    if (inspectorRef.current) {
      const top = window.scrollY + inspectorRef.current.getBoundingClientRect().top - 68;
      window.scrollTo({ top, behavior: "smooth" });
    }
    track("cta_clicked", { section: "stepper_dot", stage: layerId });
  }, [inspectorRef, setStage]);

  useEffect(() => {
    const paths = [leftPathRef.current, centerPathRef.current, rightPathRef.current];
    const lengths = paths.map((p, idx) => (p?.getTotalLength ? p.getTotalLength() : idx === 1 ? 141 : 400));
    let ticking = false;

    const onScroll = () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          updateScrollFrame(paths, lengths, setScrollProgress, setAmbientGlowY);
          ticking = false;
        });
        ticking = true;
      }
    };

    const lockRefs: LockRefs = {
      get inspectorEl() { return inspectorRef.current; },
      get activeLayer() { return activeLayerRef.current; },
      get lastChangeTime() { return lastChangeTimeRef.current; },
      get stage4UnlockedTime() { return stage4UnlockedTimeRef.current; },
      advance: () => setStage(activeLayerRef.current + 1),
      retreat: () => setStage(activeLayerRef.current - 1),
    };

    updateScrollFrame(paths, lengths, setScrollProgress, setAmbientGlowY);

    return attachLandingListeners(
      onScroll,
      (e) => processWheel(e, lockRefs),
      (e) => processKey(e, inspectorRef.current, activeLayerRef.current, setStage),
      (e) => { touchStartYRef.current = e.touches[0]?.clientY || 0; },
      (e) => processTouch(e, touchStartYRef.current, lockRefs),
    );
  }, [inspectorRef, leftPathRef, centerPathRef, rightPathRef, setStage]);

  const inspectorProgress = (activeLayer + 1) / 4;
  return { scrollProgress, inspectorProgress, activeLayer, ambientGlowY, handleSelectStage };
}
