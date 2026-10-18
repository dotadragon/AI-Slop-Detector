import { useState, useEffect, useCallback, type RefObject } from "react";
import { track } from "../utils/analytics";

interface UseLandingScrollParams {
  conduitRef: RefObject<HTMLElement | null>;
  inspectorRef: RefObject<HTMLElement | null>;
  leftPathRef: RefObject<SVGPathElement | null>;
  centerPathRef: RefObject<SVGPathElement | null>;
  rightPathRef: RefObject<SVGPathElement | null>;
}

function syncPaths(paths: (SVGPathElement | null)[], lengths: number[], clamped: number) {
  paths.forEach((pathEl, idx) => {
    if (pathEl) pathEl.style.strokeDashoffset = `${lengths[idx] * (1 - clamped)}`;
  });
}

function computeConduitProgress(element: HTMLElement, pageScrollY: number, windowHeight: number): number {
  const bounds = element.getBoundingClientRect();
  const scrollBased = pageScrollY / Math.max(180, windowHeight * 0.28);
  const viewportBased = (windowHeight * 0.72 - bounds.top) / (windowHeight * 0.38);
  return Math.max(0.18, Math.min(1, Math.max(scrollBased, viewportBased)));
}

function computeInspectorState(element: HTMLElement, windowHeight: number): { progress: number; layer: number | null } {
  const bounds = element.getBoundingClientRect();
  const totalTravel = Math.max(1, bounds.height - windowHeight);
  const progress = Math.max(0, Math.min(1, (80 - bounds.top) / totalTravel));
  const isVisible = bounds.top <= windowHeight * 0.75 && bounds.bottom >= windowHeight * 0.25;
  const layer = isVisible ? Math.min(3, Math.floor(progress * 3.99)) : null;
  return { progress, layer };
}

function scrollToStage(inspectorElement: HTMLElement, stageIdx: number) {
  const bounds = inspectorElement.getBoundingClientRect();
  const sectionTop = (window.scrollY || document.documentElement.scrollTop) + bounds.top - 72;
  const totalTravel = Math.max(0, inspectorElement.offsetHeight - window.innerHeight);
  const targetY = sectionTop + (stageIdx / 3) * (totalTravel * 0.88);
  const lenis = (window as unknown as { __lenis?: { scrollTo: (target: number, opts?: unknown) => void } }).__lenis;
  if (lenis) {
    lenis.scrollTo(targetY, { duration: 0.9 });
  } else {
    window.scrollTo({ top: targetY, behavior: "smooth" });
  }
}

export function useLandingScroll({
  conduitRef,
  inspectorRef,
  leftPathRef,
  centerPathRef,
  rightPathRef,
}: UseLandingScrollParams) {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [inspectorProgress, setInspectorProgress] = useState(0);
  const [activeLayer, setActiveLayer] = useState(0);
  const [ambientGlowY, setAmbientGlowY] = useState(25);

  const handleSelectStage = useCallback((idx: number, layerId: string) => {
    setActiveLayer(idx);
    if (inspectorRef.current) {
      scrollToStage(inspectorRef.current, idx);
    }
    track("cta_clicked", { section: "stepper_dot", stage: layerId });
  }, [inspectorRef]);

  useEffect(() => {
    const paths = [leftPathRef.current, centerPathRef.current, rightPathRef.current];
    const lengths = paths.map((p, idx) => (p?.getTotalLength ? p.getTotalLength() : idx === 1 ? 141 : 400));
    let ticking = false;

    const updateScroll = () => {
      const winH = window.innerHeight;
      const scrollY = window.scrollY || document.documentElement.scrollTop;

      if (conduitRef.current) {
        const clamped = computeConduitProgress(conduitRef.current, scrollY, winH);
        setScrollProgress(clamped);
        syncPaths(paths, lengths, clamped);
      }
      if (inspectorRef.current) {
        const { progress, layer } = computeInspectorState(inspectorRef.current, winH);
        setInspectorProgress(progress);
        if (layer !== null) setActiveLayer(layer);
      }
      const docHeight = Math.max(1, document.documentElement.scrollHeight - winH);
      setAmbientGlowY(20 + (scrollY / docHeight) * 55);
      ticking = false;
    };

    const onScroll = () => {
      if (!ticking) {
        requestAnimationFrame(updateScroll);
        ticking = true;
      }
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    const initRaf = requestAnimationFrame(updateScroll);

    return () => {
      cancelAnimationFrame(initRaf);
      window.removeEventListener("scroll", onScroll);
    };
  }, [conduitRef, inspectorRef, leftPathRef, centerPathRef, rightPathRef]);

  return { scrollProgress, inspectorProgress, activeLayer, ambientGlowY, handleSelectStage };
}
