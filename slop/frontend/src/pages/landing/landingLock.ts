import type { MutableRefObject } from "react";
import { track } from "../../utils/analytics";

export interface LockState {
  isPinnedRef: MutableRefObject<boolean>;
  hasCompletedAllRef: MutableRefObject<boolean>;
  activeLayerRef: MutableRefObject<number>;
  completedStagesRef: MutableRefObject<number>;
  stageStartTimeRef: MutableRefObject<number>;
  wheelDeltaAccumRef: MutableRefObject<number>;
  setActiveLayer: (val: number | ((prev: number) => number)) => void;
  setCompletedStages: (val: number | ((prev: number) => number)) => void;
  setHasCompletedAllStages: (val: boolean) => void;
}

export function getLockScrollY(el: HTMLElement | null): number | null {
  if (!el) return null;
  const rect = el.getBoundingClientRect();
  const scrollY = window.scrollY || document.documentElement.scrollTop;
  return Math.max(0, rect.top + scrollY - 56);
}

export function unlockScroll(): void {
  if (typeof document !== "undefined" && document.body) {
    document.body.style.overflowY = "";
  }
}

function lockScroll(): void {
  if (typeof document !== "undefined" && document.body) {
    document.body.style.overflowY = "hidden";
  }
}

export function handleScrollClamp(
  el: HTMLElement | null,
  lockY: number | null,
  state: LockState,
): void {
  if (!el || lockY === null || state.hasCompletedAllRef.current) return;

  const rect = el.getBoundingClientRect();
  // Check entry into lock zone
  if (rect.top <= 72 && rect.top >= -140) {
    if (!state.isPinnedRef.current) {
      window.scrollTo({ top: lockY, behavior: "instant" as ScrollBehavior });
      state.isPinnedRef.current = true;
      lockScroll();
      state.stageStartTimeRef.current = Date.now();
      state.wheelDeltaAccumRef.current = 0;
    }
  }
}

function finishAllStages(state: LockState): void {
  state.completedStagesRef.current = 3;
  state.setCompletedStages(3);
  state.activeLayerRef.current = 3;
  state.setActiveLayer(3);
  state.hasCompletedAllRef.current = true;
  state.setHasCompletedAllStages(true);
  state.isPinnedRef.current = false;
  unlockScroll();

  const ctaEl = document.getElementById("cta-forensics");
  if (ctaEl) {
    ctaEl.scrollIntoView({ behavior: "smooth" });
  }
}

function handleWheelAction(
  e: Event,
  deltaY: number,
  state: LockState,
): void {
  if (!state.isPinnedRef.current || state.hasCompletedAllRef.current) return;

  // Downward Wheel / Gesture
  if (deltaY > 0) {
    e.preventDefault();
    const now = Date.now();
    const elapsed = now - state.stageStartTimeRef.current;
    if (elapsed < 320) return;

    state.wheelDeltaAccumRef.current += Math.max(Math.abs(deltaY), 25);
    if (state.wheelDeltaAccumRef.current >= 40) {
      state.wheelDeltaAccumRef.current = 0;
      state.stageStartTimeRef.current = now;

      const current = state.activeLayerRef.current;
      if (current < 3) {
        const next = current + 1;
        state.activeLayerRef.current = next;
        state.setActiveLayer(next);
        state.completedStagesRef.current = next - 1;
        state.setCompletedStages(next - 1);
      } else {
        finishAllStages(state);
      }
    }
    return;
  }

  // Upward Wheel / Gesture
  if (deltaY < 0) {
    e.preventDefault();
    const now = Date.now();
    const elapsed = now - state.stageStartTimeRef.current;
    if (elapsed < 300) return;

    state.stageStartTimeRef.current = now;
    state.wheelDeltaAccumRef.current = 0;

    const current = state.activeLayerRef.current;
    if (current > 0) {
      const prev = current - 1;
      state.activeLayerRef.current = prev;
      state.setActiveLayer(prev);
      state.completedStagesRef.current = prev - 1;
      state.setCompletedStages(prev - 1);
    } else {
      // Stage 0: unlock upward back to cards
      state.isPinnedRef.current = false;
      unlockScroll();
      window.scrollBy({ top: -180, behavior: "smooth" });
    }
  }
}

export function onWheelEvent(e: WheelEvent, state: LockState): void {
  if (!state.isPinnedRef.current) return;
  handleWheelAction(e, e.deltaY, state);
}

export function onKeyEvent(e: KeyboardEvent, state: LockState): void {
  if (!state.isPinnedRef.current) return;
  const isDown = e.key === "ArrowDown" || e.key === "PageDown" || e.key === " ";
  const isUp = e.key === "ArrowUp" || e.key === "PageUp";

  if (isDown) {
    handleWheelAction(e, 60, state);
  } else if (isUp) {
    handleWheelAction(e, -60, state);
  }
}

export function onTouchEvent(
  e: TouchEvent,
  startY: number,
  state: LockState,
): void {
  if (!state.isPinnedRef.current || !e.touches[0]) return;
  const deltaY = startY - e.touches[0].clientY;
  if (Math.abs(deltaY) < 15) return;
  handleWheelAction(e, deltaY, state);
}

export function handleStageClick(
  idx: number,
  layerId: string,
  el: HTMLElement | null,
  state: LockState,
): void {
  state.activeLayerRef.current = idx;
  state.stageStartTimeRef.current = Date.now();
  state.wheelDeltaAccumRef.current = 0;
  state.completedStagesRef.current = idx - 1;
  state.setCompletedStages(idx - 1);
  state.setActiveLayer(idx);

  if (idx === 3) {
    state.hasCompletedAllRef.current = true;
    state.setHasCompletedAllStages(true);
    state.isPinnedRef.current = false;
    unlockScroll();
  } else {
    const lockY = getLockScrollY(el);
    if (lockY !== null) {
      window.scrollTo({ top: lockY, behavior: "instant" as ScrollBehavior });
      state.isPinnedRef.current = true;
      lockScroll();
    }
  }
  track("cta_clicked", { section: "stepper_dot", stage: layerId });
}

export function handleSkipInspector(state: LockState): void {
  finishAllStages(state);
  track("cta_clicked", { section: "stepper_skip", stage: "skip_all" });
}
