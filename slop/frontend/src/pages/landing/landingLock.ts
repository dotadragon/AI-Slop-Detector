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

function checkResetUpward(scrollY: number, lockY: number, state: LockState): void {
  if (scrollY < lockY - 120) {
    state.isPinnedRef.current = false;
    state.hasCompletedAllRef.current = false;
    state.setHasCompletedAllStages(false);
    state.activeLayerRef.current = 0;
    state.setActiveLayer(0);
    state.completedStagesRef.current = -1;
    state.setCompletedStages(-1);
  }
}

export function handleScrollClamp(scrollY: number, lockY: number | null, state: LockState): void {
  if (lockY === null) return;

  if (!state.hasCompletedAllRef.current && scrollY >= lockY) {
    if (!state.isPinnedRef.current) {
      state.isPinnedRef.current = true;
      state.stageStartTimeRef.current = Date.now();
    }
    if (scrollY > lockY + 2) {
      window.scrollTo({ top: lockY, behavior: "instant" as ScrollBehavior });
    }
  }

  checkResetUpward(scrollY, lockY, state);
}

function finishAllStages(state: LockState): void {
  state.completedStagesRef.current = 3;
  state.setCompletedStages(3);
  state.hasCompletedAllRef.current = true;
  state.setHasCompletedAllStages(true);
  state.isPinnedRef.current = false;

  const ctaEl = document.getElementById("cta-forensics");
  if (ctaEl) {
    ctaEl.scrollIntoView({ behavior: "smooth" });
  }
}

function advanceStage(state: LockState): void {
  state.wheelDeltaAccumRef.current = 0;
  state.stageStartTimeRef.current = Date.now();

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

function handleWheelDown(e: Event, lockY: number, curY: number, deltaY: number, state: LockState): void {
  if (state.hasCompletedAllRef.current) return;
  if (curY + deltaY < lockY - 24) return;

  e.preventDefault();
  window.scrollTo({ top: lockY, behavior: "instant" as ScrollBehavior });

  if (!state.isPinnedRef.current) {
    state.isPinnedRef.current = true;
    state.stageStartTimeRef.current = Date.now();
    state.wheelDeltaAccumRef.current = 0;
    return;
  }

  const now = Date.now();
  const timeElapsed = now - state.stageStartTimeRef.current;
  state.wheelDeltaAccumRef.current += Math.abs(deltaY);

  if (timeElapsed >= 550 && state.wheelDeltaAccumRef.current >= 50) {
    advanceStage(state);
  }
}

function retreatStage(state: LockState): void {
  state.stageStartTimeRef.current = Date.now();
  state.wheelDeltaAccumRef.current = 0;

  const prev = state.activeLayerRef.current - 1;
  state.activeLayerRef.current = prev;
  state.setActiveLayer(prev);
  state.completedStagesRef.current = prev - 1;
  state.setCompletedStages(prev - 1);
}

function handleWheelUp(e: Event, lockY: number, curY: number, state: LockState): void {
  if (Math.abs(curY - lockY) > 48) return;

  const current = state.activeLayerRef.current;
  const timeElapsed = Date.now() - state.stageStartTimeRef.current;

  if (current > 0) {
    e.preventDefault();
    window.scrollTo({ top: lockY, behavior: "instant" as ScrollBehavior });
    state.hasCompletedAllRef.current = false;
    state.setHasCompletedAllStages(false);
    state.isPinnedRef.current = true;

    if (timeElapsed >= 450) {
      retreatStage(state);
    }
    return;
  }

  if (current === 0 && timeElapsed >= 450) {
    state.isPinnedRef.current = false;
  }
}

export function onWheelEvent(e: WheelEvent, getLockY: () => number | null, state: LockState): void {
  const lockY = getLockY();
  if (lockY === null) return;
  const curY = window.scrollY || document.documentElement.scrollTop;

  if (e.deltaY > 0) {
    handleWheelDown(e, lockY, curY, e.deltaY, state);
  } else if (e.deltaY < 0) {
    handleWheelUp(e, lockY, curY, state);
  }
}

export function onKeyEvent(e: KeyboardEvent, getLockY: () => number | null, state: LockState): void {
  const lockY = getLockY();
  if (lockY === null) return;
  const curY = window.scrollY || document.documentElement.scrollTop;

  const isDown = e.key === "ArrowDown" || e.key === "PageDown";
  const isUp = e.key === "ArrowUp" || e.key === "PageUp";

  if (isDown) {
    handleWheelDown(e, lockY, curY, 60, state);
  } else if (isUp) {
    handleWheelUp(e, lockY, curY, state);
  }
}

export function onTouchEvent(
  e: TouchEvent,
  startY: number,
  getLockY: () => number | null,
  state: LockState,
): void {
  if (!e.touches[0]) return;
  const lockY = getLockY();
  if (lockY === null) return;

  const curY = window.scrollY || document.documentElement.scrollTop;
  const deltaY = startY - e.touches[0].clientY;
  if (Math.abs(deltaY) < 15) return;

  if (deltaY > 0) {
    handleWheelDown(e, lockY, curY, deltaY, state);
  } else {
    handleWheelUp(e, lockY, curY, state);
  }
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

  const lockY = getLockScrollY(el);
  if (lockY !== null) {
    window.scrollTo({ top: lockY, behavior: "smooth" });
    state.isPinnedRef.current = true;
  }
  track("cta_clicked", { section: "stepper_dot", stage: layerId });
}
