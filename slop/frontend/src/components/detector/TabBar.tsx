import { Camera, FileAudio, FileSearch, FileText, Image, Video, Clock } from "lucide-react";
import { useRef, useEffect } from "react";
import gsap from "gsap";
import type { DetectorMode } from "../../types";

/**
 * v1 launch gating: text, image and live-image are validated and enabled.
 * Document, audio AND video are feature-gated OFF server-side (config.py flags
 * False) and pending validation. Gated tabs render as inert placeholders —
 * hidden state, not deleted: flipping a flag here re-enables the tab without
 * new routes.
 */
const tabs: Array<{
  id: DetectorMode;
  label: string;
  icon: typeof FileText;
  enabled: boolean;
  note?: string;
}> = [
  { id: "text", label: "Text", icon: FileText, enabled: true },
  { id: "image", label: "Image", icon: Image, enabled: true },
  { id: "live_image", label: "Live", icon: Camera, enabled: true },
  {
    id: "video",
    label: "Video",
    icon: Video,
    enabled: true,
  },
  {
    id: "document",
    label: "Document",
    icon: FileSearch,
    enabled: true,
  },
  {
    id: "audio",
    label: "Audio",
    icon: FileAudio,
    enabled: true,
  },
];

export function TabBar({ mode, onChange }: { mode: DetectorMode; onChange: (mode: DetectorMode) => void }) {
  const tabsContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!tabsContainerRef.current) return;
    const activeBtn = tabsContainerRef.current.querySelector("button.active");
    if (activeBtn) {
      gsap.fromTo(
        activeBtn,
        { scale: 0.94 },
        { scale: 1, duration: 0.3, ease: "back.out(2)" }
      );
    }
  }, [mode]);

  return (
    <div className="tabs" ref={tabsContainerRef} role="tablist" aria-label="Detection modalities">
      {tabs.map((tab) => {
        const Icon = tab.icon;
        const isActive = mode === tab.id;

        if (!tab.enabled) {
          // Inert placeholder: rendered for discoverability, non-interactive,
          // removed from the tab order and selection model. No dead button.
          return (
            <span
              key={tab.id}
              className="disabled-tab"
              aria-disabled="true"
              title={tab.note}
            >
              <Icon size={16} aria-hidden="true" />
              {tab.label}
              <Clock size={11} aria-hidden="true" />
            </span>
          );
        }

        return (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={isActive}
            aria-label={`${tab.label} detection mode`}
            className={isActive ? "active" : ""}
            onClick={(e) => {
              gsap.to(e.currentTarget, {
                scale: 0.92,
                duration: 0.1,
                yoyo: true,
                repeat: 1,
                ease: "power2.inOut",
                onComplete: () => onChange(tab.id),
              });
            }}
          >
            <Icon size={16} aria-hidden="true" />
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
