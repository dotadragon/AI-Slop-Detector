import { Camera, FileText, Image, Video } from "lucide-react";
import { useRef, useEffect } from "react";
import gsap from "gsap";
import type { DetectorMode } from "../../types";

const tabs: Array<{ id: DetectorMode; label: string; icon: typeof FileText }> = [
  { id: "text", label: "Text", icon: FileText },
  { id: "image", label: "Image", icon: Image },
  { id: "live_image", label: "Live", icon: Camera },
  { id: "video", label: "Video", icon: Video }
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
