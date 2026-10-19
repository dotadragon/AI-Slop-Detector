import { useRef, useState } from "react";
import { useSeo } from "../hooks/useSeo";
import { useLandingScroll } from "./landing/useLandingScroll";
import { HeroSection } from "../components/landing/HeroSection";
import { ConduitSection } from "../components/landing/ConduitSection";
import { ForensicInspector } from "../components/landing/ForensicInspector";
import { CtaBanner } from "../components/landing/CtaBanner";

export function Landing() {
  const containerRef = useRef<HTMLElement>(null);
  const conduitRef = useRef<HTMLElement>(null);
  const inspectorRef = useRef<HTMLElement>(null);
  const leftPathRef = useRef<SVGPathElement>(null);
  const centerPathRef = useRef<SVGPathElement>(null);
  const rightPathRef = useRef<SVGPathElement>(null);
  const [hoveredCard, setHoveredCard] = useState<"text" | "image" | "video" | null>(null);

  useSeo({
    title: "SLOP — AI Content Detection & Digital Forensics",
    description:
      "Detect synthetic images, text, and video with neural consensus, SynthID watermark detection, and C2PA cryptographic provenance.",
    path: "/",
  });

  const {
    scrollProgress,
    inspectorProgress,
    activeLayer,
    ambientGlowY,
    handleSelectStage,
  } = useLandingScroll({
    conduitRef,
    inspectorRef,
    leftPathRef,
    centerPathRef,
    rightPathRef,
  });

  return (
    <main id="main-content" className="landing-page-wrapper" ref={containerRef}>
      <div className="openai-ambient-glow" style={{ transform: `translate(-50%, ${ambientGlowY}vh)` }} aria-hidden="true" />
      <div className="openai-subtle-mesh" aria-hidden="true" />

      <HeroSection />
      <ConduitSection
        conduitRef={conduitRef}
        leftPathRef={leftPathRef}
        centerPathRef={centerPathRef}
        rightPathRef={rightPathRef}
        hoveredCard={hoveredCard}
        setHoveredCard={setHoveredCard}
        scrollProgress={scrollProgress}
      />
      <ForensicInspector
        inspectorRef={inspectorRef}
        activeLayer={activeLayer}
        inspectorProgress={inspectorProgress}
        handleSelectStage={handleSelectStage}
      />
      <CtaBanner />
    </main>
  );
}
