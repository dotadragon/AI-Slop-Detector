import { useEffect, useRef } from "react";
import { animate, stagger } from "animejs";

interface ThreeDForensicScannerProps {
  size?: "sm" | "md" | "lg";
  interactive?: boolean;
}

export function ThreeDForensicScanner({
  size = "md",
  interactive = true,
}: ThreeDForensicScannerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const gimbalRef = useRef<HTMLDivElement>(null);
  const ring1Ref = useRef<HTMLDivElement>(null);
  const ring2Ref = useRef<HTMLDivElement>(null);
  const ring3Ref = useRef<HTMLDivElement>(null);
  const coreRef = useRef<HTMLDivElement>(null);
  const laserRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ring1 = ring1Ref.current;
    const ring2 = ring2Ref.current;
    const ring3 = ring3Ref.current;
    const laser = laserRef.current;
    const core = coreRef.current;
    if (!ring1 || !ring2 || !ring3 || !core || !laser) return;

    // Generate 3D neural nodes inside the core
    const nodeCount = 12;
    core.innerHTML = "";
    const nodes: HTMLDivElement[] = [];
    for (let i = 0; i < nodeCount; i++) {
      const node = document.createElement("div");
      node.className = "threed-core-node";
      const phi = Math.acos(-1 + (2 * i) / nodeCount);
      const theta = Math.sqrt(nodeCount * Math.PI) * phi;
      const r = size === "sm" ? 36 : size === "lg" ? 64 : 48;
      const x = r * Math.sin(phi) * Math.cos(theta);
      const y = r * Math.sin(phi) * Math.sin(theta);
      const z = r * Math.cos(phi);

      node.style.transform = `translate3d(${x}px, ${y}px, ${z}px)`;
      core.appendChild(node);
      nodes.push(node);
    }

    // Anime.js v4: Continuous 3D Ring Rotations
    let animRing1: any;
    let animRing2: any;
    let animRing3: any;
    let animLaser: any;
    let animNodes: any;

    try {
      animRing1 = animate(ring1, {
        rotateZ: [0, 360],
        duration: 16000,
        ease: "linear",
        loop: true,
      });

      animRing2 = animate(ring2, {
        rotateZ: [360, 0],
        duration: 12000,
        ease: "linear",
        loop: true,
      });

      animRing3 = animate(ring3, {
        rotateZ: [0, 360],
        duration: 20000,
        ease: "linear",
        loop: true,
      });

      // 3D Laser Scan Sweeper
      animLaser = animate(laser, {
        translateY: size === "sm" ? [-40, 40] : size === "lg" ? [-80, 80] : [-60, 60],
        opacity: [0.3, 0.95, 0.3],
        duration: 2400,
        alternate: true,
        loop: true,
        ease: "inOutQuad",
      });

      // Depth pulsation of nodes
      animNodes = animate(nodes, {
        scale: [0.7, 1.3],
        opacity: [0.35, 1],
        duration: 1800,
        delay: stagger(100),
        alternate: true,
        loop: true,
        ease: "inOutSine",
      });
    } catch (err) {
      console.warn("Anime.js 3D animation initialization fallback:", err);
    }

    // Interactive mouse tilt parallax
    const container = containerRef.current;
    let mouseMoveHandler: ((e: MouseEvent) => void) | undefined;
    let mouseLeaveHandler: (() => void) | undefined;

    if (interactive && container) {
      mouseMoveHandler = (e: MouseEvent) => {
        if (!gimbalRef.current) return;
        const rect = container.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        const deltaX = (e.clientX - centerX) / (rect.width / 2);
        const deltaY = (e.clientY - centerY) / (rect.height / 2);

        // Smooth tilt using Anime.js v4
        animate(gimbalRef.current, {
          rotateX: 25 - deltaY * 20,
          rotateY: -20 + deltaX * 25,
          duration: 300,
          ease: "outQuad",
        });
      };

      mouseLeaveHandler = () => {
        if (!gimbalRef.current) return;
        animate(gimbalRef.current, {
          rotateX: 25,
          rotateY: -20,
          duration: 600,
          ease: "outCubic",
        });
      };

      container.addEventListener("mousemove", mouseMoveHandler);
      container.addEventListener("mouseleave", mouseLeaveHandler);
    }

    return () => {
      if (animRing1?.pause) animRing1.pause();
      if (animRing2?.pause) animRing2.pause();
      if (animRing3?.pause) animRing3.pause();
      if (animLaser?.pause) animLaser.pause();
      if (animNodes?.pause) animNodes.pause();

      if (container && mouseMoveHandler && mouseLeaveHandler) {
        container.removeEventListener("mousemove", mouseMoveHandler);
        container.removeEventListener("mouseleave", mouseLeaveHandler);
      }
    };
  }, [size, interactive]);

  const dimension = size === "sm" ? 170 : size === "lg" ? 280 : 220;

  return (
    <div
      className={`threed-scanner-root threed-scanner-${size}`}
      ref={containerRef}
      style={{
        width: dimension,
        height: dimension,
        perspective: "1000px",
      }}
      aria-hidden="true"
    >
      <div className="threed-gimbal" ref={gimbalRef}>
        {/* Orbital Ring 1: Frequency domain 3D tilt */}
        <div className="threed-ring threed-ring-outer" ref={ring1Ref}>
          <span className="threed-ring-tag">2D-DCT</span>
        </div>

        {/* Orbital Ring 2: SynthID Watermark plane */}
        <div className="threed-ring threed-ring-mid" ref={ring2Ref}>
          <span className="threed-ring-tag">SYNTH_ID</span>
        </div>

        {/* Orbital Ring 3: ViT Neural perimeter */}
        <div className="threed-ring threed-ring-inner" ref={ring3Ref}>
          <span className="threed-ring-tag">ViT_L16</span>
        </div>

        {/* Sweeping Laser Plane */}
        <div className="threed-laser-plane" ref={laserRef} />

        {/* 3D Core with animated neural vertex nodes */}
        <div className="threed-neural-core" ref={coreRef} />

        {/* Central glowing pulsar */}
        <div className="threed-core-pulsar" />
      </div>
    </div>
  );
}
