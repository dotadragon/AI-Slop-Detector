import { useEffect, useRef } from "react";
import { animate, stagger } from "animejs";

type ScannerSize = "sm" | "md" | "lg";
type AnimationHandle = { pause?: () => void };

interface ThreeDForensicScannerProps {
  size?: ScannerSize;
  interactive?: boolean;
}

function createCoreNodes(core: HTMLDivElement, size: ScannerSize): HTMLDivElement[] {
  const nodeCount = 12;
  core.innerHTML = "";
  const nodes: HTMLDivElement[] = [];
  const radius = size === "sm" ? 36 : size === "lg" ? 64 : 48;

  for (let index = 0; index < nodeCount; index += 1) {
    const node = document.createElement("div");
    node.className = "threed-core-node";
    const phiAngle = Math.acos(-1 + (2 * index) / nodeCount);
    const thetaAngle = Math.sqrt(nodeCount * Math.PI) * phiAngle;
    const coordX = radius * Math.sin(phiAngle) * Math.cos(thetaAngle);
    const coordY = radius * Math.sin(phiAngle) * Math.sin(thetaAngle);
    const coordZ = radius * Math.cos(phiAngle);

    node.style.transform = `translate3d(${coordX}px, ${coordY}px, ${coordZ}px)`;
    core.appendChild(node);
    nodes.push(node);
  }
  return nodes;
}

function initScannerAnimations(
  ring1: HTMLDivElement,
  ring2: HTMLDivElement,
  ring3: HTMLDivElement,
  laser: HTMLDivElement,
  nodes: HTMLDivElement[],
  size: ScannerSize
): AnimationHandle[] {
  const laserBounds = size === "sm" ? [-40, 40] : size === "lg" ? [-80, 80] : [-60, 60];
  try {
    return [
      animate(ring1, { rotateZ: [0, 360], duration: 16000, ease: "linear", loop: true }),
      animate(ring2, { rotateZ: [360, 0], duration: 12000, ease: "linear", loop: true }),
      animate(ring3, { rotateZ: [0, 360], duration: 20000, ease: "linear", loop: true }),
      animate(laser, { translateY: laserBounds, opacity: [0.3, 0.95, 0.3], duration: 2400, alternate: true, loop: true, ease: "inOutQuad" }),
      animate(nodes, { scale: [0.7, 1.3], opacity: [0.35, 1], duration: 1800, delay: stagger(100), alternate: true, loop: true, ease: "inOutSine" }),
    ];
  } catch {
    return [];
  }
}

function setupParallaxTilt(container: HTMLDivElement, gimbal: HTMLDivElement): () => void {
  const onMouseMove = (event: MouseEvent) => {
    const bounds = container.getBoundingClientRect();
    const centerX = bounds.left + bounds.width / 2;
    const centerY = bounds.top + bounds.height / 2;
    const deltaX = (event.clientX - centerX) / (bounds.width / 2);
    const deltaY = (event.clientY - centerY) / (bounds.height / 2);

    animate(gimbal, {
      rotateX: 25 - deltaY * 20,
      rotateY: -20 + deltaX * 25,
      duration: 300,
      ease: "outQuad",
    });
  };

  const onMouseLeave = () => {
    animate(gimbal, { rotateX: 25, rotateY: -20, duration: 600, ease: "outCubic" });
  };

  container.addEventListener("mousemove", onMouseMove);
  container.addEventListener("mouseleave", onMouseLeave);

  return () => {
    container.removeEventListener("mousemove", onMouseMove);
    container.removeEventListener("mouseleave", onMouseLeave);
  };
}

const DIMENSIONS: Record<ScannerSize, number> = {
  sm: 170,
  md: 220,
  lg: 280,
};

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
    const r1 = ring1Ref.current;
    const r2 = ring2Ref.current;
    const r3 = ring3Ref.current;
    const cr = coreRef.current;
    const lr = laserRef.current;
    if (!r1 || !r2 || !r3 || !cr || !lr) return;

    const nodes = createCoreNodes(cr, size);
    const animations = initScannerAnimations(r1, r2, r3, lr, nodes, size);

    let cleanupParallax: (() => void) | undefined;
    if (interactive && containerRef.current && gimbalRef.current) {
      cleanupParallax = setupParallaxTilt(containerRef.current, gimbalRef.current);
    }

    return () => {
      animations.forEach((handle) => handle.pause?.());
      cleanupParallax?.();
    };
  }, [size, interactive]);

  const dimension = DIMENSIONS[size] ?? 220;

  return (
    <div
      className={`threed-scanner-root threed-scanner-${size}`}
      ref={containerRef}
      style={{ width: dimension, height: dimension, perspective: "1000px" }}
      aria-hidden="true"
    >
      <div className="threed-gimbal" ref={gimbalRef}>
        <div className="threed-ring threed-ring-outer" ref={ring1Ref}>
          <span className="threed-ring-tag">2D-DCT</span>
        </div>
        <div className="threed-ring threed-ring-mid" ref={ring2Ref}>
          <span className="threed-ring-tag">SYNTH_ID</span>
        </div>
        <div className="threed-ring threed-ring-inner" ref={ring3Ref}>
          <span className="threed-ring-tag">ViT_L16</span>
        </div>
        <div className="threed-laser-plane" ref={laserRef} />
        <div className="threed-neural-core" ref={coreRef} />
        <div className="threed-core-pulsar" />
      </div>
    </div>
  );
}
