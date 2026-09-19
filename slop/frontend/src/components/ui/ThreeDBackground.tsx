import { useEffect, useRef } from "react";
import { animate, stagger } from "animejs";

export function ThreeDBackground() {
  const containerRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!gridRef.current) return;

    // Create 3D floating particles
    const particleCount = 28;
    const grid = gridRef.current;
    grid.innerHTML = "";

    const particles: HTMLDivElement[] = [];
    for (let i = 0; i < particleCount; i++) {
      const p = document.createElement("div");
      p.className = "threed-particle";
      p.style.left = `${Math.random() * 100}%`;
      p.style.top = `${Math.random() * 100}%`;
      grid.appendChild(p);
      particles.push(p);
    }

    // Anime.js v4 3D float animation
    let floatAnim: any;
    try {
      floatAnim = animate(particles, {
        translateZ: () => Math.random() * 350 - 150,
        translateX: () => Math.random() * 80 - 40,
        translateY: () => Math.random() * 80 - 40,
        scale: () => Math.random() * 1.2 + 0.6,
        opacity: () => Math.random() * 0.5 + 0.2,
        ease: "inOutSine",
        duration: () => Math.random() * 3000 + 3000,
        delay: stagger(120),
        loop: true,
        alternate: true,
      });
    } catch {
      // Graceful fallback if Anime.js animation setup is skipped
    }

    // Scroll-linked 3D parallax
    const handleScroll = () => {
      const scrolled = window.scrollY;
      if (gridRef.current) {
        gridRef.current.style.transform = `perspective(1000px) rotateX(${62 - scrolled * 0.02}deg) rotateZ(${-18 + scrolled * 0.01}deg) translateY(${scrolled * 0.3}px)`;
      }
    };

    window.addEventListener("scroll", handleScroll, { passive: true });

    return () => {
      if (floatAnim?.pause) floatAnim.pause();
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  return (
    <div className="threed-bg-container" ref={containerRef} aria-hidden="true">
      <div className="threed-grid" ref={gridRef}>
        <div className="threed-plane" />
      </div>
      <div className="threed-glow-radial" />
    </div>
  );
}
