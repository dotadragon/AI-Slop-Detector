import { Link } from "react-router-dom";
import { type RefObject } from "react";
import { ArrowUpRight } from "lucide-react";
import { track } from "../../utils/analytics";

interface ConduitSectionProps {
  conduitRef: RefObject<HTMLElement>;
  leftPathRef: RefObject<SVGPathElement>;
  centerPathRef: RefObject<SVGPathElement>;
  rightPathRef: RefObject<SVGPathElement>;
  hoveredCard: "text" | "image" | "video" | null;
  setHoveredCard: (card: "text" | "image" | "video" | null) => void;
  scrollProgress: number;
}

interface ModalityCardProps {
  modality: "text" | "image" | "video";
  hovered: boolean;
  onHover: (modality: "text" | "image" | "video" | null) => void;
  badgeClass: string;
  glyph: string;
  crcTag: string;
  metaBadge: string;
  title: string;
  description: string;
  latency: string;
}

function ModalityCard({
  modality,
  hovered,
  onHover,
  badgeClass,
  glyph,
  crcTag,
  metaBadge,
  title,
  description,
  latency,
}: ModalityCardProps) {
  return (
    <Link
      to={`/detector?tab=${modality}`}
      className={`conduit-card ${hovered ? "hovered" : ""}`}
      onMouseEnter={() => onHover(modality)}
      onMouseLeave={() => onHover(null)}
      onClick={() => track("cta_clicked", { section: "conduit", modality })}
      title={`Launch ${title} Detector`}
    >
      <div className="conduit-card-dev-header">
        <div className="conduit-dev-badge-group">
          <span className={`conduit-dev-badge ${badgeClass}`}>
            <span className="conduit-ping-dot active" />
            <span className="conduit-dev-glyph">{glyph}</span>
          </span>
          <span className="conduit-dev-bracket">[+]</span>
        </div>
        <span className="conduit-crc-tag">{crcTag}</span>
      </div>
      <div className="conduit-card-meta-line">
        <span className="conduit-card-badge">{metaBadge}</span>
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
      <div className="conduit-card-footer">
        <span className="conduit-card-link-text">
          Launch {title} <ArrowUpRight size={15} />
        </span>
        <span className="conduit-card-latency">{latency}</span>
      </div>
    </Link>
  );
}

function ConduitSvgDefs() {
  return (
    <defs>
      <filter id="glow-orange" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur1" />
        <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="blur2" />
        <feMerge>
          <feMergeNode in="blur2" />
          <feMergeNode in="blur1" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
      <linearGradient id="line-gradient" x1="0" y1="36" x2="0" y2="177" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stopColor="#ff4500" stopOpacity="0.95" />
        <stop offset="30%" stopColor="#ff6a00" stopOpacity="1" />
        <stop offset="65%" stopColor="#ff8c00" stopOpacity="1" />
        <stop offset="100%" stopColor="#ffaa00" stopOpacity="1" />
      </linearGradient>
      <linearGradient id="astra-orange-grad" x1="0" y1="0" x2="0" y2="36" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stopColor="#ff4500" stopOpacity="1" />
        <stop offset="100%" stopColor="#ff6a00" stopOpacity="1" />
      </linearGradient>
    </defs>
  );
}

function ConduitBaseTracks() {
  return (
    <>
      <line x1="500" y1="0" x2="500" y2="36" stroke="url(#astra-orange-grad)" strokeWidth="3" filter="url(#glow-orange)" />
      <path d="M 500 36 C 500 110, 166 90, 166 177" stroke="var(--border-hi, #1e293b)" strokeWidth="2" strokeDasharray="4 4" fill="none" />
      <path d="M 500 36 L 500 177" stroke="var(--border-hi, #1e293b)" strokeWidth="2" strokeDasharray="4 4" fill="none" />
      <path d="M 500 36 C 500 110, 834 90, 834 177" stroke="var(--border-hi, #1e293b)" strokeWidth="2" strokeDasharray="4 4" fill="none" />
    </>
  );
}

interface ConduitBranchPathsProps {
  leftPathRef: RefObject<SVGPathElement>;
  centerPathRef: RefObject<SVGPathElement>;
  rightPathRef: RefObject<SVGPathElement>;
  hoveredCard: "text" | "image" | "video" | null;
  scrollProgress: number;
}

function ConduitBranchPaths({
  leftPathRef,
  centerPathRef,
  rightPathRef,
  hoveredCard,
  scrollProgress,
}: ConduitBranchPathsProps) {
  const isLeft = hoveredCard === "text" || scrollProgress > 0.05;
  const isCenter = hoveredCard === "image" || scrollProgress > 0.05;
  const isRight = hoveredCard === "video" || scrollProgress > 0.05;
  return (
    <>
      <path ref={leftPathRef} d="M 500 36 C 500 110, 166 90, 166 177" className={`conduit-branch-path conduit-pulse-path ${isLeft ? "active" : ""}`} stroke="url(#line-gradient)" strokeWidth="2.5" filter="url(#glow-orange)" fill="none" />
      <path ref={centerPathRef} d="M 500 36 L 500 177" className={`conduit-branch-path conduit-pulse-path ${isCenter ? "active" : ""}`} stroke="url(#line-gradient)" strokeWidth="2.5" filter="url(#glow-orange)" fill="none" />
      <path ref={rightPathRef} d="M 500 36 C 500 110, 834 90, 834 177" className={`conduit-branch-path conduit-pulse-path ${isRight ? "active" : ""}`} stroke="url(#line-gradient)" strokeWidth="2.5" filter="url(#glow-orange)" fill="none" />
    </>
  );
}

function ConduitEnergyStreams({ scrollProgress }: { scrollProgress: number }) {
  const streamStyle = { opacity: scrollProgress > 0.05 ? 0.9 : 0, transition: "opacity 0.2s ease" };
  return (
    <>
      <path d="M 500 36 C 500 110, 166 90, 166 177" className="conduit-energy-stream" fill="none" style={streamStyle} />
      <path d="M 500 36 L 500 177" className="conduit-energy-stream" fill="none" style={streamStyle} />
      <path d="M 500 36 C 500 110, 834 90, 834 177" className="conduit-energy-stream" fill="none" style={streamStyle} />
    </>
  );
}

function ConduitTerminalNodes({ scrollProgress }: { scrollProgress: number }) {
  const nodeStyle = { opacity: scrollProgress >= 0.9 ? Math.min(1, (scrollProgress - 0.9) / 0.1) : 0, transition: "opacity 0.2s ease" };
  return (
    <>
      <circle cx="166" cy="177" r="5" fill="#ffaa00" stroke="#ff5500" strokeWidth="1.5" filter="url(#glow-orange)" style={nodeStyle} />
      <circle cx="500" cy="177" r="5" fill="#ffaa00" stroke="#ff5500" strokeWidth="1.5" filter="url(#glow-orange)" style={nodeStyle} />
      <circle cx="834" cy="177" r="5" fill="#ffaa00" stroke="#ff5500" strokeWidth="1.5" filter="url(#glow-orange)" style={nodeStyle} />
    </>
  );
}

function ConduitSplittingSvg({
  leftPathRef,
  centerPathRef,
  rightPathRef,
  hoveredCard,
  scrollProgress,
}: ConduitBranchPathsProps) {
  return (
    <svg className="conduit-svg-container" viewBox="0 0 1000 180" preserveAspectRatio="none" aria-hidden="true">
      <ConduitSvgDefs />
      <ConduitBaseTracks />
      <ConduitBranchPaths leftPathRef={leftPathRef} centerPathRef={centerPathRef} rightPathRef={rightPathRef} hoveredCard={hoveredCard} scrollProgress={scrollProgress} />
      <ConduitEnergyStreams scrollProgress={scrollProgress} />
      <ConduitTerminalNodes scrollProgress={scrollProgress} />
    </svg>
  );
}

export function ConduitSection({
  conduitRef,
  leftPathRef,
  centerPathRef,
  rightPathRef,
  hoveredCard,
  setHoveredCard,
  scrollProgress,
}: ConduitSectionProps) {
  return (
    <section className="conduit-section" ref={conduitRef} aria-label="Forensic Modality Conduits">
      <h2 className="sr-only">Forensic Modalities</h2>

      <div className="conduit-spine-stem">
        <div className={`conduit-reactor-node ${scrollProgress > 0.05 ? "active" : ""}`}>
          <div className="conduit-node-core" />
          <div className="conduit-node-ring" />
        </div>
      </div>

      <ConduitSplittingSvg
        leftPathRef={leftPathRef}
        centerPathRef={centerPathRef}
        rightPathRef={rightPathRef}
        hoveredCard={hoveredCard}
        scrollProgress={scrollProgress}
      />

      <div className="conduit-grid">
        <ModalityCard
          modality="text"
          hovered={hoveredCard === "text"}
          onHover={setHoveredCard}
          badgeClass="badge-orange"
          glyph="[0xTXT]"
          crcTag="CRC32: 0x8FA1"
          metaBadge="RoBERTa + Binoculars GBDT"
          title="Text Analysis"
          description="Fine-tuned RAID RoBERTa, dual-Qwen2.5 Binoculars cross-entropy, and sentence-level attribution."
          latency="87.5% BAcc"
        />

        <ModalityCard
          modality="image"
          hovered={hoveredCard === "image"}
          onHover={setHoveredCard}
          badgeClass="badge-cyan"
          glyph="[λ_RAW]"
          crcTag="C2PA: 0x3F1A"
          metaBadge="Swin + C2PA + PRNU"
          title="Image Analysis"
          description="Cryptographic C2PA provenance, CMOS sensor PRNU wavelets, 2D-FFT lattice peaks, and Swin GradCAM."
          latency="78.9% BAcc"
        />

        <ModalityCard
          modality="video"
          hovered={hoveredCard === "video"}
          onHover={setHoveredCard}
          badgeClass="badge-orange"
          glyph="[0xVID]"
          crcTag="CRC32: 0x9C4E"
          metaBadge="YuNet FFT + Optical Flow"
          title="Video Analysis"
          description="Farnebäck optical flow residuals, YuNet facial frequency roll-off, and keyframe consensus."
          latency="87.5% BAcc"
        />
      </div>
    </section>
  );
}
