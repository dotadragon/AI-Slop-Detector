import { Link } from "react-router-dom";
import { ArrowUpRight } from "lucide-react";
import { track } from "../../utils/analytics";

export function CtaBanner() {
  return (
    <section id="cta-forensics" className="telemetry-cta-section" aria-label="Start Content Analysis">
      <div className="cta-banner-box">
        <div className="cta-banner-glow" aria-hidden="true" />
        <div className="cta-banner-content">
          <span className="cta-banner-tag">PRODUCTION READY — TEXT, IMAGE &amp; VIDEO</span>
          <h2 className="cta-banner-heading">Deploy High-Precision AI Forensics to Your Workflow</h2>
          <p className="cta-banner-sub">
            Upload images, paste prose, or analyze video to evaluate synthetic probability with calibrated explainability and verifiable evidence dossiers.
          </p>
          <div className="cta-btn-group">
            <Link
              to="/detector?tab=image"
              className="cta-primary-btn"
              onClick={() => track("cta_clicked", { section: "bottom_banner", target: "image_detector" })}
            >
              <span>Launch Image Detector</span>
              <ArrowUpRight size={16} />
            </Link>
            <Link
              to="/detector?tab=text"
              className="cta-secondary-btn"
              onClick={() => track("cta_clicked", { section: "bottom_banner", target: "text_detector" })}
            >
              <span>Launch Text Detector</span>
              <ArrowUpRight size={16} />
            </Link>
            <Link
              to="/detector?tab=video"
              className="cta-secondary-btn"
              onClick={() => track("cta_clicked", { section: "bottom_banner", target: "video_detector" })}
            >
              <span>Launch Video Detector</span>
              <ArrowUpRight size={16} />
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
