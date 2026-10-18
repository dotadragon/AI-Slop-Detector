export function HeroSection() {
  return (
    <section className="hero-centered" aria-label="SLOP Forensics Platform">
      <div className="hero-badge-centered">
        <span className="conduit-ping-dot active" />
        <span>REAL-TIME AI CONTENT DETECTION</span>
      </div>

      <h1 className="hero-title-centered">
        Identify{" "}
        <span className="hero-highlight-text">
          artificially
          <br />
          generated
        </span>{" "}
        content<span className="hero-amber-dot">.</span>
      </h1>

      <p className="hero-subtitle-centered">
        Multi-modal forensic verification across text, image, and video. Built for engineering teams that
        require calibrated probabilities and auditable signal breakdowns.
      </p>
    </section>
  );
}
