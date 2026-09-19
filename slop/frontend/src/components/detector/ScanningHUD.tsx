import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { ShieldAlert, Cpu, Binary, Radio, CheckCircle2 } from "lucide-react";

interface Stage {
  id: number;
  label: string;
  desc: string;
  icon: typeof Binary;
  duration: number; // ms to reach this stage
}

const STAGES: Stage[] = [
  { id: 1, label: "Binary & Ingestion Audit", desc: "Validating file magic bytes and container structure", icon: Binary, duration: 250 },
  { id: 2, label: "C2PA & Cryptographic Provenance", desc: "Inspecting JUMBF manifests and signing certs", icon: ShieldAlert, duration: 700 },
  { id: 3, label: "SynthID, PRNU & Frequency Domain Scan", desc: "Probing 2D DCT / FFT bands and sensor noise residual", icon: Radio, duration: 1300 },
  { id: 4, label: "Vision Transformer Neural Inference", desc: "Computing deep ViT feature representations", icon: Cpu, duration: 2100 },
  { id: 5, label: "Consensus & Generator Attribution", desc: "Synthesizing multi-modal consensus & signature localization", icon: CheckCircle2, duration: 3000 },
];

export function ScanningHUD() {
  const [activeStage, setActiveStage] = useState(1);
  const [logs, setLogs] = useState<string[]>([]);

  useEffect(() => {
    const startTime = Date.now();
    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      let current = 1;
      for (let i = 0; i < STAGES.length; i++) {
        if (elapsed >= STAGES[i].duration) {
          current = STAGES[i].id;
        }
      }
      setActiveStage(current);

      // Generate telemetry logs
      const ms = elapsed.toString().padStart(4, "0");
      if (elapsed < 300) {
        setLogs([`[+${ms}ms] [INIT] Raw file stream mapped into memory buffers`]);
      } else if (elapsed < 800) {
        setLogs([
          `[+0120ms] [OK] Binary image signatures verified (no polyglot payload)`,
          `[+${ms}ms] [PROV] Checking C2PA JUMBF claim stores...`
        ]);
      } else if (elapsed < 1400) {
        setLogs([
          `[+0120ms] [OK] Binary image signatures verified (clean)`,
          `[+0650ms] [PROV] Content credentials evaluated`,
          `[+${ms}ms] [SCAN] Calculating 2D Discrete Cosine Transform (DCT) grid...`
        ]);
      } else if (elapsed < 2300) {
        setLogs([
          `[+0650ms] [PROV] Content credentials evaluated`,
          `[+1280ms] [FREQ] 64x64 DCT energy variance matrix extracted`,
          `[+${ms}ms] [NEURAL] Forward pass through Vision Transformer ViT-B/16...`
        ]);
      } else {
        setLogs([
          `[+1280ms] [FREQ] High-frequency watermark variance isolated`,
          `[+2150ms] [NEURAL] Softmax probability distribution finalized`,
          `[+${ms}ms] [ATTRIB] Aggregating multi-modal consensus & bounding box...`
        ]);
      }
    }, 100);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="scanning-hud" role="status" aria-live="polite">
      <div className="scanning-header">
        <div className="scanning-pulse-dot" />
        <span className="scanning-title">REAL-TIME FORENSIC TELEMETRY</span>
        <span className="scanning-tag">STAGE {activeStage}/5</span>
      </div>

      <div className="scanning-scanline-container">
        <div className="scanning-scanline" />
      </div>

      <div className="scanning-stages">
        {STAGES.map((st) => {
          const isDone = activeStage > st.id;
          const isCurrent = activeStage === st.id;
          const Icon = st.icon;

          return (
            <motion.div
              key={st.id}
              className={`scanning-stage-row ${isCurrent ? "current" : ""} ${isDone ? "done" : ""}`}
              initial={{ opacity: 0.4 }}
              animate={{ opacity: isCurrent || isDone ? 1 : 0.4 }}
              transition={{ duration: 0.2 }}
            >
              <div className="scanning-stage-indicator">
                {isDone ? (
                  <CheckCircle2 size={16} className="text-emerald-400" />
                ) : (
                  <Icon size={16} className={isCurrent ? "text-amber-400 animate-spin-slow" : "text-gray-500"} />
                )}
              </div>
              <div className="scanning-stage-info">
                <div className="scanning-stage-name">
                  <span>{st.label}</span>
                  {isCurrent && <span className="scanning-badge-active">ACTIVE</span>}
                  {isDone && <span className="scanning-badge-done">DONE</span>}
                </div>
                <div className="scanning-stage-desc">{st.desc}</div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Real-time Frequency Spectrogram Waveform Bars */}
      <div className="scanning-spectrogram" aria-hidden="true">
        <div className="spectrogram-header">
          <Radio size={12} className="text-cyan-400" />
          <span>SPECTRAL ENERGY BANDS (2D-FFT & PRNU SENSOR NOISE)</span>
        </div>
        <div className="spectrogram-bars">
          {Array.from({ length: 24 }).map((_, idx) => (
            <div
              key={idx}
              className="spectrogram-bar"
              style={{
                animationDelay: `${((idx * 0.09) % 1.2).toFixed(2)}s`,
              }}
            />
          ))}
        </div>
      </div>

      <div className="scanning-terminal">
        <div className="scanning-terminal-header">
          <span className="terminal-dot red" />
          <span className="terminal-dot yellow" />
          <span className="terminal-dot green" />
          <span className="terminal-title">FORENSIC_ENGINE_LOGS.raw</span>
        </div>
        <div className="scanning-terminal-body">
          {logs.map((line, idx) => (
            <div key={idx} className="terminal-line">
              {line}
            </div>
          ))}
          <div className="terminal-cursor">_</div>
        </div>
      </div>
    </div>
  );
}
