export interface SpectralLayer {
  id: string;
  number: string;
  title: string;
  tag: string;
  channel: string;
  detectorModality: "image" | "text" | "video";
  metrics: { label: string; val: string; status: "good" | "amber" | "cyan" }[];
  description: string;
  telemetryLog: string[];
}

const SPECTRAL_LAYERS: SpectralLayer[] = [
  {
    id: "spatial",
    number: "01",
    title: "Spatial & Cryptographic Provenance",
    tag: "C2PA JUMBF + EXIF",
    channel: "CH-01: RGB PIXEL DENSITY",
    detectorModality: "image",
    description:
      "Validates CMOS hardware sensor tags, lens focal characteristics, and authenticates cryptographically signed C2PA manifests.",
    metrics: [
      { label: "C2PA Signature", val: "SHA-256 Validated", status: "good" },
      { label: "EXIF Sensor Model", val: "CMOS Hardware Verified", status: "good" },
      { label: "Bayer CFA Pattern", val: "2×2 Mosaic Present", status: "good" },
    ],
    telemetryLog: [
      "[IO] Inspecting C2PA JUMBF boxes (urn:c2pa:signature:x509)",
      "[EXIF] Camera Make & Model matches silicon serial fingerprint",
    ],
  },
  {
    id: "prnu",
    number: "02",
    title: "PRNU Sensor Noise Residual",
    tag: "Silicon Imperfection Map",
    channel: "CH-02: SENSOR FINGERPRINT",
    detectorModality: "image",
    description:
      "Extracts Photo-Response Non-Uniformity sensor noise residuals to distinguish physical silicon imperfections from diffusion generators.",
    metrics: [
      { label: "Sensor Noise Floor", val: "0.0024 σ² (Physical)", status: "good" },
      { label: "Diffusion Latent Residual", val: "< 0.0003 (Suppressed)", status: "cyan" },
      { label: "Silicon Correlation", val: "Verified", status: "good" },
    ],
    telemetryLog: [
      "[PRNU] High-pass wavelets extracted 4-channel noise residual",
      "[CMOS] Real camera sensor noise signature verified",
    ],
  },
  {
    id: "fft",
    number: "03",
    title: "2D-FFT Spatial Frequency Harmonics",
    tag: "Azimuthal Lattice Power",
    channel: "CH-03: SPECTRAL FOURIER",
    detectorModality: "image",
    description:
      "Computes 2D Fast Fourier Transform power spectrums to isolate periodic lattice spikes and azimuthal energy imbalances.",
    metrics: [
      { label: "Lattice Periodic Peak", val: "0.04 (No Grid Spike)", status: "good" },
      { label: "High-Freq Kurtosis", val: "2.38 (Natural Decay)", status: "good" },
      { label: "Azimuthal Energy Delta", val: "1.12 dB (Symmetric)", status: "cyan" },
    ],
    telemetryLog: [
      "[FFT] Shifted 2D-FFT log-magnitude spectrum computed (512×512)",
      "[RADIAL] Azimuthal angle integration shows 1/f power law distribution",
    ],
  },
  {
    id: "gradcam",
    number: "04",
    title: "Vision Transformer GradCAM Salience",
    tag: "Multi-Crop Attention",
    channel: "CH-04: NEURAL SALIENCE",
    detectorModality: "image",
    description:
      "Propagates gradients through Swin-Base and ViT attention stages to construct explainable forensic salience heatmaps.",
    metrics: [
      { label: "Consensus Model Score", val: "P95 Ensemble < 0.08", status: "good" },
      { label: "Attention Entropy", val: "0.89 (Diffuse Natural)", status: "good" },
      { label: "SynthID Marker Check", val: "No Embedded Watermark", status: "good" },
    ],
    telemetryLog: [
      "[GRADCAM] Cross-attention activation map synthesized via Swin stage 4",
      "[ENSEMBLE] 5-crop geometric variance evaluated across frame quadrants",
    ],
  },
];

export { SPECTRAL_LAYERS };
