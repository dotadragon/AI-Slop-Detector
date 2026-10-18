export interface SignalMeta {
  label: string;
  shortDesc: string;
  tooltip: string;
}

const SIGNAL_META: Record<string, SignalMeta> = {
  model_ai_probability: {
    label: "Primary Neural Score",
    shortDesc: "Deep vision/language backbone probability",
    tooltip: "Measures how strongly our primary neural network (Swin Vision Transformer or Fine-Tuned RoBERTa) matches patterns learned from thousands of confirmed AI generations.",
  },
  neural_ai_score: {
    label: "Neural Classifier Score",
    shortDesc: "Fine-tuned transformer probability",
    tooltip: "Direct probability output from our fine-tuned transformer model analyzing token sequences and structural syntax.",
  },
  multi_crop_ai_probability: {
    label: "Multi-Crop Patch Consensus",
    shortDesc: "5-region high-resolution patch scan",
    tooltip: "Inspects 5 zoomed-in regions of the image independently so fine-grained AI textures aren't lost when resizing.",
  },
  clip_probe_score: {
    label: "CLIP Semantic Probe",
    shortDesc: "Universal generative feature projection",
    tooltip: "Uses OpenAI's CLIP-ViT-L/14 feature space to detect semantic and structural artifacts shared across Midjourney, Flux, and DALL-E.",
  },
  binoculars_ai_prob: {
    label: "Binoculars Cross-Entropy",
    shortDesc: "Dual-Qwen2.5 observer/performer ratio",
    tooltip: "Compares how 'surprised' two language models are by the phrasing. AI text is unusually predictable to another LLM compared to human writing.",
  },
  fast_detectgpt_ai_prob: {
    label: "Fast-DetectGPT Curvature",
    shortDesc: "Token log-probability curvature",
    tooltip: "Checks whether word choices sit at the statistical peak of an AI generator's probability curve rather than natural human variation.",
  },
  heuristic_score: {
    label: "Physical & Statistical Heuristics",
    shortDesc: "Combined non-neural signal risk",
    tooltip: "Blends camera sensor physics, compression grid checks, and frequency patterns into an overall physical authenticity check.",
  },
  latent_grid_score: {
    label: "Diffusion Latent Grid",
    shortDesc: "8×8 VAE decoder upsampling periodicity",
    tooltip: "AI diffusion models generate images in compressed 8×8 pixel blocks. High scores mean we detected faint repeating grid seams left by the AI decoder.",
  },
  npr_score: {
    label: "Pixel Neighbor Correlation (NPR)",
    shortDesc: "Local pixel-to-pixel upsampling check",
    tooltip: "Measures how neighboring pixels relate to each other. AI upsamplers create smoother, more predictable pixel transitions than real optical lenses.",
  },
  cfa_score: {
    label: "Camera Sensor Grid (CFA)",
    shortDesc: "Physical Bayer color filter check",
    tooltip: "Real digital cameras capture light through a physical red-green-blue mosaic filter (Bayer pattern). High scores indicate this physical camera signature is missing.",
  },
  dct_block_score: {
    label: "JPEG Block Consistency (DCT)",
    shortDesc: "Quantization table & block alignment",
    tooltip: "Examines JPEG compression blocks to see if the file matches standard camera hardware encoding or synthetic software export.",
  },
  compression_artifact_score: {
    label: "Compression Divergence (ELA)",
    shortDesc: "Error level resave difference",
    tooltip: "Resaves the image at controlled quality to highlight areas where compression error is unnaturally uniform or digitally spliced.",
  },
  frequency_artifact_score: {
    label: "Spectral Harmonics (2D-FFT)",
    shortDesc: "High-frequency Fourier spectrum peaks",
    tooltip: "Converts the image into frequency waves. AI generators often leave geometric star-like spikes in high frequencies that real cameras never produce.",
  },
  noise_uniformity_score: {
    label: "Sensor Noise Uniformity",
    shortDesc: "Silicon grain vs. synthetic smoothness",
    tooltip: "Real camera sensors have tiny physical grain across shadows and highlights. AI images often have unnaturally smooth or uniform noise.",
  },
  metadata_risk: {
    label: "Container & EXIF Risk",
    shortDesc: "Camera hardware & header check",
    tooltip: "Checks for missing camera lens/shutter metadata or software headers commonly stripped or altered by AI generation tools.",
  },
  color_stability: {
    label: "Temporal Color Stability",
    shortDesc: "Frame-to-frame lighting consistency",
    tooltip: "Measures whether lighting and color histograms stay physically consistent across consecutive video frames.",
  },
  llm_pattern_score: {
    label: "LLM Phrasing Patterns",
    shortDesc: "Characteristic AI transition phrases",
    tooltip: "Scans for overused AI transition words, symmetrical bullet structures, and formulaic phrasing typical of chat models.",
  },
  stylometry_score: {
    label: "Stylometric Uniformity",
    shortDesc: "Sentence rhythm & burstiness check",
    tooltip: "Humans mix very short and very long sentences ('burstiness'). High scores mean sentence lengths and rhythm are robotically uniform.",
  },
  contraction_avoidance: {
    label: "Contraction Avoidance",
    shortDesc: "Formal 'do not' vs. natural 'don't'",
    tooltip: "AI models default to formal uncontracted phrasing ('cannot', 'it is') more often than casual human writers.",
  },
  vocabulary_richness: {
    label: "Lexical Predictability",
    shortDesc: "Word variety vs. repetitive tokens",
    tooltip: "Measures how predictable and repetitive the vocabulary choices are across paragraphs.",
  },
  hedging_ratio: {
    label: "Hedging Language",
    shortDesc: "Cautious disclaimer phrasing",
    tooltip: "Detects frequent AI hedging phrases like 'it is important to note', 'delve into', or 'on the other hand'.",
  },
  passive_voice_score: {
    label: "Passive Voice Density",
    shortDesc: "Impersonal grammatical structure",
    tooltip: "Measures how often sentences use detached passive constructions instead of direct active voice.",
  },
  zipf_deviation: {
    label: "Word Frequency Curve (Zipf)",
    shortDesc: "Deviation from natural human word law",
    tooltip: "Natural human language follows Zipf's power law for word frequency. AI sampling temperatures slightly flatten this curve.",
  },
  punctuation_entropy: {
    label: "Punctuation Rhythm",
    shortDesc: "Comma, dash & clause regularity",
    tooltip: "Checks if commas, em-dashes, and periods appear at overly regular, mechanical intervals.",
  },
  list_structure_score: {
    label: "Formulaic List Structure",
    shortDesc: "Symmetrical bullet/numbered patterns",
    tooltip: "Detects tidy 'Bold Header: Colon Explanation' list layouts heavily favored by instruction-tuned LLMs.",
  },
  sentence_start_entropy: {
    label: "Sentence Opening Variety",
    shortDesc: "Diversity of clause starters",
    tooltip: "Checks whether consecutive sentences repeatedly start with the same parts of speech or transition words.",
  },
  paragraph_uniformity: {
    label: "Paragraph Length Symmetry",
    shortDesc: "Evenly sized paragraph blocks",
    tooltip: "AI essays often produce paragraphs of nearly identical word counts, whereas human paragraphs vary widely.",
  },
  semantic_density_ratio: {
    label: "Information Density",
    shortDesc: "Content words vs. filler ratio",
    tooltip: "Measures the packing of nouns and verbs relative to connective filler words.",
  },
};
const SIGNAL_ORDER = Object.keys(SIGNAL_META);

export { SIGNAL_META, SIGNAL_ORDER };
