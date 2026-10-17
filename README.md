AI-Slop is a robust video and text detection system for authenticating user-generated content.

## Current Detection Pipeline

### Text Detection

The text classifier uses a production-ready backend pipeline with the following components:
- **Multimodal Fusion**: Combines multiple signal types for comprehensive detection.
- **Per-Sample Analysis**: Extracts 14 features for each text sample.
- **Cross-Validated Regularization**: Uses 5-fold cross-validation to tune L2 regularization strength (`alpha` parameter).
- **Strict Promotion Gate**: Requires minimum balanced accuracy (0.75), macro-F1 (0.70), and hard human recall floor (0.80) to prevent false positives.
- **Real-Time Perplexity**: Uses Intel's OpenVINO for fast perplexity scoring.
- **OOD Robustness**: Checks for "casual text" and "short text" to avoid overfitting to dataset styles.
- **Tier-0 Ready**: Fully functional for production deployment.

### Video Detection

The video classifier uses a multi-stage pipeline to detect generative video and manipulated content:

#### 1. Preprocessing
- **Frame Extraction**: Samples 20 frames for analysis.
- **Scene Detection**: Identifies screen recordings (UI content) separately from photographic scenes.
- **Face Detection**: Uses MediaPipe to find faces for specialized analysis.
- **SynthID Embedding**: Generates cryptographic watermarks for forgery detection.

#### 2. Scene-Based Analysis
- **Deepfake Detection**: Uses a specialized DeepFake ViT model to analyze each frame for manipulation artifacts.
- **Generative Video Detection**: Uses a Swin Transformer to detect AI-generated videos (diffusion models).
- **Temporal Analysis**: Analyzes frame-to-frame consistency using:
    - **Temporal Flicker**: Detects high-frequency luminance fluctuations.
    - **Temporal Jitter**: Detects camera motion artifacts.
    - **Noise Consistency**: Ensures statistical consistency across frames.
- **Artifact Detection**: Analyzes frame metadata for:
    - **Compression Artifacts**: Detects video compression patterns.
    - **Metadata Anomalies**: Checks for unusual file properties.

#### 3. Face-Based Analysis
- **Deepfake Face Detection**: Analyzes each detected face for manipulation artifacts using DeepFake ViT.
- **Face Swap Detection**: Uses specialized metrics to detect face swaps and blending inconsistencies.
- **Facial Feature Consistency**: Ensures consistent facial characteristics across frames.

#### 4. Fusion Engine
- **Rule-Based Decision Logic**: Combines all signals using a hierarchical scoring system that prioritizes high-threat indicators:
    - **Generative Video Suppression**: If video is classified as generative (e.g., diffusion model), AI score is elevated.
    - **Deepfake Face Suppression**: Face manipulations (including face swaps) are treated as high-threat signals.
    - **Screen Recording Handling**: Authentic UI screen recordings are suppressed to avoid false positives.
    - **Natural Video Correction**: Authentic photographic footage is corrected to reduce false positives.
    - **SynthID Corroboration**: Cryptographic watermarks provide strong evidence of manipulation.

#### 5. Output
- **Final AI Score**: Aggregated score between 0.00 and 1.00.
- **Confidence Band**: Categorical confidence level (low, moderate, high, very high).
- **Confidence Range**: Margin of error for the score.
- **Primary Signal**: Identifies the main detection mechanism (e.g., generative video, deepfake face, SynthID).
- **Contributory Signals**: Detailed breakdown of all contributing factors.
- **Per-Frame Metrics**: Comprehensive data for each analyzed frame.

## Prerequisites

### System Dependencies
- Python 3.8+
- Linux environment (Ubuntu recommended)
- GPU recommended for video processing (MediaPipe requires OpenGL 3.2+)

### Python Packages
```bash
pip install
    numpy
    scikit-learn
    pandas
    torch>=2.0.0
    torchvision
    transformers>=4.40.0
    accelerate>=0.25.0
    mediapipe==0.10.9
    diffusers>=0.20.0
    opencv-python-headless
    PyYAML>=6.0.0
    tqdm
    sentence-transformers>=2.2.0
    onnxruntime>=1.16.0
    tensorrt>=8.6.1
```

## Project Structure

```
AI-Slop/
├── scripts//
│   ├── train_text_fusion.py          # Text classifier training
│   ├── retrain_text_fusion.py          # Production-ready text retraining
│   ├── eval_accuracy.py              # Model evaluation
│   ├── eval_gate.py                  # CI evaluation gate
│   ├── retrain_video_classifier.py     # Video classifier training
│   └── run_video_ai_detection.py       # Video analysis inference
├── slop/backend/
│   ├── app/ml/
│   │   ├── fusion.py                 # Detection fusion logic
│   │   ├── text_detector.py            # Text classification
│   │   ├── video_detector.py           # Video classification
│   │   └── ...                       # Other ML modules
│   └── config/                       # Configuration files
└── tests//
    └── ...                           # Test files
```

## Quick Start

### Training

#### Text Classifier
```bash
python scripts/retrain_text_fusion.py
```

#### Video Classifier
```bash
python scripts/retrain_video_classifier.py
```

### Evaluation

#### Evaluate Model Accuracy
```bash
python scripts/eval_accuracy.py --split val
```

#### Run CI Gate (Automatic)
```bash
python scripts/eval_gate.py
```

### Inference

#### Run Video AI Detection
```bash
python scripts/run_video_ai_detection.py \
    --video_path /path/to/video.mp4 \
    --output_path /path/to/output.json
```

## Model Selection

The system supports several detection models that can be selected via configuration:

### Text Models
- **Neural Prob**: Transformer-based text classifier
- **Heuristic Score**: Rule-based text scoring
- **LLM Pattern Score**: Pattern matching for AI-generated text
- **Perplexity**: Language model perplexity analysis

### Video Models
- **DeepFake ViT**: For detecting manipulated faces
- **Swin Transformer**: For detecting generative video
- **VideoMAE**: Pre-trained video analysis

## Configuration

Model configurations are managed in `slop/backend/config/`. Key parameters include:
- Model paths and thresholds
- Split ratios (train/val/test)
- Regularization strengths
- Confidence thresholds

## Development

### Training Data
- **Text**: `raid_train/text`, `raid_val/text`, `heldout_set/text`
- **Video**: `raid_train/video`, `raid_val/video`, `heldout_set/video`

### Evaluation
- Use `eval_accuracy.py` for detailed metrics
- Use `eval_gate.py` for CI gate checks
- Maintain human recall >= 0.80 for text classifier

### Troubleshooting
- GPU memory errors: Reduce batch size, use mixed precision
- Model not loading: Verify paths and installations
- False positives: Adjust confidence thresholds in `slop/backend/config/`

## License

Proprietary. All rights reserved.

## Support

For issues or questions, please contact the development team.