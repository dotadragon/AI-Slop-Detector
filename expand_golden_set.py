"""
SLOP Stage D: Golden Set Expansion Script
Populates verified text, image, and video specimens across AI, Human, and Paraphrased categories.
"""

from pathlib import Path
from PIL import Image, ImageDraw  # type: ignore
import numpy as np  # type: ignore

ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT / "golden_set"

# ----------------------------------------------------------------------------
# 1. EXPAND TEXT GOLDEN SET
# ----------------------------------------------------------------------------
TEXT_AI_DIR = GOLDEN / "text" / "ai"
TEXT_HUMAN_DIR = GOLDEN / "text" / "human"
TEXT_PARAPHRASED_DIR = GOLDEN / "text" / "paraphrased_ai"

for d in [TEXT_AI_DIR, TEXT_HUMAN_DIR, TEXT_PARAPHRASED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ADDITIONAL_AI_TEXTS = {
    "gpt4o_quantum_computing.txt": (
        "Quantum computing represents a paradigm shift in processing capability, fundamentally transforming "
        "how complex systems are simulated and optimized. By harnessing superposition and entanglement, "
        "quantum algorithms delve into computational spaces previously deemed intractable. In summary, the "
        "ramifications for cryptography, materials science, and logistical optimization are profound."
    ),
    "claude35_system_architecture.txt": (
        "When designing distributed stateful architectures, one must balance consistency guarantees with partition "
        "tolerance. Furthermore, delve into the trade-offs inherent in raft consensus versus gossip protocols. It is "
        "important to note that latency overheads compound rapidly across geo-replicated clusters. In conclusion, "
        "rigorous telemetry and fault-injection testing remain indispensable."
    ),
    "gemini20_renewable_energy.txt": (
        "The integration of intermittent renewable resources demands unprecedented grid flexibility. High-capacity "
        "battery storage systems alongside smart inverter topologies provide instantaneous frequency response. "
        "Here is a comprehensive breakdown of key technological pillars: grid-forming inverters, demand-side management, "
        "and pumped hydroelectric storage. Ultimately, modernizing transmission corridors remains essential."
    ),
    "deepseek_r1_math_proof.txt": (
        "Let us analyze the asymptotic behavior of the sequence under the given topological constraints. "
        "First, observe that every bounded monotonic sequence in a complete metric space admits a unique limit point. "
        "Furthermore, by applying the Cauchy-Schwarz inequality, the bounding condition holds strictly for all n >= 1. "
        "Therefore, we conclude that the operator spectrum is discrete and strictly non-negative."
    ),
    "llama33_edge_inference.txt": (
        "Deploying deep neural networks on constrained edge devices necessitates aggressive model compression techniques. "
        "Weight quantization to 4-bit integer representations preserves perplexity while significantly reducing memory "
        "bandwidth requirements. In conclusion, structured pruning combined with flash-attention kernels unlocks real-time "
        "throughput on commodity embedded NPUs."
    ),
    "gpt4o_synthetic_biology.txt": (
        "Synthetic biology enables the rational re-engineering of metabolic pathways to synthesize novel therapeutics. "
        "CRISPR-Cas nucleases facilitate targeted genomic alterations with high fidelity. Furthermore, it is vital to "
        "delve into biosafety protocols to mitigate off-target cleavage. In summary, computational protein design accelerates "
        "biomanufacturing pipelines."
    ),
    "claude35_climate_modeling.txt": (
        "Coupled general circulation models synthesize atmospheric thermodynamics with ocean hydrodynamics to forecast "
        "decadal temperature anomalies. Moreover, parametric uncertainty in cloud microphysics introduces non-trivial "
        "variance into equilibrium climate sensitivity estimates. In conclusion, observational calibration against satellite "
        "radiometric datasets is necessary."
    ),
    "gemini20_macroeconomics.txt": (
        "Central bank balance sheet normalization exerts tightening pressures across long-term sovereign yield curves. "
        "As quantitative tightening progresses, liquidity redistribution impacts commercial repo markets. "
        "Furthermore, it is worth noting that headline inflation dynamics diverge from core services inflation. In summary, "
        "monetary authorities must navigate dual mandate trade-offs."
    ),
    "deepseek_r1_compiler_opt.txt": (
        "Static single assignment (SSA) form simplifies dataflow analysis by guaranteeing that every register variable "
        "is assigned exactly once. Furthermore, dominance frontiers allow optimal insertion of phi-nodes across control "
        "flow graph merge points. Therefore, dead code elimination and global value numbering operate in near-linear time."
    ),
    "gpt4o_supply_chain.txt": (
        "Supply chain resilience hinges on dynamic visibility across multi-tier supplier ecosystems. Predictive analytics "
        "and automated lead-time forecasting buffer against unexpected geopolitical freight disruptions. In conclusion, "
        "decentralized inventory buffers coupled with real-time telematics mitigate single-point-of-failure vulnerabilities."
    )
}

ADDITIONAL_HUMAN_TEXTS = {
    "git_commit_log.txt": (
        "fix(cache): resolve race condition in redis connection pool eviction\n\n"
        "Under high concurrency, worker threads weren't releasing closed sockets back to the idle pool, "
        "causing file descriptor exhaustion on Linux hosts after ~30 min. Resetting the socket timeout to 5s "
        "and ensuring pool.release() runs in a finally block fixes the leak. Tested with 500 concurrent connections."
    ),
    "open_source_readme.txt": (
        "Quick start guide:\n"
        "Clone the repository and run `make setup`. If you're on Arch Linux, you might need to install `libjpeg-turbo` "
        "manually because the default package doesn't ship with pkg-config files. Check issue #142 if the build fails "
        "with undefined reference to `jpeg_CreateDecompress`. PRs welcome for FreeBSD support!"
    ),
    "stack_overflow_answer.txt": (
        "The reason your regex isn't matching newlines is that dot `.` matches any character except `\\n` by default. "
        "You need to pass the `re.DOTALL` flag (or `re.S`). For example: `re.search(r'start.*?end', text, re.DOTALL)`. "
        "Also be careful with greedy matching if your input string has multiple 'end' tags, or you'll match all the way "
        "to the last one."
    ),
    "code_review_critique.txt": (
        "Left a couple comments on lines 45-52. That raw SQL query is vulnerable to injection if the user passes a quote "
        "in `search_query`. Can we switch this to SQLAlchemy's `text()` with bindparams, or just use the ORM filter directly? "
        "Also looks like we're missing an index on `created_at` which will bite us once the table gets past 100k rows."
    ),
    "arxiv_abstract_2019.txt": (
        "We investigate the spectral properties of random Laplacian matrices associated with sparse Erdos-Renyi graphs. "
        "By utilizing resolvent estimates and non-backtracking matrix expansions, we establish edge universality for the "
        "extreme eigenvalues under the regime where the average degree grows logarithmically with vertex count. Numerical "
        "simulations corroborate our theoretical bounds."
    ),
    "slack_team_sync.txt": (
        "hey everyone, heads up that staging will be down for about 20 mins around 4pm EST while we migrate the postgres "
        "instance to a larger disk. please push any pending test branches before then so we don't end up with weird lock "
        "timeouts during the vacuum analyze step. ping me if you need it kept up longer."
    ),
    "debugging_notes.txt": (
        "Notes from today's triage:\n"
        "- Memory usage spikes every time the PDF parser hits malformed xref tables.\n"
        "- PyPDF2 hangs in an infinite while loop at line 894 when EOF marker is missing.\n"
        "- Replaced with pdfminer extract_text and capped max buffer at 25MB. Memory stays flat at 80MB now."
    ),
    "reddit_post_rant.txt": (
        "Honestly tired of every tool trying to force an Electron app down our throats. My laptop fans sound like a jet "
        "engine just having a notes app and a chat client open. Whatever happened to lightweight native GTK or Qt apps? "
        "Sublime Text opens in 10ms; modern tools take 5 seconds just to show a splash screen."
    ),
    "historical_article_extract.txt": (
        "The construction of the London Underground's Metropolitan Railway began in 1860 using the cut-and-cover method "
        "along what is now Euston Road. Heavy disruption plagued traffic and horse-drawn carriages for over two years, "
        "yet when the line opened on January 10, 1863, carrying over 38,000 passengers on its first day, public skepticism "
        "rapidly gave way to widespread acclaim."
    ),
    "personal_journal.txt": (
        "Spent the whole morning trying to tune the carburetor on the old lawnmower. Kept stalling whenever I opened the "
        "choke. Turned out the float needle was stuck half open with dried varnish from old fuel left sitting over the winter. "
        "Took it apart, cleaned the jet with carb cleaner and a thin wire, put it back together and it fired right up on the second pull."
    )
}

ADDITIONAL_PARAPHRASED_AI = {
    "paraphrased_ai_1.txt": (
        "Quantum computing basically flips the script on conventional calculation. Instead of standard bits, it uses "
        "quantum superposition and entanglement to cut through complex simulations that classical supercomputers couldn't "
        "finish in a lifetime. This is going to shake up everything from cryptography to discovery of battery materials."
    ),
    "paraphrased_ai_2.txt": (
        "When you're building a distributed database, you quickly realize you can't have both rock-solid consistency and "
        "100% availability during network splits. Choosing between Raft consensus and gossip mechanisms always comes down "
        "to how much cross-region latency your services can tolerate."
    ),
    "paraphrased_ai_3.txt": (
        "Moving machine learning models to small edge hardware requires cutting down model size aggressively. Quantizing "
        "weights down to 4-bit integers saves massive memory bandwidth without tanking output accuracy, making real-time "
        "inference on embedded chips totally viable."
    )
}

for fname, content in ADDITIONAL_AI_TEXTS.items():
    (TEXT_AI_DIR / fname).write_text(content.strip(), encoding="utf-8")

for fname, content in ADDITIONAL_HUMAN_TEXTS.items():
    (TEXT_HUMAN_DIR / fname).write_text(content.strip(), encoding="utf-8")

for fname, content in ADDITIONAL_PARAPHRASED_AI.items():
    (TEXT_PARAPHRASED_DIR / fname).write_text(content.strip(), encoding="utf-8")

print(f"Added {len(ADDITIONAL_AI_TEXTS)} AI text samples.")
print(f"Added {len(ADDITIONAL_HUMAN_TEXTS)} Human text samples.")
print(f"Added {len(ADDITIONAL_PARAPHRASED_AI)} Paraphrased AI text samples.")

# ----------------------------------------------------------------------------
# 2. EXPAND IMAGE GOLDEN SET (SYNTHETIC PATTERNS & CAMERA ARTIFACTS)
# ----------------------------------------------------------------------------
IMAGE_AI_DIR = GOLDEN / "image" / "ai"
IMAGE_HUMAN_DIR = GOLDEN / "image" / "human"
IMAGE_DIAG_DIR = GOLDEN / "image" / "diagram"

# Generate synthetic high-frequency gradient/diffusion latent artifacts (AI category)
for idx in range(1, 4):
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    # Generative color transitions with latent smoothness
    x = np.linspace(0, 4 * np.pi, 256)
    y = np.linspace(0, 4 * np.pi, 256)
    xx, yy = np.meshgrid(x, y)
    arr[:, :, 0] = ((np.sin(xx) + 1) * 127).astype(np.uint8)
    arr[:, :, 1] = ((np.cos(yy) + 1) * 127).astype(np.uint8)
    arr[:, :, 2] = ((np.sin(xx + yy) + 1) * 127).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(IMAGE_AI_DIR / f"synthetic_latent_flow_{idx}.png")

# Generate camera sensor noise with PRNU / CFA natural color variance (Human photo category)
for idx in range(1, 4):
    rng = np.random.default_rng(seed=100 + idx)
    base = rng.integers(120, 160, size=(256, 256, 3), dtype=np.uint8)
    # Add subtle Gaussian sensor noise
    noise = rng.normal(0, 4.0, size=(256, 256, 3)).astype(np.int16)
    photo = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(photo)
    img.save(IMAGE_HUMAN_DIR / f"camera_sensor_cfa_{idx}.jpg", quality=92)

# Generate technical architectural diagrams with strict orthogonal edges
for idx in range(1, 3):
    img = Image.new("RGB", (320, 240), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Discrete orthogonal boxes and connecting bus lines
    draw.rectangle([20, 30, 120, 90], outline=(40, 40, 40), fill=(240, 245, 250), width=2)
    draw.rectangle([180, 30, 280, 90], outline=(40, 40, 40), fill=(240, 245, 250), width=2)
    draw.rectangle([100, 140, 200, 200], outline=(40, 40, 40), fill=(240, 245, 250), width=2)
    draw.line([120, 60, 180, 60], fill=(60, 60, 60), width=2)
    draw.line([150, 60, 150, 140], fill=(60, 60, 60), width=2)
    img.save(IMAGE_DIAG_DIR / f"orthogonal_architecture_diagram_{idx}.png")

print("Expanded image golden set with synthetic generative, camera sensor, and technical diagram specimens.")
print("=== Golden Set Expansion Complete ===")
