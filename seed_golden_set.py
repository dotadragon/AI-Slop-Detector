import os
import shutil
from pathlib import Path

root = Path(r"c:\Users\karth\Desktop\AISlop")
golden_dir = root / "golden_set"

dirs = [
    golden_dir / "text" / "ai",
    golden_dir / "text" / "human",
    golden_dir / "image" / "ai",
    golden_dir / "image" / "human",
    golden_dir / "image" / "diagram",
    golden_dir / "video" / "ai",
    golden_dir / "video" / "human_screen",
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)

# 1. Seed Images
user_up = Path(r"C:\Users\karth\.gemini\antigravity-ide\brain\ebe44b8f-a3cf-489f-a0c5-840c1ee9d007\.user_uploaded")
if user_up.exists():
    for f in ["media_1789755920467.png", "media_1789757571085.jpg", "media_1789757572032.jpg", "media_1789757573221.jpg"]:
        src = user_up / f
        if src.exists():
            shutil.copy2(src, golden_dir / "image" / "ai" / f)
    if (user_up / "media_1789756044612.jpg").exists():
        shutil.copy2(user_up / "media_1789756044612.jpg", golden_dir / "image" / "human" / "media_1789756044612.jpg")
    if (user_up / "media_1789756069231.png").exists():
        shutil.copy2(user_up / "media_1789756069231.png", golden_dir / "image" / "diagram" / "media_1789756069231.png")

# 2. Seed Videos
downloads = Path(r"C:\Users\karth\Downloads")
if (downloads / "test.mp4").exists():
    shutil.copy2(downloads / "test.mp4", golden_dir / "video" / "ai" / "test.mp4")
if (downloads / "appeal.mp4").exists():
    shutil.copy2(downloads / "appeal.mp4", golden_dir / "video" / "human_screen" / "appeal.mp4")

# 3. Seed Curated Text Samples
ai_texts = [
    ("gpt4o_overview.txt", "Furthermore, it is crucial to recognize that artificial intelligence plays a pivotal role in shaping modern scientific discovery. In conclusion, the holistic integration of generative workflows underscores a profound paradigm shift across multiple domains."),
    ("claude_delve.txt", "Delving into the intricate tapestry of distributed systems reveals a nuanced balance between consistency and availability. It is worth noting that each architecture presents distinct trade-offs that warrant careful examination."),
    ("gemini_breakdown.txt", "Here is a comprehensive breakdown of the core tenets governing quantum computing. In essence, quantum superposition facilitates exponential parallel state evaluation, fundamentally transforming cryptographic algorithms."),
    ("deepseek_reasoning.txt", "To solve this problem rigorously, we must first establish the foundational mathematical principles underpinning dynamic programming. It bears mentioning that the optimal substructure enables memoization of overlapping subproblems."),
    ("chatgpt_essay.txt", "Throughout human history, technological advancement has served as a powerful catalyst for economic transformation. Moreover, by fostering innovative solutions to complex societal challenges, modern computational frameworks continue to redefine productivity.")
]

for fname, text in ai_texts:
    (golden_dir / "text" / "ai" / fname).write_text(text, encoding="utf-8")

human_texts = [
    ("casual_chat.txt", "hey man, i just checked the logs and the server looks fine now. not really sure what happened earlier with the redis connection, but a simple restart fixed it. let me know if it crashes again."),
    ("dev_notes.txt", "debugging note: make sure to check line 45 before running migration. the old column name was user_id not account_id so alembic will throw an error if you do not rename it first."),
    ("reddit_comment.txt", "Honestly I tried using both keyboards for a few months and the tactile switches just felt way better for typing all day. The linear ones were too light and I kept getting accidental double taps."),
    ("quick_update.txt", "went to the store to grab some groceries, will be back in like twenty minutes. leave the keys on the counter if you head out before i get home."),
    ("code_review_comment.txt", "looks good to me overall, just one small nit on the error handling in auth.py where we should probably return 401 instead of 403. other than that feel free to merge.")
]

for fname, text in human_texts:
    (golden_dir / "text" / "human" / fname).write_text(text, encoding="utf-8")

print("Golden set successfully created and seeded!")
