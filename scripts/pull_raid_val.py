"""Resumable val-split puller from the RAID *extra* split (disjoint from train).

Targets attack=none blocks at known offsets; safe to re-run (skips full buckets).
"""
import json
import time
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_heldout import _fetch  # noqa: E402

STATE = Path("pull_val_state.json")
HUMAN_V, V_PER = 25, 15
# (offset, want-human?) blocks probed in the extra split
BLOCKS = [
    (0, True), (1000, True), (2000, True), (5000, False), (10000, False),
    (15000, False), (20000, False), (30000, False), (33000, False),
    (40000, False), (50000, False), (53000, False), (60000, False),
    (25000, False),  # gpt2/german domain
]
TARGETS = ["human", "gpt2", "llama-chat", "mpt", "mpt-chat", "mistral-chat", "mistral"]


def try_fetch(url, tries=3):
    for a in range(tries):
        try:
            return json.loads(_fetch(url))
        except Exception as e:
            if "429" in str(e):
                time.sleep(30 * (a + 1))
            else:
                time.sleep(5)
    return None


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"val": {}}


def main():
    st = load_state()
    val = defaultdict(list, {k: [tuple(x) for x in v] for k, v in st["val"].items()})

    for offset, want_human in BLOCKS:
        key = "human" if want_human else None
        if want_human and len(val["human"]) >= HUMAN_V:
            continue
        if not want_human:
            # skip model blocks already full for all models
            if all(len(val[m]) >= V_PER for m in TARGETS if m != "human"):
                continue
        d = try_fetch(f"https://datasets-server.huggingface.co/rows?dataset=liamdugan%2Fraid&config=raid&split=extra&offset={offset}&length=100")
        if d is None:
            print(f"fetch failed at {offset}; will retry next run", flush=True)
            continue
        for r in [x["row"] for x in d.get("rows", [])]:
            m = (r.get("model") or "").lower()
            t = r.get("generation") or ""
            if (r.get("attack") or "none") != "none" or len(t.split()) < 80:
                continue
            if m not in TARGETS:
                continue
            cap = HUMAN_V if m == "human" else V_PER
            if len(val[m]) < cap:
                val[m].append([t, r.get("domain") or "unknown"])
        STATE.write_text(json.dumps({"val": dict(val)}))
        print(f"@{offset}: val={ {k: len(v) for k, v in sorted(val.items())} }", flush=True)
        time.sleep(4)

    done = all(len(val[m]) >= (HUMAN_V if m == "human" else V_PER) for m in TARGETS)
    if not done:
        print("INCOMPLETE — re-run to continue", flush=True)

    def persist(bucket, base):
        n = 0
        doms = defaultdict(int)
        for model, items in bucket.items():
            label = "human" if model == "human" else "ai"
            safe = "".join(c if c.isalnum() else "_" for c in model)[:40]
            target = base / label / safe
            target.mkdir(parents=True, exist_ok=True)
            for i, (t, dom) in enumerate(items):
                (target / f"{safe}_{i:04d}.txt").write_text(t, encoding="utf-8")
                doms[(safe, dom)] += 1
            n += len(items)
        print(f"  persisted {n} to {base}; domains={dict(doms)}")

    persist({k: [tuple(x) for x in v] for k, v in val.items()}, Path("raid_val/text"))
    print("val split persisted to raid_val/")


if __name__ == "__main__":
    main()
