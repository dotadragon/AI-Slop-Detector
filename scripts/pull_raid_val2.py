"""Rebuild the VAL split from English RAID train-split domains (books/news).

Fixes the language-shift bug: the previous val (RAID extra split) was mostly
Czech-language text, which the English detector was never meant to score.
Keeps the czech val as a separate out-of-distribution (OOD) evaluation set.

Data hygiene:
  - RAID *train* split only, attack=none, >=80 words.
  - Offsets used: 800k-1100k (books) and 1.4M-1.6M (news) — disjoint from
    train dir (12k-60k) and test (0-5k) by construction.
  - Content-dedup against raid_val_ood (the czech set) and itself.
"""
import json
import time
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_heldout import _fetch  # noqa: E402

STATE = Path("pull_val2_state.json")
HUMAN_V, V_PER = 30, 20
TARGETS = ["human", "gpt2", "llama-chat", "mpt", "mpt-chat", "mistral", "mistral-chat"]
PROBES = list(range(800_000, 1_100_000, 2_000)) + list(range(1_400_000, 1_620_000, 2_000))


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


def norm_key(t: str) -> str:
    return " ".join(t.split())[:200]


def load_texts(base: Path) -> set:
    keys = set()
    if base.exists():
        for f in base.rglob("*.txt"):
            keys.add(norm_key(f.read_text(encoding="utf-8")))
    return keys


def main():
    st = json.loads(STATE.read_text()) if STATE.exists() else {"probe": 0, "val": {}}
    val = defaultdict(list, {k: [tuple(x) for x in v] for k, v in st["val"].items()})
    seen = load_texts(Path("raid_val_ood/text")) | load_texts(Path("heldout_set/text"))
    for items in val.values():
        for t, _ in items:
            seen.add(norm_key(t))

    start_probe = st.get("probe", 0)
    for probe_idx, offset in enumerate(PROBES):
        if probe_idx < start_probe:
            continue
        if all(len(val[m]) >= (HUMAN_V if m == "human" else V_PER) for m in TARGETS):
            print("all buckets full")
            break
        d = try_fetch(f"https://datasets-server.huggingface.co/rows?dataset=liamdugan%2Fraid&config=raid&split=train&offset={offset}&length=100")
        if d is None:
            print(f"fetch failed @{offset}; stopping (re-run to resume)", flush=True)
            break
        for r in [x["row"] for x in d.get("rows", [])]:
            m = (r.get("model") or "").lower()
            t = r.get("generation") or ""
            if (r.get("attack") or "none") != "none" or len(t.split()) < 80:
                continue
            if m not in TARGETS:
                continue
            k = norm_key(t)
            if k in seen:
                continue
            cap = HUMAN_V if m == "human" else V_PER
            if len(val[m]) < cap:
                val[m].append([t, r.get("domain") or "unknown"])
                seen.add(k)
        STATE.write_text(json.dumps({"probe": probe_idx + 1, "val": dict(val)}))
        if probe_idx % 10 == 0:
            print(f"@{offset}: val={ {kk: len(vv) for kk, vv in sorted(val.items())} }", flush=True)
        time.sleep(3)

    done = all(len(val[m]) >= (HUMAN_V if m == "human" else V_PER) for m in TARGETS)
    print("COMPLETE" if done else "INCOMPLETE — re-run to resume")

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


if __name__ == "__main__":
    main()
