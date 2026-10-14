"""Resumable bulk RAID puller — safe to re-run; picks up where it left off."""
import json
import time
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_heldout import _fetch  # noqa: E402

STATE = Path("pull_state.json")
T_PER, HUMAN_T = 60, 120
V_PER, HUMAN_V = 15, 25
VAL_START = 140_000
END = 260_000


def try_fetch(url, tries=3):
    for a in range(tries):
        try:
            return json.loads(_fetch(url))
        except Exception as e:
            if "429" in str(e):
                time.sleep(35 * (a + 1))
            else:
                time.sleep(5)
    return None


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"offset": 12_000, "train": {}, "val": {}}


def main():
    st = load_state()
    offset = st["offset"]
    train = defaultdict(list, {k: [tuple(x) for x in v] for k, v in st["train"].items()})
    val = defaultdict(list, {k: [tuple(x) for x in v] for k, v in st["val"].items()})

    while offset < END:
        # Stop early if everything is full
        full = (
            all(len(train.get(m, [])) >= (HUMAN_T if m == "human" else T_PER) for m in
                ["human", "gpt2", "llama-chat", "mpt", "mpt-chat", "mistral-chat"])
            and all(len(val.get(m, [])) >= (HUMAN_V if m == "human" else V_PER) for m in
                    ["human", "gpt2", "llama-chat", "mpt", "mpt-chat", "mistral-chat"])
        )
        if full:
            print("all buckets full, stopping scan")
            break

        d = try_fetch(f"https://datasets-server.huggingface.co/rows?dataset=liamdugan%2Fraid&config=raid&split=train&offset={offset}&length=100")
        if d is None:
            print(f"fetch failed at {offset}, saving state and exiting")
            break
        rows = [r["row"] for r in d.get("rows", [])]
        if not rows:
            print("EOF")
            break
        for r in rows:
            m = (r.get("model") or "").lower()
            t = r.get("generation") or ""
            atk = (r.get("attack") or "none")
            dom = r.get("domain") or "unknown"
            if atk != "none" or len(t.split()) < 80:
                continue
            in_val = offset >= VAL_START
            if m == "human":
                bucket, cap = (val, HUMAN_V) if in_val else (train, HUMAN_T)
                key = "human"
            else:
                bucket, cap = (val, V_PER) if in_val else (train, T_PER)
                key = m
            if len(bucket[key]) < cap:
                bucket[key].append([t, dom])
        offset += 100
        if offset % 2000 == 0:
            st = {"offset": offset, "train": dict(train), "val": dict(val)}
            STATE.write_text(json.dumps(st))
            print(f"@{offset}: train={ {k: len(v) for k, v in sorted(train.items())} }", flush=True)

    # Persist to disk
    def persist(bucket, base):
        n = 0
        doms_total = defaultdict(int)
        for model, items in bucket.items():
            label = "human" if model == "human" else "ai"
            safe = "".join(c if c.isalnum() else "_" for c in model)[:40]
            target = base / label / safe
            target.mkdir(parents=True, exist_ok=True)
            existing = len(list(target.glob("*.txt")))
            for i, (t, dom) in enumerate(items):
                (target / f"{safe}_{i:04d}.txt").write_text(t, encoding="utf-8")
                doms_total[dom] += 1
            n += len(items)
        print(f"  persisted {n} to {base}; domains={dict(doms_total)}")

    persist({k: [tuple(x) for x in v] for k, v in train.items()}, Path("raid_train/text"))
    persist({k: [tuple(x) for x in v] for k, v in val.items()}, Path("raid_val/text"))
    st = {"offset": offset, "train": dict(train), "val": dict(val)}
    STATE.write_text(json.dumps(st))
    print(f"state saved at offset {offset}")


if __name__ == "__main__":
    main()
