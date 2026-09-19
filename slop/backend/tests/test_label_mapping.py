"""
G2: Label mapping and non-inversion verification test.
Ensures id2label resolves the correct class for AI/Fake across models.
"""

def test_label_resolution_logic():
    # Synthetic id2label dictionary matching typical HuggingFace models
    roberta_labels = {0: "LABEL_0", 1: "Fake"}
    swin_labels = {0: "real", 1: "artificial"}
    vit_labels = {0: "Real", 1: "Fake"}

    def resolve_fake_idx(labels_dict):
        lowered = {int(k): v.lower() for k, v in labels_dict.items()}
        return next((i for i, l in lowered.items() if any(t in l for t in ("fake", "ai", "artificial", "synthetic", "chatgpt"))), 0)

    assert resolve_fake_idx(roberta_labels) == 1
    assert resolve_fake_idx(swin_labels) == 1
    assert resolve_fake_idx(vit_labels) == 1
