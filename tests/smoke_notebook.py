from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # before the notebook imports pyplot, so plt.show() doesn't open windows

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "train_sentiment.ipynb"

POS = ["ótimo produto chegou rápido", "excelente qualidade recomendo", "adorei funciona muito bem"]
NEU = ["produto ok nada demais", "razoável pelo preço", "cumpre o que promete mas demorou"]
NEG = ["péssimo não funciona", "veio quebrado horrível", "nunca chegou quero reembolso"]


def make_tiny_model(path: Path) -> None:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=False)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(
        POS + NEU + NEG,
        trainers.WordPieceTrainer(vocab_size=300, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]),
    )
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]",
    )
    fast.save_pretrained(path)
    config = BertConfig(vocab_size=fast.vocab_size, hidden_size=32, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=64)
    BertModel(config).save_pretrained(path)


def fake_raw(n: int = 900) -> pd.DataFrame:
    rng = random.Random(0)
    rows = []
    for i in range(n):
        rating = rng.choice([1, 2, 3, 4, 5, 5, 5, 4])
        pool = NEG if rating <= 2 else NEU if rating == 3 else POS
        # The suffix keeps texts unique; otherwise drop_duplicates collapses the dataset to 9 rows.
        rows.append({"review_title": "t", "review_text": f"{rng.choice(pool)} {i}", "overall_rating": rating})
    rows.append({"review_title": "t", "review_text": None, "overall_rating": 5})
    return pd.DataFrame(rows)


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    os.chdir(tmp)
    model_dir = tmp / "tiny-bert"
    make_tiny_model(model_dir)

    pushed: dict[str, object] = {}

    import huggingface_hub
    import transformers

    # Stub every Hub call so the push cell runs offline and we can inspect what it would upload.
    huggingface_hub.notebook_login = lambda: None
    huggingface_hub.ModelCard.push_to_hub = lambda self, repo_id, **kw: pushed.update(card=str(self), repo=repo_id)
    transformers.PreTrainedModel.push_to_hub = lambda self, repo_id, **kw: pushed.update(model=repo_id)
    transformers.PreTrainedTokenizerBase.push_to_hub = lambda self, repo_id, **kw: pushed.update(tokenizer=repo_id)

    overrides = {
        "install": "",
        "config": (
            f'MODEL_NAME = r"{model_dir}"\nHUB_MODEL_ID = "test/tiny"\nPUSH_TO_HUB = True\n'
            "SEED = 42\nMAX_LENGTH = 32\nEPOCHS = 1\nBATCH_SIZE = 16\nLEARNING_RATE = 5e-4\nSAMPLE_SIZE = 800\n"
        ),
        "load-data": "raw = fake_raw()",
    }

    nb = nbformat.read(NOTEBOOK, as_version=4)
    namespace: dict[str, object] = {"fake_raw": fake_raw, "__name__": "__main__"}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        source = overrides.get(cell.id, cell.source)
        print(f"--- running cell [{cell.id}]")
        exec(compile(source, f"<cell {cell.id}>", "exec"), namespace)

    results = namespace["results"]
    assert set(results["baseline"]) == {"accuracy", "macro_f1"}
    assert 0 <= results["bertimbau"]["macro_f1"] <= 1
    assert (tmp / "confusion_matrix.png").exists() and (tmp / "results.json").exists()
    assert pushed["model"] == pushed["tokenizer"] == pushed["repo"] == "test/tiny"
    card = pushed["card"]
    assert "| BERTimbau |" in card and "| TF-IDF + LogReg |" in card and "{rows}" not in card
    print("\nSMOKE TEST PASSED")
    print(card[card.index("## Results"):card.index("## Usage")])


if __name__ == "__main__":
    main()
