from pathlib import Path

import nbformat as nbf

CELLS: list[tuple[str, str, str]] = []


def md(cell_id: str, text: str) -> None:
    CELLS.append((cell_id, "markdown", text.strip()))


def code(cell_id: str, text: str) -> None:
    CELLS.append((cell_id, "code", text.strip()))


md("title", """
# Sentiment analysis of Brazilian e-commerce reviews

Fine-tuning **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) to classify product reviews
in Brazilian Portuguese as **negative / neutral / positive**, compared against a
**TF-IDF + Logistic Regression** baseline.

- **Data:** [B2W-Reviews01](https://huggingface.co/datasets/ruanchaves/b2w-reviews01) — ~130k real reviews from a large Brazilian marketplace (CC BY 4.0).
- **Labels:** derived from the 1–5 star rating: 1–2 → negative, 3 → neutral, 4–5 → positive.
- **Main metric:** macro-F1, because the classes are imbalanced and the neutral class is the hard one.

> Runtime: **GPU (T4)** — *Runtime → Change runtime type → T4 GPU*. Full run takes roughly 30–40 minutes.
""")

code("install", """
!pip install -q "transformers==5.22.2" "datasets==5.2.0" "accelerate==1.15.0" "huggingface_hub==2.4.0" "scikit-learn>=1.5"
""")

md("config-md", "## 1. Configuration")

code("config", """
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
HUB_MODEL_ID = "LeoGR22/bertimbau-review-sentiment-ptbr"
PUSH_TO_HUB = True

SEED = 42
MAX_LENGTH = 128
EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
SAMPLE_SIZE = None
""")

code("imports", """
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
""")

md("data-md", """
## 2. Data

Loads the dataset from the Hugging Face Hub, falling back to the parquet conversion and then to
the original CSV published by the dataset authors.
""")

code("load-data", """
def load_raw() -> pd.DataFrame:
    from datasets import load_dataset

    attempts = [
        lambda: load_dataset("ruanchaves/b2w-reviews01", split="train").to_pandas(),
        lambda: load_dataset("ruanchaves/b2w-reviews01", revision="refs/convert/parquet", split="train").to_pandas(),
        lambda: pd.read_csv(
            "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv",
            low_memory=False,
        ),
    ]
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            print("Source failed:", type(exc).__name__, str(exc)[:120])
    raise RuntimeError("Could not load the dataset from any source")


raw = load_raw()
print(raw.shape)
raw[["review_title", "review_text", "overall_rating"]].head()
""")

code("prepare-data", """
def rating_to_label(rating: int) -> str:
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


df = raw[["review_text", "overall_rating"]].copy()
df["text"] = df["review_text"].astype(str).str.strip()
df = df[raw["review_text"].notna() & (df["text"].str.len() >= 3)]
df = df.drop_duplicates(subset="text")
df["label"] = df["overall_rating"].astype(int).map(rating_to_label)
df = df[["text", "label"]].reset_index(drop=True)

if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
    df, _ = train_test_split(df, train_size=SAMPLE_SIZE, stratify=df["label"], random_state=SEED)
    df = df.reset_index(drop=True)

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED)

print(f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
dist = df["label"].value_counts(normalize=True).reindex(LABELS)
dist
""")

code("plot-dist", """
ax = dist.plot.bar(rot=0, title="Label distribution", ylabel="share of reviews")
ax.bar_label(ax.containers[0], fmt="{:.1%}")
plt.tight_layout()
plt.show()
""")

md("baseline-md", """
## 3. Baseline: TF-IDF + Logistic Regression

A strong classical baseline. If the transformer doesn't beat this by a clear margin,
it isn't worth the extra cost.
""")

code("baseline", """
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

baseline = LogisticRegression(max_iter=2000, class_weight="balanced")
baseline.fit(X_train, train_df["label"])
baseline_pred = baseline.predict(X_test)

baseline_metrics = {
    "accuracy": accuracy_score(test_df["label"], baseline_pred),
    "macro_f1": f1_score(test_df["label"], baseline_pred, average="macro"),
}
print(classification_report(test_df["label"], baseline_pred, labels=LABELS, digits=3))
""")

md("bert-md", "## 4. Fine-tuning BERTimbau")

code("tokenize", """
from datasets import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

length_sample = train_df["text"].sample(min(5000, len(train_df)), random_state=SEED).tolist()
lengths = [len(ids) for ids in tokenizer(length_sample)["input_ids"]]
print(f"Share of reviews within {MAX_LENGTH} tokens: {np.mean(np.array(lengths) <= MAX_LENGTH):.1%}")


def to_dataset(frame: pd.DataFrame) -> Dataset:
    ds = Dataset.from_dict({"text": frame["text"].tolist(), "label": frame["label"].map(LABEL2ID).tolist()})
    return ds.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH), batched=True)


train_ds, val_ds, test_ds = (to_dataset(f) for f in (train_df, val_df, test_df))
""")

code("model", """
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
)

# Inverse square-root frequency: plain inverse frequency over-weights the small neutral class.
counts = train_df["label"].value_counts().reindex(LABELS).to_numpy()
class_weights = torch.tensor((counts.sum() / counts) ** 0.5, dtype=torch.float)
class_weights = class_weights / class_weights.mean()
print(dict(zip(LABELS, class_weights.round(decimals=2).tolist())))


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "macro_f1": f1_score(labels, preds, average="macro")}


args = TrainingArguments(
    output_dir="outputs",
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    weight_decay=0.01,
    warmup_steps=0.06,  # values below 1 are read as a fraction of total steps
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    logging_steps=200,
    report_to="none",
    seed=SEED,
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)
""")

code("train", "trainer.train()")

md("eval-md", "## 5. Evaluation on the held-out test set")

code("evaluate", """
test_output = trainer.predict(test_ds)
bert_pred = [ID2LABEL[i] for i in np.argmax(test_output.predictions, axis=-1)]

bert_metrics = {
    "accuracy": accuracy_score(test_df["label"], bert_pred),
    "macro_f1": f1_score(test_df["label"], bert_pred, average="macro"),
}
print(classification_report(test_df["label"], bert_pred, labels=LABELS, digits=3))

per_class = {
    name: f1_score(test_df["label"], pred, labels=LABELS, average=None)
    for name, pred in (("TF-IDF + LogReg", baseline_pred), ("BERTimbau", bert_pred))
}
comparison = pd.DataFrame({
    "accuracy": [baseline_metrics["accuracy"], bert_metrics["accuracy"]],
    "macro_f1": [baseline_metrics["macro_f1"], bert_metrics["macro_f1"]],
    **{f"f1_{label}": [per_class["TF-IDF + LogReg"][i], per_class["BERTimbau"][i]] for i, label in enumerate(LABELS)},
}, index=["TF-IDF + LogReg", "BERTimbau"]).round(3)
comparison
""")

code("confusion", """
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, pred) in zip(axes, (("TF-IDF + LogReg", baseline_pred), ("BERTimbau", bert_pred))):
    ConfusionMatrixDisplay.from_predictions(
        test_df["label"], pred, labels=LABELS, normalize="true", values_format=".2f", ax=ax, colorbar=False
    )
    ax.set_title(name)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
""")

md("errors-md", """
### Error analysis

Where does the model still fail? Mostly on the neutral class and on reviews whose
text contradicts the star rating, a known source of label noise when deriving
sentiment from ratings.
""")

code("errors", """
errors = test_df.assign(pred=bert_pred).query("label != pred")
print(f"{len(errors):,} errors out of {len(test_df):,}")
pd.set_option("display.max_colwidth", 160)
errors.sample(min(12, len(errors)), random_state=SEED)
""")

code("save-metrics", """
results = {
    "model": MODEL_NAME,
    "dataset": "ruanchaves/b2w-reviews01",
    "test_size": len(test_df),
    "baseline": {k: round(v, 4) for k, v in baseline_metrics.items()},
    "bertimbau": {k: round(v, 4) for k, v in bert_metrics.items()},
    "per_class_f1": comparison[[f"f1_{label}" for label in LABELS]].to_dict(orient="index"),
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
""")

md("hub-md", """
## 6. Publish to the Hugging Face Hub

Creates the model repository with weights, tokenizer and a model card containing the
results above. You need a token with **write** access (huggingface.co/settings/tokens).
""")

code("push", """
if PUSH_TO_HUB:
    from huggingface_hub import ModelCard, notebook_login

    notebook_login()

    trainer.model.push_to_hub(HUB_MODEL_ID)
    tokenizer.push_to_hub(HUB_MODEL_ID)

    rows = "\\n".join(
        f"| {name} | {row.accuracy:.3f} | {row.macro_f1:.3f} | {row.f1_negative:.3f} | {row.f1_neutral:.3f} | {row.f1_positive:.3f} |"
        for name, row in comparison.iterrows()
    )
    card = ModelCard(f\"\"\"---
language: pt
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: {MODEL_NAME}
datasets:
- ruanchaves/b2w-reviews01
metrics:
- f1
- accuracy
tags:
- sentiment-analysis
- portuguese
- e-commerce
---

# BERTimbau — sentiment of Brazilian e-commerce reviews

`{MODEL_NAME}` fine-tuned to classify product reviews in Brazilian Portuguese as
**negative**, **neutral** or **positive**.

## Results (held-out test set, {len(test_df):,} reviews)

| Model | Accuracy | Macro-F1 | F1 negative | F1 neutral | F1 positive |
|---|---:|---:|---:|---:|---:|
{{rows}}

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="{HUB_MODEL_ID}")
clf("Chegou antes do prazo e funciona muito bem!")
```

## Training

- Data: B2W-Reviews01 (CC BY 4.0), labels from star rating (1–2 negative, 3 neutral, 4–5 positive), deduplicated, stratified 80/10/10 split.
- {EPOCHS} epochs, lr {LEARNING_RATE}, batch {BATCH_SIZE}, max length {MAX_LENGTH}, class-weighted cross-entropy (inverse square-root frequency), best epoch selected by validation macro-F1.

## Limitations

- Labels come from star ratings, so reviews whose text contradicts the rating add noise, especially in the neutral class.
- Trained on product reviews; performance on other domains (e.g. food delivery, support tickets) is not measured.
\"\"\".replace("{rows}", rows))
    card.push_to_hub(HUB_MODEL_ID)
    print(f"https://huggingface.co/{HUB_MODEL_ID}")
""")


def build(path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata.update({
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    })
    for cell_id, kind, source in CELLS:
        cell = nbf.v4.new_markdown_cell(source) if kind == "markdown" else nbf.v4.new_code_cell(source)
        cell.id = cell_id
        nb.cells.append(cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print(f"Wrote {path} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build(Path(__file__).parent / "notebooks" / "train_sentiment.ipynb")
