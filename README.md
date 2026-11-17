# Review Sentiment PT-BR

Fine-tuning **BERTimbau** to classify Brazilian Portuguese e-commerce reviews as **negative / neutral / positive**, benchmarked against a TF-IDF + Logistic Regression baseline, published on the Hugging Face Hub with a live demo.

| | |
|---|---|
| 🤗 **Model** | [LeoGR22/bertimbau-review-sentiment-ptbr](https://huggingface.co/LeoGR22/bertimbau-review-sentiment-ptbr) |
| 🎮 **Demo** | [Hugging Face Space](https://huggingface.co/spaces/LeoGR22/review-sentiment-ptbr) |
| 📓 **Notebook** | [`notebooks/train_sentiment.ipynb`](notebooks/train_sentiment.ipynb) · [Open in Colab](https://colab.research.google.com/github/LeoGR22/review-sentiment-ptbr/blob/main/notebooks/train_sentiment.ipynb) |

## Results

Held-out test set (10% of the data, stratified, never seen during training or model selection).

<!-- Fill in from results.json after running the notebook -->
| Model | Accuracy | Macro-F1 | F1 negative | F1 neutral | F1 positive |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LogReg | – | – | – | – | – |
| BERTimbau (fine-tuned) | – | – | – | – | – |

![Confusion matrices](assets/confusion_matrix.png)

## Approach

- **Data:** [B2W-Reviews01](https://huggingface.co/datasets/ruanchaves/b2w-reviews01), ~130k real reviews from a large Brazilian marketplace (CC BY 4.0). Labels come from the star rating: 1–2 → negative, 3 → neutral, 4–5 → positive. Duplicated texts are removed before the 80/10/10 stratified split, so no review leaks between splits.
- **Baseline first:** TF-IDF (unigrams + bigrams) with class-balanced Logistic Regression. The transformer has to beat it by a clear margin to justify its cost.
- **Model:** `neuralmind/bert-base-portuguese-cased`, 2 epochs, lr 2e-5, max length 128 tokens (covers ~95% of reviews), mixed precision on a free Colab T4.
- **Imbalance:** neutral reviews are a small minority. The loss uses class weights based on inverse *square-root* frequency, a middle ground between ignoring the imbalance and over-correcting it.
- **Model selection and metric:** best epoch picked by validation **macro-F1**, which weighs the three classes equally. Accuracy alone would reward a model that just predicts "positive".
- **Error analysis:** most remaining errors are neutral reviews and reviews whose text contradicts the star rating, which is label noise inherent to rating-derived labels.

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="LeoGR22/bertimbau-review-sentiment-ptbr")
clf("Chegou antes do prazo e funciona muito bem!")
# [{'label': 'positive', 'score': 0.98}]
```

## Reproducing

1. Open the notebook in Colab and select a **T4 GPU** runtime.
2. *Runtime → Run all.* The notebook trains both models, evaluates them, saves `results.json` and `confusion_matrix.png`, and pushes the model and its model card to the Hub (asks for a write token).
3. Deploy the demo: create a Gradio Space and upload the files in [`space/`](space).

The notebook is generated from [`build_notebook.py`](build_notebook.py), so changes are reviewable as plain Python. [`tests/smoke_notebook.py`](tests/smoke_notebook.py) runs every cell offline on CPU with a tiny random BERT and synthetic data to check the pipeline end to end:

```bash
pip install -r requirements-dev.txt
python build_notebook.py
python tests/smoke_notebook.py
```

## Project structure

```
build_notebook.py               notebook source (cells as code)
notebooks/train_sentiment.ipynb data prep, baseline, fine-tuning, evaluation, Hub upload
space/                          Gradio demo for Hugging Face Spaces
tests/smoke_notebook.py         offline end-to-end check of the notebook
assets/                         figures used in this README
```

## Stack

Python · PyTorch · Hugging Face Transformers / Datasets / Hub · scikit-learn · Gradio · Google Colab
