import os

import gradio as gr
from transformers import pipeline

MODEL_ID = os.getenv("MODEL_ID", "LeoGR22/bertimbau-review-sentiment-ptbr")
LABELS_PT = {"negative": "Negativo", "neutral": "Neutro", "positive": "Positivo"}

classifier = pipeline("text-classification", model=MODEL_ID, top_k=None)


def predict(text: str) -> dict[str, float]:
    text = (text or "").strip()
    if not text:
        return {}
    scores = classifier(text, truncation=True, max_length=128)[0]  # same MAX_LENGTH as training
    return {LABELS_PT.get(s["label"], s["label"]): float(s["score"]) for s in scores}


EXAMPLES = [
    "Chegou antes do prazo, bem embalado e funciona perfeitamente. Recomendo!",
    "O produto é razoável, cumpre o básico, mas esperava um acabamento melhor pelo preço.",
    "Veio com defeito e até agora ninguém respondeu meu pedido de troca.",
    "Bonito, mas a bateria dura bem menos do que o anunciado.",
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, label="Avaliação do produto", placeholder="Escreva uma avaliação em português..."),
    outputs=gr.Label(num_top_classes=3, label="Sentimento"),
    examples=EXAMPLES,
    title="Sentimento de avaliações de e-commerce (PT-BR)",
    description=(
        f"BERTimbau ajustado em ~130 mil avaliações reais de e-commerce. "
        f"Modelo: [{MODEL_ID}](https://huggingface.co/{MODEL_ID})"
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
