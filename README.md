
# Customer Support Sentiment & Intent Tracker

A Streamlit application built around the supplied Google Colab notebook:
**Multi-Task Learning for Joint Dialogue Act and Emotion Classification**

## What the supplied Colab model does

The notebook uses:
- `distilbert-base-uncased`
- maximum sequence length: 64
- a shared DistilBERT encoder
- dropout 0.3
- 4 dialogue-act classes:
  Inform, Question, Directive, Commissive
- 7 emotion classes:
  Neutral, Anger, Disgust, Fear, Happiness, Sadness, Surprise
- joint loss = dialogue classification loss + emotion classification loss

The notebook saves:
`model.pt`, tokenizer files, `labels.json`, `metrics.json`, and `config.json`.

## IMPORTANT: install your trained model

From Colab, run the notebook's final ZIP creation/download cells.

Extract the downloaded `multitask_model.zip`.

Copy its CONTENTS into:

`models/multitask_model/`

The final folder should look like:

models/
└── multitask_model/
    ├── model.pt
    ├── config.json
    ├── labels.json
    ├── metrics.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── special_tokens_map.json
    └── vocab.txt

Do not put another `multitask_model` folder inside it.

## Run in VS Code

Open a terminal in this project folder:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Demo mode

If `model.pt` is not installed, the app still opens and demonstrates the UI using simple fallback predictions. Message Analyzer clearly warns that it is demo mode.

For your final project, install your actual Colab model so all predictions come from your trained joint-task DistilBERT.

## Pages

1. Dashboard
2. Live Customer Support
3. Message Analyzer
4. Conversation Analysis
5. Sentiment Dashboard
6. Intent Dashboard
7. Priority & Escalation
8. Conversation History
9. Model Performance

The numbered Streamlit pages are 8 modules plus `app.py` as the home dashboard.
