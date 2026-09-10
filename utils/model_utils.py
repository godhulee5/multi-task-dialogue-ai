from pathlib import Path
import json

import streamlit as st
import torch
import torch.nn as nn

from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models" / "multitask_model"


# ============================================================
# DEFAULT LABELS
# ============================================================

DEFAULT_DIALOGUE = {
    0: "Inform",
    1: "Question",
    2: "Directive",
    3: "Commissive"
}

DEFAULT_EMOTION = {
    0: "Neutral",
    1: "Anger",
    2: "Disgust",
    3: "Fear",
    4: "Happiness",
    5: "Sadness",
    6: "Surprise"
}


# ============================================================
# MULTI-TASK CONFIGURATION
# ============================================================

class MultiTaskConfig(PretrainedConfig):

    model_type = "multitask-distilbert"

    def __init__(
        self,
        num_dialogue_labels=4,
        num_emotion_labels=7,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.num_dialogue_labels = num_dialogue_labels
        self.num_emotion_labels = num_emotion_labels


# ============================================================
# MULTI-TASK DISTILBERT MODEL
# ============================================================

class MultiTaskModel(PreTrainedModel):

    config_class = MultiTaskConfig

    def __init__(self, config):

        super().__init__(config)

        model_name = "distilbert-base-uncased"

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        self.dropout = nn.Dropout(0.3)

        hidden_size = self.encoder.config.hidden_size

        # Dialogue Act Classification Head
        self.dialogue_classifier = nn.Linear(
            hidden_size,
            config.num_dialogue_labels
        )

        # Emotion Classification Head
        self.emotion_classifier = nn.Linear(
            hidden_size,
            config.num_emotion_labels
        )


    def forward(
        self,
        input_ids,
        attention_mask,
        dialogue_labels=None,
        emotion_labels=None
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS-like representation
        pooled = self.dropout(
            outputs.last_hidden_state[:, 0]
        )

        # Dialogue prediction
        dialogue_logits = self.dialogue_classifier(
            pooled
        )

        # Emotion prediction
        emotion_logits = self.emotion_classifier(
            pooled
        )

        loss = None

        if (
            dialogue_labels is not None
            and emotion_labels is not None
        ):

            dialogue_loss = nn.CrossEntropyLoss()(
                dialogue_logits,
                dialogue_labels
            )

            emotion_loss = nn.CrossEntropyLoss()(
                emotion_logits,
                emotion_labels
            )

            loss = dialogue_loss + emotion_loss

        return {
            "loss": loss,
            "dialogue_logits": dialogue_logits,
            "emotion_logits": emotion_logits
        }


# ============================================================
# LOAD LABELS / METRICS / CONFIG
# ============================================================

def load_metadata():

    labels = {}
    metrics = {}
    config = {}

    labels_file = MODEL_DIR / "labels.json"
    metrics_file = MODEL_DIR / "metrics.json"
    config_file = MODEL_DIR / "config.json"


    # -----------------------------
    # Labels
    # -----------------------------

    if labels_file.exists():

        labels = json.loads(
            labels_file.read_text(
                encoding="utf-8"
            )
        )


    # -----------------------------
    # Metrics
    # -----------------------------

    if metrics_file.exists():

        metrics = json.loads(
            metrics_file.read_text(
                encoding="utf-8"
            )
        )


    # -----------------------------
    # Config
    # -----------------------------

    if config_file.exists():

        config = json.loads(
            config_file.read_text(
                encoding="utf-8"
            )
        )


    # -----------------------------
    # Dialogue labels
    # -----------------------------

    dialogue_labels = labels.get(
        "dialogue_act_labels",
        DEFAULT_DIALOGUE
    )

    dialogue = {
        int(k): v
        for k, v in dialogue_labels.items()
    }


    # -----------------------------
    # Emotion labels
    # -----------------------------

    emotion_labels = labels.get(
        "emotion_labels",
        DEFAULT_EMOTION
    )

    emotion = {
        int(k): v
        for k, v in emotion_labels.items()
    }


    return (
        dialogue,
        emotion,
        metrics,
        config
    )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource(
    show_spinner="Loading DistilBERT model..."
)
def load_model():

    model_file = MODEL_DIR / "model.pt"


    # --------------------------------------------------------
    # Check whether trained model exists
    # --------------------------------------------------------

    if not model_file.exists():

        return None, None, None


    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    dialogue, emotion, metrics, config_data = (
        load_metadata()
    )


    # --------------------------------------------------------
    # Create configuration
    # --------------------------------------------------------

    config = MultiTaskConfig(

        num_dialogue_labels=config_data.get(
            "num_dialogue_labels",
            len(dialogue)
        ),

        num_emotion_labels=config_data.get(
            "num_emotion_labels",
            len(emotion)
        )
    )


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = MultiTaskModel(config)


    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    state = torch.load(
        model_file,
        map_location="cpu"
    )


    model.load_state_dict(
        state
    )


    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()


    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR)
    )


    # --------------------------------------------------------
    # Return everything
    # --------------------------------------------------------

    metadata = {

        "dialogue": dialogue,

        "emotion": emotion,

        "metrics": metrics,

        "config": config_data
    }


    return (
        model,
        tokenizer,
        metadata
    )


# ============================================================
# CHECK WHETHER MODEL IS READY
# ============================================================

def model_ready():

    model_file = MODEL_DIR / "model.pt"

    tokenizer_file = (
        MODEL_DIR / "tokenizer_config.json"
    )


    return (
        model_file.exists()
        and tokenizer_file.exists()
    )