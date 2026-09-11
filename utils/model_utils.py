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

from huggingface_hub import snapshot_download


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

# ============================================================
# HUGGING FACE MODEL
# ============================================================

HF_REPO_ID = "Godhulee/customer-support-multitask-distilbert"

# Files that we need from Hugging Face
HF_ALLOW_PATTERNS = [
    "model.pt",
    "config.json",
    "labels.json",
    "metrics.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
]


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

        self.dialogue_classifier = nn.Linear(
            hidden_size,
            config.num_dialogue_labels
        )

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

        pooled = self.dropout(
            outputs.last_hidden_state[:, 0]
        )

        dialogue_logits = self.dialogue_classifier(
            pooled
        )

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
# DOWNLOAD MODEL FROM HUGGING FACE
# ============================================================

@st.cache_resource(show_spinner="Downloading trained model from Hugging Face...")
def download_model_repository():

    try:

        model_path = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="model",
            allow_patterns=HF_ALLOW_PATTERNS
        )

        return Path(model_path)

    except Exception as e:

        st.error(
            "Unable to download the trained model from "
            "Hugging Face."
        )

        st.exception(e)

        return None


# ============================================================
# LOAD LABELS / METRICS / CONFIG
# ============================================================

def load_metadata(model_dir):

    labels = {}
    metrics = {}
    config = {}

    labels_file = model_dir / "labels.json"
    metrics_file = model_dir / "metrics.json"
    config_file = model_dir / "config.json"

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    if labels_file.exists():

        try:

            labels = json.loads(
                labels_file.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:
            labels = {}

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    if metrics_file.exists():

        try:

            metrics = json.loads(
                metrics_file.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:
            metrics = {}

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    if config_file.exists():

        try:

            config = json.loads(
                config_file.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:
            config = {}

    # --------------------------------------------------------
    # DIALOGUE LABELS
    # --------------------------------------------------------

    dialogue_labels = labels.get(
        "dialogue_act_labels",
        DEFAULT_DIALOGUE
    )

    dialogue = {
        int(k): v
        for k, v in dialogue_labels.items()
    }

    # --------------------------------------------------------
    # EMOTION LABELS
    # --------------------------------------------------------

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
    show_spinner="Loading Customer Support AI model..."
)
def load_model():

    # --------------------------------------------------------
    # Download Hugging Face repository
    # --------------------------------------------------------

    model_dir = download_model_repository()

    if model_dir is None:

        return None, None, None

    # --------------------------------------------------------
    # Check model file
    # --------------------------------------------------------

    model_file = model_dir / "model.pt"

    if not model_file.exists():

        st.error(
            "model.pt was not found in the Hugging Face repository."
        )

        return None, None, None

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    (
        dialogue,
        emotion,
        metrics,
        config_data
    ) = load_metadata(model_dir)

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
    # Create model architecture
    # --------------------------------------------------------

    model = MultiTaskModel(config)

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    try:

        state = torch.load(
            model_file,
            map_location="cpu"
        )

        model.load_state_dict(
            state
        )

    except Exception as e:

        st.error(
            "Unable to load the trained model weights."
        )

        st.exception(e)

        return None, None, None

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    try:

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir)
        )

    except Exception as e:

        st.error(
            "Unable to load the tokenizer."
        )

        st.exception(e)

        return None, None, None

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "dialogue": dialogue,

        "emotion": emotion,

        "metrics": metrics,

        "config": config_data,

        "model_source": HF_REPO_ID
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

    try:

        model_dir = download_model_repository()

        if model_dir is None:
            return False

        model_file = model_dir / "model.pt"

        tokenizer_file = (
            model_dir / "tokenizer_config.json"
        )

        return (
            model_file.exists()
            and tokenizer_file.exists()
        )

    except Exception:

        return False


# ============================================================
# GET MODEL INFORMATION
# ============================================================

def get_model_info():

    return {

        "model_name":
            "Customer Support Multi-Task DistilBERT",

        "base_model":
            "distilbert-base-uncased",

        "dialogue_classes":
            4,

        "emotion_classes":
            7,

        "model_source":
            HF_REPO_ID,

        "architecture":
            "Shared DistilBERT Encoder + "
            "Dialogue Act Head + Emotion Head"
    }
