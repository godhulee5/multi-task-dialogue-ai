from pathlib import Path
import json
import hashlib

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
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]


# ============================================================
# HUGGING FACE MODEL
# ============================================================

HF_REPO_ID = "Godhulee/customer-support-multitask-distilbert"

# Download only the files required by the application
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

        # Same base model used during training
        self.encoder = AutoModel.from_pretrained(
            "distilbert-base-uncased"
        )

        # Same dropout used during training
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

        # Shared DistilBERT encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS representation
        pooled = outputs.last_hidden_state[:, 0]

        # Dropout
        pooled = self.dropout(pooled)

        # Dialogue Act prediction
        dialogue_logits = self.dialogue_classifier(
            pooled
        )

        # Emotion prediction
        emotion_logits = self.emotion_classifier(
            pooled
        )

        # Optional training loss
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

            # Multi-task joint loss
            loss = dialogue_loss + emotion_loss

        return {
            "loss": loss,
            "dialogue_logits": dialogue_logits,
            "emotion_logits": emotion_logits
        }


# ============================================================
# DOWNLOAD MODEL FROM HUGGING FACE
# ============================================================

@st.cache_resource(
    show_spinner="Downloading trained Customer Support AI model..."
)
def download_model_repository():

    try:

        model_path = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="model",
            allow_patterns=HF_ALLOW_PATTERNS,
            revision="main"
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
# MODEL FILE HASH
# ============================================================

def get_model_fingerprint(model_file):

    """
    Creates SHA256 fingerprint of model.pt.

    This helps verify that VS Code and Streamlit Cloud
    are using the same trained model weights.
    """

    sha256 = hashlib.sha256()

    try:

        with open(model_file, "rb") as f:

            while True:

                chunk = f.read(1024 * 1024)

                if not chunk:
                    break

                sha256.update(chunk)

        return sha256.hexdigest()

    except Exception:
        return "Unavailable"


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata(model_dir):

    labels = {}
    metrics = {}
    config = {}

    # --------------------------------------------------------
    # File paths
    # --------------------------------------------------------

    labels_file = model_dir / "labels.json"
    metrics_file = model_dir / "metrics.json"
    config_file = model_dir / "config.json"

    # --------------------------------------------------------
    # Load labels
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
    # Load metrics
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
    # Load configuration
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
    # Dialogue labels
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
    # Emotion labels
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
    # Download model
    # --------------------------------------------------------

    model_dir = download_model_repository()

    if model_dir is None:

        return None, None, None

    # --------------------------------------------------------
    # Model file
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

    num_dialogue_labels = config_data.get(
        "num_dialogue_labels",
        len(dialogue)
    )

    num_emotion_labels = config_data.get(
        "num_emotion_labels",
        len(emotion)
    )

    config = MultiTaskConfig(
        num_dialogue_labels=num_dialogue_labels,
        num_emotion_labels=num_emotion_labels
    )

    # --------------------------------------------------------
    # Create architecture
    # --------------------------------------------------------

    model = MultiTaskModel(config)

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    try:

        # weights_only is safer for a state_dict file.
        # Fallback supports older PyTorch versions.

        try:

            state = torch.load(
                model_file,
                map_location="cpu",
                weights_only=True
            )

        except TypeError:

            state = torch.load(
                model_file,
                map_location="cpu"
            )

        # Some training scripts save:
        # {"state_dict": ...}
        # Support both formats.

        if isinstance(state, dict):

            if "state_dict" in state:

                state = state["state_dict"]

            elif "model_state_dict" in state:

                state = state["model_state_dict"]

        model.load_state_dict(
            state,
            strict=True
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
            str(model_dir),
            use_fast=True
        )

    except Exception as e:

        st.error(
            "Unable to load the trained tokenizer."
        )

        st.exception(e)

        return None, None, None

    # --------------------------------------------------------
    # Model fingerprint
    # --------------------------------------------------------

    fingerprint = get_model_fingerprint(
        model_file
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "dialogue": dialogue,

        "emotion": emotion,

        "metrics": metrics,

        "config": config_data,

        "model_source": HF_REPO_ID,

        "model_file": str(model_file),

        "model_fingerprint": fingerprint,

        "base_model": "distilbert-base-uncased",

        "architecture":
            "Shared DistilBERT Encoder + "
            "Dialogue Act Head + Emotion Head",

        "max_length": 64
    }

    return (
        model,
        tokenizer,
        metadata
    )


# ============================================================
# CHECK MODEL READY
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

        labels_file = (
            model_dir / "labels.json"
        )

        return (
            model_file.exists()
            and tokenizer_file.exists()
            and labels_file.exists()
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

        "dialogue_labels":
            [
                "Inform",
                "Question",
                "Directive",
                "Commissive"
            ],

        "emotion_labels":
            [
                "Neutral",
                "Anger",
                "Disgust",
                "Fear",
                "Happiness",
                "Sadness",
                "Surprise"
            ],

        "model_source":
            HF_REPO_ID,

        "architecture":
            "Shared DistilBERT Encoder + "
            "Dialogue Act Head + Emotion Head",

        "max_length":
            64
    }
