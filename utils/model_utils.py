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

# IMPORTANT:
# Kept for compatibility with existing pages such as
# 8_Model_Performance.py
MODEL_DIR = BASE_DIR / "models" / "multitask_model"


# ============================================================
# HUGGING FACE MODEL
# ============================================================

HF_REPO_ID = "Godhulee/customer-support-multitask-distilbert"

HF_REVISION = "main"


# ============================================================
# FILES TO DOWNLOAD FROM HUGGING FACE
# ============================================================

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
# DEFAULT DIALOGUE ACT LABELS
# ============================================================

DEFAULT_DIALOGUE = {
    0: "Inform",
    1: "Question",
    2: "Directive",
    3: "Commissive"
}


# ============================================================
# DEFAULT EMOTION LABELS
# ============================================================

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
# MODEL CONSTANTS
# ============================================================

BASE_MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 64

DROPOUT = 0.3

NUM_DIALOGUE_LABELS = 4

NUM_EMOTION_LABELS = 7


# ============================================================
# MULTI-TASK CONFIGURATION
# ============================================================

class MultiTaskConfig(PretrainedConfig):

    model_type = "multitask-distilbert"

    def __init__(
        self,
        num_dialogue_labels=NUM_DIALOGUE_LABELS,
        num_emotion_labels=NUM_EMOTION_LABELS,
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

        # ----------------------------------------------------
        # SHARED DISTILBERT ENCODER
        # ----------------------------------------------------

        self.encoder = AutoModel.from_pretrained(
            BASE_MODEL_NAME
        )

        # ----------------------------------------------------
        # DROPOUT
        # ----------------------------------------------------

        self.dropout = nn.Dropout(DROPOUT)

        # ----------------------------------------------------
        # HIDDEN SIZE
        # ----------------------------------------------------

        hidden_size = self.encoder.config.hidden_size

        # ----------------------------------------------------
        # DIALOGUE ACT CLASSIFICATION HEAD
        # ----------------------------------------------------

        self.dialogue_classifier = nn.Linear(
            hidden_size,
            config.num_dialogue_labels
        )

        # ----------------------------------------------------
        # EMOTION CLASSIFICATION HEAD
        # ----------------------------------------------------

        self.emotion_classifier = nn.Linear(
            hidden_size,
            config.num_emotion_labels
        )


    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(
        self,
        input_ids,
        attention_mask,
        dialogue_labels=None,
        emotion_labels=None
    ):

        # ----------------------------------------------------
        # SHARED DISTILBERT ENCODER
        # ----------------------------------------------------

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # ----------------------------------------------------
        # CLS REPRESENTATION
        # ----------------------------------------------------

        pooled = outputs.last_hidden_state[:, 0]

        # ----------------------------------------------------
        # DROPOUT
        # ----------------------------------------------------

        pooled = self.dropout(pooled)

        # ----------------------------------------------------
        # DIALOGUE ACT PREDICTION
        # ----------------------------------------------------

        dialogue_logits = self.dialogue_classifier(
            pooled
        )

        # ----------------------------------------------------
        # EMOTION PREDICTION
        # ----------------------------------------------------

        emotion_logits = self.emotion_classifier(
            pooled
        )

        # ----------------------------------------------------
        # OPTIONAL TRAINING LOSS
        # ----------------------------------------------------

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

            # Joint Multi-Task Loss
            loss = dialogue_loss + emotion_loss

        # ----------------------------------------------------
        # RETURN OUTPUTS
        # ----------------------------------------------------

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
            revision=HF_REVISION
        )

        return Path(model_path)

    except Exception as e:

        st.error(
            "Unable to download the trained model "
            "from Hugging Face."
        )

        st.exception(e)

        return None


# ============================================================
# MODEL FILE HASH / FINGERPRINT
# ============================================================

def get_model_fingerprint(model_file):

    """
    Creates a SHA256 fingerprint of model.pt.

    This is useful for checking whether the model
    running locally and the model running on
    Streamlit Cloud are identical.
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
# LOAD JSON FILE SAFELY
# ============================================================

def _load_json_file(file_path):

    """
    Safely loads a JSON file.
    Returns {} if the file does not exist
    or cannot be read.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        return {}

    try:

        return json.loads(
            file_path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:

        return {}


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata(model_dir=None):

    """
    Loads:

    - labels.json
    - metrics.json
    - config.json

    model_dir is optional.

    If no directory is supplied, the Hugging Face
    repository is downloaded automatically.

    This keeps compatibility with pages that call:

        load_metadata()

    and pages that call:

        load_metadata(model_dir)
    """

    # --------------------------------------------------------
    # DOWNLOAD MODEL IF DIRECTORY WAS NOT PROVIDED
    # --------------------------------------------------------

    if model_dir is None:

        model_dir = download_model_repository()

    # --------------------------------------------------------
    # DOWNLOAD FAILED
    # --------------------------------------------------------

    if model_dir is None:

        return (
            DEFAULT_DIALOGUE.copy(),
            DEFAULT_EMOTION.copy(),
            {},
            {}
        )

    # --------------------------------------------------------
    # CONVERT TO PATH
    # --------------------------------------------------------

    model_dir = Path(model_dir)

    # --------------------------------------------------------
    # JSON FILES
    # --------------------------------------------------------

    labels_file = model_dir / "labels.json"

    metrics_file = model_dir / "metrics.json"

    config_file = model_dir / "config.json"

    # --------------------------------------------------------
    # LOAD FILES
    # --------------------------------------------------------

    labels = _load_json_file(labels_file)

    metrics = _load_json_file(metrics_file)

    config = _load_json_file(config_file)

    # --------------------------------------------------------
    # DIALOGUE ACT LABELS
    # --------------------------------------------------------

    dialogue_labels = labels.get(
        "dialogue_act_labels",
        DEFAULT_DIALOGUE
    )

    # Handle dictionary format
    if isinstance(dialogue_labels, dict):

        dialogue = {
            int(k): v
            for k, v in dialogue_labels.items()
        }

    # Handle list format
    elif isinstance(dialogue_labels, list):

        dialogue = {
            i: value
            for i, value in enumerate(dialogue_labels)
        }

    else:

        dialogue = DEFAULT_DIALOGUE.copy()

    # --------------------------------------------------------
    # EMOTION LABELS
    # --------------------------------------------------------

    emotion_labels = labels.get(
        "emotion_labels",
        DEFAULT_EMOTION
    )

    # Handle dictionary format
    if isinstance(emotion_labels, dict):

        emotion = {
            int(k): v
            for k, v in emotion_labels.items()
        }

    # Handle list format
    elif isinstance(emotion_labels, list):

        emotion = {
            i: value
            for i, value in enumerate(emotion_labels)
        }

    else:

        emotion = DEFAULT_EMOTION.copy()

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

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
    # DOWNLOAD MODEL REPOSITORY
    # --------------------------------------------------------

    model_dir = download_model_repository()

    # --------------------------------------------------------
    # CHECK DOWNLOAD
    # --------------------------------------------------------

    if model_dir is None:

        return None, None, None

    # --------------------------------------------------------
    # MODEL FILE
    # --------------------------------------------------------

    model_file = model_dir / "model.pt"

    if not model_file.exists():

        st.error(
            "model.pt was not found in the "
            "Hugging Face repository."
        )

        return None, None, None

    # --------------------------------------------------------
    # LOAD METADATA
    # --------------------------------------------------------

    (
        dialogue,
        emotion,
        metrics,
        config_data
    ) = load_metadata(model_dir)

    # --------------------------------------------------------
    # DETERMINE NUMBER OF CLASSES
    # --------------------------------------------------------

    num_dialogue_labels = config_data.get(
        "num_dialogue_labels",
        len(dialogue)
    )

    num_emotion_labels = config_data.get(
        "num_emotion_labels",
        len(emotion)
    )

    # Make sure values are integers
    try:

        num_dialogue_labels = int(
            num_dialogue_labels
        )

    except Exception:

        num_dialogue_labels = len(dialogue)

    try:

        num_emotion_labels = int(
            num_emotion_labels
        )

    except Exception:

        num_emotion_labels = len(emotion)

    # --------------------------------------------------------
    # CREATE CONFIGURATION
    # --------------------------------------------------------

    config = MultiTaskConfig(
        num_dialogue_labels=num_dialogue_labels,
        num_emotion_labels=num_emotion_labels
    )

    # --------------------------------------------------------
    # CREATE MODEL ARCHITECTURE
    # --------------------------------------------------------

    try:

        model = MultiTaskModel(config)

    except Exception as e:

        st.error(
            "Unable to create the Multi-Task "
            "DistilBERT model."
        )

        st.exception(e)

        return None, None, None

    # --------------------------------------------------------
    # LOAD TRAINED WEIGHTS
    # --------------------------------------------------------

    try:

        # ----------------------------------------------------
        # Modern PyTorch
        # ----------------------------------------------------

        try:

            state = torch.load(
                model_file,
                map_location="cpu",
                weights_only=True
            )

        # ----------------------------------------------------
        # Older PyTorch
        # ----------------------------------------------------

        except TypeError:

            state = torch.load(
                model_file,
                map_location="cpu"
            )

        # ----------------------------------------------------
        # SUPPORT DIFFERENT CHECKPOINT FORMATS
        # ----------------------------------------------------

        if isinstance(state, dict):

            if "state_dict" in state:

                state = state["state_dict"]

            elif "model_state_dict" in state:

                state = state["model_state_dict"]

        # ----------------------------------------------------
        # LOAD STATE DICTIONARY
        # ----------------------------------------------------

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
    # EVALUATION MODE
    # --------------------------------------------------------

    model.eval()

    # --------------------------------------------------------
    # LOAD TOKENIZER
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
    # MODEL FINGERPRINT
    # --------------------------------------------------------

    fingerprint = get_model_fingerprint(
        model_file
    )

    # --------------------------------------------------------
    # COMPLETE METADATA
    # --------------------------------------------------------

    metadata = {

        "dialogue": dialogue,

        "emotion": emotion,

        "metrics": metrics,

        "config": config_data,

        "model_source": HF_REPO_ID,

        "model_revision": HF_REVISION,

        "model_file": str(model_file),

        "model_fingerprint": fingerprint,

        "base_model": BASE_MODEL_NAME,

        "architecture": (
            "Shared DistilBERT Encoder + "
            "Dialogue Act Head + "
            "Emotion Head"
        ),

        "max_length": MAX_LENGTH,

        "dropout": DROPOUT,

        "num_dialogue_labels":
            num_dialogue_labels,

        "num_emotion_labels":
            num_emotion_labels
    }

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # DOWNLOAD / LOCATE MODEL
        # ----------------------------------------------------

        model_dir = download_model_repository()

        if model_dir is None:

            return False

        # ----------------------------------------------------
        # REQUIRED FILES
        # ----------------------------------------------------

        model_file = model_dir / "model.pt"

        tokenizer_file = (
            model_dir / "tokenizer_config.json"
        )

        labels_file = (
            model_dir / "labels.json"
        )

        # ----------------------------------------------------
        # CHECK
        # ----------------------------------------------------

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
            BASE_MODEL_NAME,

        "dialogue_classes":
            NUM_DIALOGUE_LABELS,

        "emotion_classes":
            NUM_EMOTION_LABELS,

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
            (
                "Shared DistilBERT Encoder + "
                "Dialogue Act Head + "
                "Emotion Head"
            ),

        "max_length":
            MAX_LENGTH,

        "dropout":
            DROPOUT
    }


# ============================================================
# OPTIONAL: GET MODEL FINGERPRINT AFTER LOADING
# ============================================================

def get_loaded_model_fingerprint():

    """
    Downloads the model repository if required
    and returns the SHA256 fingerprint of model.pt.
    """

    try:

        model_dir = download_model_repository()

        if model_dir is None:

            return "Unavailable"

        model_file = model_dir / "model.pt"

        if not model_file.exists():

            return "Unavailable"

        return get_model_fingerprint(
            model_file
        )

    except Exception:

        return "Unavailable"
