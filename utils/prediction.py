
import torch
import streamlit as st
from utils.model_utils import load_model, load_metadata

DEMO = {
    "What is the status of my order?": ("Question","Neutral"),
    "I am extremely unhappy with this service.": ("Inform","Anger"),
    "Please cancel my order immediately.": ("Directive","Anger"),
    "Thank you so much for solving my problem!": ("Commissive","Happiness"),
}

def priority_for(emotion, dialogue_act):
    if emotion in {"Anger","Disgust"}:
        return "High"
    if emotion in {"Fear","Sadness"} or dialogue_act in {"Directive"}:
        return "Medium"
    return "Low"

def predict_text(text):
    text = (text or "").strip()
    if not text:
        raise ValueError("Please enter a message.")

    model, tokenizer, meta = load_model()
    if model is None:
        dialogue, emotion = DEMO.get(text, ("Question" if "?" in text else "Inform", "Neutral"))
        return {
            "text": text, "dialogue_act": dialogue, "emotion": emotion,
            "dialogue_confidence": 0.0, "emotion_confidence": 0.0,
            "dialogue_probabilities": {}, "emotion_probabilities": {},
            "priority": priority_for(emotion, dialogue), "demo": True
        }

    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        out = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
    dprob = torch.softmax(out["dialogue_logits"], dim=1)[0]
    eprob = torch.softmax(out["emotion_logits"], dim=1)[0]
    did = int(torch.argmax(dprob).item())
    eid = int(torch.argmax(eprob).item())
    dlabels = meta["dialogue"]
    elabels = meta["emotion"]
    d_probs = {dlabels[i]: float(dprob[i]) for i in range(len(dlabels))}
    e_probs = {elabels[i]: float(eprob[i]) for i in range(len(elabels))}
    return {
        "text": text,
        "dialogue_act_id": did, "dialogue_act": dlabels[did],
        "emotion_id": eid, "emotion": elabels[eid],
        "dialogue_confidence": float(dprob[did]),
        "emotion_confidence": float(eprob[eid]),
        "dialogue_probabilities": d_probs,
        "emotion_probabilities": e_probs,
        "priority": priority_for(elabels[eid], dlabels[did]),
        "demo": False
    }

def support_reply(result):
    emotion, act = result["emotion"], result["dialogue_act"]
    if emotion in {"Anger","Disgust"}:
        return "I’m sorry about this experience. I understand your concern and will help you resolve it."
    if emotion in {"Fear","Sadness"}:
        return "I understand your concern. Let’s work through the issue step by step."
    if act == "Question":
        return "I’d be happy to help with that. Let’s look at the details of your request."
    if act == "Directive":
        return "Understood. I’ll help you with that request."
    if emotion == "Happiness":
        return "That’s great to hear! Thank you for sharing your experience."
    return "Thanks for contacting support. How can I help you further?"
