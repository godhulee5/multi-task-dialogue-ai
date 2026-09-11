encoding = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=64
)

with torch.no_grad():

    outputs = model(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"]
    )

dialogue_probs = torch.softmax(
    outputs["dialogue_logits"],
    dim=1
)

emotion_probs = torch.softmax(
    outputs["emotion_logits"],
    dim=1
)

dialogue_id = torch.argmax(
    dialogue_probs,
    dim=1
).item()

emotion_id = torch.argmax(
    emotion_probs,
    dim=1
).item()

dialogue_label = metadata["dialogue"][dialogue_id]
emotion_label = metadata["emotion"][emotion_id]

dialogue_confidence = (
    dialogue_probs[0][dialogue_id].item() * 100
)

emotion_confidence = (
    emotion_probs[0][emotion_id].item() * 100
)
