# Evaluation functions
import torch
import random

# Batch builder
def build_batch_tensors(batch_ids, pad_id, device):
    max_len = max(len(seq) for seq in batch_ids)
    input_ids = torch.full(
        (len(batch_ids), max_len),
        pad_id,
        dtype=torch.long,
        device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for i, seq in enumerate(batch_ids):
        seq = torch.tensor(seq, device=device)
        input_ids[i, :len(seq)] = seq
        attention_mask[i, :len(seq)] = 1

    return input_ids, attention_mask

# Scoring
@torch.no_grad()
def score_choices(model, input_ids, attention_mask):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    logits = outputs.logits
    log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    targets = input_ids[:, 1:]

    scores = []
    for i in range(input_ids.size(0)):
        score = 0.0
        for t in range(targets.size(1)):
            if attention_mask[i, t + 1]:
                score += log_probs[i, t, targets[i, t]]
        scores.append(score)

    return torch.stack(scores)