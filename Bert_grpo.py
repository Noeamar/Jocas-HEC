import os
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ============================================================
# 0. CONFIG
# ============================================================

@dataclass
class Config:
    model_name: str = "bert-base-uncased"
    train_path: str = "train.csv"      # CSV avec colonnes: text,label
    val_path: str = "val.csv"          # optionnel
    num_labels: int = 2                # classification binaire ici
    max_length: int = 256
    batch_size: int = 16
    num_epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Hyperparam RL
    gamma_ce: float = 0.5              # poids de la cross-entropy de stabilisation
    clip_adv: float = 5.0              # clip des advantages
    group_size: int = 4                # taille des groupes GRPO (K épisodes)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

CFG = Config()


# ============================================================
# 1. DATASET SIMPLE (text,label)
# ============================================================

import pandas as pd

class TextClsDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        df = pd.read_csv(path)
        assert "text" in df.columns and "label" in df.columns, \
            "Le CSV doit avoir les colonnes 'text' et 'label'"
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# ============================================================
# 2. GRPO : définition du “reward” + loss
# ============================================================

def reward_fn(pred_actions, true_labels):
    """
    Reward simple :
      +1 si la prédiction == label
      0 sinon

    pred_actions : (B,)
    true_labels  : (B,)
    """
    reward = (pred_actions == true_labels).float()
    return reward


def compute_grpo_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    cfg: Config,
):
    """
    logits : (B, num_labels)
    labels : (B,)
    -> renvoie: loss_grpo, metrics
    """

    # Distribution de politique = softmax sur les classes
    log_probs = torch.log_softmax(logits, dim=-1)  # (B, C)
    probs = torch.softmax(logits, dim=-1)

    # On échantillonne une action (classe) pour chaque exemple
    # (on pourrait aussi prendre argmax, mais sampling = RL plus standard)
    dist = torch.distributions.Categorical(probs)
    actions = dist.sample()        # (B,)
    chosen_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

    # Reward (B,)
    rewards = reward_fn(actions, labels)

    # ===================== GRPO : baseline de groupe ===================== #
    # On découpe le batch en groupes de taille K
    B = logits.size(0)
    K = cfg.group_size
    num_groups = max(1, B // K)

    # si B n'est pas multiple de K, on tronque (c'est plus simple)
    trunc_B = num_groups * K
    rewards = rewards[:trunc_B]
    chosen_log_probs = chosen_log_probs[:trunc_B]

    rewards_group = rewards.view(num_groups, K)
    log_probs_group = chosen_log_probs.view(num_groups, K)

    # baseline = moyenne de reward au sein de chaque groupe
    baseline = rewards_group.mean(dim=1, keepdim=True)  # (G,1)
    advantages = rewards_group - baseline               # (G,K)

    # Optionnel : clip des advantages
    advantages = torch.clamp(advantages, -cfg.clip_adv, cfg.clip_adv)

    # Loss policy = - E[ log pi(a|s) * A ]
    policy_loss = - (log_probs_group * advantages.detach()).mean()

    # ===================== Stabilisation : CE supervisée ===================== #
    ce_loss_fn = nn.CrossEntropyLoss()
    ce_loss = ce_loss_fn(logits, labels)

    # Combine : GRPO + CE
    loss = policy_loss + cfg.gamma_ce * ce_loss

    with torch.no_grad():
        acc = (actions == labels).float().mean().item()
        avg_reward = rewards.mean().item()

    metrics = {
        "policy_loss": policy_loss.item(),
        "ce_loss": ce_loss.item(),
        "acc": acc,
        "avg_reward": avg_reward,
    }
    return loss, metrics


# ============================================================
# 3. ENTRAÎNEMENT
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np_random = __import__("numpy").random
    np_random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train():
    set_seed(CFG.seed)

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG.model_name,
        num_labels=CFG.num_labels,
    )
    model.to(CFG.device)

    train_dataset = TextClsDataset(CFG.train_path, tokenizer, CFG.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
    )

    # Optim + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    num_training_steps = len(train_loader) * CFG.num_epochs
    num_warmup_steps = int(CFG.warmup_ratio * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    global_step = 0
    model.train()

    for epoch in range(CFG.num_epochs):
        print(f"\n===== Epoch {epoch+1}/{CFG.num_epochs} =====")
        running_loss = 0.0
        running_acc = 0.0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(CFG.device) for k, v in batch.items()}
            labels = batch.pop("labels")

            outputs = model(**batch)
            logits = outputs.logits  # (B, num_labels)

            loss, metrics = compute_grpo_loss(
                logits,
                labels,
                CFG,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_loss += loss.item()
            running_acc += metrics["acc"]

            if global_step % 50 == 0:
                avg_loss = running_loss / 50
                avg_acc = running_acc / 50
                running_loss = 0.0
                running_acc = 0.0

                print(
                    f"Step {global_step:05d} | "
                    f"loss={avg_loss:.4f} | "
                    f"acc={avg_acc:.3f} | "
                    f"reward={metrics['avg_reward']:.3f} | "
                    f"policy_loss={metrics['policy_loss']:.4f} | "
                    f"ce_loss={metrics['ce_loss']:.4f}"
                )

    # Sauvegarde du modèle fine-tuné
    out_dir = "bert-grpo-checkpoint"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\n✅ Modèle GRPO+BERT sauvegardé dans {out_dir}")


if __name__ == "__main__":
    train()