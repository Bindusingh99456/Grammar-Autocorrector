import os
import sys
import time
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import matplotlib.pyplot as plt

BASE_MODEL_NAME = "grammarly/coedit-large"
COEDIT_PROMPT_PREFIX = "Fix the grammar: "


class CoEditDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Format input with standard CoEdIT instruction prefix
        self.inputs = [f"{COEDIT_PROMPT_PREFIX}{text}" for text in df["input"].astype(str)]
        self.targets = [text for text in df["output"].astype(str)]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp_text = self.inputs[idx]
        tgt_text = self.targets[idx]

        model_inputs = self.tokenizer(
            inp_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = model_inputs.input_ids.squeeze(0)
        attention_mask = model_inputs.attention_mask.squeeze(0)
        label_ids = labels.input_ids.squeeze(0)

        # Replace padding token ids in labels with -100 so they are ignored in cross-entropy loss
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
        }


def check_hardware_for_training(force_cpu=False):
    """
    Enforce project strategy: 'grammarly/coedit-large' (770M) must be trained on a cloud GPU.
    Never attempt to train locally without a GPU.
    """
    has_gpu = torch.cuda.is_available()
    if not has_gpu and not force_cpu:
        print("=" * 76)
        print(" [POLICY] LOCAL GPU NOT DETECTED - LOCAL TRAINING ABORTED")
        print("=" * 76)
        print(f" Target Model: '{BASE_MODEL_NAME}' (770M parameters)")
        print(" Fine-tuning 'grammarly/coedit-large' requires a modern GPU to prevent severe")
        print(" CPU memory exhaustion and thermal throttling.")
        print()
        print(" Training must be performed on a cloud GPU (Google Colab or Kaggle) via:")
        print("   --> docs/train_colab.ipynb")
        print()
        print(" Steps:")
        print("   1. Open 'docs/train_colab.ipynb' in Google Colab (free T4 GPU).")
        print("   2. Run the staged LoRA fine-tuning cells.")
        print("   3. Download the trained LoRA adapter and place it into 'model/gec-model/'.")
        print("   4. The local application will automatically load and serve the adapter.")
        print("=" * 76)
        sys.exit(1)

    device = torch.device("cuda" if has_gpu else "cpu")
    if has_gpu:
        print(f"[OK] Cloud/Local GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARNING] Running on CPU via --force_cpu. This is not recommended for 770M params.")
    return device


def plot_loss_curve(log_df, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(log_df["epoch"], log_df["train_loss"], marker="o", label="Training Loss", color="#0e2240")
    plt.plot(log_df["epoch"], log_df["val_loss"], marker="s", label="Validation Loss", color="#d9531e")
    plt.title("CoEdIT-Large LoRA Fine-Tuning Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved loss curve visualization to {output_path}")


def train(args):
    device = check_hardware_for_training(force_cpu=args.force_cpu)
    model_name = args.model_name or BASE_MODEL_NAME

    print(f"\n=== Staged LoRA Fine-Tuning Configuration ===")
    print(f"Base Model:       {model_name} (770M parameters)")
    print(f"Device:           {device}")
    print(f"Task Prefix:      '{COEDIT_PROMPT_PREFIX}'")
    print(f"Max Token Length: {args.max_length}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Learning Rate:    {args.lr}")
    print(f"Epochs:           {args.epochs}")
    print(f"LoRA Rank (r):    {args.lora_r}")
    print(f"LoRA Alpha:       {args.lora_alpha}")

    # Load datasets
    print(f"\nLoading training data from {args.train_file}...")
    train_df = pd.read_csv(args.train_file)
    if args.max_train_samples and len(train_df) > args.max_train_samples:
        train_df = train_df.iloc[:args.max_train_samples]
    print(f"Training pairs: {len(train_df):,}")

    print(f"Loading validation data from {args.val_file}...")
    val_df = pd.read_csv(args.val_file)
    if args.max_val_samples and len(val_df) > args.max_val_samples:
        val_df = val_df.iloc[:args.max_val_samples]
    print(f"Validation pairs: {len(val_df):,}")

    # Load Tokenizer & Base Model
    print(f"\nLoading tokenizer and weights for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

    # Configure LoRA for Seq2Seq LM
    print(f"Applying LoRA adapter (rank={args.lora_r}, alpha={args.lora_alpha})...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q", "v"],
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    model.to(device)

    train_dataset = CoEditDataset(train_df, tokenizer, max_length=args.max_length)
    val_dataset = CoEditDataset(val_df, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    training_log = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Training Phase
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print(f"Epoch {epoch}/{args.epochs} | Step {step+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}", flush=True)

        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(val_loader)
        epoch_duration = time.time() - epoch_start

        print(f"\n--- Epoch {epoch} Summary ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_duration:.1f}s")

        training_log.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "time_sec": epoch_duration,
        })

        # Save Best LoRA Adapter Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"--> Best validation loss improved to {best_val_loss:.4f}. Saving LoRA adapter checkpoint to {output_dir}...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            patience_counter += 1
            print(f"No improvement. Early stopping patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # Save training log and plot curve
    log_df = pd.DataFrame(training_log)
    log_csv = output_dir.parent / "training_log.csv"
    log_df.to_csv(log_csv, index=False)
    print(f"Saved training log to {log_csv}")

    loss_curve_png = output_dir.parent / "loss_curve.png"
    plot_loss_curve(log_df, loss_curve_png)
    print("\nTraining completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Staged LoRA Fine-Tuning for grammarly/coedit-large")
    parser.add_argument("--model_name", type=str, default=BASE_MODEL_NAME)
    parser.add_argument("--train_file", type=str, default="data/processed/train.csv")
    parser.add_argument("--val_file", type=str, default="data/processed/val.csv")
    parser.add_argument("--output_dir", type=str, default="model/gec-model")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--force_cpu", action="store_true", help="Force execution on CPU (not recommended for 770M)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
