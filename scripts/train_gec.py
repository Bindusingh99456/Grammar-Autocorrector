"""Staged LoRA Fine-Tuning of grammarly/coedit-large (770M) for Grammatical Error Correction.

Designed for Google Colab (free T4 GPU) or Kaggle (T4x2 GPU).
DO NOT RUN LOCALLY ON CPU: 770M parameter models will trigger severe CPU memory exhaustion.

Phase M3 Curriculum:
- Base model: grammarly/coedit-large
- Input format: "Fix grammatical errors in this sentence: <text>"
- LoRA: r=16, alpha=32, dropout=0.05 on q, k, v, o, wi, wo
- Precision: bf16 if supported, otherwise fp32 (T5 is unstable in fp16)
- Gradient checkpointing enabled, effective batch size 32 (8 x 4 accumulation)
- Stage 1: stage1_colab.csv (150k subset of stage1.csv), 1 epoch, lr=3e-4
- Stage 2: stage2.csv (wi_train + fce_train + coedit_gec oversampled x3), 3 epochs, lr=1e-4
- Validation: evaluated each epoch on wi_dev.csv + jfleg_dev.json with early stopping
- Checkpointing: persistent saves every 500 steps with auto-resume on disconnect
- Export: merges LoRA weights into base model, saves gec-model/, zips as gec-model.zip
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, TaskType, get_peft_model


BASE_MODEL_NAME = "grammarly/coedit-large"
INPUT_PREFIX = "Fix grammatical errors in this sentence: "
MAX_LENGTH = 64


class GECDataset(Dataset):
    """Memory-efficient Dataset for sequence-to-sequence GEC."""

    def __init__(self, inputs: list[str], targets: list[str], tokenizer, max_length: int = MAX_LENGTH):
        self.inputs = [f"{INPUT_PREFIX}{str(inp).strip()}" for inp in inputs]
        self.targets = [str(tgt).strip() for tgt in targets]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp_text = self.inputs[idx]
        tgt_text = self.targets[idx]

        inp_enc = self.tokenizer(
            inp_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tgt_enc = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = tgt_enc.input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inp_enc.input_ids.squeeze(0),
            "attention_mask": inp_enc.attention_mask.squeeze(0),
            "labels": labels,
        }


def get_environment_paths(user_data_dir: str | None = None):
    """Resolve data, checkpoint, and export directories based on cloud environment."""
    if os.path.exists("/content"):
        # Google Colab environment
        try:
            from google.colab import drive
            drive_mount_point = "/content/drive"
            if not os.path.exists(f"{drive_mount_point}/MyDrive"):
                print("[COLAB] Mounting Google Drive to persist checkpoints across disconnects...")
                drive.mount(drive_mount_point)
            base_dir = Path(f"{drive_mount_point}/MyDrive/gec_training")
            print(f"[COLAB] Persistent Drive storage active: {base_dir}")
        except Exception as e:
            print(f"[COLAB] Could not mount Drive ({e}). Using /content/gec_training (non-persistent).")
            base_dir = Path("/content/gec_training")
    elif os.path.exists("/kaggle"):
        # Kaggle environment
        base_dir = Path("/kaggle/working/gec_training")
        print(f"[KAGGLE] Persistent storage active: {base_dir}")
    else:
        # Generic cloud VM or local root
        base_dir = Path("./cloud_gec_training")

    ckpt_dir = base_dir / "checkpoints"
    export_dir = base_dir / "export"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Search for data files
    search_dirs = [
        Path(user_data_dir) if user_data_dir else None,
        Path("data/processed"),
        Path("/content/Grammar_Autocorrector/data/processed"),
        Path("/content/data/processed"),
        Path("/kaggle/working/Grammar_Autocorrector/data/processed"),
        Path("/kaggle/input/grammar-autocorrector/data/processed"),
    ]
    resolved_data_dir = None
    for d in search_dirs:
        if d and d.exists() and (d / "stage1_colab.csv").exists():
            resolved_data_dir = d
            break
    if not resolved_data_dir:
        # Fallback to local data/processed
        resolved_data_dir = Path("data/processed")

    print(f"Data directory:       {resolved_data_dir}")
    print(f"Checkpoints directory: {ckpt_dir}")
    print(f"Export directory:      {export_dir}\n")

    return resolved_data_dir, ckpt_dir, export_dir, base_dir


def load_validation_pairs(data_dir: Path) -> tuple[list[str], list[str]]:
    """Load combined validation pairs from wi_dev.csv and jfleg_dev.json."""
    val_inputs = []
    val_targets = []

    # 1. W&I+LOCNESS Dev Split
    wi_dev_path = data_dir / "wi_dev.csv"
    if wi_dev_path.exists():
        df_wi_dev = pd.read_csv(wi_dev_path, encoding="utf-8")
        val_inputs.extend(df_wi_dev["input"].astype(str).tolist())
        val_targets.extend(df_wi_dev["output"].astype(str).tolist())
        print(f"Loaded W&I dev validation pairs: {len(df_wi_dev):,}")
    else:
        print(f"[WARNING] {wi_dev_path} not found.")

    # 2. JFLEG Dev Set
    jfleg_dev_path = data_dir / "jfleg_dev.json"
    if jfleg_dev_path.exists():
        with open(jfleg_dev_path, "r", encoding="utf-8") as f:
            jfleg_records = json.load(f)
        for rec in jfleg_records:
            src = rec.get("source", "")
            refs = rec.get("references", [])
            if src and refs:
                # Pair with primary reference
                val_inputs.append(src)
                val_targets.append(refs[0])
        print(f"Loaded JFLEG dev validation pairs: {len(jfleg_records):,}")
    else:
        print(f"[WARNING] {jfleg_dev_path} not found.")

    print(f"Total combined validation pairs: {len(val_inputs):,}\n")
    return val_inputs, val_targets


def train_gec():
    parser = argparse.ArgumentParser(description="Stage-trained LoRA Fine-Tuning for grammarly/coedit-large")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing processed CSV files")
    parser.add_argument("--skip_stage1", action="store_true", help="Skip Stage 1 and train Stage 2 only")
    parser.add_argument("--use_targeted", action="store_true", help="Optionally mix data/processed/targeted.csv into Stage 2 (keeping real data >= 50%)")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU execution (not recommended)")
    args = parser.parse_args()

    # Hardware check
    has_gpu = torch.cuda.is_available()
    if not has_gpu and not args.force_cpu:
        print("=" * 80)
        print(" [POLICY] NO CUDA GPU DETECTED - TRAINING ABORTED ON LOCAL CPU")
        print("=" * 80)
        print(f" Target Model: '{BASE_MODEL_NAME}' (T5-Large, 770M parameters)")
        print(" Fine-tuning this model locally on CPU will cause severe system memory exhaustion.")
        print()
        print(" This training pipeline is engineered for cloud GPUs (Google Colab T4 or Kaggle).")
        print(" Instructions to run:")
        print("   1. Open Google Colab (https://colab.research.google.com).")
        print("   2. Upload 'notebooks/train_gec.ipynb'.")
        print("   3. Select 'Runtime > Change runtime type > T4 GPU'.")
        print("   4. Run the notebook to produce 'gec-model.zip'.")
        print("   5. Download 'gec-model.zip' and run 'python scripts/install_model.py'.")
        print("=" * 80)
        sys.exit(0)

    device_name = torch.cuda.get_device_name(0) if has_gpu else "CPU"
    is_bf16 = has_gpu and torch.cuda.is_bf16_supported()
    precision_mode = "bfloat16" if is_bf16 else "float32 (T5 fp16 instability avoided)"

    print("=" * 80)
    print("GRAMMAR AUTOCORRECTOR: STAGED LoRA FINE-TUNING")
    print("=" * 80)
    print(f"Base Model:       {BASE_MODEL_NAME} (770M)")
    print(f"Compute Device:   {device_name}")
    print(f"Precision Mode:   {precision_mode}")
    print(f"Input Prompt:     '{INPUT_PREFIX}<sentence>'")
    print(f"Batch Strategy:   Effective Batch 32 (8 per device x 4 accumulation steps)")
    print(f"LoRA Hyperparams: r=16, alpha=32, dropout=0.05 on [q, k, v, o, wi, wo]")
    print("=" * 80 + "\n")

    data_dir, ckpt_dir, export_dir, base_dir = get_environment_paths(args.data_dir)

    # 1. Load Tokenizer & Base Model
    print(f"Loading tokenizer and base model weights from '{BASE_MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    torch_dtype = torch.bfloat16 if is_bf16 else torch.float32

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch_dtype,
    )

    # 2. Configure LoRA
    print("Applying LoRA adapter to attention and feed-forward layers...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "k", "v", "o", "wi", "wo"],
        bias="none",
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing to conserve VRAM on 16GB GPUs
    model.gradient_checkpointing_enable()

    # Load validation data (wi_dev + jfleg_dev)
    val_inputs, val_targets = load_validation_pairs(data_dir)
    val_dataset = GECDataset(val_inputs, val_targets, tokenizer)

    # =========================================================================
    # STAGE 1: Broad Grammar, Spelling & Noise Pre-training
    # =========================================================================
    stage1_ckpt_dir = ckpt_dir / "stage1"
    if not args.skip_stage1:
        print("=" * 80)
        print("STAGE 1: 150k Synthetic Noise & General GEC Pairs (1 Epoch, lr=3e-4)")
        print("=" * 80)

        stage1_file = data_dir / "stage1_colab.csv"
        if not stage1_file.exists():
            stage1_file = data_dir / "stage1.csv"

        print(f"Loading Stage 1 training data from {stage1_file}...")
        df_stage1 = pd.read_csv(stage1_file, encoding="utf-8")
        if len(df_stage1) > 150_000:
            df_stage1 = df_stage1.iloc[:150_000]
        print(f"Stage 1 active pairs: {len(df_stage1):,}")

        train_dataset_s1 = GECDataset(
            df_stage1["input"].astype(str).tolist(),
            df_stage1["output"].astype(str).tolist(),
            tokenizer,
        )

        stage1_args = Seq2SeqTrainingArguments(
            output_dir=str(stage1_ckpt_dir),
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,  # 8 * 4 = 32 effective batch size
            per_device_eval_batch_size=8,
            learning_rate=3e-4,
            num_train_epochs=1,
            weight_decay=0.01,
            warmup_steps=500,
            logging_steps=50,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            evaluation_strategy="epoch",
            save_safetensors=True,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            bf16=is_bf16,
            fp16=False,
            gradient_checkpointing=True,
            dataloader_num_workers=2 if os.name != "nt" else 0,
            report_to="none",
        )

        trainer_s1 = Seq2SeqTrainer(
            model=model,
            args=stage1_args,
            train_dataset=train_dataset_s1,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # Auto-resume check for Stage 1
        s1_checkpoint = get_last_checkpoint(str(stage1_ckpt_dir))
        if s1_checkpoint:
            print(f"[RESUME] Found existing Stage 1 checkpoint at '{s1_checkpoint}'. Resuming...")
        else:
            print("[START] Starting Stage 1 training from scratch...")

        trainer_s1.train(resume_from_checkpoint=s1_checkpoint)
        print("Stage 1 training completed successfully.\n")
    else:
        print("[NOTICE] Skipping Stage 1 as requested via --skip_stage1.")

    # =========================================================================
    # STAGE 2: High-Quality In-Domain Adaptation (W&I + FCE + CoEdIT)
    # =========================================================================
    print("=" * 80)
    print("STAGE 2: Domain-Targeted GEC (3 Epochs, lr=1e-4, x3 Gold Oversampling)")
    print("=" * 80)

    stage2_file = data_dir / "stage2.csv"
    print(f"Loading Stage 2 training data from {stage2_file}...")
    df_stage2 = pd.read_csv(stage2_file, encoding="utf-8")
    print(f"Stage 2 base gold pairs: {len(df_stage2):,}")

    if args.use_targeted:
        targeted_file = data_dir / "targeted.csv"
        if not targeted_file.exists():
            targeted_file = Path("data/processed/targeted.csv")
        if targeted_file.exists():
            df_targeted = pd.read_csv(targeted_file, encoding="utf-8")
            print(f"[STAGE 2 MIXING] Loaded {len(df_targeted):,} targeted error pairs from {targeted_file.name}")
            n_real = len(df_stage2)
            n_target = len(df_targeted)
            real_pct = (n_real / (n_real + n_target)) * 100
            print(f"[STAGE 2 MIXING] Real gold: {n_real:,} | Targeted synthetic: {n_target:,} | Real data ratio: {real_pct:.1f}% (Policy requirement: >= 50%)")
            assert real_pct >= 50.0, f"Real data ratio {real_pct:.1f}% is below required 50%!"

            df_stage2 = pd.concat([df_stage2[["input", "output"]], df_targeted[["input", "output"]]], ignore_index=True)
            df_stage2 = df_stage2.sample(frac=1.0, random_state=42).reset_index(drop=True)
            print(f"[STAGE 2 MIXING] Combined Stage 2 active pairs: {len(df_stage2):,}")
        else:
            print(f"[WARNING] --use_targeted specified but '{targeted_file}' was not found. Continuing with pure stage2.csv.")

    train_dataset_s2 = GECDataset(
        df_stage2["input"].astype(str).tolist(),
        df_stage2["output"].astype(str).tolist(),
        tokenizer,
    )

    stage2_ckpt_dir = ckpt_dir / "stage2"
    stage2_args = Seq2SeqTrainingArguments(
        output_dir=str(stage2_ckpt_dir),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,  # 8 * 4 = 32 effective batch size
        per_device_eval_batch_size=8,
        learning_rate=1e-4,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_steps=300,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        evaluation_strategy="epoch",
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=is_bf16,
        fp16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=2 if os.name != "nt" else 0,
        report_to="none",
    )

    trainer_s2 = Seq2SeqTrainer(
        model=model,
        args=stage2_args,
        train_dataset=train_dataset_s2,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Auto-resume check for Stage 2
    s2_checkpoint = get_last_checkpoint(str(stage2_ckpt_dir))
    if s2_checkpoint:
        print(f"[RESUME] Found existing Stage 2 checkpoint at '{s2_checkpoint}'. Resuming...")
    else:
        print("[START] Starting Stage 2 training from scratch...")

    trainer_s2.train(resume_from_checkpoint=s2_checkpoint)
    print("Stage 2 training completed successfully.\n")

    # =========================================================================
    # EXPORT: Merge LoRA Weights & Package Model
    # =========================================================================
    print("=" * 80)
    print("FINAL EXPORT: MERGING LoRA WEIGHTS & CREATING ARCHIVE")
    print("=" * 80)

    export_model_dir = export_dir / "gec-model"
    print("Merging trained LoRA adapter weights directly into the base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged standalone model to '{export_model_dir}'...")
    merged_model.save_pretrained(str(export_model_dir))
    tokenizer.save_pretrained(str(export_model_dir))

    zip_base = base_dir / "gec-model"
    print(f"Creating compressed distribution archive: '{zip_base}.zip'...")
    shutil.make_archive(str(zip_base), "zip", root_dir=str(export_model_dir))

    final_zip = base_dir / "gec-model.zip"
    if final_zip.exists():
        size_mb = final_zip.stat().st_size / (1024 * 1024)
        print("\n" + "=" * 80)
        print(f"[SUCCESS] Packaging complete: {final_zip.name}")
        print(f"  * File Path: {final_zip}")
        print(f"  * File Size: {size_mb:.2f} MB ({size_mb / 1024:.2f} GB)")
        print("=" * 80)
        print("\nNext Steps:")
        print("  1. Download 'gec-model.zip' from Google Drive or Colab sidebar to your laptop.")
        print("  2. Place 'gec-model.zip' in your project root or Downloads folder.")
        print("  3. Run 'python scripts/install_model.py' to unpack it into 'model/gec-model/'.")
        print("  4. Start the app: 'run.bat' (Windows) or './run.sh' (Linux/macOS).")


if __name__ == "__main__":
    train_gec()
