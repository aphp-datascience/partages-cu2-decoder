"""Fine-tune the decoder on your own coded stays (LoRA, one GPU), then merge the adapter into a new model folder.

    uv run --extra finetune python -m scripts.finetune --model models/cu2-icd10-dp-decoder \
        --data my_train.csv --out models/decoder-finetuned
    uv run python -m scripts.predict --model models/decoder-finetuned/merged --data my_test.csv --out preds.csv

Input: the stays file of the PARTAGES guidelines, with `primary_diag` (the PMSI DP of the RUM, Z codes
included) as the target. Same prompt, same cut of long records and same answer as the original training; the
loss covers the answer only. The merged model gets a copy of `resources/`, its predictable codes extended with
the codes of your training stays (otherwise an answer on such a code would be mapped back to another code).
Several GPUs: `uv run --extra finetune torchrun --standalone --nproc_per_node N -m scripts.finetune ...`.
Keep the test stays out of the training file.
"""

import argparse
import inspect
import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decoder.acts import load_ccam, with_acts  # noqa: E402
from decoder.icd10 import ICD10  # noqa: E402
from decoder.model import MAX_MODEL_LEN, fit_prompt, target  # noqa: E402
from decoder.stays import read_stays  # noqa: E402


class Stays(Dataset):
    def __init__(self, texts, answers, tok):
        end = tok.convert_tokens_to_ids("<|im_end|>")
        self.ex, cut = [], 0
        for text, ans in zip(texts, answers):
            a = tok(ans, add_special_tokens=False)["input_ids"] + [end]
            p, was_cut = fit_prompt(tok, text, MAX_MODEL_LEN - len(a))
            cut += was_cut
            self.ex.append((p + a, [-100] * len(p) + a))
        print(f"  {len(self.ex)} stays, {cut} cut at their end", flush=True)

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        ids, lab = self.ex[i]
        return {"input_ids": ids, "labels": lab}


def collate(batch, pad_id: int):
    n = max(len(x["input_ids"]) for x in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, x in enumerate(batch):
        k = len(x["input_ids"])
        ids[i, :k], lab[i, :k], att[i, :k] = torch.tensor(x["input_ids"]), torch.tensor(x["labels"]), 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, type=Path, help="the downloaded decoder folder")
    ap.add_argument("--data", required=True, type=Path, help="training stays (CSV of the guidelines, with primary_diag)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-acts", action="store_true", help="do not add the CCAM line")
    ap.add_argument("--n-dev", type=int, default=300, help="stays held out for the validation loss")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(a.seed)
    icd = ICD10(a.model / "resources")
    df = read_stays(a.data)
    if "primary_diag" not in df:
        raise SystemExit(f"{a.data}: no primary_diag column (the target of the fine-tuning)")
    df["primary_diag"] = [icd.canon(c) if isinstance(c, str) else None for c in df["primary_diag"]]
    n0 = len(df)
    df = df[df["primary_diag"].isin(icd.cim)]
    print(f"{n0} stays ; {len(df)} with a DP of the ICD-10 ; {df['primary_diag'].nunique()} codes", flush=True)
    if not a.no_acts:
        ccam = load_ccam()
        df["text"] = [with_acts(t, c, ccam) for t, c in zip(df["text"], df["procedure"])]
    df = df.sample(frac=1.0, random_state=0)
    dev, train = df.head(a.n_dev), df.iloc[a.n_dev:]

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_ds = Stays(train["text"], [target(c, icd) for c in train["primary_diag"]], tok)
    dev_ds = Stays(dev["text"], [target(c, icd) for c in dev["primary_diag"]], tok)

    from peft import LoraConfig, get_peft_model

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    kw = dict(
        output_dir=str(a.out), seed=a.seed, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum,
        per_device_eval_batch_size=a.batch, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, logging_steps=20, group_by_length=True, eval_strategy="steps",
        eval_steps=200, save_strategy="no", report_to=[], remove_unused_columns=False, dataloader_num_workers=2)
    args = TrainingArguments(**{k: v for k, v in kw.items() if k in accepted})
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id))
    trainer.train()
    trainer.save_model(str(a.out / "adapter"))
    if args.process_index == 0:
        dest = a.out / "merged"
        trainer.model.merge_and_unload().save_pretrained(dest, safe_serialization=True)
        tok.save_pretrained(dest)
        shutil.copytree(a.model / "resources", dest / "resources", dirs_exist_ok=True)
        labels = sorted(icd.labels | set(train["primary_diag"]))
        (dest / "resources" / "labels_cim.json").write_text(json.dumps(labels))
        print(f"-> {dest} ({len(labels)} predictable codes)", flush=True)


if __name__ == "__main__":
    main()
