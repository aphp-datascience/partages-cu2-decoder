"""Code the principal diagnosis (DP) of every stay of a file with the decoder (one generation per stay).

    uv run python -m scripts.predict --model models/cu2-icd10-dp-decoder --data my_test.csv --out preds.csv

Input: the stays file of the PARTAGES guidelines (decoder/stays.py): CSV with `id`, `speciality`, `text`,
`procedure`, `primary_diag` (the last one optional, only used to print the accuracy). The CCAM codes of
`procedure` are written as the first line of the text, as in training (`--no-acts` to skip).
Output (CSV, or parquet if --out ends with .parquet): one row per stay with the suggested DP
(`primary_diag_pred`) and the ICD-10 label written by the model (`label`).
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decoder.acts import load_ccam, with_acts  # noqa: E402
from decoder.icd10 import ICD10  # noqa: E402
from decoder.model import ANSWER_TOKENS, MAX_MODEL_LEN, PRIME, fit_prompt, parse_answer  # noqa: E402
from decoder.stays import read_stays  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="local dir or Hugging Face repo id of the model")
    ap.add_argument("--data", required=True, type=Path, help="stays file (CSV or parquet)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-acts", action="store_true", help="do not add the CCAM line")
    ap.add_argument("--gpus", type=int, default=1, help="tensor parallel GPUs")
    ap.add_argument("--quantization", choices=["fp8"], help="load the weights in FP8 (about 8.5 GB instead of 16 GB)")
    a = ap.parse_args()

    model = Path(a.model)
    if not model.is_dir():  # Hugging Face repo id ("org/name" or "org/name@revision")
        from huggingface_hub import snapshot_download

        repo, _, rev = a.model.partition("@")
        model = Path(snapshot_download(repo, revision=rev or None))
    icd = ICD10(model / "resources")

    df = read_stays(a.data)
    texts = list(df["text"])
    if not a.no_acts:
        ccam = load_ccam()
        texts = [with_acts(t, c, ccam) for t, c in zip(texts, df["procedure"])]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model)
    prompts, cut = [], 0
    for t in texts:
        ids, was_cut = fit_prompt(tok, t, MAX_MODEL_LEN - 128)
        cut += was_cut
        prompts.append(tok.decode(ids) + PRIME)
    print(f"{len(texts)} stays ; {cut} cut at their end to fit {MAX_MODEL_LEN} tokens", flush=True)
    # cascade attention off: vLLM produced degenerate outputs with it on small batches sharing a long prefix
    llm = LLM(model=str(model), max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.9, seed=0, dtype="bfloat16",
              tensor_parallel_size=a.gpus, quantization=a.quantization, disable_cascade_attn=True)
    stops = [i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")) if i]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=ANSWER_TOKENS, stop_token_ids=stops))
    answers = [parse_answer(o.outputs[0].text, icd) for o in outs]
    out = pd.DataFrame({
        "id": df["id"].to_numpy(),
        "primary_diag_pred": [dp for dp, _ in answers],
        "label": [lab for _, lab in answers],
    })
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.suffix.lower() == ".parquet":
        out.to_parquet(a.out, index=False)
    else:
        out.to_csv(a.out, index=False)
    print(f"-> {a.out} ({time.time() - t0:.0f} s) ; no answer {out['primary_diag_pred'].isna().mean():.1%}")

    if "primary_diag" in df:
        g = df["primary_diag"]
        pred = pd.Series([p if isinstance(p, str) else "" for p in out["primary_diag_pred"]])
        s_ = g.notna().to_numpy()
        gg, pp = g[s_].reset_index(drop=True), pred[s_].reset_index(drop=True)
        print(f"suggested DP vs primary_diag, n={int(s_.sum())}: exact {(pp == gg).mean():.1%}, "
              f"4 characters {(pp.str[:4] == gg.str[:4]).mean():.1%}, 3 characters {(pp.str[:3] == gg.str[:3]).mean():.1%}")


if __name__ == "__main__":
    main()
