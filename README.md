# PARTAGES-CU2: LLM decoder for French ICD-10 medical coding

Given the reports of a French hospital stay, this model codes the **principal diagnosis (DP)** of
the stay as in the PMSI (the French hospital activity database), in ICD-10 (CIM-10, ATIH version).
A fine-tuned 8-billion-parameter model (Qwen3-8B) reads the record and writes the ICD-10 label and
code of the DP in a single generation, without tools. It was developed in the PARTAGES project (use
case 2, coding assistance for medical information departments, DIM). The model is released on the
Hugging Face Hub:
**[`PARTAGES-CU2/cu2-icd10-dp-decoder`](https://huggingface.co/PARTAGES-CU2/cu2-icd10-dp-decoder)**.

**Terms used below**

- **DP** (*diagnostic principal*): the health problem that motivated the stay.
- **Z code**: an ICD-10 code of chapter XXI, which describes a reason for contact rather than a
  disease (e.g. `Z51.1` chemotherapy session, `Z09.0` follow-up after surgery).
- **CRC / CRH / CR-OP**: consultation reports, discharge summaries, operative reports.

**In short**

- **For:** suggesting the DP of a stay to the DIM, fast, on large volumes. **Not for:** automatic
  coding without validation; no associated diagnoses (DAS), no related diagnosis (DR).
- **Runs locally** with [vLLM](https://github.com/vllm-project/vllm): no text leaves your
  infrastructure. The model was trained without any real patient data.
- **Fast:** one short generation per stay; on one A100, about 950 synthetic stays per minute.
- **Hardware:** one GPU with ≥ 24 GB of memory.
- **Can be fine-tuned on your own coded stays** with the script of this repository (one GPU).
- **Two CU2 models code the DP from the text:** this decoder, and the
  [CU2 agent](https://github.com/aphp-datascience/partages-cu2-agent), slower, which searches the
  ICD-10 and the MCO guide with tools, also gives the DR and justifies its choice.
- **Mostly trained on surgical stays**, but covers the whole ICD-10 (see
  [Limitations](#limitations)).

## 1. Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/); Linux x86_64 with an NVIDIA GPU and a driver ≥ 580 (CUDA 13).
`uv sync` installs the exact versions of the tested environment (`uv.lock`).

```bash
git clone https://github.com/aphp-datascience/partages-cu2-decoder.git
cd partages-cu2-decoder
uv sync
```

**Model.** The Hub repository is private for now: ask the CU2 team for access, log in, then
download the model once. It ships the ICD-10 index it needs too (`resources/`):

```bash
uv run hf auth login
uv run hf download PARTAGES-CU2/cu2-icd10-dp-decoder --local-dir models/cu2-icd10-dp-decoder
```

On a server without internet access, download on another machine and copy the model folder over.

## 2. Input data

One **CSV file, one row per hospital stay (RUM)**, with the columns of the PARTAGES guidelines
(the format of the PARTAGES Codabench challenge):

| Column | Needed for | Content | Example |
|---|---|---|---|
| `id` | always | a unique id of the stay | `RUM_000123` |
| `speciality` | — | the specialty of the stay (not used by the model) | `CHIRURGIE VISCERALE` |
| `text` | always | the reports of the stay (CRC, CRH, CR-OP) concatenated, separated by a blank line or by the `[...]` marker of the guidelines | `Compte rendu de consultation …` |
| `procedure` | strongly recommended | the CCAM codes of the stay, separated by spaces; empty if none | `NDPA014 NDPA008` |
| `primary_diag` | evaluation, fine-tuning | the PMSI principal diagnosis (DP) of the RUM, as coded, Z codes included | `M6534`, `Z511` |

The model reads at most 4 096 tokens (about 2 500 words): longer records are cut at their end (the
procedures and the beginning of the reports are kept). **Make the test set as clean as possible:** a
noisy test set scores the model against coding errors. If you can, use stays reviewed by the DIM,
ideally coded twice independently, and keep only the stays where both codings agree. A raw accuracy
must not be read as a ranking of hospitals, whose coding practices differ.

## 3. Running and evaluating the decoder

```bash
uv run python -m scripts.predict --model models/cu2-icd10-dp-decoder --data my_test.csv --out preds.csv
```

| Option | What it does |
|---|---|
| `--model` | the downloaded model folder |
| `--data`, `--out` | your stays file, and the file where predictions are written (CSV, or parquet if the name ends with `.parquet`) |
| `--no-acts` | do not write the CCAM line (by default it is built from `procedure`, as in training) |
| `--gpus` | number of GPUs (tensor parallelism) |
| `--quantization fp8` | load the weights in FP8 (about 8.5 GB instead of 16 GB) |

`preds.csv` has one row per stay:

| Column | Meaning |
|---|---|
| `id` | the id of the stay |
| `primary_diag_pred` | the suggested DP, to compare with your `primary_diag` |
| `label` | the ICD-10 label written by the model |

If the file has a `primary_diag` column, the script prints the share of stays where the suggested
DP equals it: exact code, first 4 characters (the Codabench metric) and first 3 characters.

## 4. Fine-tuning on your stays

Coding practices differ between hospitals, and real reports differ from the synthetic texts of the
training. With a few thousand stays coded by your DIM (the same CSV, `primary_diag` filled), adapt
the model on one GPU with ≥ 40 GB of memory:

```bash
uv sync --extra finetune
uv run python -m scripts.finetune --model models/cu2-icd10-dp-decoder --data my_train.csv --out models/decoder-ft
uv run python -m scripts.predict --model models/decoder-ft/merged --data my_test.csv --out preds_ft.csv
```

The script keeps the prompt and the answer of the original training and trains a LoRA adapter (rank
32, one epoch, learning rate 1e-4), merged into a new model folder `merged/`. The codes of your
training stays are added to the codes the model may predict. **Keep the test stays out of the
training file**, and compare the model before and after on the same test stays.

## Model and training

### How the decoder works

The *system prompt* (in French) asks the model to identify the DP of the stay as a DIM physician
would; the *user message* is the record of the stay: the CCAM line
(`Actes (CCAM) : CODE - label ; …`), then the reports. The model answers with one line of JSON,
`{"libelle": "Appendicite aiguë", "code": "K358"}`. Generation is greedy (two runs give the same
codes) and starts from `{"libelle": "`, so the answer is always in this format.

The model may only suggest the codes of its list (`resources/labels_cim.json` of the model, 13 439
ICD-10 codes). If it writes another code (a 3-character category, an unknown code), the code is replaced by the
first allowed code found for the label it wrote (BM25 over ICD-10 labels, inclusion notes and
synonyms, `decoder/icd10.py`); this concerns very few stays.

### Training data

No real patient data was used for training. The model learns from the training data of the
[CU2 encoder](https://github.com/aphp-datascience/partages-cu2-encoder), the target being the DP of
the stay:

| Source | Stays | What it is | Codes used for training |
|---|---:|---|---|
| **MISTRAL** | 100&nbsp;038 | synthetic surgical stays generated by an LLM | original code; recoded by an LLM agent for about half of them |
| **PARHAF** | 4&nbsp;254 | public patients of [PARHAF](https://huggingface.co/datasets/HealthDataHub/PARHAF) (CC BY 4.0): fictitious patients written by physicians | recoded by an LLM agent; for 954 patients, a reference coding by Claude Opus, partly reviewed by physicians |
| **PARHAF rewrites** | 3&nbsp;140 | synthetic stays rewritten from PARHAF patients | recoded by an LLM agent |
| **Synonyms** | 20&nbsp;000 texts | ICD-10 labels and synonyms | the code itself |

PARHAF and its rewrites are seen 4 times (about 150 000 training examples in all).

**Why recode?** The codes shipped with these sources are noisy: a synthetic stay does not always
match the code it was generated from, and PARHAF reports follow the code of their scenario more or
less closely. The stays were therefore recoded by **Mistral Large 4** used as an agent, with offline
ICD-10 and MCO-guide tools and a mandatory PMSI coding procedure; where the reference coding exists,
it is used instead. Only fictitious or synthetic texts were sent to the APIs.

### Training parameters

- **Base model:** `Qwen/Qwen3-8B` (Apache-2.0).
- **Fine-tuning:** full fine-tuning (all weights), one epoch, learning rate 1e-5 (cosine), batch of
  64 examples, 4 096 tokens; weights released in bf16.
- **Loss:** on the answer only (the prompt and the record are not learnt).

## Results (before any fine-tuning on real hospital data)

Accuracy of the suggested **DP** against the PMSI DP of the stay, i.e. what `primary_diag`
contains: exact code, first 4 characters (the metric of the PARTAGES Codabench challenge), and
3-character category.

**Physician-reviewed AP-HP stays** (May 2026, text = acts + CRC + CRH + CR-OP).

Views:

- **flattened**: every review is a row.
- **agreement + single**: one row per stay, reviewed once or by several reviewers who agree.
- **agreement**: only stays with several agreeing reviewers.

Scope: **ICD-10** = all stays, **surgery** = stays coded in the surgical scope of the use case
(~3 500 codes).

| View | Scope | Stays | Exact | 4 char. | 3 char. |
|---|---|---:|---:|---:|---:|
| flattened | ICD-10 | 310 | 43.2 % | 44.8 % | 54.2 % |
| flattened | surgery | 259 | 48.6 % | 50.2 % | 59.9 % |
| agreement + single | ICD-10 | 195 | 44.1 % | 46.2 % | 54.4 % |
| agreement + single | surgery | 167 | 48.5 % | 50.3 % | 59.3 % |
| agreement | ICD-10 | 47 | 68.1 % | 68.1 % | 72.3 % |
| agreement | surgery | 42 | 71.4 % | 71.4 % | 73.8 % |

- **Z codes:** when the DP is a Z code (sessions, follow-up…), the decoder finds it about as often
  as other DPs: 46.1 % exact (76 stays, flattened).
- **Compared with the other CU2 models** on the same 310 flattened stays (exact code): encoder
  41.0 %, decoder 43.2 %, agent 45.5 %. The differences are within the uncertainty of such a set.
- **Small sets:** 95 % confidence interval about ±7 points on ~200 stays, ±13 points on ~50.
- **The reference coding is not certain either:** even reviewed PMSI coding contains errors, and
  some stays can legitimately be coded in more than one way. The **agreement** view, where
  reviewers concur, is the most reliable.

## Limitations

- **Mostly trained on surgery.** MISTRAL, by far the largest source, only covers ~4 000 codes,
  almost all surgical. The model is clearly more accurate on surgical codes (compare the
  **surgery** rows above); it rarely suggests a code it never saw in training.
- Coding **assistance** only; no justification is given.
- One DP per stay; no DR, no associated diagnoses.
- Records beyond 4 096 tokens are cut at their end.
- Trained on synthetic or fictitious notes: expect a domain shift on real hospital notes.
- **Reference codes are not certain.** The codes used for training and evaluation are not 100 %
  sure: synthetic and LLM-recoded labels contain errors, reviewed PMSI coding too, and some stays
  can legitimately be coded in more than one way.
- Evaluated on real stays of a single hospital group (AP-HP, single-RUM stays with CRC, CRH or
  CR-OP reports; AP-HP data used for evaluation only, never for training). Other report types
  and multi-RUM stays have not been evaluated specifically.

## Repository layout

```
decoder/model.py       prompt, cut of long records, answer parsing (as in training)
decoder/icd10.py       ICD-10 index of the model: predictable codes, search used to map back an answer
decoder/acts.py        CCAM procedure line of the input text
decoder/stays.py       reading the stays file (CSV of the guidelines)
scripts/predict.py     inference (and accuracy) on a file
scripts/finetune.py    fine-tuning on your own coded stays (LoRA)
data/ccam_2026.csv     CCAM 2026 nomenclature (ATIH)
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
