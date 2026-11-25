"""Prompt and answer of the decoder, exactly as in its training.

The model reads a system prompt (in French) and the record of the stay, and writes one line of JSON:
`{"libelle": "<ICD-10 label>", "code": "<code without dot>"}`. Generation is greedy and starts from
`{"libelle": "` so that the model always answers in this format. Records longer than the context are cut
at their END (the CCAM line and the beginning of the reports are kept), as in training.
"""

import json
import re

from .icd10 import ICD10

SYSTEM = (
    "Tu es médecin DIM, expert du codage PMSI MCO. À partir du dossier d'un séjour hospitalier, tu "
    "identifies le DIAGNOSTIC PRINCIPAL (DP) et son code CIM-10 FR.\n"
    "Réponds UNIQUEMENT par un objet JSON sur une seule ligne :\n"
    '{"libelle": "<libellé CIM-10 du diagnostic>", "code": "<code CIM-10 sans point>"}'
)
PRIME = '{"libelle": "'
MAX_MODEL_LEN = 4096  # context length of the training
ANSWER_TOKENS = 96


def prompt_of(tok, text: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Dossier :\n{text.strip()}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def fit_prompt(tok, text: str, budget: int) -> tuple[list[int], bool]:
    """Token ids of the prompt within `budget` tokens; if too long, the TEXT is cut at its end."""
    ids = tok(prompt_of(tok, text), add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return ids, False
    body = tok(text.strip(), add_special_tokens=False)["input_ids"]
    keep = body[: max(0, budget - (len(ids) - len(body)) - 8)]
    return tok(prompt_of(tok, tok.decode(keep)), add_special_tokens=False)["input_ids"][:budget], True


def target(code: str, icd: ICD10) -> str:
    """The training answer for a code (its ICD-10 label, then the code)."""
    return json.dumps({"libelle": icd.label(code), "code": code}, ensure_ascii=False)


def parse_answer(generated: str, icd: ICD10) -> tuple[str | None, str]:
    """(DP, label written by the model). A code the model may not predict (unknown, non-leaf) is replaced by
    the first predictable code found for its label, except for a Z code; None if there is no code."""
    raw = PRIME + generated
    m = re.search(r"\{[^{}]*\}", raw)
    try:
        j = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        j = {}
    label = str(j.get("libelle") or "")
    code = icd.canon(str(j.get("code") or ""))
    if code and code not in icd.labels and label and not code.startswith("Z"):
        hits = icd.search(label)
        if hits:
            code = hits[0]
    return code or None, label
