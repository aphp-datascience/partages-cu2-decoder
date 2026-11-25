"""ICD-10 FR (ATIH) index shipped with the model, in its `resources/` folder: codes, labels, the codes the
model may predict (`labels_cim.json`), and a BM25 search over labels, inclusion notes and synonyms.

The search is only used to map back an answer whose code is not predictable (an unknown or non-leaf code):
the code is replaced by the first predictable code found for the label the model wrote.
"""

import json
import re
import unicodedata
from pathlib import Path

from rank_bm25 import BM25Okapi

WORD = re.compile(r"[a-z0-9]+")
STOP = set(
    "de la le les des du un une et ou a au aux en d l par pour sur sans avec dans est sont "
    "non autre autres precise precisee sai".split()
)


def norm_tokens(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return [w for w in WORD.findall(t) if w not in STOP and len(w) > 1]


def norm_code(code: str) -> str:
    """Upper case, dots and spaces removed; the `+` of PMSI extensions (C189+8) is kept."""
    return re.sub(r"[^A-Z0-9+]", "", str(code).upper())


class ICD10:
    def __init__(self, resources: Path):
        resources = Path(resources)
        self.cim = {}
        with open(resources / "index" / "cim10.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                self.cim[r["code"]] = r
        self.labels = set(json.loads((resources / "labels_cim.json").read_text()))
        # a model may drop the `+` of an extension (C1898 for C189+8): map it back when unambiguous
        self.alias = {c.replace("+", ""): c for c in self.cim if "+" in c and c.replace("+", "") not in self.cim}
        self._bm25 = None
        self._synonyms = resources / "syn_all.parquet"

    def canon(self, code: str) -> str:
        c = norm_code(code)
        return c if c in self.cim else self.alias.get(c, c)

    def label(self, code: str) -> str:
        r = self.cim.get(self.canon(code)) or {}
        return r.get("libelle") or r.get("libelle_pmsi") or ""

    def _index(self) -> None:
        terms, codes = [], []
        for code, r in self.cim.items():
            if r.get("mco_had") == "3":  # not codable (non-leaf)
                continue
            for t in [r.get("libelle") or ""] + list(r.get("inclusions", [])):
                if t:
                    terms.append(t), codes.append(code)
        if self._synonyms.exists():
            import pandas as pd
            import pyarrow.parquet as pq

            col = "dp" if "dp" in pq.read_schema(self._synonyms).names else "affection"
            syn = pd.read_parquet(self._synonyms, columns=["note_text", col]).drop_duplicates().dropna()
            for t, c in zip(syn["note_text"], syn[col]):
                if c in self.cim:
                    terms.append(str(t)), codes.append(c)
        self._codes = codes
        self._bm25 = BM25Okapi([norm_tokens(t) for t in terms])

    def search(self, text: str, k: int = 10) -> list[str]:
        """Among the k codes whose label, inclusion note or synonym best matches `text`, the predictable ones."""
        q = norm_tokens(text)
        if not q:
            return []
        if self._bm25 is None:
            self._index()
        scores = self._bm25.get_scores(q)
        best: list[str] = []  # the k best codes, predictable or not, as in the evaluation
        for i in scores.argsort()[::-1][: 40 * k]:
            if scores[i] <= 0 or len(best) >= k:
                break
            if self._codes[i] not in best:
                best.append(self._codes[i])
        return [c for c in best if c in self.labels]
