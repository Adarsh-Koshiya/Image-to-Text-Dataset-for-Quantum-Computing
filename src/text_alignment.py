"""
text_alignment.py

Extract descriptive text passages related to figures and compute
character-level text positions.

Definition of text_positions (IMPORTANT FOR DOCUMENTATION)
---------------------------------------------------------
Each (begin, end) tuple refers to 0-based character offsets into the
deterministic string `TextAligner.full_text`, which is constructed by
concatenating all .tex files in sorted path order, separated by '\\n'.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple, Optional

from pylatexenc.latex2text import LatexNodes2Text
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextAligner:
    """
    Aligns LaTeX textual descriptions with figures and computes
    character offsets in a deterministic global text representation.
    """
    # \ref{...}, \cref{...}, \Cref{...}, \autoref{...}
    _REF_CMD_RE = r"(?:ref|cref|Cref|autoref)"

    def __init__(self, latex_root: str):
        self.latex_root = latex_root
        self.full_text, self.file_offsets = self._build_global_text()

        # LaTeX -> plain text converter (general-purpose)
        self._l2t = LatexNodes2Text()  # [web:98]

        # Sentence splitter (lightweight, deterministic)
        self._nlp = spacy.blank("en")
        self._nlp.add_pipe("sentencizer")  # [web:126]

    def extract_descriptions(
        self,
        figure_index: int,
        labels: Optional[List[str]] = None,
        caption: Optional[str] = None,
        max_spans: int = 3,
        top_k_sentences: int = 2,
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Returns (descriptions, text_positions).

        descriptions: cleaned natural-language text (plain text)
        text_positions: offsets (begin,end) into self.full_text for the *source LaTeX spans*
        """
        descriptions: List[str] = []
        positions: List[Tuple[int, int]] = []

        # Pre-clean caption into plain text too (so TF-IDF compares clean strings)
        cap_plain = self._latex_to_text(caption or "")

        # Collect candidate (start,end) spans in full_text
        spans: List[Tuple[int, int]] = []

        # 1) label-based: find \ref/\cref occurrences and take paragraph spans around them
        if labels:
            for lab in self._unique_preserve_order([x.strip() for x in labels if x and x.strip()]):
                ref_pat = re.compile(
                    rf"\\{self._REF_CMD_RE}\s*\{{\s*{re.escape(lab)}\s*\}}",
                    re.IGNORECASE,
                )
                
                for match in ref_pat.finditer(self.full_text):
                    s = self._expand_to_paragraph_start(match.start())
                    e = self._expand_to_paragraph_end(match.end())
                    spans.append((s, e))

        # 2) fallback: "Fig. N" textual matching
        if not spans:
            fig_pat = re.compile(
                rf"(?:\bfig(?:ure)?\.?\s*~?\s*{int(figure_index)}\b)(?:\s*\(?\s*[a-z]\s*\)?)?",
                re.IGNORECASE,
            )
            for match in fig_pat.finditer(self.full_text):
                s = self._expand_to_paragraph_start(match.start())
                e = self._expand_to_paragraph_end(match.end())
                spans.append((s, e))

        # Deduplicate spans, keep deterministic order, limit count
        spans = self._unique_preserve_order(spans)[:max_spans]

        for (s, e) in spans:
            raw_span = self.full_text[s:e].strip()
            if not raw_span:
                continue

            plain = self._latex_to_text(raw_span)
            if not plain:
                continue

            # NLP step: sentence selection by caption relevance
            selected = self._select_relevant_sentences(
                plain_text=plain,
                caption_text=cap_plain,
                k=top_k_sentences,
            )
            if not selected:
                continue

            if selected not in descriptions:
                descriptions.append(selected)
                positions.append((s, e))

        return descriptions, positions

    # -----------------------------
    # NLP / conversion helpers
    # -----------------------------
    def _latex_to_text(self, s: str) -> str:
        # pylatexenc latex_to_text converts LaTeX to readable plain text [web:98]
        out = self._l2t.latex_to_text(s or "")
        out = re.sub(r"<[^>]+>", " ", out)
        out = re.sub(r"\s+", " ", out).strip()
        return out

    def _select_relevant_sentences(self, plain_text: str, caption_text: str, k: int) -> str:
        doc = self._nlp(plain_text)
        sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) >= 25]

        if not sentences:
            return ""

        # If caption missing, just return first 1-2 sentences deterministically
        if not caption_text:
            return " ".join(sentences[: max(1, k)])

        vec = TfidfVectorizer(stop_words="english")  # [web:121]
        X = vec.fit_transform([caption_text] + sentences)
        sims = cosine_similarity(X[0:1], X[1:]).ravel()  # [web:121]

        # Choose top-k by similarity, deterministic tie-breaking by index
        idx = sims.argsort()[::-1][: max(1, k)]
        chosen = [sentences[i] for i in idx if sims[i] > 0.03]

        if not chosen:
            # fallback: first sentence
            return sentences[0]

        return " ".join(chosen)

    # -----------------------------
    # Text span helpers
    # -----------------------------
    def _build_global_text(self) -> Tuple[str, Dict[str, int]]:
        full_text = ""
        file_offsets: Dict[str, int] = {}
        current_offset = 0

        tex_files = sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(self.latex_root)
            for f in files
            if f.endswith(".tex")
        )

        for tex_file in tex_files:
            try:
                with open(tex_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue

            file_offsets[tex_file] = current_offset
            full_text += content + "\n"
            current_offset = len(full_text)

        return full_text, file_offsets

    def _expand_to_paragraph_start(self, index: int) -> int:
        while index > 0 and self.full_text[index - 1] != "\n":
            index -= 1
        while index > 1:
            if self.full_text[index - 2:index] == "\n\n":
                break
            index -= 1
            while index > 0 and self.full_text[index - 1] != "\n":
                index -= 1
        return max(0, index)

    def _expand_to_paragraph_end(self, index: int) -> int:
        n = len(self.full_text)
        while index < n and self.full_text[index] != "\n":
            index += 1
        while index < n - 1:
            if self.full_text[index:index + 2] == "\n\n":
                index += 2
                break
            index += 1
            while index < n and self.full_text[index] != "\n":
                index += 1
        return min(n, index)

    @staticmethod
    def _unique_preserve_order(items):
        seen = set()
        out = []
        for x in items:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out
