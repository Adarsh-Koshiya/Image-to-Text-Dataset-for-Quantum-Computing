"""
latex_extractor.py

Extract and process LaTeX source files from downloaded arXiv source archives.
"""

from __future__ import annotations

import os
import tarfile
from typing import Optional


class LatexSourceExtractor:
    """
    Extracts and processes LaTeX source files from arXiv source tarballs.
    """

    def __init__(self, source_extract_dir: str):
        """
        Parameters
        ----------
        source_extract_dir : str
            Directory where extracted LaTeX sources will be stored.
        """
        self.source_extract_dir = source_extract_dir
        self.extract_root = source_extract_dir
        os.makedirs(self.source_extract_dir, exist_ok=True)

    def extract(self, archive_path: str, arxiv_id: str) -> Optional[str]:
        """
        Extract the downloaded arXiv source archive to:
            <source_extract_dir>/<safe_arxiv_id>/

        Parameters
        ----------
        archive_path : str 
            Path to the downloaded .tar.gz file.
        arxiv_id : str
            arXiv identifier from the paper list (often includes 'arXiv:' prefix).

        Returns
        -------
        Optional[str]
            Path to extracted directory, or None if failed.
        """
        if not archive_path or not os.path.exists(archive_path):
            print(f"[ERROR] Archive does not exist: {archive_path}")
            return None

        safe_id = self._safe_dirname(arxiv_id)
        extract_dir = os.path.join(self.source_extract_dir, safe_id)

        if os.path.exists(extract_dir) and os.path.isdir(extract_dir):
            return extract_dir  # already extracted

        os.makedirs(extract_dir, exist_ok=True)

        try:
            with tarfile.open(archive_path, "r:*") as tar:
                members = tar.getmembers()
                safe_members = [m for m in members if self._is_safe_member(m, extract_dir)]
                if not safe_members:
                    print(f"[ERROR] No safe members found in archive for {arxiv_id}")
                    return None

                # Use safe extraction compatible across Python versions
                for m in safe_members:
                    tar.extract(m, path=extract_dir)

            return extract_dir

        except Exception as exc:
            print(f"[ERROR] Failed to extract {arxiv_id}: {exc}")
            return None

    def extract_main_tex(self, arxiv_id: str) -> Optional[str]:
        """
        Heuristically find a main .tex file in the extracted directory.

        Note: This project currently parses all .tex files, so main-tex detection
        is optional, but keeping it can be useful for debugging.

        Returns
        -------
        Optional[str]
            Path to a likely main .tex file, or None.
        """
        safe_id = self._safe_dirname(arxiv_id)
        extract_dir = os.path.join(self.source_extract_dir, safe_id)
        if not os.path.exists(extract_dir):
            return None

        # prefer common names
        candidates = []
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.endswith(".tex"):
                    candidates.append(os.path.join(root, f))

        if not candidates:
            return None

        # deterministic order
        candidates = sorted(candidates)

        # try to prefer main.tex
        for c in candidates:
            if os.path.basename(c).lower() == "main.tex":
                return c

        return candidates[0] #

    @staticmethod
    def _safe_dirname(arxiv_id: str) -> str:
        """
        Convert arXiv ID to a filesystem-safe directory name.
        """
        s = (arxiv_id or "").strip()
        s = s.replace(":", "_").replace("/", "_")
        return s

    @staticmethod
    def _is_safe_member(member: tarfile.TarInfo, extract_dir: str) -> bool:
        """
        Prevent path traversal during tar extraction.
        """
        # member.name can be something like "../../etc/passwd"
        member_path = os.path.join(extract_dir, member.name)
        abs_extract_dir = os.path.abspath(extract_dir)
        abs_member_path = os.path.abspath(member_path)
        return abs_member_path.startswith(abs_extract_dir)
