"""
arxiv_downloader.py

Download and manage arXiv source archives (.tar.gz).
- Use /src/<id> instead of /e-print/<id> for more reliable "source tarfile" downloads.
- Sanitize arXiv IDs for filesystem paths.
- Add a small delay to be polite to arXiv services (single connection).
"""

from __future__ import annotations

import os
import time
import urllib.request
from typing import Optional


class ArxivSourceDownloader:
    """
    Downloads LaTeX source archives for arXiv papers.

    Notes
    -----
    This class downloads arXiv source packages to ensure access to
    original figure files and LaTeX structure.
    """

    # /src/<id> is the typical "source tarfile" endpoint pattern.
    BASE_URL = "https://arxiv.org/src/"

    def __init__(self, target_dir: str):
        """
        Parameters
        ----------
        target_dir : str
            Directory where downloaded source archives are stored.
        """
        self.target_dir = target_dir
        os.makedirs(self.target_dir, exist_ok=True)

    def download(self, arxiv_id: str) -> Optional[str]:
        """
        Download the source archive for a given arXiv ID.

        Parameters
        ----------
        arxiv_id : str
            arXiv identifier as in the list .

        Returns
        -------
        Optional[str]
            Path to downloaded .tar.gz file, or None if failed.
        """
        arxiv_id = (arxiv_id or "").strip()
        if not arxiv_id:
            return None

        # arXiv list entries often include "arXiv:" prefix; /src/ expects the short id
        short_id = self._to_short_id(arxiv_id)

        url = f"{self.BASE_URL}{short_id}"
        out_path = os.path.join(self.target_dir, f"{self._safe_filename(arxiv_id)}.tar.gz")

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path

        try:
            # Add a user-agent to avoid being blocked by some servers
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "NLP-QuantumCircuitDataset/1.0 (educational use)"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp, open(out_path, "wb") as f:
                f.write(resp.read())
            # Basic sanity check
            if os.path.getsize(out_path) < 100:
                # Check if it's a withdrawn paper
                try:
                    with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if "withdrawn" in content.lower() or "unavailable" in content.lower():
                            print(f"[INFO] Paper {arxiv_id} is withdrawn or unavailable.")
                            os.remove(out_path)
                            return "withdrawn"
                        else:
                            print(f"[ERROR] Small response for {arxiv_id}, likely error page.")
                except Exception:
                    pass
                return None 
            return out_path 

        except Exception as exc:
            print(f"[ERROR] Failed to download {arxiv_id} from {url}: {exc}")
            # If a partial file was written, remove it
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                pass
            return None

    @staticmethod
    def _to_short_id(arxiv_id: str) -> str:
        """
        Convert 'arXiv:2509.16049' -> '2509.16049'.
        Leaves already-short IDs unchanged.
        """
        s = arxiv_id.strip()
        if s.lower().startswith("arxiv:"):
            return s.split(":", 1)[1]
        return s

    @staticmethod
    def _safe_filename(arxiv_id: str) -> str:
        """
        Convert an arXiv ID into a filesystem-safe filename stem.
        """
        s = arxiv_id.strip().replace(":", "_").replace("/", "_")
        return s

class ArxivPdfDownloader:
    """
    Downloads and caches arXiv PDFs for page-number extraction.
    """

    BASE_URL = "https://arxiv.org/pdf/"

    def __init__(self, target_dir: str):
        self.target_dir = target_dir
        os.makedirs(self.target_dir, exist_ok=True)

    def download(self, arxiv_id: str) -> Optional[str]:
        arxiv_id = (arxiv_id or "").strip()
        if not arxiv_id:
            return None

        short_id = ArxivSourceDownloader._to_short_id(arxiv_id)
        url = f"{self.BASE_URL}{short_id}.pdf"

        out_path = os.path.join(
            self.target_dir, f"{ArxivSourceDownloader._safe_filename(arxiv_id)}.pdf"
        )

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "NLP-QuantumCircuitDataset/1.0 (educational use)"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp, open(out_path, "wb") as f:
                f.write(resp.read())

            # sanity check: tiny files are often HTML error pages
            if os.path.getsize(out_path) < 10_000:
                try:
                    os.remove(out_path)
                except Exception:
                    pass
                return None

            time.sleep(0.4)
            return out_path

        except Exception as exc:
            print(f"[ERROR] Failed to download PDF {arxiv_id} from {url}: {exc}")
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                pass
            return None

