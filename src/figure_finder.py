"""
figure_finder.py

Locate LaTeX figure environments and associated image files.

Output format (per figure):
- figure_index : int (global, sequential within a paper)
- caption      : Optional[str]
- image_paths  : List[str] (resolved to existing files)
- tex_file     : str (source .tex file)
- latex_block  : str (raw LaTeX for the figure environment)
- paper_tex    : str (concatenated full paper LaTeX, lowercased)
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional


class FigureFinder:
    """
    Parse LaTeX source to identify figure environments, captions,
    and referenced image files.

    Notes
    -----
    This is a heuristic parser (regex-based). It is designed for robustness
    and determinism rather than perfect TeX parsing.
    """

    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".pdf", ".eps")

    # Support figure and figure* environments
    FIGURE_PATTERN = re.compile(
        r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}",
        re.DOTALL | re.IGNORECASE,
    )

    # Support \caption{...} and \caption[short]{long}
    CAPTION_PATTERN = re.compile(
        r"\\caption(?:\[[^\]]*\])?\{(.+?)\}",
        re.DOTALL | re.IGNORECASE,
    )

    # Support \label{...}
    LABEL_PATTERN = re.compile(
        r"\\label\{([^}]+)\}",
        re.IGNORECASE,
    )

    # Support \includegraphics[...]{path}
    INCLUDEGRAPHICS_PATTERN = re.compile(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
        re.IGNORECASE,
    )

    def __init__(self, latex_root: str):
        """
        Parameters
        ----------
        latex_root : str
            Root directory of extracted LaTeX source for one paper.
        """
        self.latex_root = latex_root
        self.full_paper_tex: str = ""

    def find_figures(self) -> List[Dict]:
        """
        Find all figures in LaTeX source.

        Returns
        -------
        List[Dict]
            List of figure dicts (see module docstring).
        """
        figures: List[Dict] = []
        figure_counter = 0
        self.full_paper_tex = self._read_full_paper_tex()

        # Deterministic traversal order
        for root, _, files in os.walk(self.latex_root):
            for file in sorted(files):
                if not file.endswith(".tex"):
                    continue
                tex_path = os.path.join(root, file)
                parsed = self._parse_tex_file(tex_path, start_index=figure_counter)
                figures.extend(parsed)
                figure_counter = len(figures)

        return figures

    def _read_full_paper_tex(self) -> str:
        """
        Read and concatenate all LaTeX files of the paper.

        Returned text is lowercased for robust matching and reproducibility.
        """
        parts: List[str] = []

        # Deterministic order
        tex_files = sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(self.latex_root)
            for f in files
            if f.endswith(".tex")
        )

        for path in tex_files:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    parts.append(fh.read().lower())
            except Exception:
                continue

        return "\n".join(parts)

    def _parse_tex_file(self, tex_path: str, start_index: int) -> List[Dict]:
        """
        Parse a single .tex file for figures.

        Parameters
        ----------
        tex_path : str
            Path to LaTeX file.
        start_index : int
            Starting index for figure numbering.

        Returns
        -------
        List[Dict]
            Parsed figure metadata.
        """
        try:
            with open(tex_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return []

        blocks = self.FIGURE_PATTERN.findall(content)
        out: List[Dict] = []

        for i, fig_block in enumerate(blocks):
            caption = self._extract_caption(fig_block)
            image_paths = self._extract_image_paths(fig_block, tex_path)
            labels = self._extract_labels(fig_block)

            if not image_paths:
                continue

            out.append(
                {
                    "figure_index": start_index + i + 1,
                    "caption": caption,
                    "image_paths": image_paths,
                    "tex_file": tex_path,
                    "latex_block": fig_block,
                    "paper_tex": self.full_paper_tex,
                    "labels": labels,
                }
            )

        return out

    def _extract_labels(self, fig_block: str) -> List[str]:
        """
        Extract label texts from a figure block.
        """
        matches = self.LABEL_PATTERN.findall(fig_block)
        return [m.strip() for m in matches]

    def _extract_caption(self, fig_block: str) -> Optional[str]:
        """
        Extract caption text from a figure block.
        """
        m = self.CAPTION_PATTERN.search(fig_block)
        if not m:
            return None
        return m.group(1).strip()

    def _extract_image_paths(self, fig_block: str, tex_path: str) -> List[str]:
        """
        Resolve image paths referenced in includegraphics.

        Returns only paths that exist on disk.
        """
        tex_dir = os.path.dirname(tex_path)
        matches = self.INCLUDEGRAPHICS_PATTERN.findall(fig_block)

        images: List[str] = []
        for base_path in matches:
            resolved = self._resolve_image_file(base_path.strip(), tex_dir)
            if resolved:
                images.append(resolved)

        # Deduplicate while preserving order
        seen = set()
        uniq: List[str] = []
        for p in images:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        return uniq

    def _resolve_image_file(self, base_path: str, tex_dir: str) -> Optional[str]:
        """
        Resolve an image file by checking common extensions and search locations.

        Search order (deterministic):
        1) relative to tex_dir
        2) relative to latex_root
        """
        # Normalize ./ prefix
        if base_path.startswith("./"):
            base_path = base_path[2:]

        search_dirs = [tex_dir, self.latex_root]

        # If extension already provided
        for d in search_dirs:
            candidate = os.path.join(d, base_path)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        # Try adding known extensions
        for d in search_dirs:
            for ext in self.IMAGE_EXTENSIONS:
                candidate = os.path.join(d, base_path + ext)
                if os.path.exists(candidate):
                    return os.path.abspath(candidate)

        return None
