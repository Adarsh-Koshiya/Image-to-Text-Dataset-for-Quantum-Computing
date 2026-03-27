"""
quantum_circuit_filter.py

Quantum circuit detection logic refactored from
extract_only_quantum_circuits.py.

This module ONLY decides whether a given image
represents a quantum circuit, using:

1) global LaTeX context (text_gate)
2) weak visual wire detection (wire_ratio_score)

No file I/O, no JSON, no CSV here.
"""

import os
import re
from typing import Dict, List

import cv2
import numpy as np


class QuantumCircuitFilter:
    """
    Pipeline-compatible quantum circuit filter.
    Expected figure dict fields:
    - image_paths : List[str]
    - paper_tex_lower : str (full paper LaTeX, lowercased)
    """
    WIRE_RATIO_THRESH = 0.015
    KERNEL_DIV = 35

    STRONG_KW = {
        "quantikz",
        "qcircuit",
        "quantum circuit",
        "quantum-circuit",
        "quantumcircuit",
        "circuit",
        "yquant",
        }

    GATE_KW = {
        "cnot", "cx", "cz", "swap", "toffoli","ecr","rzx"
        "\\gate", "\\ctrl", "\\targ", "\\meter",
        "\\qw", "\\lstick", "\\rstick", "\\qwbundle",
        "\\multigate",
        "\\ghost",}
    EXCLUDE_KW = {
        "plot", "histogram", "hist", "chart", "graph",
        "bar", "bar chart", "line", "line plot", "scatter", "scatter plot",
        "heatmap", "confusion", "contour", "contour plot","error","target",
        "spectrum", "spectra", "spectrogram", "curve", "backend","improvement",
        "axis", "axes", "accuracy", "loss", "latency", "runtime","lambda",
        "time", "frequency", "probability", "distribution", "distributions",
        "table", "layout", "micrograph", "microscope", "measurement", "measurements",
        "map", "data","Data", "results", "fig", "figure", "diagram", "illustration", 
        "overview", "schematic", "process", "fridge", "spreading", "nonloc",
        "error", "error bar", "calibration", "benchmark", "performance","duration",
        "density", "correlation", "correlation plot", "matrix", "matrices","comparison","communication"
    }

    def is_quantum_circuit(self, figure: Dict) -> bool:
        """
        Decide whether ANY image of this figure
        represents a quantum circuit.
        """
        image_paths = figure.get("image_paths", [])
        tex = figure.get("paper_tex", "")
        if not image_paths or not tex:
            return False

        tex_lower = tex.lower()
        for img_path in image_paths:
            if self._is_circuit_image(img_path, tex_lower):
                return True
        return False

    def extract_gates(self, caption: str, latex_block: str) -> List[str]:
        """
        Extract quantum gates from caption and latex_block.
        """
        gates = []
        text = (caption or "") + " " + (latex_block or "")
        for gate in self.GATE_KW:
            if gate.lower() in text.lower():
                gates.append(gate.replace('\\', ''))
        return list(set(gates))

    # =============================
    # Core logic (from script)
    # =============================
    def _is_circuit_image(self, img_path: str, tex_lower: str) -> bool:
        """
        Apply text gate first, then weak visual check.
        """
        if not self._text_gate(img_path, tex_lower):
            return False

        img = self._load_image(img_path)
        if img is None or img.size == 0:
            return False

        return self._wire_ratio_ok(img)

    # -----------------------------
    # Text gate (PRIMARY SIGNAL)
    # -----------------------------
    def _text_gate(self, img_path: str, tex_lower: str) -> bool:
        name = os.path.basename(img_path).lower()

        # filename hit
        flat_name = name.replace(" ", "")
        if any(k.replace(" ", "") in flat_name for k in self.EXCLUDE_KW):
            return False
        if any(k.replace(" ", "") in flat_name for k in self.STRONG_KW | self.GATE_KW) or any(k.replace(" ", "") in flat_name for k in {"cnot", "cx", "cz", "toffoli", "ecr", "rzx"}) or any(k.replace(" ", "") in flat_name for k in {"quantikz", "qcircuit"}):
            return True
        if not tex_lower:
            return False

        # check occurrences of stem in tex (handles includegraphics without extension)
        stem = os.path.splitext(name)[0].lower()
        if len(stem) < 3:
            return False

        # search for stem OR stem without common suffixes
        pat = re.escape(stem)
        for m in re.finditer(pat, tex_lower):
            a = max(0, m.start() - 2000)
            b = min(len(tex_lower), m.end() + 2000)
            ctx = tex_lower[a:b]

            if any(k in ctx for k in self.EXCLUDE_KW):
                continue
            if any(k in ctx for k in self.STRONG_KW) or any(g in ctx for g in self.GATE_KW):
                return True

        return False

    def _load_image(self, img_path: str) -> np.ndarray:
        """
        Load image, handling PDFs if possible.
        """
        suffix = os.path.splitext(img_path)[1].lower()
        if suffix == '.pdf':
            try:
                from pdf2image import convert_from_path
                pages = convert_from_path(img_path, dpi=150, first_page=1, last_page=1)
                if pages:
                    pil_img = pages[0]
                    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
            except (ImportError, Exception):
                return None
        elif suffix == '.eps':
            return None
        else:
            try:
                return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            except Exception:
                return None

    # -----------------------------
    # visual check (SECONDARY)
    # -----------------------------
    def wire_ratio_score(self, cv_img: np.ndarray) -> float:
        if cv_img is None or cv_img.size == 0:
            return 0.0

        h, w = cv_img.shape[:2]
        if max(h, w) > 1600:
            s = 1600 / max(h, w)
            cv_img = cv2.resize(cv_img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            h, w = cv_img.shape[:2]

        bw = cv2.adaptiveThreshold(
            cv_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 7
        )
        inv = 255 - bw

        k = max(15, w // self.KERNEL_DIV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))

        horiz = cv2.erode(inv, kernel, iterations=1)
        horiz = cv2.dilate(horiz, kernel, iterations=1)

        return float(np.count_nonzero(horiz)) / float(h * w)

    def _wire_ratio_ok(self, img: np.ndarray) -> bool:
        return self.wire_ratio_score(img) >= self.WIRE_RATIO_THRESH
