"""
dataset_builder.py

Assemble the final image-to-text dataset for quantum circuits.

Outputs:
- dataset JSON: {filename: {arxiv_id, page, figure_number, quantum_gates,
  quantum_problem, descriptions, text_positions}}
- paper_list_counts_<exam_id>.csv in SAME ORDER as paper_list_20.txt:
  processed papers => integer count (0 allowed)
  unprocessed papers (because pipeline stopped at 250) => blank
"""

from __future__ import annotations

import csv
import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Tuple

import spacy
from PIL import Image

from .figure_finder import FigureFinder
from .quantum_circuit_filter import QuantumCircuitFilter
from .text_alignment import TextAligner


class DatasetBuilder:
    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".pdf", ".eps")

    def __init__(
        self,
        exam_id: str,
        images_dir: str,
        output_json: str,
        paper_list_path: str,
        source_root: str,
        pdf_dir: str,
    ):
        self.exam_id = str(exam_id)
        self.images_dir = images_dir
        self.output_json = output_json
        self.paper_list_path = paper_list_path
        self.source_root = source_root
        self.pdf_dir = pdf_dir

        os.makedirs(self.images_dir, exist_ok=True)

        self.circuit_filter = QuantumCircuitFilter()

        self.dataset: Dict[str, Dict] = {}
        self.paper_counts: Dict[str, int] = {}
        self.processed_papers: set[str] = set()
        self.paper_list_order: List[str] = self._read_paper_list()

    # -----------------------------
    # Public API used by pipeline
    # -----------------------------

    def process_paper(self, arxiv_id: str, extracted_dir: str, remaining: int) -> int:
        """
        Process a single paper: find figures, filter quantum circuits, extract text, save images.
        Returns the number of quantum circuit images added to the dataset.
        """
        finder = FigureFinder(extracted_dir)
        figures = finder.find_figures()
        aligner = TextAligner(extracted_dir)

        added_count = 0
        figure_number = 0

        for fig in figures:
            if added_count >= remaining:
                break

            tex = fig.get("paper_tex", "")
            if not tex:
                continue
            tex_lower = tex.lower()

            for image_index, img_path in enumerate(fig.get("image_paths", []) or []):
                if added_count >= remaining:
                    break

                # Check each image individually
                if not self.circuit_filter._is_circuit_image(img_path, tex_lower):
                    continue

                out_filename = self._make_output_filename(
                    arxiv_id, figure_number, img_path, image_index
                )
                out_path = os.path.join(self.images_dir, out_filename)

                try:
                    metadata = self._extract_metadata_v3(
                        arxiv_id=arxiv_id,
                        fig=fig,
                        aligner=aligner,
                        figure_number=figure_number,
                    )
                except Exception as e:
                    print(f"Warning: Failed to extract metadata for {arxiv_id} fig {figure_number}: {e}")
                    continue

                # If problem is unknown, try to infer a general quantum circuit
                caption = fig.get("caption", "") or ""
                if metadata["quantum_problem"] == "unknown":
                    metadata["quantum_problem"] = "Unknown Quantum Problem"

                if not self._export_as_png(img_path, out_path):
                    continue

                self.dataset[out_filename] = {
                    "arxiv_id": arxiv_id,
                    "page": metadata["page_number"],
                    "figure_number": metadata["figure_number"],
                    "quantum_gates": self._normalize_gate_list(metadata["quantum_gates"]),
                    "quantum_problem": metadata["quantum_problem"],
                    "descriptions": metadata["descriptions"],
                    "text_positions": metadata["text_positions"],
                }

                added_count += 1
                figure_number += 1

        self.mark_processed(arxiv_id, added_count)
        return added_count

    def mark_processed(self, arxiv_id: str, extracted_images: int) -> None:
        """Mark a paper as processed even if it produced 0 images or failed."""
        arxiv_id = (arxiv_id or "").strip()
        if not arxiv_id:
            return
        self.processed_papers.add(arxiv_id)
        self.paper_counts[arxiv_id] = int(extracted_images)

    def finalize(self) -> None:
        """Write final JSON and counts CSV to disk."""
        self._write_dataset_json()
        self._write_counts_csv()

    # -----------------------------
    # Metadata extraction
    # -----------------------------

    def _extract_metadata_v3(
        self,
        arxiv_id: str,
        fig: Dict,
        aligner: TextAligner,
        figure_number: int,
    ) -> Dict:
        caption = fig.get("caption", "") or ""

        # Figure number from labels (best-effort)
        fig_num = figure_number
        labels = fig.get("labels", []) or []
        for label in labels:
            num_match = re.search(r"(\d+)", str(label))
            if num_match:
                fig_num = int(num_match.group(1))
                break

        latex_block = fig.get("latex_block", "") or ""

        # 1) NLP extraction (preferred)
        descriptions, positions = aligner.extract_descriptions(
            figure_index=fig_num,
            labels=labels,
            caption=caption,
            max_spans=3,
            top_k_sentences=2,
        )

        # 2) Deterministic fallback: caption if NLP gave nothing
        if not descriptions and caption.strip():
            descriptions = [caption.strip()]
            if aligner.full_text:
                m = re.search(re.escape(caption.strip()), aligner.full_text)
                if m:
                    positions = [(m.start(), m.end())]
                else:
                    positions = []
            else:
                positions = []
        elif not descriptions:
            # last-resort fallback keeps schema consistent
            descriptions = ["unknown"]
            positions = []

        # Gates: search in caption + latex_block + descriptions with regex
        gate_text = " ".join([caption, latex_block] + (descriptions or []))
        gates: List[str] = []
        for gate in self.circuit_filter.GATE_KW:
            # Check substring
            if gate.lower() in gate_text.lower():
                gates.append(gate.replace("\\", ""))
            # Also check regex patterns for common gates
            if re.search(rf"\b{re.escape(gate)}\b", gate_text, re.IGNORECASE):
                gates.append(gate.replace("\\", ""))
        gates = list(sorted(set(gates)))

        problem = self._infer_problem(caption, fig, descriptions)
        page = self._find_pdf_page_number(arxiv_id, caption, fig_num, descriptions)

        return {
            "arxiv_number": arxiv_id,
            "page_number": page,
            "figure_number": fig_num,
            "quantum_gates": gates,
            "quantum_problem": problem,
            "descriptions": descriptions,
            "text_positions": positions,
        }



    # -----------------------------
    # JSON / CSV output
    # -----------------------------

    def _read_paper_list(self) -> List[str]:
        if not self.paper_list_path or not os.path.exists(self.paper_list_path):
            return []
        with open(self.paper_list_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _write_dataset_json(self) -> None:
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, indent=2, ensure_ascii=False)

    def _write_counts_csv(self) -> None:
        csv_path = f"paper_list_counts_{self.exam_id}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["arxiv_id", "extracted_images"])
            for arxiv_id in self.paper_list_order:
                if arxiv_id in self.processed_papers:
                    writer.writerow([arxiv_id, self.paper_counts.get(arxiv_id, 0)])
                else:
                    writer.writerow([arxiv_id, ""])

    # -----------------------------
    # Image / text helpers
    # -----------------------------

    def _make_output_filename(
        self, arxiv_id: str, figure_number: int, img_path: str, image_index: int
    ) -> str:
        base = os.path.basename(img_path)
        base = base.replace(":", "_").replace(" ", "_")
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
        base_no_ext = os.path.splitext(base)[0]
        safe_arxiv = arxiv_id.replace(":", "_").replace("/", "_")
        return f"{safe_arxiv}_fig{figure_number}_img{image_index}_{base_no_ext}.png"

    def _export_as_png(self, img_path: str, out_path: str) -> bool:
        suffix = os.path.splitext(img_path)[1].lower()

        if suffix == ".pdf":
            try:
                from pdf2image import convert_from_path

                pages = convert_from_path(img_path, dpi=200, first_page=1, last_page=1)
                if not pages:
                    return False
                pages[0].save(out_path, "PNG")
                return True
            except Exception:
                return False

        try:
            with Image.open(img_path) as im:
                im = im.convert("RGBA")
                im.save(out_path, "PNG")
            return True
        except Exception:
            return False

    def _infer_problem(self, caption: str, fig: Dict, descriptions: List[str]) -> str:
        import re
        
        # Initialize spaCy for sentence splitting
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        
        # Quantum context keywords
        quantum_keywords = {
            'quantum', 'qubit', 'circuit', 'gate', 'algorithm', 'protocol', 
            'computation', 'computing', 'entanglement', 'superposition', 
            'measurement', 'state', 'qft', 'vqe', 'qaoa', 'teleportation',
            'error', 'correction', 'optimization', 'search', 'fourier'
        }
        
        # Define patterns with synonyms and variations
        patterns = {
            r'\bgrover(?:\'?s)?\b': "Grover's algorithm",
            r'\bshor(?:\'?s)?\b': "Shor's algorithm", 
            r'\b(?:quantum\s+)?teleport(?:ation)?\b': "Quantum teleportation",
            r'\bvqe\b|\bvariational\s+quantum\s+eigensolver\b': "Variational Quantum Eigensolver",
            r'\bqaoa\b|\bquantum\s+approximate\s+optimization\b': "Quantum Approximate Optimization Algorithm",
            r'\bqft\b|\bquantum\s+fourier\s+transform\b': "Quantum Fourier Transform",
            r'\bbell\s+(?:state|measurement|preparation|pair)\b': "Bell state preparation",
            r'\bsuperdense\s+coding\b|\bsuper\s+dense\s+coding\b': "Superdense coding",
            r'\b(?:quantum\s+)?error\s+correct(?:ion)?\b': "Quantum error correction",
            r'\bhadamard\s+test\b': "Hadamard test",
            r'\bquantum\s+walk\b': "Quantum walk",
            r'\bquantum\s+phase\s+estimation\b': "Quantum phase estimation",
            r'\bbernstein.?vazirani\b': "Bernstein-Vazirani algorithm",
            r'\bsimon(?:\'?s)?\b': "Simon's algorithm",
            r'\bdeutsch(?:.?jozsa)?\b': "Deutsch-Jozsa algorithm",
            r'\bquantum\s+key\s+distribution\b|\bqkd\b': "Quantum key distribution",
            r'\bquantum\s+cryptography\b': "Quantum cryptography",
            r'\bquantum\s+simulation\b': "Quantum simulation",
            r'\bquantum\s+machine\s+learning\b|\bquantum\s+ml\b': "Quantum machine learning",
            r'\bquantum\s+chemistry\b': "Quantum chemistry",
            r'\bquantum\s+annealing\b': "Quantum annealing",
            r'\badiabatic\s+quantum\s+computation\b': "Adiabatic quantum computation",
            r'\btopological\s+quantum\s+computation\b': "Topological quantum computation",
            r'\bquantum\s+supremacy\b': "Quantum supremacy",
            r'\bquantum\s+advantage\b': "Quantum advantage",
            r'\bquantum\s+volume\b': "Quantum volume",
            r'\bquantum\s+random\s+walk\b': "Quantum random walk",
            r'\bquantum\s+search\b': "Quantum search",
            r'\bquantum\s+counting\b': "Quantum counting",
            r'\bquantum\s+amplitude\s+estimation\b': "Quantum amplitude estimation",
            r'\bquantum\s+approximate\s+counting\b': "Quantum approximate counting",
        }
        
        def has_quantum_context(sentence: str) -> bool:
            """Check if sentence contains quantum-related keywords."""
            words = set(re.findall(r'\b\w+\b', sentence.lower()))
            return bool(words & quantum_keywords)
        
        def find_problem_in_text(text: str) -> str:
            """Find quantum problem in text with context awareness."""
            if not text:
                return "unknown"
            
            # Process text with spaCy
            doc = nlp(text)
            
            for sent in doc.sents:
                sent_text = sent.text.lower()
                
                # Check each pattern
                for pattern, problem in patterns.items():
                    if re.search(pattern, sent_text, re.IGNORECASE):
                        # Verify quantum context
                        if has_quantum_context(sent_text):
                            return problem
            
            return "unknown"
        
        # Search caption first
        problem = find_problem_in_text(caption)
        if problem != "unknown":
            return problem
        
        # Search in descriptions
        for desc in descriptions:
            problem = find_problem_in_text(desc)
            if problem != "unknown":
                return problem
        
        # Get context around the figure in the tex file
        tex_file_path = fig["tex_file"]
        latex_block = fig["latex_block"]
        tex_content = self._load_tex_file_text(tex_file_path)
        
        # Find position of latex_block in tex_content
        block_start = tex_content.find(latex_block.lower())
        if block_start != -1:
            # Take 2000 chars before and after the figure block
            start = max(0, block_start - 2000)
            end = min(len(tex_content), block_start + len(latex_block) + 2000)
            context_text = tex_content[start:end]
            
            # Clean the context text
            context_text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', context_text)  # remove commands
            context_text = re.sub(r'\\[a-zA-Z]+', '', context_text)  # remove other commands
            context_text = re.sub(r'\s+', ' ', context_text).strip()
            
            problem = find_problem_in_text(context_text)
            if problem != "unknown":
                return problem
        
        # Fallback: if caption contains quantum and circuit, assume general quantum circuit
        # Removed as per user feedback: quantum circuit is not a specific problem
        
        return "unknown"

    def _load_tex_file_text(self, tex_file_path: str) -> str:
        """Load the text content of the specific tex file containing the figure."""
        try:
            with open(tex_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content.lower()
        except Exception:
            return ""

    def _normalize_gate_list(self, gates: List[str]) -> List[str]:
        if not gates:
            return []

        mapping = {
            "cx": "CNOT",
            "cnot": "CNOT",
            "cz": "CZ",
            "swap": "SWAP",
            "toffoli": "TOFFOLI",
            "measure": "MEASURE",
            "meter": "MEASURE",
        }
        drop = {"gate", "ctrl", "targ", "qw", "lstick", "rstick", "controlled"}

        norm: List[str] = []
        for g in gates:
            t = (g or "").strip()
            if not t:
                continue
            t_low = t.lower()
            if t_low in drop:
                continue
            norm.append(mapping.get(t_low, t.upper()))
        return sorted(set(norm))

    # -----------------------------
    # PDF page matching (PyMuPDF)
    # -----------------------------

    def _safe_pdf_filename(self, arxiv_id: str) -> str:
        # Must match how your ArxivPdfDownloader writes PDFs
        return arxiv_id.strip().replace(":", "_").replace("/", "_") + ".pdf"

    @lru_cache(maxsize=256)
    def _load_pdf_pages_text(self, pdf_path: str) -> List[str]:
        try:
            import fitz  # PyMuPDF
        except Exception:
            return []

        if not pdf_path or not os.path.exists(pdf_path):
            return []

        pages: List[str] = []
        with fitz.open(pdf_path) as doc:
            for i in range(doc.page_count):
                txt = doc.load_page(i).get_text("text") or ""
                txt = txt.lower()
                txt = re.sub(r"\s+", " ", txt).strip()
                pages.append(txt)
        return pages

    def _normalize_caption_for_match(self, caption: str) -> str:
        cap = (caption or "").lower()
        cap = re.sub(r"\\[a-z]+\{[^}]*\}", "", cap)  # Remove LaTeX commands
        cap = re.sub(r"\\[a-z]+", "", cap)  # Remove other commands
        cap = re.sub(r"\s+", " ", cap).strip()
        cap = cap.replace("{", "").replace("}", "").replace("\\", "")
        return cap

    def _find_pdf_page_number(self, arxiv_id: str, caption: str, fig_num: int, descriptions: List[str] = None) -> int:
        """
        Find 1-based PDF page number where the figure appears.

        1) Exact caption substring match if caption is long enough.
        2) Search in descriptions for matches.
        3) Fallback: require ("fig. <n>" or "figure <n>") AND caption-token overlap.
        """
        pdf_path = os.path.join(self.pdf_dir, self._safe_pdf_filename(arxiv_id))
        pages = self._load_pdf_pages_text(pdf_path)
        if not pages:
            return -1

        cap = self._normalize_caption_for_match(caption)
        cap_ok = cap if len(cap) >= 20 else ""

        # Also prepare descriptions
        desc_texts = []
        if descriptions:
            for desc in descriptions[:2]:  # Use first 2 descriptions
                desc_norm = self._normalize_caption_for_match(desc)
                if len(desc_norm) >= 20:
                    desc_texts.append(desc_norm)

        fig_patterns = [
            rf"\bfig\.?\s*{fig_num}\b",
            rf"\bfigure\s*{fig_num}\b",
            rf"\bfig\s*{fig_num}\b",
            rf"\bfigs?\.\s*{fig_num}\b",
        ]

        # 1) caption match (best)
        if cap_ok:
            for idx, page_text in enumerate(pages):
                if cap_ok in page_text:
                    return idx + 1

        # 2) description matches
        for desc in desc_texts:
            for idx, page_text in enumerate(pages):
                if desc in page_text:
                    return idx + 1

        # 3) fallback with token overlap (reduces false positives)
        cap_tokens = [t for t in re.split(r"[^a-z0-9]+", cap) if len(t) >= 5]
        cap_tokens = cap_tokens[:6]

        for idx, page_text in enumerate(pages):
            fig_hit = any(re.search(pat, page_text) for pat in fig_patterns)
            if not fig_hit:
                continue
            token_hits = sum(1 for t in cap_tokens if t and (t in page_text))
            if token_hits >= 0:  # Allow any token hit for more matches
                return idx + 1

        return -1
