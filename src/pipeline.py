"""
pipeline.py

Main orchestration pipeline for compiling an image-to-text dataset of
quantum circuit diagrams from arXiv sources.

Key requirements handled here:
- Process papers strictly in the given order.
- Stop after collecting max_images (default: 250) circuit images.
"""

from __future__ import annotations

from .arxiv_downloader import ArxivSourceDownloader, ArxivPdfDownloader
from .latex_extractor import LatexSourceExtractor
from .dataset_builder import DatasetBuilder


class QuantumCircuitDatasetPipeline:
    """
    End-to-end pipeline for constructing a quantum circuit image-to-text dataset
    from arXiv LaTeX sources.
    """

    def __init__(
        self,
        exam_id: str,
        paper_list_path: str,
        source_archive_dir: str,
        source_extract_dir: str,
        pdf_dir: str,
        images_dir: str,
        output_json: str,
        max_images: int = 25,
    ):
        self.exam_id = str(exam_id)
        self.paper_list_path = paper_list_path

        self.downloader = ArxivSourceDownloader(source_archive_dir)
        self.pdf_downloader = ArxivPdfDownloader(pdf_dir)
        self.extractor = LatexSourceExtractor(source_extract_dir)

        self.pdf_dir = pdf_dir
        self.images_dir = images_dir
        self.output_json = output_json
        self.max_images = int(max_images)

    def run(self) -> None:
        """
        Execute the complete dataset compilation pipeline.
        """
        builder = DatasetBuilder(
            exam_id=self.exam_id,
            images_dir=self.images_dir,
            output_json=self.output_json,
            paper_list_path=self.paper_list_path,
            source_root=self.extractor.extract_root,
            pdf_dir=self.pdf_dir,
        )

        with open(self.paper_list_path, "r", encoding="utf-8") as f:
            arxiv_ids = [line.strip() for line in f if line.strip()]

        total_images = len(builder.dataset)
        withdrawn_count = 0
        download_failure_count = 0
        extraction_failure_count = 0

        for arxiv_id in arxiv_ids:
            if total_images >= self.max_images:
                break

            # Cache PDF for later page-number extraction (best-effort).
            _ = self.pdf_downloader.download(arxiv_id)

            archive = self.downloader.download(arxiv_id)
            if archive == "withdrawn":
                withdrawn_count += 1
                builder.mark_processed(arxiv_id, extracted_images=0)
                continue
            elif not archive:
                download_failure_count += 1
                builder.mark_processed(arxiv_id, extracted_images=0)
                continue

            extracted_dir = self.extractor.extract(archive, arxiv_id)
            if not extracted_dir:
                extraction_failure_count += 1
                builder.mark_processed(arxiv_id, extracted_images=0)
                continue

            added = builder.process_paper(
                arxiv_id=arxiv_id,
                extracted_dir=extracted_dir,
                remaining=(self.max_images - total_images),
            )
            total_images += added

        builder.finalize()
        print(f"Pipeline complete. Total images collected: {total_images}")
        print(f"Withdrawn/unavailable papers: {withdrawn_count}")
        print(f"Download failures: {download_failure_count}")
        print(f"Extraction failures: {extraction_failure_count}")