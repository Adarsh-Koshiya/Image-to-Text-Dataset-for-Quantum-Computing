"""
main.py

Entry point for the quantum circuit dataset pipeline.
"""

from src.pipeline import QuantumCircuitDatasetPipeline

def main() -> None:
    pipeline = QuantumCircuitDatasetPipeline(
        exam_id="20",
        paper_list_path="paper_list_20.txt",
        source_archive_dir="data/arxiv_sources",
        source_extract_dir="data/extracted_sources",
        pdf_dir="data/pdf_source",
        images_dir="images_20",
        output_json="dataset_20.json",
        max_images=250,
    )

    pipeline.run()

if __name__ == "__main__":
    main()