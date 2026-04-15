#!/usr/bin/env python3
"""
split_score.py — Split a PDF into N parts by horizontally slicing each page.

Each output PDF contains the same horizontal band from every page of the input.
Designed for extracting musical scores from multi-part scans (e.g. IMSLP files
where multiple parts are stacked on each page).

Usage:
    python split_score.py input.pdf 4
    python split_score.py input.pdf 2 -o my_output_folder/
"""

import fitz  # PyMuPDF
import argparse
import os
from pathlib import Path


def split_pdf_into_parts(input_pdf: str, n_parts: int, output_dir: str = None):
    """
    Split a PDF into n_parts by horizontally slicing each page.

    Args:
        input_pdf:  Path to the source PDF.
        n_parts:    Number of horizontal slices (and output PDFs) to produce.
        output_dir: Directory to write output files into.
                    Defaults to <input_stem>_parts/ next to the input file.
    """
    doc = fitz.open(input_pdf)
    n_pages = len(doc)

    if n_pages == 0:
        raise ValueError("Input PDF has no pages.")

    if output_dir is None:
        output_dir = str(Path(input_pdf).parent / (Path(input_pdf).stem + "_parts"))

    os.makedirs(output_dir, exist_ok=True)
    input_stem = Path(input_pdf).stem

    print(f"Input:  {input_pdf}  ({n_pages} pages)")
    print(f"Parts:  {n_parts}")
    print(f"Output: {output_dir}/\n")

    for part_idx in range(n_parts):
        out_doc = fitz.open()

        for page_num in range(n_pages):
            # Copy the page into the new document
            out_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            new_page = out_doc[-1]

            rect = new_page.rect  # full page bounds
            slice_height = rect.height / n_parts
            y0 = rect.y0 + part_idx * slice_height
            y1 = rect.y0 + (part_idx + 1) * slice_height

            # Crop to just this horizontal band
            clip = fitz.Rect(rect.x0, y0, rect.x1, y1)
            new_page.set_cropbox(clip)

        label = f"part{part_idx + 1}_of_{n_parts}"
        output_path = os.path.join(output_dir, f"{input_stem}_{label}.pdf")
        out_doc.save(output_path, garbage=4, deflate=True)
        out_doc.close()
        print(f"  Saved: {output_path}")

    doc.close()
    print(f"\nDone — {n_parts} parts written to '{output_dir}/'")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split a PDF into N parts by horizontally slicing each page.\n"
            "Each output PDF is one band (strip) across all pages of the input."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to the input PDF file.")
    parser.add_argument(
        "n_parts",
        type=int,
        help="Number of horizontal parts to split each page into.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        metavar="DIR",
        help="Directory to write output PDFs into (default: <input_stem>_parts/).",
    )

    args = parser.parse_args()

    if args.n_parts < 2:
        parser.error("n_parts must be at least 2.")

    if not os.path.isfile(args.input):
        parser.error(f"File not found: {args.input}")

    split_pdf_into_parts(args.input, args.n_parts, args.output_dir)


if __name__ == "__main__":
    main()
