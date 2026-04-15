#!/usr/bin/env python3
"""
split_score.py — Split a PDF into N parts by horizontally slicing each page,
then reassemble each part's slices onto A4 portrait pages.

Each output PDF contains the slices from one horizontal band of each source page,
packed sequentially onto A4 pages at full width (aspect-ratio preserved). If a
slice does not fit in the remaining space on a page, the rest is left blank and
the slice starts on a new page.

Designed for extracting individual instrument parts from multi-part score scans
(e.g. IMSLP files where multiple parts are stacked on each page).

Usage:
    python split_score.py input.pdf 4
    python split_score.py input.pdf 2 -o my_output_folder/
"""

import fitz  # PyMuPDF
import argparse
import os
from pathlib import Path

# A4 portrait in PDF points (1 pt = 1/72 inch)
A4_W = 595.0
A4_H = 842.0


def split_pdf_into_parts(input_pdf: str, n_parts: int, output_dir: str = None):
    """
    Split a PDF into n_parts by horizontally slicing each page, then pack
    each part's slices onto A4 portrait pages at full width.

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

        # Start the first A4 page
        current_page = out_doc.new_page(width=A4_W, height=A4_H)
        current_y = 0.0

        for page_num in range(n_pages):
            src_page = doc[page_num]
            src_rect = src_page.rect

            # The horizontal band for this part on this source page
            slice_height = src_rect.height / n_parts
            y0 = src_rect.y0 + part_idx * slice_height
            y1 = src_rect.y0 + (part_idx + 1) * slice_height
            clip = fitz.Rect(src_rect.x0, y0, src_rect.x1, y1)

            # Scale to full A4 width, preserving aspect ratio
            slice_w = src_rect.width
            scale = A4_W / slice_w
            scaled_h = slice_height * scale

            if scaled_h > A4_H:
                # Slice is taller than a full page even after scaling —
                # clamp to page height (unavoidable distortion-free loss)
                scaled_h = A4_H

            # If it doesn't fit on the current page, start a new one
            if current_y + scaled_h > A4_H:
                current_page = out_doc.new_page(width=A4_W, height=A4_H)
                current_y = 0.0

            dest = fitz.Rect(0, current_y, A4_W, current_y + scaled_h)
            current_page.show_pdf_page(dest, doc, page_num, clip=clip)
            current_y += scaled_h

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
            "Slices for each part are packed onto A4 portrait pages at full\n"
            "width (aspect-ratio preserved). Slices that don't fit on the\n"
            "current page are moved to the next."
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
