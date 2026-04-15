#!/usr/bin/env python3
"""
preprocess.py — Split each page of a PDF into P sub-pages at whitespace boundaries,
trim surrounding whitespace from each, and write them sequentially.

For P=2 on a 30-page PDF, output is 60 pages ordered: 1a, 1b, 2a, 2b, ...

The split position is found by image analysis: each page is rendered in greyscale,
and the split is placed at the centre of the largest continuous white band found
within a search window around the expected geometric split point. Whitespace is
then trimmed from the top and bottom of each sub-page before writing.

Designed as a pre-processing step before split_score.py, for scores where each
page contains multiple full stave systems (e.g. a quartet score with 2 systems
per page — set P=2 to separate them before extracting individual parts).

Usage:
    python preprocess.py input.pdf 2
    python preprocess.py input.pdf 2 -o preprocessed.pdf --dpi 200
"""

import fitz  # PyMuPDF
import numpy as np
import argparse
import os
from pathlib import Path

# Pixel value (0–255 greyscale) at or above which a pixel is considered white.
WHITE_THRESHOLD = 250

# Default rendering resolution for image analysis (does not affect output quality).
DEFAULT_DPI = 150


# ---------------------------------------------------------------------------
# Image-analysis helpers
# ---------------------------------------------------------------------------

def _render_gray(page: fitz.Page, dpi: int) -> np.ndarray:
    """Render a page (or clipped region) to a greyscale numpy array."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def _render_gray_clip(page: fitz.Page, clip: fitz.Rect, dpi: int) -> np.ndarray:
    """Render a clipped region of a page to a greyscale numpy array."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def _longest_white_run(row_is_white: np.ndarray, start: int, end: int):
    """
    Find the longest contiguous run of True values in row_is_white[start:end].
    Returns (run_start, run_end) in absolute indices, or (None, None) if none found.
    """
    best_start = best_end = best_len = None
    run_start = None

    for i in range(start, end):
        if row_is_white[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if best_len is None or run_len > best_len:
                    best_start, best_end, best_len = run_start, i, run_len
                run_start = None

    # Close a run that reaches the search boundary
    if run_start is not None:
        run_len = end - run_start
        if best_len is None or run_len > best_len:
            best_start, best_end, best_len = run_start, end, run_len

    return best_start, best_end


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def find_split_positions(page: fitz.Page, p_sub: int, dpi: int) -> list[float]:
    """
    Find P-1 horizontal y-coordinates (in page space, PDF points) at which to
    split the page into p_sub equal parts.

    Each split is placed at the centre of the largest white band found within a
    search window of ±(1 / (3 * p_sub)) of the page height around the geometric
    split point. If no white band is found, the geometric split is used.

    Args:
        page:  Source PDF page.
        p_sub: Number of sub-pages to create.
        dpi:   Rendering resolution for analysis.

    Returns:
        List of p_sub - 1 y-coordinates in PDF points.
    """
    img = _render_gray(page, dpi)
    h_px = img.shape[0]
    row_is_white = np.all(img >= WHITE_THRESHOLD, axis=1)

    scale = 72.0 / dpi  # pixels → PDF points
    page_y0 = page.rect.y0

    split_ys = []
    for k in range(1, p_sub):
        center_px = int(k / p_sub * h_px)
        half_window = max(1, h_px // (p_sub * 3))
        search_start = max(0, center_px - half_window)
        search_end = min(h_px, center_px + half_window)

        run_start, run_end = _longest_white_run(row_is_white, search_start, search_end)

        if run_start is None:
            # No whitespace found — fall back to geometric split
            split_px = center_px
        else:
            split_px = (run_start + run_end) // 2

        split_y = page_y0 + split_px * scale
        split_ys.append(split_y)

    return split_ys


def find_content_bounds(page: fitz.Page, clip: fitz.Rect, dpi: int):
    """
    Trim whitespace from the top and bottom of a clipped region.

    Args:
        page: Source PDF page.
        clip: Region of the page to analyse (PDF points).
        dpi:  Rendering resolution for analysis.

    Returns:
        (y_top, y_bottom) in PDF points, with leading/trailing white rows removed.
        Returns the original clip bounds if the region is entirely white.
    """
    img = _render_gray_clip(page, clip, dpi)
    row_is_white = np.all(img >= WHITE_THRESHOLD, axis=1)
    content_rows = np.where(~row_is_white)[0]

    if len(content_rows) == 0:
        return clip.y0, clip.y1  # fully white — keep as-is

    scale = 72.0 / dpi
    y_top = clip.y0 + int(content_rows[0]) * scale
    y_bottom = clip.y0 + (int(content_rows[-1]) + 1) * scale
    return y_top, y_bottom


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess_pdf(input_pdf: str, p_sub: int, output_path: str = None, dpi: int = DEFAULT_DPI):
    """
    Split each page of input_pdf into p_sub sub-pages, trim whitespace, and
    write to output_path ordered as 1a, 1b, ..., 2a, 2b, ...

    Args:
        input_pdf:   Path to source PDF.
        p_sub:       Number of sub-pages per source page (≥ 2).
        output_path: Output PDF path. Defaults to <stem>_p<P>.pdf.
        dpi:         Resolution for image analysis.
    """
    doc = fitz.open(input_pdf)
    n_pages = len(doc)

    if n_pages == 0:
        raise ValueError("Input PDF has no pages.")

    if output_path is None:
        stem = Path(input_pdf).stem
        output_path = str(Path(input_pdf).parent / f"{stem}_p{p_sub}.pdf")

    print(f"Input:  {input_pdf}  ({n_pages} pages)")
    print(f"P (sub-pages per page): {p_sub}")
    print(f"Analysis DPI: {dpi}")
    print(f"Output: {output_path}\n")

    out_doc = fitz.open()
    labels = [chr(ord('a') + i) for i in range(p_sub)]

    for page_num in range(n_pages):
        page = doc[page_num]
        rect = page.rect

        split_ys = find_split_positions(page, p_sub, dpi)
        y_bounds = [rect.y0] + split_ys + [rect.y1]

        for i in range(p_sub):
            clip = fitz.Rect(rect.x0, y_bounds[i], rect.x1, y_bounds[i + 1])

            # Trim surrounding whitespace
            y_top, y_bottom = find_content_bounds(page, clip, dpi)
            content_height = max(1.0, y_bottom - y_top)
            trimmed_clip = fitz.Rect(rect.x0, y_top, rect.x1, y_bottom)

            # New page matches the source width, trimmed height
            out_page = out_doc.new_page(width=rect.width, height=content_height)
            out_page.show_pdf_page(
                fitz.Rect(0, 0, rect.width, content_height),
                doc,
                page_num,
                clip=trimmed_clip,
            )

            print(f"  Page {page_num + 1}{labels[i]}: "
                  f"split at y={y_bounds[i + 1]:.1f}, "
                  f"content y=[{y_top:.1f}, {y_bottom:.1f}]")

    out_doc.save(output_path, garbage=4, deflate=True)
    out_doc.close()
    doc.close()
    print(f"\nDone — {n_pages * p_sub} pages written to '{output_path}'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pre-processing step: split each PDF page into P sub-pages at\n"
            "whitespace boundaries, trim surrounding whitespace, and write\n"
            "sequentially (1a, 1b, 2a, 2b, ...). Feed the output into\n"
            "split_score.py for part extraction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to the input PDF file.")
    parser.add_argument(
        "p_sub",
        type=int,
        help="Number of sub-pages to split each page into (e.g. 2).",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Output PDF path (default: <input_stem>_p<P>.pdf next to input).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Resolution for whitespace analysis in DPI (default: {DEFAULT_DPI}). "
             "Higher values are slower but more accurate on fine scores.",
    )

    args = parser.parse_args()

    if args.p_sub < 2:
        parser.error("p_sub must be at least 2.")
    if not os.path.isfile(args.input):
        parser.error(f"File not found: {args.input}")

    preprocess_pdf(args.input, args.p_sub, args.output, args.dpi)


if __name__ == "__main__":
    main()
