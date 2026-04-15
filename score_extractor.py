#!/usr/bin/env python3
"""
score_extractor.py — Extract individual instrument parts from multi-part PDF scores.

Combines two steps into one command:

  1. Pre-processing (optional, --systems P > 1):
     Each page is split into P sub-pages at whitespace boundaries found by image
     analysis. Whitespace is trimmed from all four sides of each sub-page.
     Pages are ordered sequentially: 1a, 1b, 2a, 2b, ...

  2. Part extraction (--parts N):
     Each (sub-)page is divided horizontally into N equal strips, one per
     instrument/voice. Each part's strips are assembled onto A4 portrait pages
     within printer-safe margins, with a configurable gap between sections.

Usage:
    python score_extractor.py input.pdf --parts 4
    python score_extractor.py input.pdf --parts 4 --systems 2 --gap 0.5
    python score_extractor.py input.pdf --parts 4 --systems 2 --gap 1 --margin 15 --dpi 200
"""

import fitz  # PyMuPDF
import numpy as np
import argparse
import os
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

A4_W = 595.0          # A4 portrait width in PDF points
A4_H = 842.0          # A4 portrait height in PDF points
MM_TO_PT = 72.0 / 25.4

WHITE_THRESHOLD = 250  # greyscale value (0–255) at/above which a pixel is white
DEFAULT_DPI = 150      # rendering resolution for whitespace analysis
DEFAULT_GAP = 1.0      # gap between sections as fraction of preceding section height
DEFAULT_MARGIN_MM = 15.0  # printer-safe margin in mm


# ── Image-analysis helpers ─────────────────────────────────────────────────────

def _render_gray_clip(page: fitz.Page, clip: fitz.Rect, dpi: int) -> np.ndarray:
    """Render a clipped region of a page as a greyscale numpy array (H×W, uint8)."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def _longest_white_run(mask: np.ndarray, start: int, end: int):
    """
    Return (run_start, run_end) of the longest contiguous True run in mask[start:end].
    Returns (None, None) if no True values exist in that range.
    """
    best_start = best_end = best_len = None
    run_start = None

    for i in range(start, end):
        if mask[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                length = i - run_start
                if best_len is None or length > best_len:
                    best_start, best_end, best_len = run_start, i, length
                run_start = None

    if run_start is not None:
        length = end - run_start
        if best_len is None or length > best_len:
            best_start, best_end = run_start, end

    return best_start, best_end


def find_split_positions(page: fitz.Page, p_sub: int, dpi: int) -> list:
    """
    Find P−1 horizontal y-coordinates (PDF points) for dividing a page into p_sub parts.

    For each split, searches within ±⅓ of a segment around the geometric midpoint
    for the longest continuous run of all-white rows. The split is placed at the
    centre of that band. Falls back to the geometric midpoint if no whitespace is found.
    """
    img = _render_gray_clip(page, page.rect, dpi)
    h_px = img.shape[0]
    row_is_white = np.all(img >= WHITE_THRESHOLD, axis=1)
    scale = 72.0 / dpi

    split_ys = []
    for k in range(1, p_sub):
        center_px = int(k / p_sub * h_px)
        half_window = max(1, h_px // (p_sub * 3))
        s0 = max(0, center_px - half_window)
        s1 = min(h_px, center_px + half_window)

        run_start, run_end = _longest_white_run(row_is_white, s0, s1)
        split_px = center_px if run_start is None else (run_start + run_end) // 2
        split_ys.append(page.rect.y0 + split_px * scale)

    return split_ys


def find_content_bounds_2d(page: fitz.Page, clip: fitz.Rect, dpi: int):
    """
    Find tight content bounds within a clipped region, trimming whitespace on all
    four sides (top, bottom, left, right).

    Returns (x_left, y_top, x_right, y_bottom) in PDF points.
    Falls back to the original clip bounds if the region is entirely white.

    Top/bottom trimming: strict — a row is white if every pixel >= WHITE_THRESHOLD.

    Left/right trimming: robust two-stage approach to handle 1-pixel grey borders
    (e.g. scan-line artefacts) that would defeat a strict per-column check:

      Stage 1 — coarse cut using lenient column whiteness (>=98% of pixels white):
        • Find the largest "white" column run in the left third  → x_left_cut
        • Find the largest "white" column run in the right third → x_right_cut
        Cuts are placed at the centres of those runs, so we land squarely inside
        the whitespace rather than at its boundary.

      Stage 2 — fine trim: within [x_left_cut, x_right_cut] apply a strict
        all-white column check to remove any residual whitespace at both edges.
    """
    img = _render_gray_clip(page, clip, dpi)
    h_px, w_px = img.shape
    scale = 72.0 / dpi

    # ── Top / bottom (rows) — strict ──────────────────────────────────────────
    row_is_white  = np.all(img >= WHITE_THRESHOLD, axis=1)
    content_rows  = np.where(~row_is_white)[0]

    if len(content_rows) == 0:
        return clip.x0, clip.y0, clip.x1, clip.y1  # entirely white

    y_top    = clip.y0 + int(content_rows[0])        * scale
    y_bottom = clip.y0 + (int(content_rows[-1]) + 1) * scale

    # ── Left / right (columns) — robust two-stage ─────────────────────────────
    # Lenient: a column is "white" if ≥98 % of its pixels are white.
    # This tolerates the single grey border pixel that breaks np.all().
    col_white_frac = np.mean(img >= WHITE_THRESHOLD, axis=0)  # shape (w_px,)
    col_is_white_lenient = col_white_frac >= 0.98

    # Stage 1 — coarse cuts from the largest white run in each outer third
    left_third  = max(1, w_px // 3)
    right_third = w_px - left_third

    ls, le = _longest_white_run(col_is_white_lenient, 0,           left_third)
    rs, re = _longest_white_run(col_is_white_lenient, right_third, w_px)

    x_left_px  = (ls + le) // 2 if ls is not None else 0
    x_right_px = (rs + re) // 2 if rs is not None else w_px

    # Stage 2 — fine trim: strict all-white check within the coarse bounds
    if x_left_px < x_right_px:
        inner           = img[:, x_left_px:x_right_px]
        inner_col_white = np.all(inner >= WHITE_THRESHOLD, axis=0)
        inner_content   = np.where(~inner_col_white)[0]
        if len(inner_content) > 0:
            coarse_left = x_left_px
            x_left_px   = coarse_left + int(inner_content[0])
            x_right_px  = coarse_left + int(inner_content[-1]) + 1

    x_left  = clip.x0 + x_left_px  * scale
    x_right = clip.x0 + x_right_px * scale

    return x_left, y_top, x_right, y_bottom


# ── Pipeline ───────────────────────────────────────────────────────────────────

def build_sub_page_clips(doc: fitz.Document, p_sub: int, dpi: int) -> list:
    """
    Build the ordered list of (page_num, clip_rect) sub-pages, where clip_rect
    is trimmed of whitespace on all four sides.

    When p_sub == 1 no split analysis is performed, but whitespace is still trimmed.
    Output order for p_sub == 2: (0,'a'), (0,'b'), (1,'a'), (1,'b'), ...
    """
    sub_pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        rect = page.rect

        if p_sub == 1:
            y_bounds = [rect.y0, rect.y1]
        else:
            split_ys = find_split_positions(page, p_sub, dpi)
            y_bounds = [rect.y0] + split_ys + [rect.y1]

        for i in range(p_sub):
            band = fitz.Rect(rect.x0, y_bounds[i], rect.x1, y_bounds[i + 1])
            x_left, y_top, x_right, y_bottom = find_content_bounds_2d(page, band, dpi)
            clip = fitz.Rect(x_left, y_top, x_right, y_bottom)

            # Skip degenerate clips (entirely white or zero-area bands)
            if clip.width <= 1 or clip.height <= 1:
                continue

            sub_pages.append((page_num, clip))

    return sub_pages


def extract_parts(
    input_pdf: str,
    n_parts: int,
    p_sub: int = 1,
    gap_fraction: float = DEFAULT_GAP,
    margin_mm: float = DEFAULT_MARGIN_MM,
    output_dir: str = None,
    dpi: int = DEFAULT_DPI,
):
    """
    Main pipeline: preprocess pages into sub-pages (optional), then extract N
    instrument parts assembled onto A4 pages within printer-safe margins.

    Args:
        input_pdf:     Path to source PDF.
        n_parts:       Number of instrument parts to extract (horizontal slices).
        p_sub:         Sub-pages per source page (≥ 1; use 1 to skip preprocessing).
        gap_fraction:  Gap between sections as a fraction of the preceding section
                       height. 0 = no gap; 1 = gap equals the preceding section.
        margin_mm:     Printer-safe margin in mm on all four sides of each A4 page.
        output_dir:    Directory for output PDFs.
        dpi:           Rendering resolution for whitespace analysis.
    """
    doc = fitz.open(input_pdf)
    if len(doc) == 0:
        raise ValueError("Input PDF has no pages.")

    if output_dir is None:
        output_dir = str(Path(input_pdf).parent / (Path(input_pdf).stem + "_parts"))
    os.makedirs(output_dir, exist_ok=True)

    margin   = margin_mm * MM_TO_PT
    content_w = A4_W - 2 * margin
    content_h = A4_H - 2 * margin

    print(f"Input:         {input_pdf}  ({len(doc)} pages)")
    print(f"Systems/page:  {p_sub}")
    print(f"Parts:         {n_parts}")
    print(f"Gap fraction:  {gap_fraction}")
    print(f"Print margin:  {margin_mm} mm  ({margin:.1f} pt)")
    print(f"Analysis DPI:  {dpi}")
    print(f"Output dir:    {output_dir}/\n")

    sub_pages = build_sub_page_clips(doc, p_sub, dpi)
    print(f"Sub-pages after preprocessing: {len(sub_pages)}\n")

    input_stem = Path(input_pdf).stem

    for part_idx in range(n_parts):
        out_doc = fitz.open()
        current_page = out_doc.new_page(width=A4_W, height=A4_H)
        current_y    = 0.0
        last_h       = 0.0  # height of the most recently placed section

        for page_num, clip in sub_pages:
            band_h = clip.height / n_parts
            if band_h <= 0:
                continue

            band_y0   = clip.y0 + part_idx * band_h
            band_y1   = clip.y0 + (part_idx + 1) * band_h
            band_clip = fitz.Rect(clip.x0, band_y0, clip.x1, band_y1)

            # Scale band to content width, preserving aspect ratio
            scale    = content_w / clip.width
            scaled_h = min(band_h * scale, content_h)

            # Gap after the previous section (0 at the very start of each page)
            gap = gap_fraction * last_h

            # Start a new A4 page if this section (plus gap) would overflow
            if last_h > 0 and current_y + gap + scaled_h > content_h:
                current_page = out_doc.new_page(width=A4_W, height=A4_H)
                current_y = 0.0
                gap = 0.0  # no leading gap at the top of a fresh page

            current_y += gap

            dest = fitz.Rect(
                margin,
                margin + current_y,
                margin + content_w,
                margin + current_y + scaled_h,
            )
            current_page.show_pdf_page(dest, doc, page_num, clip=band_clip)
            current_y += scaled_h
            last_h     = scaled_h

        label    = f"part{part_idx + 1}_of_{n_parts}"
        out_path = os.path.join(output_dir, f"{input_stem}_{label}.pdf")
        out_doc.save(out_path, garbage=4, deflate=True)
        out_doc.close()
        print(f"  Saved: {out_path}")

    doc.close()
    print(f"\nDone — {n_parts} parts written to '{output_dir}/'")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract individual instrument parts from a multi-part PDF score.\n\n"
            "Each page is optionally split into P systems (--systems), whitespace\n"
            "is trimmed on all four sides, then each system is sliced into N parts\n"
            "(--parts) and assembled onto printer-safe A4 portrait pages."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # 4-part score, 1 system per page\n"
            "  python score_extractor.py quartet.pdf --parts 4\n\n"
            "  # 4-part score, 2 systems per page, half-section gap\n"
            "  python score_extractor.py quartet.pdf --parts 4 --systems 2 --gap 0.5\n\n"
            "  # Same with custom output directory and tighter margins\n"
            "  python score_extractor.py quartet.pdf --parts 4 --systems 2 -o ~/parts/ --margin 10"
        ),
    )

    parser.add_argument(
        "input",
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "-n", "--parts",
        type=int,
        required=True,
        metavar="N",
        help="Number of instrument parts to extract (horizontal slices per system). Required.",
    )
    parser.add_argument(
        "-p", "--systems",
        type=int,
        default=1,
        metavar="P",
        help=(
            "Number of stave systems per page (default: 1, skips system-splitting). "
            "When > 1, each page is split into P sub-pages at the largest whitespace "
            "band near each expected division point."
        ),
    )
    parser.add_argument(
        "-g", "--gap",
        type=float,
        default=DEFAULT_GAP,
        metavar="FRAC",
        help=(
            f"Gap between consecutive sections on a page, expressed as a fraction of "
            f"the preceding section's height (default: {DEFAULT_GAP}). "
            "0 = no gap. 0.5 = half a section height. 1 = a full section height."
        ),
    )
    parser.add_argument(
        "-m", "--margin",
        type=float,
        default=DEFAULT_MARGIN_MM,
        metavar="MM",
        help=(
            f"Printer-safe margin in mm applied to all four sides of each A4 page "
            f"(default: {DEFAULT_MARGIN_MM} mm)."
        ),
    )
    parser.add_argument(
        "-o", "--output-dir",
        metavar="DIR",
        help="Directory to write output PDFs into (default: <input_stem>_parts/ next to input).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        metavar="DPI",
        help=(
            f"Resolution in DPI for whitespace analysis (default: {DEFAULT_DPI}). "
            "Increase for fine-grained scores; higher values are slower."
        ),
    )

    args = parser.parse_args()

    if args.parts < 2:
        parser.error("--parts must be at least 2.")
    if args.systems < 1:
        parser.error("--systems must be at least 1.")
    if args.gap < 0:
        parser.error("--gap must be ≥ 0.")
    if args.margin < 0:
        parser.error("--margin must be ≥ 0.")
    if not os.path.isfile(args.input):
        parser.error(f"File not found: {args.input}")

    extract_parts(
        input_pdf=args.input,
        n_parts=args.parts,
        p_sub=args.systems,
        gap_fraction=args.gap,
        margin_mm=args.margin,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
