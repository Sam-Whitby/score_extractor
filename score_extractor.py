#!/usr/bin/env python3
"""
score_extractor.py — Extract individual instrument parts from multi-part PDF scores.

Combines two steps into one command:

  1. Pre-processing (optional, --systems P > 1):
     Each page is split into P sub-pages at whitespace boundaries found by image
     analysis. Whitespace is trimmed from all four sides of each sub-page.
     Pages are ordered sequentially: 1a, 1b, 2a, 2b, ...

  2. Part extraction (--parts N):
     Each (sub-)page is divided into N strips using stave-connectivity analysis:
       • Bar lines (tall vertical black lines spanning all parts) are detected and
         masked out so they don't merge all parts into one connected component.
       • The 5×N stave lines are identified and grouped into N sets of 5.
       • A flood-fill (connected component) from each part's stave lines captures
         all ink directly attached to those staves: notes, stems, beams, ledger
         lines, accidentals, etc.
       • Free-standing objects (slurs, dynamics, text) are assigned to the part
         whose connected content is nearest in 2-D Euclidean distance.
       • The cut between adjacent parts is placed at the midpoint of the vertical
         whitespace between the two parts' content extents.
     Each part's strips are assembled onto A4 portrait pages within printer-safe
     margins, with a configurable gap between sections.

Usage:
    python score_extractor.py input.pdf --parts 4
    python score_extractor.py input.pdf --parts 4 --systems 2 --gap 0.5
    python score_extractor.py input.pdf --parts 4 --systems 2 --gap 1 --margin 15 --dpi 200
"""

import fitz  # PyMuPDF
import numpy as np
from scipy import ndimage
import argparse
import os
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

A4_W = 595.0          # A4 portrait width in PDF points
A4_H = 842.0          # A4 portrait height in PDF points
MM_TO_PT = 72.0 / 25.4

WHITE_THRESHOLD = 250  # pixel value (0–255) at/above which a pixel is white
DARK_THRESHOLD  = 128  # pixel value (0–255) below which a pixel is dark/black
DEFAULT_DPI = 150      # rendering resolution for image analysis
DEFAULT_GAP = 1.0      # gap between sections as fraction of preceding section height
DEFAULT_MARGIN_MM = 15.0  # printer-safe margin in mm

# 8-connectivity structure for connected-component labelling (includes diagonals,
# needed to follow diagonal beam lines between notes).
_CC_STRUCTURE = np.ones((3, 3), dtype=np.int32)


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
    row_is_white = np.all(img >= WHITE_THRESHOLD, axis=1)
    content_rows = np.where(~row_is_white)[0]

    if len(content_rows) == 0:
        return clip.x0, clip.y0, clip.x1, clip.y1  # entirely white

    y_top    = clip.y0 + int(content_rows[0])        * scale
    y_bottom = clip.y0 + (int(content_rows[-1]) + 1) * scale

    # ── Left / right (columns) — robust two-stage ─────────────────────────────
    # Lenient: a column is "white" if ≥98% of its pixels are white.
    # This tolerates the single grey border pixel that breaks np.all().
    col_white_frac      = np.mean(img >= WHITE_THRESHOLD, axis=0)
    col_is_white_lenient = col_white_frac >= 0.98

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


# ── Stave-connectivity part-boundary detection ─────────────────────────────────

def _detect_barline_columns(dark_mask: np.ndarray) -> np.ndarray:
    """
    Return a boolean array of shape (w_px,) where True marks a bar-line column.

    Bar lines are nearly-solid vertical strokes spanning all stave systems.
    Stave lines are horizontal, so any given column only has dark pixels at each
    of the 5×N stave line rows — widely scattered through the full height.

    We exploit this with two criteria that must both hold:

      span    — topmost to bottommost dark pixel > 30% of image height.
                Filters out short note stems and beams that stay within one part.

      density — (number of dark pixels) / span > 0.40.
                A true bar line column is nearly solid ink (density ≈ 0.8–0.95).
                A stave-line column has only ~5×N dark pixels spread over almost
                the full height (density ≈ 0.02–0.05), so it is safely excluded.
    """
    h_px = dark_mask.shape[0]

    dark_count   = dark_mask.sum(axis=0).astype(np.float32)   # dark pixels per col
    col_has_dark = dark_count > 0

    first_dark = np.argmax(dark_mask,          axis=0)
    last_dark  = h_px - 1 - np.argmax(dark_mask[::-1, :], axis=0)

    span    = np.where(col_has_dark, (last_dark - first_dark + 1).astype(np.float32), 1.0)
    density = dark_count / span

    return (span > h_px * 0.30) & (density > 0.40)


def _detect_stave_groups(dark_no_barlines: np.ndarray, n_parts: int) -> list:
    """
    Identify the y-pixel positions of all 5×n_parts stave lines and group them
    into n_parts sets of 5, ordered top-to-bottom.

    Strategy:
      1. Compute row darkness (mean fraction of dark pixels per row). Stave lines
         appear as sharp peaks — they span nearly the full image width, giving a
         much higher row-darkness than isolated notes or beams.
      2. Threshold at 30% of the maximum row darkness to find candidate rows, then
         cluster consecutive candidate rows into individual line centres.
      3. Keep the 5×n_parts most prominent line centres (by peak darkness).
      4. Find the n_parts−1 largest consecutive gaps between sorted centres; those
         are the inter-part boundaries. Everything else is an intra-part gap.

    Returns:
        List of n_parts lists, each containing 5 integer y-pixel indices
        (centres of the stave lines), ordered top-to-bottom.

    Raises:
        ValueError if fewer than 5×n_parts stave line candidates are found.
    """
    row_darkness = dark_no_barlines.mean(axis=1)
    threshold    = max(0.15, row_darkness.max() * 0.30)
    candidate    = row_darkness > threshold

    # Find runs of consecutive candidate rows
    padded  = np.concatenate([[False], candidate, [False]])
    changes = np.diff(padded.astype(np.int8))
    starts  = np.where(changes ==  1)[0]
    ends    = np.where(changes == -1)[0]

    centers  = [(int(s) + int(e)) // 2 for s, e in zip(starts, ends)]
    darkness = [float(row_darkness[s:e].max()) for s, e in zip(starts, ends)]

    n_lines = 5 * n_parts
    if len(centers) < n_lines:
        raise ValueError(
            f"Stave detection found only {len(centers)} line candidates "
            f"(need {n_lines}). Try increasing --dpi or check that the page "
            f"contains a full score system."
        )

    # Keep the n_lines most prominent, re-sorted by y-position
    if len(centers) > n_lines:
        order   = np.argsort(darkness)[::-1][:n_lines]
        order   = np.sort(order)
        centers = [centers[i] for i in order]

    # Find the n_parts−1 largest consecutive gaps → inter-part boundaries
    gaps           = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    split_after    = set(np.argsort(gaps)[::-1][: n_parts - 1].tolist())

    groups: list   = []
    current: list  = []
    for i, c in enumerate(centers):
        current.append(c)
        if i in split_after:
            groups.append(current)
            current = []
    groups.append(current)

    if len(groups) != n_parts or any(len(g) != 5 for g in groups):
        raise ValueError(
            f"Stave grouping produced {[len(g) for g in groups]} lines per group; "
            f"expected {n_parts} groups of 5."
        )
    return groups


def find_part_boundaries(
    page: fitz.Page, clip: fitz.Rect, n_parts: int, dpi: int
) -> list:
    """
    Compute n_parts+1 y-coordinates (PDF points) that divide the clip into n_parts
    strips using stave-connectivity analysis.

    Returns [clip.y0, cut_1, cut_2, ..., clip.y1] where each cut is the midpoint
    of the vertical whitespace between the two adjacent parts' content extents.

    Algorithm:
      1. Render the clip region as greyscale; threshold to a dark-pixel mask.
      2. Detect and mask bar-line columns so they don't bridge all parts.
      3. Detect stave-line groups (5 lines per part).
      4. Label connected components on the bar-line-masked image (8-connectivity).
      5. For each part, collect all component labels that touch its 5 stave lines
         via flood-fill — this captures notes, stems, beams, ledger lines, etc.
      6. Identify satellite components (slurs, dynamics, text) not connected to any
         stave, and assign each to the part whose connected-content bounding box is
         nearest in 2-D Euclidean distance.
      7. Recompute each part's y-extent (connected + satellites).
      8. Place the cut between parts i and i+1 at the midpoint of
         (bottom of part i content, top of part i+1 content).

    Falls back to equal-height division if stave detection fails.
    """
    img     = _render_gray_clip(page, clip, dpi)
    h_px, w_px = img.shape
    scale   = 72.0 / dpi
    dark_mask = img < DARK_THRESHOLD

    # ── 1. Detect and mask bar lines ──────────────────────────────────────────
    barline_cols       = _detect_barline_columns(dark_mask)
    dark_no_barlines   = dark_mask.copy()
    dark_no_barlines[:, barline_cols] = False

    # ── 2. Detect stave groups ────────────────────────────────────────────────
    stave_groups = _detect_stave_groups(dark_no_barlines, n_parts)

    # ── 3. Label connected components (8-connectivity) ────────────────────────
    labeled, _ = ndimage.label(dark_no_barlines, structure=_CC_STRUCTURE)

    # ── 4. Collect component labels touching each part's stave lines ──────────
    part_label_sets: list[set] = []
    for group in stave_groups:
        labels: set = set()
        for y_px in group:
            for dy in range(-2, 3):           # ±2 px tolerance for line thickness
                row = int(np.clip(y_px + dy, 0, h_px - 1))
                row_labels = labeled[row, :]
                labels.update(row_labels[row_labels > 0].tolist())
        part_label_sets.append(labels)

    # ── 5. Find satellite components ──────────────────────────────────────────
    all_stave_labels = set().union(*part_label_sets)
    all_labels       = set(np.unique(labeled).tolist()) - {0}
    satellite_labels = all_labels - all_stave_labels

    # ── 6. Compute per-part connected-content bounding boxes ──────────────────
    # Build a lookup array for fast mask construction: label → part index (-1 = none)
    max_label   = int(labeled.max())
    label_owner = np.full(max_label + 1, -1, dtype=np.int32)
    for i, labels in enumerate(part_label_sets):
        for lbl in labels:
            if 0 < lbl <= max_label:
                label_owner[lbl] = i

    part_bboxes: list = []       # (r0, c0, r1, c1) in pixel coords
    for i, labels in enumerate(part_label_sets):
        if labels:
            mask = label_owner[labeled] == i
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            if len(rows) > 0 and len(cols) > 0:
                part_bboxes.append(
                    (int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1]))
                )
                continue
        # Fallback: use stave y-range
        ys = stave_groups[i]
        part_bboxes.append((ys[0], 0, ys[-1], w_px - 1))

    # ── 7. Assign satellites to the nearest part (2-D Euclidean to bbox) ──────
    for sat_lbl in satellite_labels:
        sat_mask = labeled == int(sat_lbl)
        rows = np.where(sat_mask.any(axis=1))[0]
        cols = np.where(sat_mask.any(axis=0))[0]
        if len(rows) == 0:
            continue
        sat_cy = (rows[0]  + rows[-1])  / 2.0
        sat_cx = (cols[0]  + cols[-1])  / 2.0

        min_dist = float("inf")
        nearest  = 0
        for i, (r0, c0, r1, c1) in enumerate(part_bboxes):
            ny   = float(np.clip(sat_cy, r0, r1))
            nx   = float(np.clip(sat_cx, c0, c1))
            dist = np.hypot(sat_cy - ny, sat_cx - nx)
            if dist < min_dist:
                min_dist = dist
                nearest  = i

        part_label_sets[nearest].add(int(sat_lbl))
        # Update the owner lookup so the y-range recompute below is correct
        if 0 < int(sat_lbl) <= max_label:
            label_owner[int(sat_lbl)] = nearest

    # ── 8. Recompute per-part y-ranges including satellites ───────────────────
    part_y_ranges: list = []
    for i in range(n_parts):
        mask = label_owner[labeled] == i
        rows = np.where(mask.any(axis=1))[0]
        if len(rows) > 0:
            part_y_ranges.append((int(rows[0]), int(rows[-1])))
        else:
            ys = stave_groups[i]
            part_y_ranges.append((ys[0], ys[-1]))

    # ── 9. Cut boundaries: midpoint of whitespace between adjacent parts ───────
    cuts_px: list = []
    for i in range(n_parts - 1):
        y_bottom_i    = part_y_ranges[i][1]
        y_top_next    = part_y_ranges[i + 1][0]
        cuts_px.append((y_bottom_i + y_top_next) / 2.0)

    boundaries = [clip.y0]
    for cut_px in cuts_px:
        boundaries.append(clip.y0 + cut_px * scale)
    boundaries.append(clip.y1)

    return boundaries


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
    Main pipeline: preprocess pages into sub-pages (optional), detect part boundaries
    using stave-connectivity analysis, then assemble each part onto A4 pages.
    """
    doc = fitz.open(input_pdf)
    if len(doc) == 0:
        raise ValueError("Input PDF has no pages.")

    if output_dir is None:
        output_dir = str(Path(input_pdf).parent / (Path(input_pdf).stem + "_parts"))
    os.makedirs(output_dir, exist_ok=True)

    margin    = margin_mm * MM_TO_PT
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
    print(f"Sub-pages after preprocessing: {len(sub_pages)}")
    print("Detecting stave-connectivity boundaries...\n")

    # Precompute part boundaries for every sub-page so the per-part loop can
    # reuse them without re-running the (slow) connectivity analysis N times.
    sub_page_boundaries: list = []
    for idx, (page_num, clip) in enumerate(sub_pages):
        try:
            bounds = find_part_boundaries(doc[page_num], clip, n_parts, dpi)
            print(f"  Sub-page {idx + 1}: cuts at "
                  f"{[f'{y:.1f}' for y in bounds[1:-1]]} pt")
        except ValueError as exc:
            print(f"  Sub-page {idx + 1}: stave detection failed ({exc}) "
                  f"— falling back to equal division.")
            bounds = [
                clip.y0 + i * clip.height / n_parts for i in range(n_parts + 1)
            ]
        sub_page_boundaries.append(bounds)

    print()
    input_stem = Path(input_pdf).stem

    for part_idx in range(n_parts):
        out_doc      = fitz.open()
        current_page = out_doc.new_page(width=A4_W, height=A4_H)
        current_y    = 0.0
        last_h       = 0.0

        for (page_num, clip), boundaries in zip(sub_pages, sub_page_boundaries):
            band_y0   = boundaries[part_idx]
            band_y1   = boundaries[part_idx + 1]
            band_h    = band_y1 - band_y0
            if band_h <= 0:
                continue

            band_clip = fitz.Rect(clip.x0, band_y0, clip.x1, band_y1)

            # Scale band to content width, preserving aspect ratio
            scale    = content_w / clip.width
            scaled_h = min(band_h * scale, content_h)

            gap = gap_fraction * last_h

            if last_h > 0 and current_y + gap + scaled_h > content_h:
                current_page = out_doc.new_page(width=A4_W, height=A4_H)
                current_y    = 0.0
                gap          = 0.0

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
            "is trimmed on all four sides, then stave-connectivity analysis locates\n"
            "the precise boundary between each of the N parts (--parts) before\n"
            "assembling each part onto printer-safe A4 portrait pages."
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

    parser.add_argument("input", help="Path to the input PDF file.")
    parser.add_argument(
        "-n", "--parts",
        type=int, required=True, metavar="N",
        help="Number of instrument parts to extract (horizontal slices per system). Required.",
    )
    parser.add_argument(
        "-p", "--systems",
        type=int, default=1, metavar="P",
        help=(
            "Number of stave systems per page (default: 1, skips system-splitting). "
            "When > 1, each page is split into P sub-pages at the largest whitespace "
            "band near each expected division point."
        ),
    )
    parser.add_argument(
        "-g", "--gap",
        type=float, default=DEFAULT_GAP, metavar="FRAC",
        help=(
            f"Gap between consecutive sections on a page, as a fraction of the "
            f"preceding section's height (default: {DEFAULT_GAP}). "
            "0 = no gap. 0.5 = half a section height. 1 = a full section height."
        ),
    )
    parser.add_argument(
        "-m", "--margin",
        type=float, default=DEFAULT_MARGIN_MM, metavar="MM",
        help=(
            f"Printer-safe margin in mm on all four sides of each A4 page "
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
        type=int, default=DEFAULT_DPI, metavar="DPI",
        help=(
            f"Resolution in DPI for image analysis (default: {DEFAULT_DPI}). "
            "Increase for finer stave detection; higher values are slower."
        ),
    )

    args = parser.parse_args()

    if args.parts < 2:
        parser.error("--parts must be at least 2.")
    if args.systems < 1:
        parser.error("--systems must be at least 1.")
    if args.gap < 0:
        parser.error("--gap must be >= 0.")
    if args.margin < 0:
        parser.error("--margin must be >= 0.")
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
