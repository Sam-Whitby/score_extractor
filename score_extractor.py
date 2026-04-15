#!/usr/bin/env python3
"""
score_extractor.py — Extract individual instrument parts from multi-part PDF scores.

Pipeline:
  1. Pre-processing (optional, --systems P > 1)
     Each page is split into P sub-pages at horizontal whitespace boundaries.
     Whitespace is trimmed from all four sides of each sub-page.

  2. Part extraction (--parts N)
     Each sub-page is analysed to find exact per-part boundaries using stave
     connectivity, then each part is assembled onto A4 portrait pages.

Usage:
    python score_extractor.py input.pdf --parts 4
    python score_extractor.py input.pdf --parts 4 --systems 2 --gap 0.5 --dpi 200
"""

import io
import struct
import zlib
import fitz
import numpy as np
from scipy import ndimage
import argparse
import os
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

A4_W = 595.0
A4_H = 842.0
MM_TO_PT = 72.0 / 25.4

WHITE_THRESHOLD = 250   # pixel ≥ this → white background
DARK_THRESHOLD  = 128   # pixel < this → black ink
DEFAULT_DPI     = 150
DEFAULT_GAP     = 1.0
DEFAULT_MARGIN_MM = 15.0

# 8-connectivity so diagonal beams/stems are followed during flood-fill.
_CC_STRUCTURE = np.ones((3, 3), dtype=np.int32)


# ── Low-level image helpers ────────────────────────────────────────────────────

def _render_gray_clip(page: fitz.Page, clip: fitz.Rect, dpi: int) -> np.ndarray:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def _numpy_to_pixmap(arr: np.ndarray) -> fitz.Pixmap:
    """Convert a greyscale uint8 (H, W) numpy array to a fitz.Pixmap."""
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape
    return fitz.Pixmap(fitz.csGRAY, w, h, arr.tobytes(), False)


def _longest_white_run(mask: np.ndarray, start: int, end: int):
    """Longest contiguous True run in mask[start:end]. Returns (run_start, run_end)."""
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


# ── System-split helpers (Step 1) ─────────────────────────────────────────────

def find_split_positions(page: fitz.Page, p_sub: int, dpi: int) -> list:
    """P-1 horizontal y-coordinates (PDF pts) for splitting a page into p_sub systems."""
    img = _render_gray_clip(page, page.rect, dpi)
    h_px = img.shape[0]
    row_is_white = np.all(img >= WHITE_THRESHOLD, axis=1)
    scale = 72.0 / dpi
    split_ys = []
    for k in range(1, p_sub):
        center_px = int(k / p_sub * h_px)
        half_win  = max(1, h_px // (p_sub * 3))
        s0, s1    = max(0, center_px - half_win), min(h_px, center_px + half_win)
        rs, re    = _longest_white_run(row_is_white, s0, s1)
        split_px  = center_px if rs is None else (rs + re) // 2
        split_ys.append(page.rect.y0 + split_px * scale)
    return split_ys


def find_content_bounds_2d(page: fitz.Page, clip: fitz.Rect, dpi: int):
    """
    Tight (x_left, y_top, x_right, y_bottom) in PDF pts, whitespace trimmed on all sides.

    Top/bottom: strict per-row check (all pixels white).
    Left/right: two-stage robust check tolerating 1-pixel grey borders.
      Stage 1 — find largest white column-run in each outer third (≥98% white per column).
      Stage 2 — strict fine-trim within the coarse cut bounds.
    """
    img     = _render_gray_clip(page, clip, dpi)
    h_px, w_px = img.shape
    scale   = 72.0 / dpi

    # Top / bottom
    row_is_white = np.all(img >= WHITE_THRESHOLD, axis=1)
    content_rows = np.where(~row_is_white)[0]
    if len(content_rows) == 0:
        return clip.x0, clip.y0, clip.x1, clip.y1
    y_top    = clip.y0 + int(content_rows[0])        * scale
    y_bottom = clip.y0 + (int(content_rows[-1]) + 1) * scale

    # Left / right
    col_white_frac       = np.mean(img >= WHITE_THRESHOLD, axis=0)
    col_is_white_lenient = col_white_frac >= 0.98
    lt = max(1, w_px // 3);  rt = w_px - lt
    ls, le = _longest_white_run(col_is_white_lenient, 0,  lt)
    rs, re = _longest_white_run(col_is_white_lenient, rt, w_px)
    x_left_px  = (ls + le) // 2 if ls is not None else 0
    x_right_px = (rs + re) // 2 if rs is not None else w_px
    if x_left_px < x_right_px:
        inner         = img[:, x_left_px:x_right_px]
        inner_content = np.where(~np.all(inner >= WHITE_THRESHOLD, axis=0))[0]
        if len(inner_content) > 0:
            cl          = x_left_px
            x_left_px   = cl + int(inner_content[0])
            x_right_px  = cl + int(inner_content[-1]) + 1
    return clip.x0 + x_left_px * scale, y_top, clip.x0 + x_right_px * scale, y_bottom


# ── Bar-line and stave detection ───────────────────────────────────────────────

def _detect_barline_columns(dark_mask: np.ndarray) -> np.ndarray:
    """
    Boolean (w_px,) mask of candidate bar-line columns.

    Criterion: span (topmost→bottommost dark pixel) > 30% of image height
    AND density (dark pixels / span) > 0.40.

    This distinguishes bar lines (span ≈ 100%, density ≈ 0.85) from stave-line
    columns (span ≈ 100% but density ≈ 0.02 — only 5×N sparse rows are dark)
    and most note stems (span < 30%).  A second-pass verification against stave
    positions is required to reject stems from multiple aligned parts.
    """
    h_px         = dark_mask.shape[0]
    dark_count   = dark_mask.sum(axis=0).astype(np.float32)
    col_has_dark = dark_count > 0
    first_dark   = np.argmax(dark_mask,          axis=0)
    last_dark    = h_px - 1 - np.argmax(dark_mask[::-1, :], axis=0)
    span         = np.where(col_has_dark, (last_dark - first_dark + 1).astype(np.float32), 1.0)
    density      = dark_count / span
    return (span > h_px * 0.30) & (density > 0.40)


def _verify_barline_columns(
    barline_cols_candidate: np.ndarray,
    dark_mask: np.ndarray,
    stave_groups: list,
) -> np.ndarray:
    """
    Filter candidate bar-line columns to those that are dark at EVERY stave
    line position AND dark in the gap between every pair of adjacent stave
    groups (all with a ±3 px tolerance window).

    True bar lines are continuous through the entire system; note stems have
    breaks in the inter-stave gaps, so they fail the gap check even when
    multiple aligned stems coincidentally pass the initial density criterion.
    """
    h_px     = dark_mask.shape[0]
    verified = barline_cols_candidate.copy()

    # Must be dark at every individual stave line
    for group in stave_groups:
        for y_line in group:
            y_lo = max(0, y_line - 3)
            y_hi = min(h_px, y_line + 4)
            verified = verified & dark_mask[y_lo:y_hi, :].any(axis=0)

    # Must be dark at the midpoint of every inter-part gap
    for i in range(len(stave_groups) - 1):
        gap_mid = (stave_groups[i][-1] + stave_groups[i + 1][0]) // 2
        y_lo = max(0, gap_mid - 3)
        y_hi = min(h_px, gap_mid + 4)
        if y_lo < y_hi:
            verified = verified & dark_mask[y_lo:y_hi, :].any(axis=0)

    return verified


def _group_barline_cols(barline_cols: np.ndarray) -> list:
    """
    Group consecutive True entries in barline_cols into (start_col, end_col)
    inclusive pairs, one per physical bar line.
    """
    groups = []
    start  = None
    for c, is_bl in enumerate(barline_cols.tolist()):
        if is_bl and start is None:
            start = c
        elif not is_bl and start is not None:
            groups.append((start, c - 1))
            start = None
    if start is not None:
        groups.append((start, len(barline_cols) - 1))
    return groups


def _remove_system_bracket(
    barline_cols: np.ndarray,
    barline_groups: list,
    w_px: int,
) -> tuple:
    """
    Detect and remove the ornate system bracket at the far left of each system.

    The bracket is a thick connected vertical line (typically 4-20 px wide) that
    appears before the first thin barline and within the leftmost 15% of the image.
    Regular barlines are 1-2 px wide; the bracket is conspicuously wider.

    Returns (updated_barline_cols, bracket_cols_range) where bracket_cols_range
    is (c_start, c_end) inclusive, or None if no bracket was found.
    """
    if not barline_groups:
        return barline_cols, None

    widths    = [end - start + 1 for start, end in barline_groups]
    median_w  = float(np.median(widths)) if len(widths) >= 3 else 1.0
    first     = barline_groups[0]
    first_w   = first[1] - first[0] + 1
    first_pos = first[0] / w_px

    # Bracket is significantly wider than typical barlines and near the left edge
    if first_w > max(3, median_w * 2.0) and first_pos < 0.15:
        updated = barline_cols.copy()
        updated[first[0]: first[1] + 1] = False
        return updated, first

    return barline_cols, None


def _detect_stave_groups(dark_no_barlines: np.ndarray, n_parts: int) -> list:
    """
    Find the y-pixel positions of all 5×n_parts stave lines, grouped into n_parts
    lists of 5 (ordered top-to-bottom).

    Raises ValueError if fewer than 5×n_parts candidates are found.
    """
    row_darkness = dark_no_barlines.mean(axis=1)
    threshold    = max(0.15, row_darkness.max() * 0.30)
    candidate    = row_darkness > threshold

    padded  = np.concatenate([[False], candidate, [False]])
    changes = np.diff(padded.astype(np.int8))
    starts  = np.where(changes ==  1)[0]
    ends    = np.where(changes == -1)[0]

    centers  = [(int(s) + int(e)) // 2 for s, e in zip(starts, ends)]
    darkness = [float(row_darkness[s:e].max()) for s, e in zip(starts, ends)]

    n_lines = 5 * n_parts
    if len(centers) < n_lines:
        raise ValueError(
            f"Found only {len(centers)} stave-line candidates (need {n_lines}). "
            "Try --dpi 200 or check the page contains a full score system."
        )
    if len(centers) > n_lines:
        order   = np.argsort(darkness)[::-1][:n_lines]
        centers = [centers[i] for i in np.sort(order)]

    gaps        = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    split_after = set(np.argsort(gaps)[::-1][: n_parts - 1].tolist())

    groups: list = []
    current: list = []
    for i, c in enumerate(centers):
        current.append(c)
        if i in split_after:
            groups.append(current)
            current = []
    groups.append(current)

    if len(groups) != n_parts or any(len(g) != 5 for g in groups):
        raise ValueError(
            f"Stave grouping gave {[len(g) for g in groups]} lines per group; "
            f"expected {n_parts} groups of 5."
        )
    return groups


# ── Per-part mask computation ──────────────────────────────────────────────────

def compute_part_data(
    page: fitz.Page, clip: fitz.Rect, n_parts: int, dpi: int
):
    """
    Return (part_masks, img) for the given clip region.

    part_masks[i] is a boolean (H, W) array that is True for every pixel
    belonging to part i:

      • Pixels from connected components (8-connected) that touch part i's 5 stave
        lines — captures notes, stems, beams, ledger lines, accidentals. Full extent
        is kept (no clipping at part boundaries).

      • Bar-line pixels within part i's stave y-span (with ±2 px buffer). Bar lines
        outside this span are excluded → they appear white in the rendered output,
        matching how extracted parts normally look.

      • Satellite pixels (components not touching any stave) assigned to the part
        whose connected-content bounding box is nearest in 2-D Euclidean distance.
        Satellites are included as their full connected component (not clipped).

      • Components touching multiple stave groups ('shared') are clipped to the
        cut boundary (midpoint between adjacent stave groups) so they are split
        between the two parts rather than duplicated.

    img is the greyscale source image at dpi.

    Raises ValueError if stave detection fails.
    """
    img        = _render_gray_clip(page, clip, dpi)
    h_px, w_px = img.shape
    dark_mask  = img < DARK_THRESHOLD

    # ── 1. Initial bar-line detection ─────────────────────────────────────────
    barline_cols = _detect_barline_columns(dark_mask)
    dark_nb      = dark_mask.copy()
    dark_nb[:, barline_cols] = False

    # ── 2. Stave groups (using initial bar-line mask) ─────────────────────────
    stave_groups = _detect_stave_groups(dark_nb, n_parts)

    # ── 2a. Verify bar-line columns ───────────────────────────────────────────
    # Reject false positives (e.g. note stems from multiple aligned parts) by
    # requiring each candidate to be dark at every stave line AND in every
    # inter-part gap.
    barline_cols = _verify_barline_columns(barline_cols, dark_mask, stave_groups)

    # ── 2b. Remove system bracket ─────────────────────────────────────────────
    # The ornate vertical bracket at the far left of each system connects all
    # parts.  It passes verification (it is continuous through all staves and
    # gaps) but is much wider than a real barline.  Remove it from the analysis
    # and white it out in the source image so it never appears in any part.
    _pre_groups  = _group_barline_cols(barline_cols)
    barline_cols, bracket_range = _remove_system_bracket(barline_cols, _pre_groups, w_px)
    if bracket_range is not None:
        c0, c1 = bracket_range
        img[:, c0: c1 + 1]       = 255
        dark_mask[:, c0: c1 + 1] = False

    dark_nb = dark_mask.copy()
    dark_nb[:, barline_cols] = False

    # ── 2c. Crossing bar-line pixels ─────────────────────────────────────────
    # Pixels in bar-line columns that belong to objects crossing the bar line
    # (slurs, crescendos, ties) have non-bar-line dark content on BOTH sides.
    # These must not be whited out — they must remain continuous in the output.
    #
    # From non-bar-line dark pixels (±3 rows vertical tolerance for curved
    # objects), propagate rightward through consecutive dark bar-line columns
    # → left_reach; similarly leftward → right_reach.  Pixels reachable from
    # both directions are true crossing pixels.
    barline_cols_2d = np.zeros((h_px, w_px), dtype=bool)
    barline_cols_2d[:, barline_cols] = True

    dark_nb_exp = ndimage.binary_dilation(
        dark_nb, structure=np.ones((7, 1), dtype=np.int32)   # ±3 row tolerance
    )

    left_reach = np.zeros((h_px, w_px), dtype=bool)
    _seed = np.zeros((h_px, w_px), dtype=bool)
    _seed[:, 1:] = dark_nb_exp[:, :-1]
    left_reach |= _seed & dark_mask & barline_cols_2d
    for _ in range(6):
        _step = np.zeros((h_px, w_px), dtype=bool)
        _step[:, 1:] = left_reach[:, :-1]
        _new = _step & dark_mask & barline_cols_2d & ~left_reach
        if not _new.any():
            break
        left_reach |= _new

    right_reach = np.zeros((h_px, w_px), dtype=bool)
    _seed = np.zeros((h_px, w_px), dtype=bool)
    _seed[:, :-1] = dark_nb_exp[:, 1:]
    right_reach |= _seed & dark_mask & barline_cols_2d
    for _ in range(6):
        _step = np.zeros((h_px, w_px), dtype=bool)
        _step[:, :-1] = right_reach[:, 1:]
        _new = _step & dark_mask & barline_cols_2d & ~right_reach
        if not _new.any():
            break
        right_reach |= _new

    crossing_bl = dark_mask & barline_cols_2d & left_reach & right_reach

    # ── 3. Connected components ───────────────────────────────────────────────
    labeled, _ = ndimage.label(dark_nb, structure=_CC_STRUCTURE)
    max_label  = int(labeled.max())

    # ── 4. Labels touching each part's stave lines ───────────────────────────
    part_label_sets: list[set] = [set() for _ in range(n_parts)]
    for i, group in enumerate(stave_groups):
        for y_px in group:
            for dy in range(-2, 3):
                row = int(np.clip(y_px + dy, 0, h_px - 1))
                row_lbls = labeled[row, :]
                part_label_sets[i].update(row_lbls[row_lbls > 0].tolist())

    # ── 5. Classify labels ────────────────────────────────────────────────────
    label_parts: dict = {}
    for i, lbls in enumerate(part_label_sets):
        for lbl in lbls:
            label_parts.setdefault(lbl, []).append(i)

    all_stave_labels = set(label_parts.keys())
    all_labels       = set(np.unique(labeled).tolist()) - {0}
    satellite_labels = all_labels - all_stave_labels

    # label_owner: single-part → part index
    #              shared       → -2
    #              satellite    → assigned below
    label_owner  = np.full(max_label + 1, -1, dtype=np.int32)
    shared_labels: set = set()
    for lbl, parts in label_parts.items():
        if 0 < lbl <= max_label:
            if len(parts) == 1:
                label_owner[lbl] = parts[0]
            else:
                shared_labels.add(lbl)
                label_owner[lbl] = -2

    # ── 6. Per-part bboxes from single-part content (for satellite distances) ─
    part_bboxes: list = []
    for i in range(n_parts):
        m    = label_owner[labeled] == i
        rows = np.where(m.any(axis=1))[0]
        cols = np.where(m.any(axis=0))[0]
        if len(rows) > 0 and len(cols) > 0:
            part_bboxes.append((int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])))
        else:
            ys = stave_groups[i]
            part_bboxes.append((ys[0], 0, ys[-1], w_px - 1))

    # ── 7. Assign satellites to nearest part (2-D Euclidean to bbox) ─────────
    for sat_lbl in satellite_labels:
        lbl_int = int(sat_lbl)
        if lbl_int < 1 or lbl_int > max_label:
            continue
        sat_m = labeled == lbl_int
        rows  = np.where(sat_m.any(axis=1))[0]
        cols  = np.where(sat_m.any(axis=0))[0]
        if len(rows) == 0:
            continue
        sat_cy = (rows[0] + rows[-1]) / 2.0
        sat_cx = (cols[0] + cols[-1]) / 2.0
        nearest = min(
            range(n_parts),
            key=lambda i: np.hypot(
                sat_cy - float(np.clip(sat_cy, part_bboxes[i][0], part_bboxes[i][2])),
                sat_cx - float(np.clip(sat_cx, part_bboxes[i][1], part_bboxes[i][3])),
            ),
        )
        label_owner[lbl_int] = nearest

    # ── 8. Cut positions between stave groups (for shared-label clipping) ─────
    cut_px = [
        (stave_groups[i][-1] + stave_groups[i + 1][0]) // 2
        for i in range(n_parts - 1)
    ]
    y_tops = [0]       + cut_px
    y_bots = cut_px    + [h_px]

    # ── 9. Build per-part masks ───────────────────────────────────────────────
    part_masks: list = []
    for i in range(n_parts):
        # Single-part connected content + satellites: full extent (not clipped)
        mask_i = (label_owner[labeled] == i)

        # Shared components: clipped to this part's cut y-range
        for lbl in shared_labels:
            if i in label_parts.get(lbl, []):
                lbl_m = (labeled == lbl).copy()
                if y_tops[i] > 0:
                    lbl_m[: y_tops[i], :] = False
                if y_bots[i] < h_px:
                    lbl_m[y_bots[i] :,  :] = False
                mask_i = mask_i | lbl_m

        # Bar-line pixels within this part's stave span (±2 px buffer)
        st    = stave_groups[i][0]
        sb    = stave_groups[i][-1]
        bl_y0 = max(0,    st - 2)
        bl_y1 = min(h_px, sb + 3)
        bl_mask              = np.zeros((h_px, w_px), dtype=bool)
        bl_mask[bl_y0:bl_y1, barline_cols] = dark_mask[bl_y0:bl_y1, barline_cols]
        mask_i = mask_i | bl_mask

        # Crossing bar-line pixels in this part's y-range (slurs, crescendos etc.
        # that pass through the bar line outside the stave span must not be broken)
        cross_i = crossing_bl.copy()
        if y_tops[i] > 0:    cross_i[:y_tops[i], :] = False
        if y_bots[i] < h_px: cross_i[y_bots[i]:,  :] = False
        mask_i = mask_i | cross_i

        # Dilate 1 px to include antialiased ink edges, then restrict to
        # pixels that are actually non-white (avoids pulling in background noise).
        mask_i = ndimage.binary_dilation(mask_i, structure=_CC_STRUCTURE, iterations=1)
        mask_i = mask_i & (img < WHITE_THRESHOLD)

        part_masks.append(mask_i)

    barline_groups = _group_barline_cols(barline_cols)
    return part_masks, img, barline_groups, stave_groups


# ── Sub-page clip builder (Step 1 output) ─────────────────────────────────────

def build_sub_page_clips(doc: fitz.Document, p_sub: int, dpi: int) -> list:
    """
    Returns ordered list of (page_num, clip_rect) after system-splitting and 4-side
    whitespace trimming. Order: 1a, 1b, 2a, 2b, ...
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
            if clip.width > 1 and clip.height > 1:
                sub_pages.append((page_num, clip))
    return sub_pages



# ── Bar-number annotation helper ──────────────────────────────────────────────

BAR_FONTSIZE = 5.5   # pt — small, unobtrusive


def _annotate_bar_numbers(
    page: fitz.Page,
    dest: fitz.Rect,
    barline_groups: list,
    bar_start: int,
    part_idx: int,
    w_px: int,
    content_w: float,
):
    """
    Write small grey bar numbers above each internal bar-line in this section.

    Skips:
      • The LAST barline of every section — it is the same physical barline as the
        FIRST of the next system, which will be labelled there instead.
      • The FIRST barline of part 0 on every section — the printed score already
        shows this number at the beginning of each system for the top part.
    """
    n = len(barline_groups)
    for bl_idx, (bl_start, bl_end) in enumerate(barline_groups):
        if bl_idx == n - 1:
            continue   # closing barline → will be the opening of the next system
        if part_idx == 0 and bl_idx == 0:
            continue   # top part already has this number printed in the score

        bar_num = bar_start + bl_idx
        bl_cx   = (bl_start + bl_end) / 2.0
        x_pdf   = dest.x0 + bl_cx / w_px * content_w
        x_pdf  -= len(str(bar_num)) * BAR_FONTSIZE * 0.28   # approximate centering
        y_pdf   = dest.y0 + BAR_FONTSIZE + 1.5

        page.insert_text(
            (x_pdf, y_pdf),
            str(bar_num),
            fontsize=BAR_FONTSIZE,
            color=(0.35, 0.35, 0.35),
        )


# ── Main pipeline ──────────────────────────────────────────────────────────────────────────────

def extract_parts(
    input_pdf: str,
    n_parts: int,
    p_sub: int = 1,
    margin_mm: float = DEFAULT_MARGIN_MM,
    output_dir: str = None,
    dpi: int = DEFAULT_DPI,
):
    """
    Full pipeline: system-split → stave-connectivity part detection →
    mask-based rendering → equal-spacing A4 assembly.

    Two phases:
      Phase 1 — collect all per-part section images and bar-number metadata.
      Phase 2 — pack sections onto A4 pages (greedy, minimum 2 pt gap) and
                distribute with equal whitespace between staves on each page.
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
    MIN_GAP   = 2.0   # pt — minimum whitespace between sections on a page

    print(f"Input:         {input_pdf}  ({len(doc)} pages)")
    print(f"Systems/page:  {p_sub}")
    print(f"Parts:         {n_parts}")
    print(f"Print margin:  {margin_mm} mm ({margin:.1f} pt)")
    print(f"Analysis DPI:  {dpi}")
    print(f"Output dir:    {output_dir}/\n")

    sub_pages  = build_sub_page_clips(doc, p_sub, dpi)
    print(f"Sub-pages after preprocessing: {len(sub_pages)}\n")

    input_stem = Path(input_pdf).stem

    # ── Phase 1: collect sections ──────────────────────────────────────────────────────────
    # Each section dict: 'cropped', 'scaled_h', 'w_px', 'barline_groups', 'bar_start'
    all_sections: list = [[] for _ in range(n_parts)]
    bar_number = 1

    for sp_idx, (page_num, clip) in enumerate(sub_pages):
        print(f"  Processing sub-page {sp_idx + 1}/{len(sub_pages)} "
              f"(source page {page_num + 1})…", end="", flush=True)

        try:
            part_masks, img, barline_groups, _ = compute_part_data(
                doc[page_num], clip, n_parts, dpi
            )
            print(" stave detection OK")
        except ValueError as exc:
            print(f" WARNING: {exc} — using equal split fallback.")
            img   = _render_gray_clip(doc[page_num], clip, dpi)
            h_px  = img.shape[0]
            equal_masks = []
            for i in range(n_parts):
                m  = np.zeros(img.shape, dtype=bool)
                r0 = int(i / n_parts * h_px)
                r1 = int((i + 1) / n_parts * h_px)
                m[r0:r1, :] = img[r0:r1, :] < WHITE_THRESHOLD
                equal_masks.append(m)
            part_masks     = equal_masks
            barline_groups = []   # no reliable bar info in fallback

        w_px  = img.shape[1]
        scale = content_w / clip.width

        for part_idx in range(n_parts):
            img_part = np.full_like(img, 255)
            mask     = part_masks[part_idx]
            img_part[mask] = img[mask]

            dark_rows = np.where((img_part < WHITE_THRESHOLD).any(axis=1))[0]
            if len(dark_rows) == 0:
                continue
            r0, r1  = int(dark_rows[0]), int(dark_rows[-1])
            cropped = img_part[r0: r1 + 1, :]

            h_pt     = cropped.shape[0] * (72.0 / dpi)
            scaled_h = min(h_pt * scale, content_h)
            if scaled_h <= 0:
                continue

            all_sections[part_idx].append({
                'cropped':        cropped,
                'scaled_h':       scaled_h,
                'w_px':           w_px,
                'barline_groups': barline_groups,
                'bar_start':      bar_number,
            })

        # The closing barline of this system IS the opening barline of the next,
        # so bar_number advances by len(barline_groups) - 1.
        if barline_groups:
            bar_number += len(barline_groups) - 1

    doc.close()

    # ── Phase 2: equal-spacing layout ───────────────────────────────────────────────────────────
    print()
    for part_idx in range(n_parts):
        sections = all_sections[part_idx]
        if not sections:
            continue

        out_doc = fitz.open()

        # Pack sections onto pages greedily (add as many as fit with MIN_GAP).
        pages: list = []
        current: list = []

        for sec in sections:
            n_now  = len(current) + 1
            total_h = sum(s['scaled_h'] for s in current) + sec['scaled_h']
            needed  = total_h + (n_now - 1) * MIN_GAP
            if needed <= content_h or not current:
                current.append(sec)
            else:
                pages.append(current)
                current = [sec]
        if current:
            pages.append(current)

        for pg_secs in pages:
            page    = out_doc.new_page(width=A4_W, height=A4_H)
            n_pg    = len(pg_secs)
            total_h = sum(s['scaled_h'] for s in pg_secs)
            # Equal whitespace between sections; never below MIN_GAP.
            gap = (content_h - total_h) / (n_pg - 1) if n_pg > 1 else 0.0
            gap = max(gap, MIN_GAP)

            y = 0.0
            for sec in pg_secs:
                dest = fitz.Rect(
                    margin,
                    margin + y,
                    margin + content_w,
                    margin + y + sec['scaled_h'],
                )
                page.insert_image(dest, pixmap=_numpy_to_pixmap(sec['cropped']))
                _annotate_bar_numbers(
                    page, dest,
                    sec['barline_groups'], sec['bar_start'],
                    part_idx, sec['w_px'], content_w,
                )
                y += sec['scaled_h'] + gap

        label    = f"part{part_idx + 1}_of_{n_parts}"
        out_path = os.path.join(output_dir, f"{input_stem}_{label}.pdf")
        out_doc.save(out_path, garbage=4, deflate=True)
        out_doc.close()
        print(f"  Saved: {out_path}  ({len(pages)} page(s), {len(sections)} system(s))")

    print(f"\nDone — {n_parts} parts written to '{output_dir}'")


# ── CLI ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract individual instrument parts from a multi-part PDF score.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python score_extractor.py quartet.pdf --parts 4\n"
            "  python score_extractor.py quartet.pdf --parts 4 --systems 2\n"
            "  python score_extractor.py quartet.pdf --parts 4 --systems 2 --dpi 200 -o ~/parts/"
        ),
    )
    parser.add_argument("input", help="Path to input PDF.")
    parser.add_argument("-n", "--parts",   type=int, required=True, metavar="N",
        help="Number of instrument parts to extract (horizontal slices per system).")
    parser.add_argument("-p", "--systems", type=int, default=1, metavar="P",
        help="Systems per page (default 1). >1 triggers horizontal page-splitting at whitespace.")
    parser.add_argument("-m", "--margin",  type=float, default=DEFAULT_MARGIN_MM, metavar="MM",
        help=f"Printer-safe margin in mm on all A4 sides (default {DEFAULT_MARGIN_MM}).")
    parser.add_argument("-o", "--output-dir", metavar="DIR",
        help="Output directory (default: <input_stem>_parts/).")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, metavar="DPI",
        help=f"Rendering/analysis resolution (default {DEFAULT_DPI}). "
             "Higher gives better output quality and detection accuracy but is slower.")

    args = parser.parse_args()
    if args.parts   < 2: parser.error("--parts must be >= 2.")
    if args.systems < 1: parser.error("--systems must be >= 1.")
    if args.margin  < 0: parser.error("--margin must be >= 0.")
    if not os.path.isfile(args.input): parser.error(f"File not found: {args.input}")

    extract_parts(
        input_pdf=args.input,
        n_parts=args.parts,
        p_sub=args.systems,
        margin_mm=args.margin,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
