# score_extractor

Command-line tools to extract individual instrument parts from multi-part PDF scores — the same problem that [Partifi](https://partifi.com) aims to solve.

The pipeline has two steps:

1. **`preprocess.py`** — split each page into P sub-pages at whitespace boundaries (for scores with multiple systems per page), trimming surrounding whitespace from each.
2. **`split_score.py`** — slice each page horizontally into N parts (one per instrument/voice) and assemble each part onto readable A4 pages.

Either step can be used independently.

---

## Pipeline overview

```
original.pdf
    │
    ▼  preprocess.py  (if each page has multiple systems stacked)
preprocessed.pdf  (2× pages, ordered 1a, 1b, 2a, 2b, ...)
    │
    ▼  split_score.py  (split into N instrument parts)
output/
  score_part1_of_4.pdf
  score_part2_of_4.pdf
  ...
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Step 1 — `preprocess.py`

Splits each page into P sub-pages by finding the largest horizontal whitespace band near the expected geometric split point. Whitespace is trimmed from the top and bottom of each resulting sub-page.

**When to use:** your source PDF has multiple complete systems stacked on each page (e.g. a quartet score with 2 systems per page — use P=2).

### Usage

```bash
python preprocess.py <input.pdf> <P> [-o output.pdf] [--dpi N]
```

| Argument | Description |
|---|---|
| `input.pdf` | Path to the input PDF |
| `P` | Sub-pages per page (must be ≥ 2) |
| `-o FILE` | Output PDF path (default: `<input_stem>_p<P>.pdf`) |
| `--dpi N` | Analysis resolution in DPI (default: 150; higher = slower but more accurate) |

### Example

A score with 2 systems per page, 30 pages → 60-page output ordered 1a, 1b, 2a, 2b, ...:

```bash
python preprocess.py quartet.pdf 2
# → quartet_p2.pdf  (60 pages)
```

### How the split is found

Each page is rendered as a greyscale image. For each of the P−1 split points, the tool searches for the longest continuous run of white rows within a window of ±⅓ of a segment around the expected geometric position. The split is placed at the centre of that white band. If no whitespace is found in the window, the geometric centre is used as a fallback.

---

## Step 2 — `split_score.py`

Slices each page horizontally into N equal strips (one per instrument/voice) and assembles each part's strips onto A4 portrait pages.

**When to use:** each page has N instrument parts stacked top-to-bottom in equal bands.

### Usage

```bash
python split_score.py <input.pdf> <N> [-o output_dir]
```

| Argument | Description |
|---|---|
| `input.pdf` | Path to the input PDF (can be the output of `preprocess.py`) |
| `N` | Number of horizontal parts (must be ≥ 2) |
| `-o DIR` | Output directory (default: `<input_stem>_parts/`) |

### Example

Extract 4 parts from a preprocessed score:

```bash
python split_score.py quartet_p2.pdf 4 -o quartet_parts/
# → quartet_parts/quartet_p2_part1_of_4.pdf  (Violin I)
# → quartet_parts/quartet_p2_part2_of_4.pdf  (Violin II)
# → quartet_parts/quartet_p2_part3_of_4.pdf  (Viola)
# → quartet_parts/quartet_p2_part4_of_4.pdf  (Cello)
```

### How A4 assembly works

Each strip is scaled to the **full width of A4 portrait** with aspect ratio preserved (no stretching). Strips are stacked top-to-bottom in page order. If a strip does not fit in the remaining space on a page, the rest is left as whitespace and the strip begins on a fresh page. This keeps the part readable sequentially with minimal page turns.

All operations are at the PDF vector level using [PyMuPDF](https://pymupdf.readthedocs.io/) — no rasterisation, so output is sharp at any zoom.

---

## Full worked example

```bash
# Step 1: split each page into 2 systems
python preprocess.py quartet_score.pdf 2

# Step 2: extract 4 instrument parts
python split_score.py quartet_score_p2.pdf 4
```

---

## Notes

- The preprocessing split is **equal-area** by design; the whitespace search only adjusts within a window around the expected position. If systems are very unevenly distributed, increase `--dpi` for finer analysis.
- The part-extraction split in `split_score.py` is strictly equal-height. If parts occupy unequal space on the page, manual source cropping may be needed.
- Output pages from `split_score.py` are **A4 portrait** (595 × 842 pt).
- Output files from `split_score.py` are named `<input_stem>_part<N>_of_<total>.pdf`.
