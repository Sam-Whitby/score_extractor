# score_extractor

A command-line tool to split a PDF into **N horizontal parts**, one output PDF per part. Every output PDF contains the same strip from each page of the input.

This is intended for extracting individual instrument parts from multi-part score scans — the same problem that [Partifi](https://partifi.com) aims to solve.

---

## How it works

Given a 30-page PDF with 4 parts stacked on each page, the tool produces 4 output PDFs:

| Output file | Content |
|---|---|
| `score_part1_of_4.pdf` | Top quarter of every page |
| `score_part2_of_4.pdf` | Second quarter of every page |
| `score_part3_of_4.pdf` | Third quarter of every page |
| `score_part4_of_4.pdf` | Bottom quarter of every page |

Slicing is done at the PDF vector level using [PyMuPDF](https://pymupdf.readthedocs.io/), so the output is sharp at any zoom — no rasterisation.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python split_score.py <input.pdf> <n_parts> [-o output_dir]
```

### Arguments

| Argument | Description |
|---|---|
| `input.pdf` | Path to the input PDF |
| `n_parts` | Number of horizontal slices (must be ≥ 2) |
| `-o DIR` | Output directory (default: `<input_stem>_parts/` next to the input file) |

### Examples

Split a score with 4 stacked parts into 4 separate PDFs:

```bash
python split_score.py quartet_score.pdf 4
```

Split into 2 parts and write to a custom directory:

```bash
python split_score.py duet.pdf 2 -o ~/Desktop/duet_parts/
```

---

## Notes

- The slices are **equal-height** divisions of each page. If the staves aren't evenly distributed, you may need to crop or adjust the source PDF manually first.
- Output files are named `<input_stem>_part<N>_of_<total>.pdf`.
- Tested with IMSLP-format PDFs (A4 and letter).
