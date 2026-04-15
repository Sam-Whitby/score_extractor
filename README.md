# score_extractor

A command-line tool to extract individual instrument parts from multi-part PDF scores — the same problem that [Partifi](https://partifi.com) aims to solve.

Handles two layout patterns common on IMSLP:

- **Multiple parts stacked per page** — e.g. four instrument staves in four horizontal bands per page
- **Multiple systems per page** — e.g. two full stave groups stacked on a single page (`--systems 2`)

Both can be combined. All output is on A4 portrait pages within printer-safe margins.

---

## How it works

The tool runs up to two stages in a single command:

**Stage 1 — System splitting** (`--systems P`, optional)

Each source page is rendered as a greyscale image. For each of the P−1 required splits, the tool finds the longest continuous run of all-white rows within a search window centred on the expected geometric split point (±⅓ of a segment). The split is placed at the centre of that white band. Whitespace is then trimmed from **all four sides** of each sub-page. Output order: 1a, 1b, 2a, 2b, …

**Stage 2 — Part extraction** (`--parts N`)

Each (sub-)page is divided into N equal horizontal strips. Each part's strips are assembled sequentially onto A4 pages, scaled to fill the full printable width with aspect ratio preserved. A configurable gap (`--gap`) is inserted between sections. If a section (plus its gap) would overflow the page, the remainder is left blank and the section starts on a fresh page.

All operations are at the PDF vector level using [PyMuPDF](https://pymupdf.readthedocs.io/) — no rasterisation, output is sharp at any zoom.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```
python score_extractor.py <input.pdf> --parts N [options]
```

### Required

| Flag | Description |
|---|---|
| `input` | Path to the input PDF |
| `-n` / `--parts N` | Number of instrument parts to extract (horizontal slices per system) |

### Optional

| Flag | Default | Description |
|---|---|---|
| `-p` / `--systems P` | `1` | Systems (stave groups) per page. When > 1, each page is split at whitespace boundaries before extraction. |
| `-g` / `--gap FRAC` | `1.0` | Gap between sections on a page as a fraction of the preceding section's height. `0` = no gap, `1` = a full section-height gap. |
| `-m` / `--margin MM` | `15` | Printer-safe margin in mm on all four sides of each A4 page. |
| `-o` / `--output-dir DIR` | `<stem>_parts/` | Directory to write output PDFs into. |
| `--dpi DPI` | `150` | Resolution for whitespace analysis. Higher is slower but more accurate. |

---

## Examples

**4-part score, one system per page:**
```bash
python score_extractor.py quartet.pdf --parts 4
```

**4-part score, two systems per page, half-section gap:**
```bash
python score_extractor.py quartet.pdf --parts 4 --systems 2 --gap 0.5
```

**2-part score, custom output directory, tighter margins:**
```bash
python score_extractor.py duet.pdf --parts 2 -o ~/Desktop/duet_parts/ --margin 10
```

**Finer whitespace detection for a dense score:**
```bash
python score_extractor.py score.pdf --parts 4 --systems 2 --dpi 200
```

---

## Output

Each part is written as `<input_stem>_part<k>_of_<N>.pdf` in the output directory.

For a 30-page input with `--parts 4 --systems 2` the output is:

```
quartet_parts/
  quartet_part1_of_4.pdf   ← top strip of every sub-page (e.g. Violin I)
  quartet_part2_of_4.pdf   ← second strip                 (e.g. Violin II)
  quartet_part3_of_4.pdf   ← third strip                  (e.g. Viola)
  quartet_part4_of_4.pdf   ← bottom strip                 (e.g. Cello)
```

---

## Notes

- The system split (`--systems`) searches for whitespace near the geometric midpoint. If your systems are unevenly distributed, try increasing `--dpi` for finer detection.
- Part slices (`--parts`) are strictly equal-height divisions of each (trimmed) sub-page. If staves occupy unequal space between parts, some manual source cropping may be needed.
- The default gap of `1.0` inserts a full section-height of space between each system on the output pages. Lower values (e.g. `0.2`) give a tighter layout.
