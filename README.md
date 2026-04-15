# score_extractor

Extracts individual instrument parts from multi-part PDF scores (e.g. IMSLP quartet scores) into separate, readable A4 PDFs — the same goal as [Partifi](https://partifi.com).

---

## How it works

### Step 1 — System splitting (optional, `--systems P`)

If each page contains multiple stave systems stacked vertically (e.g. two complete quartet systems per page), each page is split into P sub-pages. The split point is found by rendering the page as a greyscale image and locating the largest continuous white band within a search window centred on the expected geometric split. Whitespace is trimmed from all four sides of each sub-page.

Left/right trimming uses a two-stage approach: a lenient column-whiteness test (≥98% white pixels) finds the coarse cut inside each outer third of the page, then a strict all-white check fine-trims within those bounds. This tolerates the 1-pixel grey borders common in scanned PDFs.

### Step 2 — Stave-connectivity part detection

For each sub-page:

1. **Bar-line detection** — columns where the vertical span of dark pixels covers >30% of the image height *and* the pixel density within that span is >40% are classified as bar lines. This density criterion distinguishes bar lines (nearly solid ink, density ≈ 0.85) from stave-line columns (only ~20 sparse rows dark across the full height, density ≈ 0.02). Detected bar lines are temporarily masked out so they do not bridge parts during connectivity analysis.

2. **Stave detection** — after masking bar lines, rows with above-threshold darkness form peaks corresponding to the 5×N stave lines. The N×5 peaks with the highest row-darkness are kept and grouped into N sets of 5 by identifying the N−1 largest consecutive gaps between sorted line positions.

3. **Flood-fill** — connected components (8-connectivity, so diagonal beams are followed) are labelled on the bar-line-masked image. Each component is classified by which stave group(s) it touches:
   - *Single-part*: touches only one stave group → assigned fully to that part (full extent retained).
   - *Shared*: touches multiple stave groups → clipped at the midpoint between the adjacent stave groups, so each part receives only its portion.
   - *Satellite*: touches no stave (slurs, dynamics, text) → assigned to the part whose connected-content bounding box is nearest in 2-D Euclidean distance, and included as its full connected component.

4. **Bar-line trimming** — bar-line pixels are added back to each part's mask, but only within that part's stave y-span (±2 px buffer). This removes the bar lines above the top stave and below the bottom stave of each part, matching the appearance of conventionally extracted parts.

5. **Antialiasing** — the mask is dilated by 1 pixel and intersected with non-white pixels to include antialiased ink edges without pulling in background noise.

### Step 3 — A4 assembly

For each part, the sub-page image is rendered at the analysis DPI, the per-part mask is applied (non-part pixels set to white), and the result is cropped to a tight bounding box. Because satellites may extend above or below the main stave body, the crop is a rectangle that includes the satellite content with white space around it — so each section has flat rectangular edges. Sections are then scaled to fill the full printable width of A4 (aspect ratio preserved) and stacked with a configurable gap, with page breaks inserted as needed.

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

| Flag | Default | Description |
|---|---|---|
| `-n` / `--parts N` | *(required)* | Number of instrument parts per system |
| `-p` / `--systems P` | `1` | Systems stacked per source page (triggers Step 1 splitting) |
| `-g` / `--gap FRAC` | `1.0` | Gap between sections as a fraction of the preceding section's height |
| `-m` / `--margin MM` | `15` | Printer-safe margin in mm on all four sides of each A4 page |
| `-o` / `--output-dir DIR` | `<stem>_parts/` | Output directory |
| `--dpi DPI` | `150` | Rendering resolution — controls both analysis accuracy and output image quality |

---

## Examples

```bash
# 4-part score, one system per page
python score_extractor.py quartet.pdf --parts 4

# 4-part score, two systems per page, compact gap, higher quality
python score_extractor.py quartet.pdf --parts 4 --systems 2 --gap 0.3 --dpi 200

# Write to a specific directory
python score_extractor.py quartet.pdf --parts 4 --systems 2 -o ~/Desktop/parts/
```

Output files are named `<input_stem>_part<k>_of_<N>.pdf`.
