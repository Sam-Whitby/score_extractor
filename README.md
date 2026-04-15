# score_extractor

Extracts individual instrument parts from multi-part PDF scores (e.g. IMSLP quartet scores) into separate, readable A4 PDFs — the same goal as [Partifi](https://partifi.com).

---

## How it works

### Step 1 — System splitting (optional, `--systems P`)

If each page contains multiple stave systems stacked vertically (e.g. two complete quartet systems per page), each page is split into P sub-pages. The split point is found by rendering the page as a greyscale image and locating the largest continuous white band within a search window centred on the expected geometric split. Whitespace is trimmed from all four sides of each sub-page.

Left/right trimming uses a two-stage approach: a lenient column-whiteness test (≥98% white pixels) finds the coarse cut inside each outer third of the page, then a strict all-white check fine-trims within those bounds. This tolerates the 1-pixel grey borders common in scanned PDFs.

### Step 2 — Stave-connectivity part detection

For each sub-page:

1. **Bar-line detection** — columns where the vertical span of dark pixels covers >30% of the image height *and* the pixel density within that span is >40% are classified as candidate bar lines. Stave-line columns have the same span but far lower density (~0.02), so they are excluded.

2. **Verification against stave positions** — candidates are checked to be dark at every stave line position and at the midpoint of every inter-stave gap. Note stems — which can align across multiple parts but break at stave gaps — are rejected here.

3. **System bracket removal** — the ornate vertical bracket at the far left of each system (connecting all parts) passes the bar-line criteria but is conspicuously wider than real bar lines. It is detected as the leftmost barline group whose width is more than twice the median barline width, then removed: its columns are whited out in the source image so it never appears in any part output.

4. **Stave detection** — after masking bar lines, rows with above-threshold darkness form peaks corresponding to the 5×N stave lines. The N×5 peaks with the highest row-darkness are kept and grouped into N sets of 5 by identifying the N−1 largest consecutive gaps between sorted line positions.

5. **Flood-fill** — connected components (8-connectivity, so diagonal beams are followed) are labelled on the bar-line-masked image. Each component is classified by which stave group(s) it touches:
   - *Single-part*: touches only one stave group → assigned fully to that part (full extent retained).
   - *Shared*: touches multiple stave groups → clipped at the midpoint between adjacent stave groups.
   - *Satellite*: touches no stave (slurs, dynamics, text) → assigned to the part whose connected-content bounding box is nearest in 2-D Euclidean distance, included as its full connected component.

6. **Bar-line trimming** — bar-line pixels are added back to each part's mask only within that part's stave y-span (±2 px buffer). This removes bar lines above and below each part's own staves.

7. **Crossing object preservation** — pixels in bar-line columns that have non-bar-line dark content on both sides (slurs, crescendos, ties that cross a bar line) are identified by bilateral propagation and preserved in the owning part's mask, so crossing objects appear continuous rather than interrupted.

8. **Antialiasing** — the mask is dilated by 1 pixel and intersected with non-white pixels to include antialiased ink edges.

### Step 3 — A4 assembly with equal stave spacing

For each part, all system section images are collected in Phase 1. In Phase 2 they are packed onto A4 pages and positioned with equal whitespace between adjacent staves on each page, producing uniform, professionally-spaced output. Pages are filled greedily (add sections until the next would overflow at minimum 2 pt spacing), then the gap on each page is equalized.

Bar numbers are written in small grey text above each bar line. The closing bar line of each system (which is the same physical bar line as the opening of the next system) is labelled on the next system only, preventing double-counting. The top part (part 1) already has bar numbers printed in the source score at the start of each system, so these are omitted there.

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
| `-m` / `--margin MM` | `15` | Printer-safe margin in mm on all four sides of each A4 page |
| `-o` / `--output-dir DIR` | `<stem>_parts/` | Output directory |
| `--dpi DPI` | `150` | Rendering resolution — controls both analysis accuracy and output image quality |

---

## Examples

```bash
# 4-part score, one system per page
python score_extractor.py quartet.pdf --parts 4

# 4-part score, two systems per page, higher quality
python score_extractor.py quartet.pdf --parts 4 --systems 2 --dpi 200

# Write to a specific directory
python score_extractor.py quartet.pdf --parts 4 --systems 2 -o ~/Desktop/parts/
```

Output files are named `<input_stem>_part<k>_of_<N>.pdf`.
