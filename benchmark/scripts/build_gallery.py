#!/usr/bin/env python
"""Build an HTML gallery for timelapse segmentation visualizations.

Scans the ``benchmark/outputs/visuals/`` directory tree for PNG figures
produced by ``visualize_timelapse.py`` and generates a standalone HTML
page with:

  - Navigation sidebar grouped by dataset and FOV group
  - Display-mode toggle (overlay vs. grayscale)
  - Figure-type sections (comparison grid, contour overlay, temporal montage)
  - Click-to-enlarge lightbox
  - Summary statistics table

The generated HTML uses relative paths to PNGs so it must live inside
(or be opened from) the visuals directory.

Usage
  python build_gallery.py                          # default output
  python build_gallery.py -o /tmp/gallery.html     # custom output path
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
VISUALS_DIR = BENCHMARK_DIR / "outputs" / "visuals"

# ---------------------------------------------------------------------------
# FOV labels (same as visualize_timelapse.py)
# ---------------------------------------------------------------------------

FOV_LABELS: dict[str, str] = {
    "20171107_C03_001_Airyscan_Processing_scene01": "Airyscan 40x",
    "20180315_N01_0011_scene00": "ZSD 20x",
    "20180227_N01_0011_scene00": "ZSD 100x",
    "20170424_C01_001_Airyscan_Processing_scene01": "Airyscan",
    "20170718_C01_001_Airyscan_Processing_scene01": "CZI 1 Scene 1",
    "20171031_C01_001_Airyscan_Processing_scene01": "CZI 2 Scene 1",
}

DATASET_ORDER = ["NPM1", "Mitochondria", "Tubulin_MultiCh"]
FIGURE_TYPE_LABELS = {
    "comparison": "Comparison Grid",
    "overlay": "Contour Overlay",
    "temporal": "Temporal Montage",
}

DATASET_METHODS: dict[str, list[str]] = {
    "NPM1": ["Classic", "Individual", "MegaSeg"],
    "Mitochondria": ["Classic", "Individual", "MegaSeg"],
    "Tubulin_MultiCh": ["Classic", "MegaSeg"],
}

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Comparison: {fov_stem}_T{NNNN}_{mode}.png  where mode = overlay|gray
_COMPARISON_RE = re.compile(r"^(.+)_T(\d+)_(overlay|gray)\.png$")

# Overlay (contour): {fov_stem}_T{NNNN}.png  (no mode suffix)
_OVERLAY_RE = re.compile(r"^(.+)_T(\d+)\.png$")

# Temporal: {fov_stem}_(overlay|gray).png  (no timepoint)
_TEMPORAL_RE = re.compile(r"^(.+)_(overlay|gray)\.png$")


def parse_comparison(name: str) -> dict | None:
    m = _COMPARISON_RE.match(name)
    if m is None:
        return None
    return {"fov_stem": m.group(1), "timepoint": int(m.group(2)), "mode": m.group(3)}


def parse_overlay(name: str) -> dict | None:
    m = _OVERLAY_RE.match(name)
    if m is None:
        return None
    return {"fov_stem": m.group(1), "timepoint": int(m.group(2))}


def parse_temporal(name: str) -> dict | None:
    m = _TEMPORAL_RE.match(name)
    if m is None:
        return None
    return {"fov_stem": m.group(1), "mode": m.group(2)}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_figures(
    visuals_dir: Path,
) -> dict[str, dict[str, dict[str, list]]]:
    """Scan visuals directory and organize figures hierarchically.

    Returns
    -------
    {dataset: {fov_stem: {figure_type: [entry, ...]}}}

    Each entry is a dict with keys depending on figure_type:
      comparison: {path, timepoint, mode}
      overlay:    {path, timepoint}
      temporal:   {path, mode}
    """
    result: dict[str, dict[str, dict[str, list]]] = {}

    for fig_type in ("comparison", "overlay", "temporal"):
        fig_dir = visuals_dir / fig_type
        if not fig_dir.is_dir():
            continue

        for ds_dir in sorted(fig_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            dataset = ds_dir.name

            for png in sorted(ds_dir.glob("*.png")):
                name = png.name

                if fig_type == "comparison":
                    parsed = parse_comparison(name)
                elif fig_type == "overlay":
                    parsed = parse_overlay(name)
                else:
                    parsed = parse_temporal(name)

                if parsed is None:
                    continue

                fov_stem = parsed.pop("fov_stem")
                parsed["path"] = str(png.relative_to(visuals_dir))

                result.setdefault(dataset, {}).setdefault(fov_stem, {}).setdefault(
                    fig_type, []
                ).append(parsed)

    return result


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def _fov_label(dataset: str, fov_stem: str) -> str:
    """Human-readable label for a FOV group."""
    return FOV_LABELS.get(fov_stem, fov_stem)


def _slug(text: str) -> str:
    """Convert text to a URL-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def build_html(figures: dict, visuals_dir: Path) -> str:
    """Render the full gallery HTML."""

    # Compute stats
    total_pngs = 0
    dataset_counts: dict[str, int] = {}
    for ds in figures:
        count = 0
        for fov in figures[ds]:
            for fig_type in figures[ds][fov]:
                n = len(figures[ds][fov][fig_type])
                count += n
                total_pngs += n
        dataset_counts[ds] = count

    # Build sidebar nav and content sections
    nav_items = []
    content_sections = []

    for dataset in DATASET_ORDER:
        if dataset not in figures:
            continue

        ds_slug = _slug(dataset)
        fov_stems = sorted(figures[dataset].keys())

        # Dataset nav group
        nav_items.append(f'<div class="nav-dataset">{dataset}</div>')

        for fov_stem in fov_stems:
            fov_label = _fov_label(dataset, fov_stem)
            section_id = f"{ds_slug}--{_slug(fov_stem)}"
            nav_items.append(f'<a class="nav-fov" href="#{section_id}">{fov_label}</a>')

            fov_data = figures[dataset][fov_stem]
            methods_str = ", ".join(DATASET_METHODS.get(dataset, []))

            # Section header
            section_html = [
                f'<section id="{section_id}" class="fov-section">',
                f"<h2>{dataset} - {fov_label}</h2>",
                f'<p class="methods-tag">Methods: {methods_str}</p>',
            ]

            # -- Comparison grids --
            if "comparison" in fov_data:
                entries = fov_data["comparison"]
                # Group by timepoint
                by_tp: dict[int, dict[str, str]] = defaultdict(dict)
                for e in entries:
                    by_tp[e["timepoint"]][e["mode"]] = e["path"]

                section_html.append('<div class="fig-group">')
                section_html.append("<h3>Comparison Grid</h3>")
                section_html.append(
                    '<div class="mode-toggle">'
                    '<button class="mode-btn active" data-mode="overlay">'
                    "Overlay</button>"
                    '<button class="mode-btn" data-mode="gray">'
                    "Grayscale</button></div>"
                )
                section_html.append('<div class="image-grid">')
                for tp in sorted(by_tp.keys()):
                    modes = by_tp[tp]
                    overlay_path = modes.get("overlay", "")
                    gray_path = modes.get("gray", "")
                    section_html.append(
                        f'<div class="image-card" '
                        f'data-overlay="{overlay_path}" '
                        f'data-gray="{gray_path}">'
                        f'<img src="{overlay_path}" alt="T{tp:04d}" loading="lazy">'
                        f'<div class="image-label">T={tp}</div>'
                        f"</div>"
                    )
                section_html.append("</div>")  # image-grid
                section_html.append("</div>")  # fig-group

            # -- Contour overlays --
            if "overlay" in fov_data:
                entries = sorted(fov_data["overlay"], key=lambda e: e["timepoint"])
                section_html.append('<div class="fig-group">')
                section_html.append("<h3>Contour Overlay</h3>")
                section_html.append('<div class="image-grid">')
                for e in entries:
                    section_html.append(
                        f'<div class="image-card">'
                        f'<img src="{e["path"]}" alt="T{e["timepoint"]:04d}" '
                        f'loading="lazy">'
                        f'<div class="image-label">T={e["timepoint"]}</div>'
                        f"</div>"
                    )
                section_html.append("</div>")  # image-grid
                section_html.append("</div>")  # fig-group

            # -- Temporal montages --
            if "temporal" in fov_data:
                entries = fov_data["temporal"]
                modes_map: dict[str, str] = {}
                for e in entries:
                    modes_map[e["mode"]] = e["path"]

                section_html.append('<div class="fig-group">')
                section_html.append("<h3>Temporal Montage</h3>")
                section_html.append(
                    '<div class="mode-toggle">'
                    '<button class="mode-btn active" data-mode="overlay">'
                    "Overlay</button>"
                    '<button class="mode-btn" data-mode="gray">'
                    "Grayscale</button></div>"
                )
                overlay_path = modes_map.get("overlay", "")
                gray_path = modes_map.get("gray", "")
                section_html.append(
                    f'<div class="temporal-card" '
                    f'data-overlay="{overlay_path}" '
                    f'data-gray="{gray_path}">'
                    f'<img src="{overlay_path}" alt="Temporal montage" '
                    f'loading="lazy">'
                    f"</div>"
                )
                section_html.append("</div>")  # fig-group

            section_html.append("</section>")
            content_sections.append("\n".join(section_html))

    nav_html = "\n".join(nav_items)
    body_html = "\n\n".join(content_sections)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Summary table rows
    summary_rows = []
    for ds in DATASET_ORDER:
        if ds not in figures:
            continue
        n_fov = len(figures[ds])
        n_figs = dataset_counts[ds]
        methods = ", ".join(DATASET_METHODS.get(ds, []))
        summary_rows.append(
            f"<tr><td>{ds}</td><td>{n_fov}</td><td>{methods}</td><td>{n_figs}</td></tr>"
        )

    summary_table = "\n".join(summary_rows)

    return dedent(f"""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Timelapse Segmentation Gallery - MegaSeg</title>
    <style>
    :root {{
      --bg: #0d1117;
      --surface: #161b22;
      --surface2: #21262d;
      --border: #30363d;
      --text: #c9d1d9;
      --text-muted: #8b949e;
      --accent: #58a6ff;
      --accent-hover: #79c0ff;
      --orange: #ff6b35;
      --green: #2ecc71;
      --blue: #3498db;
      --sidebar-w: 240px;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial,
                   sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      min-height: 100vh;
    }}

    /* Sidebar */
    .sidebar {{
      position: fixed;
      top: 0; left: 0;
      width: var(--sidebar-w);
      height: 100vh;
      overflow-y: auto;
      background: var(--surface);
      border-right: 1px solid var(--border);
      padding: 1rem 0;
      z-index: 10;
    }}
    .sidebar h1 {{
      font-size: 1rem;
      padding: 0 1rem 0.75rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 0.5rem;
      color: var(--accent);
    }}
    .nav-dataset {{
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      padding: 0.75rem 1rem 0.25rem;
    }}
    .nav-fov {{
      display: block;
      padding: 0.35rem 1rem 0.35rem 1.5rem;
      color: var(--text);
      text-decoration: none;
      font-size: 0.85rem;
      border-left: 2px solid transparent;
      transition: all 0.15s;
    }}
    .nav-fov:hover {{
      background: var(--surface2);
      color: var(--accent);
      border-left-color: var(--accent);
    }}

    /* Main content */
    .main {{
      margin-left: var(--sidebar-w);
      flex: 1;
      padding: 2rem;
      max-width: calc(100vw - var(--sidebar-w));
    }}
    .main > h1 {{
      font-size: 1.5rem;
      margin-bottom: 0.25rem;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 0.85rem;
      margin-bottom: 1.5rem;
    }}

    /* Summary table */
    .summary-table {{
      width: 100%;
      max-width: 700px;
      border-collapse: collapse;
      margin-bottom: 2rem;
      font-size: 0.85rem;
    }}
    .summary-table th {{
      text-align: left;
      padding: 0.5rem 0.75rem;
      background: var(--surface2);
      border-bottom: 2px solid var(--border);
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.75rem;
      letter-spacing: 0.04em;
    }}
    .summary-table td {{
      padding: 0.5rem 0.75rem;
      border-bottom: 1px solid var(--border);
    }}

    /* FOV section */
    .fov-section {{
      margin-bottom: 3rem;
      scroll-margin-top: 1rem;
    }}
    .fov-section h2 {{
      font-size: 1.2rem;
      margin-bottom: 0.25rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid var(--border);
    }}
    .methods-tag {{
      color: var(--text-muted);
      font-size: 0.8rem;
      margin-bottom: 1rem;
    }}

    /* Figure group */
    .fig-group {{
      margin-bottom: 2rem;
    }}
    .fig-group h3 {{
      font-size: 1rem;
      color: var(--text-muted);
      margin-bottom: 0.5rem;
    }}

    /* Mode toggle */
    .mode-toggle {{
      display: inline-flex;
      gap: 0;
      margin-bottom: 0.75rem;
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
    }}
    .mode-btn {{
      background: var(--surface);
      color: var(--text-muted);
      border: none;
      padding: 0.35rem 0.85rem;
      font-size: 0.8rem;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .mode-btn:not(:last-child) {{
      border-right: 1px solid var(--border);
    }}
    .mode-btn.active {{
      background: var(--accent);
      color: #fff;
    }}
    .mode-btn:hover:not(.active) {{
      background: var(--surface2);
      color: var(--text);
    }}

    /* Image grid */
    .image-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 0.75rem;
    }}
    .image-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
      cursor: pointer;
      transition: border-color 0.15s;
    }}
    .image-card:hover {{
      border-color: var(--accent);
    }}
    .image-card img {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .image-label {{
      padding: 0.35rem 0.5rem;
      font-size: 0.8rem;
      color: var(--text-muted);
      text-align: center;
      background: var(--surface2);
    }}

    /* Temporal card (full width) */
    .temporal-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
      cursor: pointer;
      transition: border-color 0.15s;
    }}
    .temporal-card:hover {{
      border-color: var(--accent);
    }}
    .temporal-card img {{
      width: 100%;
      height: auto;
      display: block;
    }}

    /* Lightbox */
    .lightbox {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.92);
      z-index: 100;
      justify-content: center;
      align-items: center;
      cursor: zoom-out;
    }}
    .lightbox.open {{
      display: flex;
    }}
    .lightbox img {{
      max-width: 95vw;
      max-height: 95vh;
      object-fit: contain;
      border-radius: 4px;
    }}

    /* Color legend */
    .color-legend {{
      display: flex;
      gap: 1.25rem;
      margin-bottom: 1.5rem;
      font-size: 0.85rem;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}
    .legend-swatch {{
      width: 14px;
      height: 14px;
      border-radius: 3px;
      display: inline-block;
    }}

    /* Responsive */
    @media (max-width: 768px) {{
      .sidebar {{ display: none; }}
      .main {{ margin-left: 0; padding: 1rem; }}
      .image-grid {{ grid-template-columns: 1fr; }}
    }}
    </style>
    </head>
    <body>
    <nav class="sidebar">
      <h1>Timelapse Gallery</h1>
      {nav_html}
    </nav>

    <div class="main">
      <h1>Timelapse Segmentation Gallery</h1>
      <p class="subtitle">
        Generated {timestamp} - {total_pngs} figures across
        {sum(len(figures[d]) for d in figures)} FOV groups
      </p>

      <div class="color-legend">
        <div class="legend-item">
          <span class="legend-swatch" style="background:var(--orange)"></span>
          Classic
        </div>
        <div class="legend-item">
          <span class="legend-swatch" style="background:var(--blue)"></span>
          Individual
        </div>
        <div class="legend-item">
          <span class="legend-swatch" style="background:var(--green)"></span>
          MegaSeg
        </div>
      </div>

      <table class="summary-table">
        <thead>
          <tr>
            <th>Dataset</th>
            <th>FOV Groups</th>
            <th>Methods</th>
            <th>Figures</th>
          </tr>
        </thead>
        <tbody>
          {summary_table}
        </tbody>
      </table>

      {body_html}
    </div>

    <div class="lightbox" id="lightbox">
      <img id="lightbox-img" src="" alt="Enlarged view">
    </div>

    <script>
    // Display mode toggling
    document.querySelectorAll('.mode-toggle').forEach(toggle => {{
      toggle.querySelectorAll('.mode-btn').forEach(btn => {{
        btn.addEventListener('click', () => {{
          const mode = btn.dataset.mode;
          // Update active button
          toggle.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          // Update images in the parent fig-group
          const group = toggle.closest('.fig-group');
          // Handle image-card elements (comparison grids)
          group.querySelectorAll('.image-card[data-' + mode + ']').forEach(card => {{
            const src = card.getAttribute('data-' + mode);
            if (src) card.querySelector('img').src = src;
          }});
          // Handle temporal-card elements
          group.querySelectorAll('.temporal-card[data-' + mode + ']').forEach(card => {{
            const src = card.getAttribute('data-' + mode);
            if (src) card.querySelector('img').src = src;
          }});
        }});
      }});
    }});

    // Lightbox
    const lb = document.getElementById('lightbox');
    const lbImg = document.getElementById('lightbox-img');

    document.querySelectorAll('.image-card, .temporal-card').forEach(card => {{
      card.addEventListener('click', () => {{
        const src = card.querySelector('img').src;
        lbImg.src = src;
        lb.classList.add('open');
      }});
    }});
    lb.addEventListener('click', () => {{
      lb.classList.remove('open');
      lbImg.src = '';
    }});
    document.addEventListener('keydown', e => {{
      if (e.key === 'Escape' && lb.classList.contains('open')) {{
        lb.classList.remove('open');
        lbImg.src = '';
      }}
    }});

    // Highlight active nav on scroll
    const sections = document.querySelectorAll('.fov-section');
    const navLinks = document.querySelectorAll('.nav-fov');
    const observer = new IntersectionObserver(entries => {{
      entries.forEach(entry => {{
        if (entry.isIntersecting) {{
          const id = entry.target.id;
          navLinks.forEach(link => {{
            link.style.borderLeftColor =
              link.getAttribute('href') === '#' + id
                ? 'var(--accent)' : 'transparent';
            link.style.color =
              link.getAttribute('href') === '#' + id
                ? 'var(--accent)' : '';
          }});
        }}
      }});
    }}, {{ threshold: 0.2 }});
    sections.forEach(s => observer.observe(s));
    </script>
    </body>
    </html>
    """)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build HTML gallery for timelapse visualizations."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=("Output HTML path. Defaults to benchmark/outputs/visuals/gallery.html"),
    )
    parser.add_argument(
        "--visuals-dir",
        type=Path,
        default=None,
        help="Visuals directory to scan. Defaults to benchmark/outputs/visuals/",
    )
    args = parser.parse_args()

    visuals_dir = args.visuals_dir or VISUALS_DIR
    output_path = args.output or (visuals_dir / "gallery.html")

    if not visuals_dir.is_dir():
        raise SystemExit(f"Visuals directory not found: {visuals_dir}")

    figures = discover_figures(visuals_dir)
    if not figures:
        raise SystemExit("No PNG figures found. Run visualize_timelapse.py first.")

    html = build_html(figures, visuals_dir)
    output_path.write_text(html, encoding="utf-8")

    # Print summary
    total = sum(
        len(entries)
        for ds in figures.values()
        for fov in ds.values()
        for entries in fov.values()
    )
    print(f"Gallery written to {output_path}")
    print(f"  {total} figures across {len(figures)} datasets")


if __name__ == "__main__":
    main()
