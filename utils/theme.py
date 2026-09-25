"""
Visual identity for the GenAI Music Recommender.

The app's one real differentiator is that it plots tracks on the
valence/energy plane and reads off a mood quadrant instead of a release
year (this dataset has no release-year column — see the README). So the
design leans on that plane as its signature visual: the same four mood
colors defined here are reused for the quadrant diagram, the
recommendation cards, the badges, and the mood chart — nowhere else — so
the color-coding reads as a system, not decoration.

Colors are chosen to match what each mood *feels* like, not just the
valence/energy math, so they're readable at a glance without needing the
legend: gold for upbeat energy, green for calm content, red for
anger/intensity, blue for sadness/mellowness.
"""

MOOD_COLORS = {
    "Happy & Energetic": "#FFC94D",  # high valence, high energy — warm gold
    "Chill & Content": "#4FD1A5",    # high valence, low energy — calm green
    "Angry & Intense": "#FF5C5C",    # low valence, high energy — red
    "Sad & Mellow": "#5A8DEE",       # low valence, low energy — blue
}
MOOD_FALLBACK = "#8C96AD"

INK = "#10131C"
SURFACE = "#161B29"
SURFACE_RAISED = "#1E2536"
BORDER = "#293349"
TEXT = "#E7EBF5"
TEXT_MUTED = "#8C96AD"
ACCENT = MOOD_COLORS["Happy & Energetic"]


def inject_css() -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');

:root {{
  --ink: {INK};
  --surface: {SURFACE};
  --surface-raised: {SURFACE_RAISED};
  --border: {BORDER};
  --text: {TEXT};
  --text-muted: {TEXT_MUTED};
  --accent: {ACCENT};
}}

html, body, [class*="css"], .stApp, p, span, div, label {{
  font-family: 'Inter', -apple-system, sans-serif;
  font-size: 0.92rem;
}}

h1, h2, h3, .app-hero__title {{
  font-family: 'Fraunces', Georgia, serif !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em !important;
}}

/* Compact layout: trim the generous default padding/margins so the app
   fits a normal screen without excess scrolling. */
.block-container {{
  padding-top: 1.6rem !important;
  padding-bottom: 1.5rem !important;
  max-width: 1100px !important;
}}
h2 {{ font-size: 1.15rem !important; margin-top: 0.4rem !important; margin-bottom: 0.4rem !important; }}
h3 {{ font-size: 1rem !important; margin-top: 0.3rem !important; margin-bottom: 0.3rem !important; }}
div[data-testid="stVerticalBlock"] {{ gap: 0.5rem; }}
div[data-testid="stCaptionContainer"] p {{ font-size: 0.8rem !important; }}

.stApp {{
  background: var(--ink) !important;
}}

section[data-testid="stSidebar"] {{
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}}

.app-hero {{
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 0.1rem;
}}
.app-hero__title {{
  font-size: 1.5rem;
  color: var(--text);
  margin: 0;
  line-height: 1.1;
}}

/* Recommendation cards: colored left stripe set per-card via a
   generated rule (see mood_stripe_css), plus a shared, compact base look. */
div[data-testid="stVerticalBlockBorderWrapper"]:has(div.rec-card-marker) {{
  background: var(--surface);
  border: 1px solid var(--border) !important;
  border-left-width: 3px !important;
  border-radius: 8px !important;
  padding: 0.15rem 0 !important;
}}

span.badge {{
  display: inline-block;
  padding: 0.05rem 0.5rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 500;
  margin-right: 0.3rem;
  border: 1px solid transparent;
}}
span.badge--mood {{
  color: var(--ink);
}}
span.badge--genre {{
  background: var(--surface-raised);
  color: var(--text-muted);
  border-color: var(--border);
}}

.quadrant-caption {{
  color: var(--text-muted);
  font-size: 0.76rem;
  margin-top: 0.2rem;
  line-height: 1.3;
}}
</style>
"""


def mood_stripe_css(index: int, mood: str) -> str:
    """CSS rule giving recommendation card `index` a left border in its
    mood's color. Relies on Streamlit's `key=` param putting an
    `st-key-<key>` class on the container's wrapper div."""
    color = MOOD_COLORS.get(mood, MOOD_FALLBACK)
    return (
        f'<style>[class*="st-key-rec-card-{index}"] '
        f'div[data-testid="stVerticalBlockBorderWrapper"] '
        f'{{ border-left-color: {color} !important; }}</style>'
    )


def mood_badge(mood: str) -> str:
    color = MOOD_COLORS.get(mood, MOOD_FALLBACK)
    return f'<span class="badge badge--mood" style="background:{color}">{mood}</span>'


def genre_badge(genre: str) -> str:
    return f'<span class="badge badge--genre">{genre}</span>'


def quadrant_diagram(selected_moods=None, size: int = 190, labels: bool = True) -> str:
    """An SVG of the valence/energy plane with the four mood quadrants,
    highlighting any currently-selected moods. This is the app's one
    drawn glyph — it explains the mechanism instead of decorating the
    page. With `labels=False` it's just the colored 2x2 mark (used as a
    small logo next to the title)."""
    selected_moods = set(selected_moods or [])
    s = size
    half = s / 2
    quads = [
        ("Sad & Mellow", 0, half, half, half),
        ("Angry & Intense", half, half, half, half),
        ("Chill & Content", 0, 0, half, half),
        ("Happy & Energetic", half, 0, half, half),
    ]
    rects = []
    for mood, x, y, w, h in quads:
        color = MOOD_COLORS[mood]
        selected = mood in selected_moods
        opacity = "0.95" if (not selected_moods or selected) else "0.35"
        stroke = f'stroke="{TEXT}" stroke-width="2"' if selected else 'stroke="none"'
        rects.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" '
                      f'fill-opacity="{opacity}" {stroke} />')
    rects_svg = "".join(rects)
    lines_svg = (
        f'<line x1="0" y1="{half}" x2="{s}" y2="{half}" stroke="{INK}" stroke-width="2" />'
        f'<line x1="{half}" y1="0" x2="{half}" y2="{s}" stroke="{INK}" stroke-width="2" />'
    )
    if not labels:
        return (
            f'<svg width="{s}" height="{s}" viewBox="0 0 {s} {s}" '
            f'xmlns="http://www.w3.org/2000/svg" style="border-radius:6px;flex-shrink:0;">'
            f'<g>{rects_svg}</g>{lines_svg}</svg>'
        )
    return f"""
<svg width="{s}" height="{s+34}" viewBox="0 0 {s} {s+34}" xmlns="http://www.w3.org/2000/svg">
  <g>{rects_svg}</g>
  {lines_svg}
  <text x="{s/2}" y="{s+16}" text-anchor="middle" fill="{TEXT_MUTED}" font-size="11" font-family="Inter">valence \u2192</text>
  <text x="6" y="12" fill="{TEXT_MUTED}" font-size="11" font-family="Inter" writing-mode="vertical-rl">energy \u2191</text>
</svg>
"""