"""Shared paths and constants. Every module should import paths from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "assets" / "fonts"
DATA = ROOT / "data"          # downloaded / generated corpora (git-ignored)
RESULTS = ROOT / "results"    # experiment outputs (images + caches git-ignored)

DEJAVU = Path("/usr/share/fonts/truetype/dejavu")
if not DEJAVU.exists():   # no system DejaVu (e.g. Windows): DejaVu 2.37 TTFs copied into assets/fonts
    DEJAVU = FONTS
LATIN_FONT = DEJAVU / "DejaVuSans.ttf"              # default Latin font (0.51 em/char)
LATIN_MONO_FONT = DEJAVU / "DejaVuSansMono.ttf"     # 0.60 em/char
CONDENSED_FONT = FONTS / "ArchivoNarrow.ttf"        # 0.37 em/char  (scripts/fetch_fonts.sh)
CONDENSED_FONT_2 = FONTS / "RobotoCondensed.ttf"    # 0.40 em/char
CJK_FONT = FONTS / "NotoSansCJKsc-Regular.otf"      # CJK ideographs are 1.0 em wide; has Latin too

# Claude vision cost model. Measured on this runner (docs/claude/claude-eval-notes.md):
# tokens = ceil(W/28) * ceil(H/28) + ~3  (~784 px^2/token). Use canvas sides that are multiples of 28.
# Nothing is downscaled up to 1568x1568; the limits below are conservative.
CLAUDE_PATCH = 28
CLAUDE_PX_PER_TOKEN = 784
CLAUDE_MAX_EDGE = 1568
CLAUDE_MAX_AREA = 1_150_000
# High-resolution tier (Claude 4.7 and later, incl. Opus 5.5; vision docs 2026-11): images up to a 2576-px long edge
# and 4784 visual tokens (ceil(W/28) x ceil(H/28)) are not downscaled. Opt in per call (claude_eval.transcribe(hires=True)).
CLAUDE_HIRES_MAX_EDGE = 2576
CLAUDE_HIRES_MAX_TOKENS = 4784
