#!/usr/bin/env bash
# Downloads the fonts used by the experiments into assets/fonts (not committed: large binaries).
set -euo pipefail
cd "$(dirname "$0")/../assets/fonts"
get() { [ -s "$2" ] || curl -sSfL -o "$2" "$1"; echo "ok $2"; }
get https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf NotoSansCJKsc-Regular.otf
get "https://raw.githubusercontent.com/google/fonts/main/ofl/robotocondensed/RobotoCondensed%5Bwght%5D.ttf" RobotoCondensed.ttf
get "https://raw.githubusercontent.com/google/fonts/main/ofl/archivonarrow/ArchivoNarrow%5Bwght%5D.ttf" ArchivoNarrow.ttf
