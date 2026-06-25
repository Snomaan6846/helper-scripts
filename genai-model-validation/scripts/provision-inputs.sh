#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INPUTS_DIR="$BASE_DIR/inputs"

SCENERY_URL="https://upload.wikimedia.org/wikipedia/commons/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
AURORA_URL="https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg"
CAT_URL="https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg"

AUDIO_TEXT="The quick brown fox jumps over the lazy dog"

ALL_FILES=(
    "test-scenery.jpg"
    "test-scenery-4k.jpg"
    "test-image-2.jpg"
    "test-audio-en.wav"
    "test-audio-en.mp3"
    "test-audio-en.flac"
)

if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning provisioned inputs from $INPUTS_DIR ..."
    for f in "${ALL_FILES[@]}"; do
        if [[ -f "$INPUTS_DIR/$f" ]]; then
            rm -f "$INPUTS_DIR/$f"
            echo "  removed $f"
        fi
    done
    rm -f "$INPUTS_DIR"/*.tmp
    echo "Done."
    exit 0
fi

declare -i created=0
declare -i skipped=0
declare -i failed=0
failures=()

mkdir -p "$INPUTS_DIR"

download_and_resize() {
    local url="$1" dest="$2" width="$3" height="$4"
    if [[ -f "$dest" && -s "$dest" ]]; then
        echo "SKIP  $dest (already exists)"
        skipped+=1
        return 0
    fi
    local tmp="${dest}.tmp"
    echo "FETCH $url"
    if ! curl -fSL --retry 3 --retry-delay 2 -o "$tmp" "$url"; then
        echo "FAIL  could not download $url"
        failures+=("$dest")
        failed+=1
        return 1
    fi
    echo "RESIZE $tmp -> ${width}x${height}"
    if ! python3 -c "
from PIL import Image
im = Image.open('$tmp')
im = im.resize(($width, $height), Image.LANCZOS)
im.save('$dest', 'JPEG', quality=90)
"; then
        echo "FAIL  could not resize $tmp"
        rm -f "$tmp"
        failures+=("$dest")
        failed+=1
        return 1
    fi
    rm -f "$tmp"
    echo "OK    $dest"
    created+=1
}

download_raw() {
    local url="$1" dest="$2" min_bytes="$3"
    if [[ -f "$dest" && -s "$dest" ]]; then
        echo "SKIP  $dest (already exists)"
        skipped+=1
        return 0
    fi
    echo "FETCH $url"
    if ! curl -fSL --retry 3 --retry-delay 2 -o "$dest" "$url"; then
        echo "FAIL  could not download $url"
        failures+=("$dest")
        failed+=1
        return 1
    fi
    local size
    size=$(stat --printf='%s' "$dest" 2>/dev/null || stat -f '%z' "$dest" 2>/dev/null)
    if (( size < min_bytes )); then
        echo "FAIL  $dest is ${size} bytes (need >=${min_bytes})"
        rm -f "$dest"
        failures+=("$dest")
        failed+=1
        return 1
    fi
    echo "OK    $dest (${size} bytes)"
    created+=1
}

generate_audio() {
    local dest="$1"
    if [[ -f "$dest" && -s "$dest" ]]; then
        echo "SKIP  $dest (already exists)"
        skipped+=1
        return 0
    fi
    echo "GEN   $dest via espeak-ng"
    if ! espeak-ng -v en -s 130 -w "$dest" "$AUDIO_TEXT"; then
        echo "FAIL  espeak-ng failed"
        failures+=("$dest")
        failed+=1
        return 1
    fi
    echo "OK    $dest"
    created+=1
}

transcode() {
    local src="$1" dest="$2"
    shift 2
    local ffargs=("$@")
    if [[ -f "$dest" && -s "$dest" ]]; then
        echo "SKIP  $dest (already exists)"
        skipped+=1
        return 0
    fi
    echo "XCODE $src -> $dest"
    if ! ffmpeg -y -i "$src" "${ffargs[@]}" "$dest" 2>/dev/null; then
        echo "FAIL  ffmpeg transcode failed for $dest"
        failures+=("$dest")
        failed+=1
        return 1
    fi
    echo "OK    $dest"
    created+=1
}

verify_nonzero() {
    local f="$1"
    if [[ ! -s "$f" ]]; then
        echo "FAIL  $f is missing or zero-length"
        failures+=("$f")
        failed+=1
        return 1
    fi
    return 0
}

echo "=== Provisioning test inputs into $INPUTS_DIR ==="
echo

download_and_resize "$SCENERY_URL"  "$INPUTS_DIR/test-scenery.jpg"    1280 720 || true
download_raw        "$AURORA_URL"   "$INPUTS_DIR/test-scenery-4k.jpg" 2097152  || true
download_and_resize "$CAT_URL"      "$INPUTS_DIR/test-image-2.jpg"    1280 720 || true

generate_audio "$INPUTS_DIR/test-audio-en.wav" || true

if [[ -f "$INPUTS_DIR/test-audio-en.wav" && -s "$INPUTS_DIR/test-audio-en.wav" ]]; then
    transcode "$INPUTS_DIR/test-audio-en.wav" "$INPUTS_DIR/test-audio-en.mp3"  -b:a 128k || true
    transcode "$INPUTS_DIR/test-audio-en.wav" "$INPUTS_DIR/test-audio-en.flac" || true
fi

echo
echo "=== Verifying files ==="
for f in "${ALL_FILES[@]}"; do
    verify_nonzero "$INPUTS_DIR/$f" || true
done

echo
echo "=== Summary ==="
echo "  Created : $created"
echo "  Skipped : $skipped"
echo "  Failed  : $failed"

if (( failed > 0 )); then
    echo
    echo "Failed files:"
    for f in "${failures[@]}"; do
        echo "  - $f"
    done
    exit 1
fi

echo
echo "All inputs provisioned successfully."
