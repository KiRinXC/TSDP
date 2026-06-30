#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WEIGHT_DIR="${WEIGHT_DIR:-$SCRIPT_DIR}"

download() {
    local name="$1"
    local url="$2"
    local output="$WEIGHT_DIR/$name"
    local partial="$output.part"

    if [[ -f "$output" ]]; then
        printf '[pre-train] exists: %s\n' "$output"
        return 0
    fi

    printf '[pre-train] downloading: %s\n' "$name"
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 5 --retry-delay 5 --connect-timeout 30 \
            --continue-at - --output "$partial" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=5 --timeout=30 --continue --output-document="$partial" "$url"
    else
        printf '[pre-train] error: curl or wget is required\n' >&2
        exit 1
    fi
    mv "$partial" "$output"
}

mkdir -p "$WEIGHT_DIR"

download mobilenet_v2-b0353104.pth https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
download resnet18-5c106cde.pth https://download.pytorch.org/models/resnet18-5c106cde.pth
download resnet50-19c8e357.pth https://download.pytorch.org/models/resnet50-19c8e357.pth
download vgg16_bn-6c64b313.pth https://download.pytorch.org/models/vgg16_bn-6c64b313.pth

printf '[pre-train] done\n'
