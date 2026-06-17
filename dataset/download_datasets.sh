#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${DATASET_DIR:-$SCRIPT_DIR}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$DATASET_DIR/_archives}"
FORCE="${FORCE:-0}"

usage() {
    cat <<'USAGE'
Download the four datasets used by TensorShield into this project's dataset/ dir.

Usage:
  bash dataset/download_datasets.sh [all|cifar10|cifar100|stl10|tiny-imagenet ...]

By default, all four datasets are downloaded and extracted:
  dataset/cifar10/cifar-10-batches-py
  dataset/cifar100/cifar-100-python
  dataset/stl10/stl10_binary
  dataset/tiny-imagenet-200

Environment:
  DATASET_DIR=/path/to/dataset   Override output directory.
  ARCHIVE_DIR=/path/to/archives  Override archive cache directory.
  FORCE=1                        Re-download and re-extract existing files.

Examples:
  bash dataset/download_datasets.sh
  bash dataset/download_datasets.sh cifar10 cifar100
  FORCE=1 bash dataset/download_datasets.sh tiny-imagenet
USAGE
}

log() {
    printf '[download-datasets] %s\n' "$*"
}

die() {
    printf '[download-datasets] error: %s\n' "$*" >&2
    exit 1
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

need_downloader() {
    if command -v curl >/dev/null 2>&1; then
        DOWNLOADER="curl"
    elif command -v wget >/dev/null 2>&1; then
        DOWNLOADER="wget"
    else
        die "required command not found: curl or wget"
    fi
}

md5_ok() {
    local file="$1"
    local expected="$2"

    [[ -f "$file" ]] || return 1
    printf '%s  %s\n' "$expected" "$file" | md5sum -c - >/dev/null 2>&1
}

fetch_url() {
    local url="$1"
    local output="$2"

    if [[ "$DOWNLOADER" == "curl" ]]; then
        curl -L --fail --retry 5 --retry-delay 5 --connect-timeout 30 \
            --continue-at - --output "$output" "$url"
    else
        wget --tries=5 --timeout=30 --continue --output-document="$output" "$url"
    fi
}

download_archive() {
    local name="$1"
    local url="$2"
    local expected_md5="$3"
    local archive="$ARCHIVE_DIR/$name"
    local partial="$archive.part"

    mkdir -p "$ARCHIVE_DIR"

    if [[ "$FORCE" == "1" ]]; then
        rm -f "$archive" "$partial"
    fi

    if md5_ok "$archive" "$expected_md5"; then
        log "archive ok: $archive"
        return 0
    fi

    if [[ -f "$archive" ]]; then
        log "removing archive with unexpected checksum: $archive"
        rm -f "$archive"
    fi

    log "downloading $name"
    fetch_url "$url" "$partial"
    mv "$partial" "$archive"

    if ! md5_ok "$archive" "$expected_md5"; then
        rm -f "$archive"
        die "checksum failed for $name"
    fi

    log "checksum ok: $name"
}

extract_tar_gz() {
    local archive="$1"
    local output_dir="$2"
    local expected_dir="$3"

    need_cmd tar

    if [[ "$FORCE" == "1" && -d "$expected_dir" ]]; then
        rm -rf "$expected_dir"
    fi

    if [[ -d "$expected_dir" ]]; then
        log "dataset ok: $expected_dir"
        return 0
    fi

    mkdir -p "$output_dir"
    log "extracting $(basename "$archive")"
    tar -xzf "$archive" -C "$output_dir"

    [[ -d "$expected_dir" ]] || die "expected directory missing after extraction: $expected_dir"
}

extract_zip() {
    local archive="$1"
    local output_dir="$2"
    local expected_dir="$3"

    need_cmd unzip

    if [[ "$FORCE" == "1" && -d "$expected_dir" ]]; then
        rm -rf "$expected_dir"
    fi

    if [[ -d "$expected_dir" ]]; then
        log "dataset ok: $expected_dir"
        return 0
    fi

    mkdir -p "$output_dir"
    log "extracting $(basename "$archive")"
    unzip -q "$archive" -d "$output_dir"

    [[ -d "$expected_dir" ]] || die "expected directory missing after extraction: $expected_dir"
}

flatten_tiny_imagenet_train() {
    local root="$1"
    local class_dir
    local image_file

    [[ -d "$root/train" ]] || die "Tiny-ImageNet train directory missing: $root/train"

    for class_dir in "$root"/train/*; do
        [[ -d "$class_dir/images" ]] || continue
        while IFS= read -r -d '' image_file; do
            mv -n "$image_file" "$class_dir/"
        done < <(find "$class_dir/images" -maxdepth 1 -type f -print0)
        rmdir "$class_dir/images" 2>/dev/null || true
    done
}

prepare_tiny_imagenet_val() {
    local root="$1"
    local val_dir="$root/val"
    local images_dir="$val_dir/images"
    local annotations="$val_dir/val_annotations.txt"
    local image_name
    local label
    local remainder
    local label_dir

    [[ -f "$annotations" ]] || die "Tiny-ImageNet val annotations missing: $annotations"

    if [[ -d "$images_dir" ]]; then
        log "rearranging Tiny-ImageNet validation images"
        while read -r image_name label remainder; do
            [[ -n "${image_name:-}" && -n "${label:-}" ]] || continue
            [[ -f "$images_dir/$image_name" ]] || continue
            label_dir="$val_dir/$label"
            mkdir -p "$label_dir"
            mv -n "$images_dir/$image_name" "$label_dir/"
        done < "$annotations"
        rmdir "$images_dir" 2>/dev/null || true
    fi

    if [[ ! -e "$root/val2" ]]; then
        ln -s val "$root/val2" 2>/dev/null || cp -al "$val_dir" "$root/val2"
    fi
}

prepare_tiny_imagenet() {
    local root="$DATASET_DIR/tiny-imagenet-200"

    flatten_tiny_imagenet_train "$root"
    prepare_tiny_imagenet_val "$root"
    log "Tiny-ImageNet layout prepared for both val/ and val2/ loaders"
}

download_cifar10() {
    local name="cifar-10-python.tar.gz"
    local url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    local md5="c58f30108f718f92721af3b95e74349a"
    local root="$DATASET_DIR/cifar10"

    download_archive "$name" "$url" "$md5"
    extract_tar_gz "$ARCHIVE_DIR/$name" "$root" "$root/cifar-10-batches-py"
}

download_cifar100() {
    local name="cifar-100-python.tar.gz"
    local url="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    local md5="eb9058c3a382ffc7106e4002c42a8d85"
    local root="$DATASET_DIR/cifar100"

    download_archive "$name" "$url" "$md5"
    extract_tar_gz "$ARCHIVE_DIR/$name" "$root" "$root/cifar-100-python"
}

download_stl10() {
    local name="stl10_binary.tar.gz"
    local url="https://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    local md5="91f7769df0f17e558f3565bffb0c7dfb"
    local root="$DATASET_DIR/stl10"

    download_archive "$name" "$url" "$md5"
    extract_tar_gz "$ARCHIVE_DIR/$name" "$root" "$root/stl10_binary"
}

download_tiny_imagenet() {
    local name="tiny-imagenet-200.zip"
    local url="https://cs231n.stanford.edu/tiny-imagenet-200.zip"
    local md5="90528d7ca1a48142e341f4ef8d21d0de"
    local root="$DATASET_DIR/tiny-imagenet-200"

    download_archive "$name" "$url" "$md5"
    extract_zip "$ARCHIVE_DIR/$name" "$DATASET_DIR" "$root"
    prepare_tiny_imagenet
}

normalize_target() {
    local raw="${1,,}"

    case "$raw" in
        all)
            printf '%s\n' cifar10 cifar100 stl10 tiny-imagenet
            ;;
        cifar10|cifar-10)
            printf '%s\n' cifar10
            ;;
        cifar100|cifar-100)
            printf '%s\n' cifar100
            ;;
        stl10|stl-10)
            printf '%s\n' stl10
            ;;
        tiny|tinyimagenet|tiny-imagenet|tinyimagenet200|tiny-imagenet-200)
            printf '%s\n' tiny-imagenet
            ;;
        *)
            die "unknown dataset target: $1"
            ;;
    esac
}

main() {
    local requested=()
    local target

    need_cmd md5sum
    need_cmd find
    need_downloader
    mkdir -p "$DATASET_DIR"

    if [[ $# -eq 0 ]]; then
        requested=(cifar10 cifar100 stl10 tiny-imagenet)
    else
        while [[ $# -gt 0 ]]; do
            case "$1" in
                -h|--help|help)
                    usage
                    exit 0
                    ;;
            esac
            while IFS= read -r target; do
                requested+=("$target")
            done < <(normalize_target "$1")
            shift
        done
    fi

    for target in "${requested[@]}"; do
        case "$target" in
            cifar10) download_cifar10 ;;
            cifar100) download_cifar100 ;;
            stl10) download_stl10 ;;
            tiny-imagenet) download_tiny_imagenet ;;
            *) die "internal error, unsupported target: $target" ;;
        esac
    done

    log "done"
}

main "$@"
