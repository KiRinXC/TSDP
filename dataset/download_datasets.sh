#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${DATASET_DIR:-$SCRIPT_DIR/public}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$DATASET_DIR/_archives}"
FORCE="${FORCE:-0}"
DOWNLOADER="${DOWNLOADER:-}"

usage() {
    cat <<'USAGE'
Download the four public datasets into this project's canonical dataset/public/ layout.

Usage:
  bash dataset/download_datasets.sh [all|c10|c100|s10|t200 ...]

By default, all four datasets are downloaded and extracted:
  dataset/public/c10/cifar-10-batches-py
  dataset/public/c100/cifar-100-python
  dataset/public/s10/stl10_binary
  dataset/public/t200

Environment:
  DATASET_DIR=/path/to/dataset/public
                                 Override public dataset directory.
  ARCHIVE_DIR=/path/to/archives  Override archive cache directory.
  FORCE=1                        Re-download and re-extract existing files.
  DOWNLOADER=curl|wget           Choose downloader explicitly. Default: first available.

Examples:
  bash dataset/download_datasets.sh
  bash dataset/download_datasets.sh c10 c100
  FORCE=1 bash dataset/download_datasets.sh t200
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
    if [[ -n "$DOWNLOADER" ]]; then
        case "$DOWNLOADER" in
            curl|wget)
                command -v "$DOWNLOADER" >/dev/null 2>&1 || die "required command not found: $DOWNLOADER"
                return 0
                ;;
            *)
                die "unknown DOWNLOADER: $DOWNLOADER (valid: curl wget)"
                ;;
        esac
    fi

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

fetch_url_with() {
    local downloader="$1"
    local url="$2"
    local output="$3"

    if [[ "$downloader" == "curl" ]]; then
        curl -L --fail --retry 5 --retry-delay 5 --connect-timeout 30 \
            --output "$output" "$url"
    elif [[ "$downloader" == "wget" ]]; then
        wget --tries=5 --timeout=30 --output-document="$output" "$url"
    else
        die "unsupported downloader: $downloader"
    fi
}

download_md5_archive() {
    local name="$1"
    local expected_md5="$2"
    local preferred_downloader="$3"
    shift 3
    local urls=("$@")
    local archive="$ARCHIVE_DIR/$name"
    local part="$archive.part"
    local url

    mkdir -p "$ARCHIVE_DIR"

    if [[ "$preferred_downloader" == "wget" ]]; then
        need_cmd wget
    elif [[ "$preferred_downloader" == "curl" ]]; then
        need_cmd curl
    else
        die "unsupported downloader: $preferred_downloader"
    fi

    if [[ "$FORCE" == "1" ]]; then
        rm -f "$archive" "$part"
    fi

    if md5_ok "$archive" "$expected_md5"; then
        log "archive ok: $archive"
        return 0
    fi

    if [[ -f "$archive" ]]; then
        log "removing archive with unexpected checksum: $archive"
        rm -f "$archive"
    fi

    for url in "${urls[@]}"; do
        rm -f "$part"
        log "downloading $name from $url"
        if fetch_url_with "$preferred_downloader" "$url" "$part"; then
            if md5_ok "$part" "$expected_md5"; then
                mv "$part" "$archive"
                log "checksum ok: $name"
                return 0
            fi
            log "checksum failed for $name from $url"
            rm -f "$part"
        else
            log "download failed for $name from $url"
            rm -f "$part"
        fi
    done

    die "could not download a valid archive for $name"
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

official_cifar10_available() {
    local root="$1/cifar-10-batches-py"
    [[ -f "$root/data_batch_1" && -f "$root/data_batch_2" && -f "$root/data_batch_3" && -f "$root/data_batch_4" && -f "$root/data_batch_5" && -f "$root/test_batch" && -f "$root/batches.meta" ]]
}

official_cifar100_available() {
    local root="$1/cifar-100-python"
    [[ -f "$root/train" && -f "$root/test" && -f "$root/meta" ]]
}

flatten_tiny_imagenet_train() {
    local root="$1"
    local class_dir
    local image_file

    [[ -d "$root/train" ]] || die "Tiny-ImageNet train directory missing: $root/train"

    for class_dir in "$root"/train/*; do
        [[ -d "$class_dir/images" ]] || continue
        while IFS= read -r -d '' image_file; do
            [[ -f "$image_file" ]] || continue
            mv -n "$image_file" "$class_dir/" 2>/dev/null || [[ ! -f "$image_file" ]]
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
            mv -n "$images_dir/$image_name" "$label_dir/" 2>/dev/null || [[ ! -f "$images_dir/$image_name" ]]
        done < "$annotations"
        rmdir "$images_dir" 2>/dev/null || true
    fi

    if [[ ! -e "$root/val2" ]]; then
        ln -s val "$root/val2" 2>/dev/null || cp -al "$val_dir" "$root/val2"
    fi
}

prepare_tiny_imagenet() {
    local root="$DATASET_DIR/t200"

    flatten_tiny_imagenet_train "$root"
    prepare_tiny_imagenet_val "$root"
    log "Tiny-ImageNet layout prepared for both val/ and val2/ loaders"
}

verify_stl10_integrity() {
    local root="$1"

    md5_ok "$root/train_X.bin" "918c2871b30a85fa023e0c44e0bee87f" || die "STL10 train_X.bin checksum failed"
    md5_ok "$root/train_y.bin" "5a34089d4802c674881badbb80307741" || die "STL10 train_y.bin checksum failed"
    md5_ok "$root/unlabeled_X.bin" "5242ba1fed5e4be9e1e742405eb56ca4" || die "STL10 unlabeled_X.bin checksum failed"
    md5_ok "$root/test_X.bin" "7f263ba9f9e0b06b93213547f721ac82" || die "STL10 test_X.bin checksum failed"
    md5_ok "$root/test_y.bin" "36f9794fa4beb8a2c72628de14fa638e" || die "STL10 test_y.bin checksum failed"
    log "STL10 inner file checksums ok"
}

download_cifar10() {
    local root="$DATASET_DIR/c10"
    local name="cifar-10-python.tar.gz"
    local md5="c58f30108f718f92721af3b95e74349a"

    if official_cifar10_available "$root"; then
        log "dataset ok: $root/cifar-10-batches-py"
        return 0
    fi

    download_md5_archive "$name" "$md5" "$DOWNLOADER" \
        "https://dataset.bj.bcebos.com/cifar/$name" \
        "https://www.cs.toronto.edu/~kriz/$name"
    extract_tar_gz "$ARCHIVE_DIR/$name" "$root" "$root/cifar-10-batches-py"
}

download_cifar100() {
    local root="$DATASET_DIR/c100"
    local name="cifar-100-python.tar.gz"
    local md5="eb9058c3a382ffc7106e4002c42a8d85"

    if official_cifar100_available "$root"; then
        log "dataset ok: $root/cifar-100-python"
        return 0
    fi

    download_md5_archive "$name" "$md5" "$DOWNLOADER" \
        "https://dataset.bj.bcebos.com/cifar/$name" \
        "https://www.cs.toronto.edu/~kriz/$name"
    extract_tar_gz "$ARCHIVE_DIR/$name" "$root" "$root/cifar-100-python"
}

download_stl10() {
    local name="stl10_binary.tar.gz"
    local url="http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    local md5="91f7769df0f17e558f3565bffb0c7dfb"
    local root="$DATASET_DIR/s10"

    download_md5_archive "$name" "$md5" wget "$url"
    extract_tar_gz "$ARCHIVE_DIR/$name" "$root" "$root/stl10_binary"
    verify_stl10_integrity "$root/stl10_binary"
}

download_tiny_imagenet() {
    local name="tiny-imagenet-200.zip"
    local url="https://cs231n.stanford.edu/tiny-imagenet-200.zip"
    local md5="90528d7ca1a48142e341f4ef8d21d0de"
    local root="$DATASET_DIR/t200"

    download_md5_archive "$name" "$md5" "$DOWNLOADER" "$url"
    if [[ "$FORCE" == "1" && -d "$root" ]]; then
        rm -rf "$root"
    fi
    if [[ ! -d "$root" ]]; then
        extract_zip "$ARCHIVE_DIR/$name" "$DATASET_DIR" "$DATASET_DIR/tiny-imagenet-200"
        mv "$DATASET_DIR/tiny-imagenet-200" "$root"
    else
        log "dataset ok: $root"
    fi
    prepare_tiny_imagenet
}

main() {
    local requested=()
    local target

    need_cmd md5sum
    need_cmd find
    need_downloader
    mkdir -p "$DATASET_DIR"

    if [[ $# -eq 0 ]]; then
        requested=(c10 c100 s10 t200)
    else
        while [[ $# -gt 0 ]]; do
            case "$1" in
                -h|--help|help)
                    usage
                    exit 0
                    ;;
            esac
            target="${1,,}"
            case "$target" in
                all)
                    requested+=(c10 c100 s10 t200)
                    ;;
                c10|c100|s10|t200)
                    requested+=("$target")
                    ;;
                *)
                    die "unknown dataset target: $1 (valid: all c10 c100 s10 t200)"
                    ;;
            esac
            shift
        done
    fi

    for target in "${requested[@]}"; do
        case "$target" in
            c10) download_cifar10 ;;
            c100) download_cifar100 ;;
            s10) download_stl10 ;;
            t200) download_tiny_imagenet ;;
            *) die "internal error, unsupported target: $target" ;;
        esac
    done

    log "done"
}

main "$@"
