#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/release-check.sh [options]

Run the local DeepForge release qualification matrix: importer-only, Release,
ASan, UBSan, and a baseline/auto benchmark smoke test.

Options:
  --build-root DIR   Parent build directory (default: build-release-check).
  --jobs N           Limit parallel build jobs.
  --no-clean         Reuse existing build trees instead of cleaning first.
  -h, --help         Show this help text.

Dependency discovery is delegated to scripts/build.sh. Set
LLVM_INSTALL_PREFIX and CUDNN_FRONTEND_SOURCE_DIR, or use CMAKE_PREFIX_PATH.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(cd -- "$script_dir/.." && pwd)
build_root=${DEEPFORGE_RELEASE_BUILD_ROOT:-build-release-check}
jobs=${DEEPFORGE_BUILD_JOBS:-}
clean=ON

while (($# > 0)); do
    case "$1" in
        --build-root)
            (($# >= 2)) || die "--build-root requires a directory"
            build_root=$2
            shift 2
            ;;
        --jobs)
            (($# >= 2)) || die "--jobs requires a positive integer"
            jobs=$2
            shift 2
            ;;
        --no-clean)
            clean=OFF
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1 (try --help)"
            ;;
    esac
done

if [[ -n "$jobs" && ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    die "--jobs must be a positive integer"
fi
case "$build_root" in
    /*) ;;
    *) build_root="$source_dir/$build_root" ;;
esac
build_root=$(realpath -m -- "$build_root")
[[ "$build_root" != / && "$build_root" != "$source_dir" ]] ||
    die "unsafe build root: $build_root"

common_args=(--build-type Release)
if [[ -n "$jobs" ]]; then
    common_args+=(--jobs "$jobs")
fi
if [[ "$clean" == ON ]]; then
    common_args+=(--clean)
fi

printf 'Importer-only qualification\n'
"$script_dir/build.sh" --importer-only \
    --build-dir "$build_root/importer" "${common_args[@]}"

printf 'Full Release qualification\n'
"$script_dir/build.sh" --build-dir "$build_root/release" \
    "${common_args[@]}"

printf 'AddressSanitizer qualification\n'
ASAN_OPTIONS=${ASAN_OPTIONS:-detect_leaks=1:halt_on_error=1} \
    "$script_dir/build.sh" --build-dir "$build_root/address" \
    --sanitizer address --no-tools "${common_args[@]}"

printf 'UndefinedBehaviorSanitizer qualification\n'
UBSAN_OPTIONS=${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1} \
    "$script_dir/build.sh" --build-dir "$build_root/undefined" \
    --sanitizer undefined --no-tools "${common_args[@]}"

printf 'Schedule A/B benchmark smoke\n'
"$build_root/release/tools/deepforge-benchmark" \
    --profile=small --iterations=1 --schedule=both

printf 'DeepForge release qualification passed\n'
