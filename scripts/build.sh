#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/build.sh [options]

Configure, build, and optionally test DeepForge.

Options:
  --importer-only       Build the CPU-only importer without MLIR.
  --build-dir DIR       Build directory (default: build or build-importer).
  --build-type TYPE     CMake build type (default: Release).
  --generator NAME      CMake generator (default: Ninja).
  --jobs N              Limit parallel build jobs.
  --no-tests            Do not build or run the test suite.
  --no-tools            Do not build command-line tools.
  --configure-only      Configure the build tree without compiling.
  --install PREFIX      Install after a successful build.
  --clean               Remove the selected build directory first.
  -h, --help            Show this help text.

Dependency discovery:
  CMAKE_PREFIX_PATH             Existing CMake prefix list.
  LLVM_INSTALL_PREFIX           Optional MLIR/LLVM install prefix.
  CUDNN_FRONTEND_SOURCE_DIR     Optional cuDNN Frontend checkout.

The existing CMAKE_PREFIX_PATH environment is preserved. Optional dependency
variables are supplied as additional CMake prefixes. No machine-specific paths
are assumed.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(cd -- "$script_dir/.." && pwd)

mode=full
build_dir=${DEEPFORGE_BUILD_DIR:-}
build_type=${DEEPFORGE_BUILD_TYPE:-Release}
generator=${DEEPFORGE_GENERATOR:-Ninja}
jobs=${DEEPFORGE_BUILD_JOBS:-}
build_tests=${DEEPFORGE_BUILD_TESTS:-ON}
build_tools=${DEEPFORGE_BUILD_TOOLS:-ON}
configure_only=OFF
install_prefix=
clean=OFF

cmake_value_is_true() {
    case "${1^^}" in
        1|ON|YES|TRUE|Y) return 0 ;;
        *) return 1 ;;
    esac
}

while (($# > 0)); do
    case "$1" in
        --importer-only)
            mode=importer
            shift
            ;;
        --build-dir)
            (($# >= 2)) || die "--build-dir requires a directory"
            build_dir=$2
            shift 2
            ;;
        --build-type)
            (($# >= 2)) || die "--build-type requires a value"
            build_type=$2
            shift 2
            ;;
        --generator)
            (($# >= 2)) || die "--generator requires a name"
            generator=$2
            shift 2
            ;;
        --jobs)
            (($# >= 2)) || die "--jobs requires a positive integer"
            jobs=$2
            shift 2
            ;;
        --no-tests)
            build_tests=OFF
            shift
            ;;
        --no-tools)
            build_tools=OFF
            shift
            ;;
        --configure-only)
            configure_only=ON
            shift
            ;;
        --install)
            (($# >= 2)) || die "--install requires a prefix"
            install_prefix=$2
            shift 2
            ;;
        --clean)
            clean=ON
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

if [[ -z "$build_dir" ]]; then
    if [[ "$mode" == importer ]]; then
        build_dir=build-importer
    else
        build_dir=build
    fi
fi

case "$build_dir" in
    /*) ;;
    *) build_dir="$source_dir/$build_dir" ;;
esac
build_dir=$(realpath -m -- "$build_dir")

if [[ -n "$install_prefix" ]]; then
    case "$install_prefix" in
        /*) ;;
        *) install_prefix="$source_dir/$install_prefix" ;;
    esac
    install_prefix=$(realpath -m -- "$install_prefix")
fi

command -v cmake >/dev/null 2>&1 || die "cmake is not available on PATH"
if [[ "$generator" == Ninja ]]; then
    command -v ninja >/dev/null 2>&1 || die "ninja is not available on PATH"
fi
if cmake_value_is_true "$build_tests"; then
    command -v ctest >/dev/null 2>&1 || die "ctest is not available on PATH"
fi

if [[ -n "$jobs" && ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    die "--jobs must be a positive integer"
fi

if [[ "$clean" == ON ]]; then
    [[ "$build_dir" != / && "$build_dir" != "$source_dir" ]] ||
        die "refusing to clean unsafe build directory: $build_dir"
    [[ ! -e "$build_dir" || -f "$build_dir/CMakeCache.txt" ]] ||
        die "refusing to clean a directory without CMakeCache.txt: $build_dir"
    rm -rf -- "$build_dir"
fi

prefix_path=
append_prefix() {
    local prefix=$1
    [[ -n "$prefix" ]] || return 0
    prefix=$(realpath -m -- "$prefix")
    if [[ -n "$prefix_path" ]]; then
        prefix_path="${prefix_path};${prefix}"
    else
        prefix_path=$prefix
    fi
}

append_prefix "${LLVM_INSTALL_PREFIX:-}"
append_prefix "${CUDNN_FRONTEND_SOURCE_DIR:-}"

if [[ "$mode" == importer ]]; then
    build_tools=OFF
fi

cmake_args=(
    -S "$source_dir"
    -B "$build_dir"
    -G "$generator"
    "-DCMAKE_BUILD_TYPE=$build_type"
    "-DDEEPFORGE_BUILD_TESTS=$build_tests"
    "-DDEEPFORGE_BUILD_TOOLS=$build_tools"
)

if [[ "$mode" == importer ]]; then
    cmake_args+=(-DDEEPFORGE_ENABLE_MLIR=OFF)
fi

if [[ -n "$prefix_path" ]]; then
    cmake_args+=("-DCMAKE_PREFIX_PATH=$prefix_path")
fi

printf 'Configuring DeepForge (%s) in %s\n' "$mode" "$build_dir"
cmake "${cmake_args[@]}"

if [[ "$configure_only" == ON ]]; then
    exit 0
fi

build_args=(--build "$build_dir")
if [[ -n "$jobs" ]]; then
    build_args+=(--parallel "$jobs")
fi

printf 'Building DeepForge\n'
cmake "${build_args[@]}"

if cmake_value_is_true "$build_tests"; then
    printf 'Running tests\n'
    ctest --test-dir "$build_dir" --output-on-failure
fi

if [[ -n "$install_prefix" ]]; then
    printf 'Installing DeepForge in %s\n' "$install_prefix"
    cmake --install "$build_dir" --prefix "$install_prefix"
fi
