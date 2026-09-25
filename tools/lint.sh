#!/usr/bin/env bash
# Formatting and static analysis for grad.cpp's C++ sources.
#
#   tools/lint.sh format          check every tracked C++ file against .clang-format
#   tools/lint.sh format --fix    reformat them in place
#   tools/lint.sh tidy [BUILD]    run clang-tidy (.clang-tidy) over the project's
#                                 translation units in BUILD/compile_commands.json
#                                 (default: build; configure with
#                                 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
#
# Environment:
#   CLANG_FORMAT, CLANG_TIDY   tool binaries (default: clang-format, clang-tidy).
#                              CI pins clang-format 19.1.7 and clang-tidy 19.1.0;
#                              other versions may format or diagnose differently.
#   JOBS                       parallel clang-tidy processes (default: 4)
#   TIDY_OBJCXX=1              also analyze the Objective-C++ Metal backend (macOS)
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
CLANG_FORMAT=${CLANG_FORMAT:-clang-format}
CLANG_TIDY=${CLANG_TIDY:-clang-tidy}
JOBS=${JOBS:-4}

cxx_files() {
    git ls-files -- '*.cpp' '*.h' '*.mm'
}

run_format() {
    if [[ ${1:-} == --fix ]]; then
        cxx_files | xargs "$CLANG_FORMAT" -i
    else
        cxx_files | xargs "$CLANG_FORMAT" --dry-run --Werror
        echo "clang-format: $(cxx_files | wc -l | tr -d ' ') files clean"
    fi
}

run_tidy() {
    local build=${1:-build}
    local db="$build/compile_commands.json"
    if [[ ! -f $db ]]; then
        echo "$db not found: configure with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON" >&2
        exit 2
    fi

    # Project translation units: tracked sources that the build compiles
    # (tests/package/main.cpp belongs to a separate project and is not).
    local pattern='\.cpp$'
    if [[ ${TIDY_OBJCXX:-0} == 1 ]]; then
        pattern='\.(cpp|mm)$'
    fi
    local tus=()
    while IFS= read -r f; do
        if grep -qF "\"$PWD/$f\"" "$db"; then
            tus+=("$f")
        fi
    done < <(git ls-files | grep -E "$pattern")

    # Project headers are analyzed through those TUs. Report this
    # checkout's only: an absolute prefix, so SDK and /usr/include paths,
    # which also contain "/include/", never match.
    local root_re
    root_re=$(printf '%s' "$PWD" | sed 's/[][\\.^$*+?(){}|]/\\&/g')
    local extra=("--header-filter=^$root_re/(include|src|tests)/")
    if [[ $(uname) == Darwin ]]; then
        # A standalone clang-tidy (e.g. from pip) does not know what
        # AppleClang assumes implicitly: the Xcode SDK (without it <array>
        # and Accelerate are missing) and a deployment target of the host
        # macOS (without it libc++ marks std::from_chars unavailable).
        extra+=("--extra-arg=-isysroot$(xcrun --show-sdk-path)")
        extra+=("--extra-arg=-mmacosx-version-min=$(sw_vers -productVersion)")
    fi

    echo "clang-tidy: ${#tus[@]} translation units, $JOBS jobs"
    # clang prints "N warnings generated." for every TU, counting the
    # system-header diagnostics it then suppresses; drop that noise.
    printf '%s\n' "${tus[@]}" \
        | xargs -P "$JOBS" -n 1 "$CLANG_TIDY" -p "$build" --quiet "${extra[@]}" 2>&1 \
        | { grep -Ev '^[0-9]+ warnings? generated\.$' || true; }
    echo "clang-tidy: clean"
}

case ${1:-} in
    format) shift; run_format "$@" ;;
    tidy) shift; run_tidy "$@" ;;
    *)
        awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
        exit 2
        ;;
esac
