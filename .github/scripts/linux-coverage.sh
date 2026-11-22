#!/usr/bin/env bash
set -euo pipefail

# Keep the original source paths in the compiler's .lst reports. Each group
# uses the project's recipe and dependency selections in its own build directory.
project=$PWD
if [[ ${1:-} == --stable-build-path ]]; then
    build="${RUNNER_TEMP:?}/mir-stat-coverage"
    mkdir -- "$build"
else
    build=$(mktemp -d)
fi
trap 'rm -rf -- "$build"' EXIT
for group in descriptive remaining; do
    mkdir "$build/$group"
    cd "$build/$group"
    ln -s "$project"/{source,tests,benchmarks} .
    cp "$project/dub.sdl" .
    if [[ -f "$project/dub.selections.json" ]]; then
        cp "$project/dub.selections.json" .
    fi
    # DUB adds source/ by default unless sourcePaths is set at package scope.
    if [[ "$group" == descriptive ]]; then
        cat >> dub.sdl <<'SDL'

sourcePaths "source/mir/stat/descriptive"
sourceFiles "source/mir/stat/internal/borrow.d"
importPaths "source"
SDL
    else
        cat >> dub.sdl <<'SDL'

excludedSourceFiles "source/mir/stat/descriptive/*" "source/mir/stat/internal/borrow.d"
SDL
    fi

    echo "Running Linux general coverage ($group)"
    bash "$project/.github/scripts/linux-resources.sh" run dub test --build=unittest-cov
    # A fresh checkout resolves dependencies on its first build. Reuse that
    # selection for the remaining group and subsequent workflow steps.
    if [[ -f dub.selections.json && ! -f "$project/dub.selections.json" ]]; then
        cp dub.selections.json "$project/"
    fi
    for report in source-*.lst; do
        if [[ -e "$project/$report" ]]; then
            echo "::error::Duplicate general coverage report $report"
            exit 1
        fi
        mv "$report" "$project/"
    done
done
