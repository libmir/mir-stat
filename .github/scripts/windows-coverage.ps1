# Match the Linux coverage groups while retaining package-mode template coverage.
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$project = (Get-Location).Path
$temporaryRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$build = Join-Path $temporaryRoot ('mir-stat-coverage-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $build | Out-Null

try {
    foreach ($group in @('descriptive', 'remaining')) {
        $directory = Join-Path $build $group
        New-Item -ItemType Directory -Path $directory | Out-Null
        # Copies avoid Windows symlink privileges and preserve relative source paths.
        foreach ($name in @('source', 'tests', 'benchmarks')) {
            Copy-Item -LiteralPath (Join-Path $project $name) -Destination $directory -Recurse
        }
        Copy-Item -LiteralPath (Join-Path $project 'dub.sdl') -Destination $directory
        $selections = Join-Path $project 'dub.selections.json'
        if (Test-Path -LiteralPath $selections) {
            Copy-Item -LiteralPath $selections -Destination $directory
        }

        # Set sourcePaths at package scope; configuration paths add to source/.
        $groupRecipe = if ($group -eq 'descriptive') {
            @'

sourcePaths "source/mir/stat/descriptive"
sourceFiles "source/mir/stat/internal/borrow.d"
importPaths "source"
'@
        } else {
            @'

excludedSourceFiles "source/mir/stat/descriptive/*" "source/mir/stat/internal/borrow.d"
'@
        }
        Add-Content -LiteralPath (Join-Path $directory 'dub.sdl') -Value $groupRecipe

        Set-Location -LiteralPath $directory
        Write-Host "Running Windows LDC general coverage ($group)"
        & (Join-Path $project '.github/scripts/windows-resources.ps1') Run dub test --build=unittest-cov
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        # Reuse the first group's dependency resolution for later builds.
        $resolved = Join-Path $directory 'dub.selections.json'
        if ((Test-Path -LiteralPath $resolved) -and !(Test-Path -LiteralPath $selections)) {
            Copy-Item -LiteralPath $resolved -Destination $selections
        }
        $reports = @(Get-ChildItem -LiteralPath $directory -Filter 'source-*.lst' -File)
        if ($reports.Count -eq 0) { throw "No coverage reports found for $group" }
        foreach ($report in $reports) {
            if ($report.Length -eq 0) { throw "Empty coverage report: $($report.Name)" }
            if (Test-Path -LiteralPath (Join-Path $project $report.Name)) {
                throw "Duplicate general coverage report: $($report.Name)"
            }
        }
        foreach ($report in $reports) {
            Move-Item -LiteralPath $report.FullName -Destination $project
        }
    }
} finally {
    Set-Location -LiteralPath $project
    # Only remove the unique scratch directory created above.
    $resolvedBuild = [IO.Path]::GetFullPath($build)
    $expectedParent = $temporaryRoot.TrimEnd([IO.Path]::DirectorySeparatorChar)
    if ([IO.Path]::GetDirectoryName($resolvedBuild) -ne $expectedParent -or
        [IO.Path]::GetFileName($resolvedBuild) -notmatch '^mir-stat-coverage-[0-9a-f]{32}$') {
        throw "Unexpected coverage scratch path: $resolvedBuild"
    }
    Remove-Item -LiteralPath $resolvedBuild -Recurse -Force
}
