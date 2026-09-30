param(
    [string]$DataRoot = 'D:\uwb_gestures',
    [string]$CodeRoot = (Join-Path $env:USERPROFILE '.codex\worktrees\uwb-user-calibration\uwb_gestures'),
    [string]$OutDir = 'D:\uwb-cloud-package-20260930',
    [string]$Python = 'D:\D_conda\envs\pytorch\python.exe',
    [switch]$CodeOnly,
    [switch]$IncludePrepared
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$expectedBranch = 'codex/uwb-direction1-cloud-20260930'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python. Pass -Python with your Python 3.9+ executable." }
if (-not (Test-Path -LiteralPath $CodeRoot -PathType Container)) { throw "Code worktree not found: $CodeRoot. Pass -CodeRoot with the experimental worktree." }
if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) { throw "Data root not found: $DataRoot" }
$branch = (& git -C $repo branch --show-current)
if ($LASTEXITCODE -ne 0 -or $branch -ne $expectedBranch) { throw "Run from the clone on branch $expectedBranch." }
$remote = (& git -C $repo remote get-url origin)
if ($LASTEXITCODE -ne 0 -or $remote -notmatch '^(https://github\.com/|git@github\.com:)Lixuu2001/git-local-demo(\.git)?$') { throw 'Unexpected Git remote; expected Lixuu2001/git-local-demo.' }
$status = (& git -C $repo status --porcelain=v1)
if ($LASTEXITCODE -ne 0 -or $status) { throw 'The transfer clone must be clean before migration. Commit or preserve your edits first.' }
$target = Join-Path $repo 'uwb_direction1\original'
if (Test-Path -LiteralPath $target) { throw 'An original source snapshot already exists in this clone. Preserve it and use a fresh migration branch for another snapshot.' }
$arguments = @((Join-Path $PSScriptRoot 'transfer.py'), 'pack', '--code-root', $CodeRoot, '--data-root', $DataRoot, '--out', $OutDir)
if ($CodeOnly) { $arguments += '--code-only' }
if ($IncludePrepared) { $arguments += '--include-prepared' }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Export failed; no Git files were staged.' }
Copy-Item -LiteralPath (Join-Path $OutDir 'code') -Destination $target -Recurse
& git -C $repo add -- uwb_direction1/original
if ($LASTEXITCODE -ne 0) { throw 'Git add failed.' }
& git -C $repo diff --cached --stat
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the staged snapshot.' }
& git -C $repo commit -m 'research: import original UWB source and migration provenance'
if ($LASTEXITCODE -ne 0) { throw 'Git commit failed. Configure your Git author locally, then commit the staged snapshot.' }
& git -C $repo push origin $expectedBranch
if ($LASTEXITCODE -ne 0) { throw 'Push failed; the local commit and data package are preserved. Authenticate Git locally and retry git push.' }
Write-Host "Source snapshot pushed to $expectedBranch."
Write-Host "Transfer package: $OutDir"
Write-Host 'Send TRANSFER_MANIFEST.json first, then the chunks directory files. Data arrays remain outside Git.'
