$ErrorActionPreference = 'Stop'
$workDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifestPath = Join-Path $workDir 'whole_network_manifest.json'
$sourceDir = Join-Path $workDir 'sources'
New-Item -ItemType Directory -Force -Path $sourceDir | Out-Null
$items = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$results = @()
foreach ($item in $items) {
    $target = Join-Path $sourceDir ($item.id + '.pdf')
    if ((Test-Path -LiteralPath $target) -and ((Get-Item -LiteralPath $target).Length -gt 10000)) {
        $results += [pscustomobject]@{ id = $item.id; status = 'cached'; bytes = (Get-Item -LiteralPath $target).Length }
        continue
    }
    $partial = $target + '.part'
    if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
    & curl.exe -L --fail --retry 2 --retry-delay 2 --connect-timeout 20 --max-time 180 -o $partial $item.pdf
    if ($LASTEXITCODE -ne 0) {
        $results += [pscustomobject]@{ id = $item.id; status = 'failed'; bytes = 0 }
        continue
    }
    Move-Item -LiteralPath $partial -Destination $target -Force
    $results += [pscustomobject]@{ id = $item.id; status = 'downloaded'; bytes = (Get-Item -LiteralPath $target).Length }
}
$results | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $workDir 'download_results.json') -Encoding UTF8
$results | Format-Table -AutoSize
