$ErrorActionPreference = 'Continue'
$work = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceDir = Join-Path $work 'sources'
New-Item -ItemType Directory -Force -Path $sourceDir | Out-Null
$items = Get-Content -Raw -LiteralPath (Join-Path $work 'sources_manifest.json') | ConvertFrom-Json
$jobs = @()

foreach ($item in $items) {
    $url = [string]$item.pdf
    if ($item.id -eq 'vit') { $url = 'https://arxiv.org/pdf/2010.11929' }
    if ($item.id -eq 'unet') { $url = 'https://arxiv.org/pdf/1505.04597' }
    $out = Join-Path $sourceDir ($item.id + '.pdf')
    $part = $out + '.part'
    $log = Join-Path $sourceDir ($item.id + '.download.log')
    $err = Join-Path $sourceDir ($item.id + '.download.err')
    if (Test-Path -LiteralPath $part) { Remove-Item -LiteralPath $part -Force }
    $args = @('-L', '--fail', '--retry', '4', '--retry-delay', '2', '--connect-timeout', '20', '--max-time', '300', '-A', 'Mozilla/5.0', '-o', $part, $url)
    $process = Start-Process -FilePath 'curl.exe' -ArgumentList $args -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $err -PassThru
    $jobs += [pscustomobject]@{ Id = [string]$item.id; Output = $out; Part = $part; Process = $process }
}

foreach ($job in $jobs) {
    $job.Process.WaitForExit()
    if ($job.Process.ExitCode -eq 0 -and (Test-Path -LiteralPath $job.Part)) {
        Move-Item -LiteralPath $job.Part -Destination $job.Output -Force
        Write-Output ('OK ' + $job.Id)
    } else {
        Write-Output ('FAILED ' + $job.Id + ' exit=' + $job.Process.ExitCode)
    }
}

Get-ChildItem -LiteralPath $sourceDir -Filter '*.pdf' | Sort-Object Name | Select-Object Name, Length
