$ErrorActionPreference = 'Continue'
$sourceDir = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'sources'
$targets = @(
    @{ Id = 'resnet'; Url = 'https://arxiv.org/pdf/1512.03385' },
    @{ Id = 'mae'; Url = 'https://arxiv.org/pdf/2111.06377' },
    @{ Id = 'detr'; Url = 'https://arxiv.org/pdf/2005.12872' },
    @{ Id = 'maskrcnn'; Url = 'https://arxiv.org/pdf/1703.06870' },
    @{ Id = 'sam'; Url = 'https://arxiv.org/pdf/2304.02643' },
    @{ Id = 'nerf'; Url = 'https://arxiv.org/pdf/2003.08934' }
)
$jobs = @()
foreach ($target in $targets) {
    $out = Join-Path $sourceDir ($target.Id + '.pdf')
    $part = $out + '.fallback'
    if (Test-Path -LiteralPath $part) { Remove-Item -LiteralPath $part -Force }
    $args = @('-L', '--fail', '--retry', '3', '--retry-delay', '2', '--connect-timeout', '20', '--max-time', '240', '-A', 'Mozilla/5.0', '-o', $part, $target.Url)
    $process = Start-Process -FilePath 'curl.exe' -ArgumentList $args -WindowStyle Hidden -PassThru
    $jobs += [pscustomobject]@{ Id = $target.Id; Output = $out; Part = $part; Process = $process }
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
