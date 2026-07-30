$ErrorActionPreference = "Stop"

$ReleaseRoot = Split-Path -Parent $PSScriptRoot
$AllFiles = @(Get-ChildItem -LiteralPath $ReleaseRoot -Recurse -File -Force)
$WavFiles = @($AllFiles | Where-Object { $_.Extension -eq ".wav" })

if ($WavFiles.Count -ne 897) {
    throw "Expected 897 generated WAV files, found $($WavFiles.Count)"
}

$Round1Wav = @($WavFiles | Where-Object { $_.FullName -like "*\experiments\round1_5epoch\*" })
$Round2Wav = @($WavFiles | Where-Object { $_.FullName -like "*\experiments\round2_8epoch_cosine\*" })
if ($Round1Wav.Count -ne 345) {
    throw "Expected 345 round-1 WAV files, found $($Round1Wav.Count)"
}
if ($Round2Wav.Count -ne 552) {
    throw "Expected 552 round-2 WAV files, found $($Round2Wav.Count)"
}

$WavOutsideExperiments = @($WavFiles | Where-Object { $_.FullName -notlike "*\experiments\*" })
if ($WavOutsideExperiments.Count -ne 0) {
    throw "Found WAV files outside experiments/: $($WavOutsideExperiments.FullName -join ', ')"
}

foreach ($Wav in $WavFiles) {
    $Stream = [IO.File]::OpenRead($Wav.FullName)
    try {
        if ($Stream.Length -lt 44) {
            throw "WAV file is too short: $($Wav.FullName)"
        }
        $Header = New-Object byte[] 12
        [void]$Stream.Read($Header, 0, 12)
        $Riff = [Text.Encoding]::ASCII.GetString($Header, 0, 4)
        $Wave = [Text.Encoding]::ASCII.GetString($Header, 8, 4)
        if ($Riff -ne "RIFF" -or $Wave -ne "WAVE") {
            throw "Invalid WAV header: $($Wav.FullName)"
        }
    } finally {
        $Stream.Dispose()
    }
}

$ForbiddenDirectories = @(Get-ChildItem -LiteralPath $ReleaseRoot -Recurse -Directory -Force | Where-Object {
    $_.Name -eq "checkpoints" -or
    $_.Name -like "checkpoint-epoch-*" -or
    $_.Name -eq "__pycache__" -or
    $_.Name -like "*failed*" -or
    $_.Name -like "*interruption*"
})
if ($ForbiddenDirectories.Count -ne 0) {
    throw "Forbidden directories found: $($ForbiddenDirectories.FullName -join ', ')"
}

$ForbiddenFiles = @($AllFiles | Where-Object {
    $_.Extension -eq ".pid" -or
    $_.Extension -eq ".pyc" -or
    $_.Name -eq "ref_audio_031.wav"
})
if ($ForbiddenFiles.Count -ne 0) {
    throw "Forbidden files found: $($ForbiddenFiles.FullName -join ', ')"
}

$LossSteps = @(Import-Csv -LiteralPath (Join-Path $ReleaseRoot "metrics\loss_steps.csv"))
$LossEpochs = @(Import-Csv -LiteralPath (Join-Path $ReleaseRoot "metrics\loss_epoch_summary.csv"))
$Matrix = @(Import-Csv -LiteralPath (Join-Path $ReleaseRoot "metrics\experiment_matrix.csv"))
if ($LossSteps.Count -ne 3510) {
    throw "Expected 3510 loss rows, found $($LossSteps.Count)"
}
if ($LossEpochs.Count -ne 39) {
    throw "Expected 39 epoch summaries, found $($LossEpochs.Count)"
}
if ($Matrix.Count -ne 6) {
    throw "Expected 6 experiment rows, found $($Matrix.Count)"
}

$MissingReadmeTargets = New-Object System.Collections.Generic.List[string]
$Readme = Get-Content -LiteralPath (Join-Path $ReleaseRoot "README.md") -Raw -Encoding UTF8
$LinkPattern = [regex]'\[[^\]]+\]\(([^)]+)\)'
foreach ($Match in $LinkPattern.Matches($Readme)) {
    $Target = [Uri]::UnescapeDataString($Match.Groups[1].Value)
    if ($Target -match '^(https?://|#)') {
        continue
    }
    $ResolvedTarget = Join-Path $ReleaseRoot ($Target -replace '/', '\')
    if (-not (Test-Path -LiteralPath $ResolvedTarget)) {
        $MissingReadmeTargets.Add($Target)
    }
}
if ($MissingReadmeTargets.Count -ne 0) {
    throw "README targets are missing: $($MissingReadmeTargets -join ', ')"
}

$Inventory = @($AllFiles | Where-Object { $_.Name -notin @("FILE_INVENTORY.csv", "SHA256SUMS") } | Sort-Object FullName | ForEach-Object {
    [pscustomobject]@{
        relative_path = $_.FullName.Substring($ReleaseRoot.Length + 1).Replace("\", "/")
        bytes = $_.Length
        size_mib = [math]::Round($_.Length / 1MB, 6)
        extension = $_.Extension
    }
})
$Inventory | Export-Csv -LiteralPath (Join-Path $ReleaseRoot "FILE_INVENTORY.csv") -NoTypeInformation -Encoding UTF8

$FilesToHash = @(Get-ChildItem -LiteralPath $ReleaseRoot -Recurse -File -Force | Where-Object { $_.Name -ne "SHA256SUMS" } | Sort-Object FullName)
$ChecksumLines = foreach ($File in $FilesToHash) {
    $RelativePath = $File.FullName.Substring($ReleaseRoot.Length + 1).Replace("\", "/")
    $Hash = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$Hash  $RelativePath"
}
Set-Content -LiteralPath (Join-Path $ReleaseRoot "SHA256SUMS") -Value $ChecksumLines -Encoding ASCII

$FinalFiles = @(Get-ChildItem -LiteralPath $ReleaseRoot -Recurse -File -Force)
$TotalBytes = ($FinalFiles | Measure-Object Length -Sum).Sum
$LargestFile = $FinalFiles | Sort-Object Length -Descending | Select-Object -First 1
if ($LargestFile.Length -ge 100MB) {
    throw "A file exceeds GitHub's 100 MiB limit: $($LargestFile.FullName)"
}

Write-Output "Release verification passed."
Write-Output ("Files: {0}" -f $FinalFiles.Count)
Write-Output ("WAV files: {0} (round1={1}, round2={2})" -f $WavFiles.Count, $Round1Wav.Count, $Round2Wav.Count)
Write-Output ("Loss rows: {0}; epoch summaries: {1}; experiments: {2}" -f $LossSteps.Count, $LossEpochs.Count, $Matrix.Count)
Write-Output ("Total size: {0:N1} MiB" -f ($TotalBytes / 1MB))
Write-Output ("Largest file: {0:N2} MiB - {1}" -f ($LargestFile.Length / 1MB), $LargestFile.Name)
