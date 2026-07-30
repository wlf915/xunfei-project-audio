$ErrorActionPreference = "Stop"

$ReleaseRoot = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $ReleaseRoot
$Round1Root = Join-Path $WorkspaceRoot "qwen3tts_ref031_3lr_5ep_seed42\ref031_3lr_5ep_seed42"
$Round2Root = Join-Path $WorkspaceRoot "ref031_cosine_8ep_seed42_20260730\ref031_cosine_8ep_seed42_20260730"

function Ensure-Directory {
    param([Parameter(Mandatory = $true)][string]$Path)
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Copy-RequiredFile {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Required file is missing: $Source"
    }
    Ensure-Directory (Split-Path -Parent $Destination)
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

function Copy-OptionalFile {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    if (Test-Path -LiteralPath $Source -PathType Leaf) {
        Ensure-Directory (Split-Path -Parent $Destination)
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
    }
}

$Directories = @(
    "data",
    "environment\round1_5epoch",
    "environment\round2_8epoch_cosine",
    "experiments\round1_5epoch",
    "experiments\round2_8epoch_cosine",
    "figures\round1_5epoch",
    "figures\round2_8epoch_cosine",
    "metrics",
    "scripts\round1_5epoch",
    "scripts\round2_8epoch_cosine"
)

foreach ($RelativePath in $Directories) {
    Ensure-Directory (Join-Path $ReleaseRoot $RelativePath)
}

$Runs = @(
    [pscustomobject]@{
        Round = "round1_5epoch"
        RunId = "lr1e-6"
        DisplayName = "5 epochs / lr=1e-6"
        LearningRate = "1e-6"
        Epochs = 5
        Scheduler = "constant"
        WarmupRatio = "0"
        Source = Join-Path $Round1Root "hparam_runs\ep5_lr1e-6_ref031_seed42"
        ExpectedAudio = 115
        ExpectedCompare = 15
        ExpectedTest = 100
    },
    [pscustomobject]@{
        Round = "round1_5epoch"
        RunId = "lr2e-6"
        DisplayName = "5 epochs / lr=2e-6"
        LearningRate = "2e-6"
        Epochs = 5
        Scheduler = "constant"
        WarmupRatio = "0"
        Source = Join-Path $Round1Root "hparam_runs\ep5_lr2e-6_ref031_seed42"
        ExpectedAudio = 115
        ExpectedCompare = 15
        ExpectedTest = 100
    },
    [pscustomobject]@{
        Round = "round1_5epoch"
        RunId = "lr5e-6"
        DisplayName = "5 epochs / lr=5e-6"
        LearningRate = "5e-6"
        Epochs = 5
        Scheduler = "constant"
        WarmupRatio = "0"
        Source = Join-Path $Round1Root "hparam_runs\ep5_lr5e-6_ref031_seed42"
        ExpectedAudio = 115
        ExpectedCompare = 15
        ExpectedTest = 100
    },
    [pscustomobject]@{
        Round = "round2_8epoch_cosine"
        RunId = "lr5e-7"
        DisplayName = "8 epochs / lr=5e-7 / cosine"
        LearningRate = "5e-7"
        Epochs = 8
        Scheduler = "cosine"
        WarmupRatio = "0.05"
        Source = Join-Path $Round2Root "hparam_runs\A_lr5e-7_cosine8_ref031_seed42"
        ExpectedAudio = 184
        ExpectedCompare = 24
        ExpectedTest = 160
    },
    [pscustomobject]@{
        Round = "round2_8epoch_cosine"
        RunId = "lr1e-6"
        DisplayName = "8 epochs / lr=1e-6 / cosine"
        LearningRate = "1e-6"
        Epochs = 8
        Scheduler = "cosine"
        WarmupRatio = "0.05"
        Source = Join-Path $Round2Root "hparam_runs\B_lr1e-6_cosine8_ref031_seed42"
        ExpectedAudio = 184
        ExpectedCompare = 24
        ExpectedTest = 160
    },
    [pscustomobject]@{
        Round = "round2_8epoch_cosine"
        RunId = "lr1.5e-6"
        DisplayName = "8 epochs / lr=1.5e-6 / cosine"
        LearningRate = "1.5e-6"
        Epochs = 8
        Scheduler = "cosine"
        WarmupRatio = "0.05"
        Source = Join-Path $Round2Root "hparam_runs\C_lr1.5e-6_cosine8_ref031_seed42"
        ExpectedAudio = 184
        ExpectedCompare = 24
        ExpectedTest = 160
    }
)

$LossRecords = New-Object System.Collections.Generic.List[object]
$EpochSummaries = New-Object System.Collections.Generic.List[object]
$RunInventory = New-Object System.Collections.Generic.List[object]
$LossPattern = [regex]'Epoch\s+(\d+)\s+\|\s+Step\s+(\d+)\s+\|\s+Loss:\s+([0-9.eE+\-]+)(?:\s+\|\s+LR:\s+([0-9.eE+\-]+))?'

foreach ($Run in $Runs) {
    if (-not (Test-Path -LiteralPath $Run.Source -PathType Container)) {
        throw "Successful run directory is missing: $($Run.Source)"
    }

    $Destination = Join-Path $ReleaseRoot ("experiments\{0}\{1}" -f $Run.Round, $Run.RunId)
    Ensure-Directory $Destination

    $SourceResults = Join-Path $Run.Source "results"
    $DestinationAudio = Join-Path $Destination "audio"
    if (Test-Path -LiteralPath $DestinationAudio) {
        Remove-Item -LiteralPath $DestinationAudio -Recurse -Force
    }
    Copy-Item -LiteralPath $SourceResults -Destination $DestinationAudio -Recurse -Force

    Copy-RequiredFile (Join-Path $Run.Source "train.log") (Join-Path $Destination "train.log")
    Copy-RequiredFile (Join-Path $Run.Source "config.txt") (Join-Path $Destination "config.original.txt")
    Copy-OptionalFile (Join-Path $Run.Source "infer_all_epochs_23.log") (Join-Path $Destination "inference.log")
    Copy-OptionalFile (Join-Path $Run.Source "ref_audio.sha256") (Join-Path $Destination "ref_audio.sha256")
    Copy-OptionalFile (Join-Path $Run.Source "train_jsonl.sha256") (Join-Path $Destination "train_jsonl.sha256")
    Copy-OptionalFile (Join-Path $Run.Source "test_jsonl.sha256") (Join-Path $Destination "test_jsonl.sha256")

    $SourceLossFigure = Get-ChildItem -LiteralPath $Run.Source -Filter "loss_*.png" -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $SourceLossFigure) {
        Copy-Item -LiteralPath $SourceLossFigure.FullName -Destination (Join-Path $ReleaseRoot ("figures\{0}\{1}.png" -f $Run.Round, $Run.RunId)) -Force
    }

    $AllWav = @(Get-ChildItem -LiteralPath $DestinationAudio -Recurse -Filter "*.wav" -File)
    $CompareWav = @(Get-ChildItem -LiteralPath (Join-Path $DestinationAudio "compare_audio") -Filter "*.wav" -File)
    $TestWav = @(Get-ChildItem -LiteralPath (Join-Path $DestinationAudio "test20") -Recurse -Filter "*.wav" -File)

    if ($AllWav.Count -ne $Run.ExpectedAudio) {
        throw "$($Run.DisplayName): expected $($Run.ExpectedAudio) WAV files, found $($AllWav.Count)"
    }
    if ($CompareWav.Count -ne $Run.ExpectedCompare) {
        throw "$($Run.DisplayName): expected $($Run.ExpectedCompare) compare WAV files, found $($CompareWav.Count)"
    }
    if ($TestWav.Count -ne $Run.ExpectedTest) {
        throw "$($Run.DisplayName): expected $($Run.ExpectedTest) test WAV files, found $($TestWav.Count)"
    }

    $RunInventory.Add([pscustomobject]@{
        round = $Run.Round
        run_id = $Run.RunId
        learning_rate = $Run.LearningRate
        epochs = $Run.Epochs
        scheduler = $Run.Scheduler
        warmup_ratio = $Run.WarmupRatio
        compare_wav = $CompareWav.Count
        test_wav = $TestWav.Count
        total_wav = $AllWav.Count
        seed = 42
        batch_size = 2
        gradient_accumulation_steps = 4
        effective_batch_size = 8
        training_examples = 180
        held_out_test_examples = 20
        reference_audio_id = "wavs/031.wav"
    })

    $RunLossRecords = New-Object System.Collections.Generic.List[object]
    foreach ($Line in Get-Content -LiteralPath (Join-Path $Run.Source "train.log")) {
        $Match = $LossPattern.Match($Line)
        if (-not $Match.Success) {
            continue
        }

        $EpochZeroBased = [int]$Match.Groups[1].Value
        $Step = [int]$Match.Groups[2].Value
        $Loss = [double]::Parse($Match.Groups[3].Value, [Globalization.CultureInfo]::InvariantCulture)
        $LoggedLearningRate = if ($Match.Groups[4].Success) {
            [double]::Parse($Match.Groups[4].Value, [Globalization.CultureInfo]::InvariantCulture)
        } else {
            [double]::Parse($Run.LearningRate, [Globalization.CultureInfo]::InvariantCulture)
        }

        $Record = [pscustomobject]@{
            round = $Run.Round
            run_id = $Run.RunId
            epoch = $EpochZeroBased + 1
            epoch_zero_based = $EpochZeroBased
            step = $Step
            loss = $Loss
            learning_rate = $LoggedLearningRate
            scheduler = $Run.Scheduler
        }
        $LossRecords.Add($Record)
        $RunLossRecords.Add($Record)
    }

    $ExpectedLossRecords = $Run.Epochs * 90
    if ($RunLossRecords.Count -ne $ExpectedLossRecords) {
        throw "$($Run.DisplayName): expected $ExpectedLossRecords loss records, found $($RunLossRecords.Count)"
    }

    foreach ($EpochGroup in ($RunLossRecords | Group-Object epoch | Sort-Object { [int]$_.Name })) {
        $Values = @($EpochGroup.Group | ForEach-Object { [double]$_.loss })
        $Mean = ($Values | Measure-Object -Average).Average
        $Minimum = ($Values | Measure-Object -Minimum).Minimum
        $Maximum = ($Values | Measure-Object -Maximum).Maximum
        $Variance = (($Values | ForEach-Object { [math]::Pow($_ - $Mean, 2) }) | Measure-Object -Average).Average

        $EpochSummaries.Add([pscustomobject]@{
            round = $Run.Round
            run_id = $Run.RunId
            learning_rate = $Run.LearningRate
            scheduler = $Run.Scheduler
            epoch = [int]$EpochGroup.Name
            observations = $Values.Count
            mean_loss = [math]::Round($Mean, 6)
            standard_deviation = [math]::Round([math]::Sqrt($Variance), 6)
            minimum_loss = [math]::Round($Minimum, 6)
            maximum_loss = [math]::Round($Maximum, 6)
            first_loss = [math]::Round([double]$EpochGroup.Group[0].loss, 6)
            last_loss = [math]::Round([double]$EpochGroup.Group[-1].loss, 6)
        })
    }
}

$RunInventory | Export-Csv -LiteralPath (Join-Path $ReleaseRoot "metrics\experiment_matrix.csv") -NoTypeInformation -Encoding UTF8
$LossRecords | Export-Csv -LiteralPath (Join-Path $ReleaseRoot "metrics\loss_steps.csv") -NoTypeInformation -Encoding UTF8
$EpochSummaries | Export-Csv -LiteralPath (Join-Path $ReleaseRoot "metrics\loss_epoch_summary.csv") -NoTypeInformation -Encoding UTF8

$Round1Scripts = @(
    "sft_12hz_ref031_save_all.py",
    "run_one_lr_experiment.sh",
    "infer_one_epoch_compare3_test20.py",
    "infer_all_epochs_23_and_cleanup.sh"
)
foreach ($Name in $Round1Scripts) {
    Copy-RequiredFile (Join-Path $Round1Root "scripts\$Name") (Join-Path $ReleaseRoot "scripts\round1_5epoch\$Name")
}

$Round2Scripts = @(
    "sft_12hz_seed42_cosine_infer_cleanup.py",
    "run_one_lr.sh",
    "run_all_3lr.sh",
    "resume_remaining.sh",
    "plot_final_results.py"
)
foreach ($Name in $Round2Scripts) {
    Copy-RequiredFile (Join-Path $Round2Root "scripts\$Name") (Join-Path $ReleaseRoot "scripts\round2_8epoch_cosine\$Name")
}

$DataFiles = @(
    "metadata\test_ids.txt",
    "metadata\test_raw.jsonl",
    "metadata\text_200.txt",
    "metadata\train_ids.txt",
    "metadata\train_raw.jsonl",
    "processed\train_with_codes.jsonl"
)
foreach ($RelativePath in $DataFiles) {
    Copy-RequiredFile (Join-Path $Round1Root "data\$RelativePath") (Join-Path $ReleaseRoot "data\$RelativePath")
}

$PortableTrain = Get-Content -LiteralPath (Join-Path $Round1Root "data\metadata\train_raw.jsonl") -Encoding UTF8 | ForEach-Object {
    $Item = $_ | ConvertFrom-Json
    [pscustomobject]@{
        split = "train"
        audio_id = [IO.Path]::GetFileNameWithoutExtension([string]$Item.audio)
        text = [string]$Item.text
        reference_audio_id = "wavs/031.wav"
    }
}
$PortableTest = Get-Content -LiteralPath (Join-Path $Round1Root "data\metadata\test_raw.jsonl") -Encoding UTF8 | ForEach-Object {
    $Item = $_ | ConvertFrom-Json
    [pscustomobject]@{
        split = "test"
        audio_id = [IO.Path]::GetFileNameWithoutExtension([string]$Item.audio)
        text = [string]$Item.text
        reference_audio_id = "wavs/031.wav"
    }
}
$PortableTrain | Export-Csv -LiteralPath (Join-Path $ReleaseRoot "data\train_texts.csv") -NoTypeInformation -Encoding UTF8
$PortableTest | Export-Csv -LiteralPath (Join-Path $ReleaseRoot "data\test_texts.csv") -NoTypeInformation -Encoding UTF8

Copy-RequiredFile (Join-Path $Round1Root "core_versions.txt") (Join-Path $ReleaseRoot "environment\round1_5epoch\core_versions.txt")
Copy-RequiredFile (Join-Path $Round1Root "environment_freeze.txt") (Join-Path $ReleaseRoot "environment\round1_5epoch\pip_freeze.txt")
Copy-RequiredFile (Join-Path $Round1Root "nvidia_smi.txt") (Join-Path $ReleaseRoot "environment\round1_5epoch\nvidia_smi.txt")
Copy-RequiredFile (Join-Path $Round2Root "reproducibility\core_versions.txt") (Join-Path $ReleaseRoot "environment\round2_8epoch_cosine\core_versions.txt")
Copy-RequiredFile (Join-Path $Round2Root "reproducibility\pip_freeze.txt") (Join-Path $ReleaseRoot "environment\round2_8epoch_cosine\pip_freeze.txt")
Copy-RequiredFile (Join-Path $Round2Root "reproducibility\nvidia_smi.txt") (Join-Path $ReleaseRoot "environment\round2_8epoch_cosine\nvidia_smi.txt")

Copy-RequiredFile (Join-Path $Round1Root "loss_lr_comparison_ref031.png") (Join-Path $ReleaseRoot "figures\round1_5epoch\loss_comparison.png")
Copy-RequiredFile (Join-Path $Round2Root "loss_comparison.png") (Join-Path $ReleaseRoot "figures\round2_8epoch_cosine\loss_comparison.png")
Copy-RequiredFile (Join-Path $Round2Root "lr_schedule.png") (Join-Path $ReleaseRoot "figures\round2_8epoch_cosine\lr_schedule.png")
Copy-OptionalFile (Join-Path $Round2Root "loss_A_lr5e-7.png") (Join-Path $ReleaseRoot "figures\round2_8epoch_cosine\lr5e-7.png")
Copy-OptionalFile (Join-Path $Round2Root "loss_B_lr1e-6.png") (Join-Path $ReleaseRoot "figures\round2_8epoch_cosine\lr1e-6.png")
Copy-OptionalFile (Join-Path $Round2Root "loss_C_lr1_5e-6.png") (Join-Path $ReleaseRoot "figures\round2_8epoch_cosine\lr1.5e-6.png")

$MarkdownLines = New-Object System.Collections.Generic.List[string]
$MarkdownLines.Add("| Round | Learning rate | Scheduler | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Epoch 6 | Epoch 7 | Epoch 8 |")
$MarkdownLines.Add("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
foreach ($Run in $Runs) {
    $RunEpochs = @($EpochSummaries | Where-Object { $_.round -eq $Run.Round -and $_.run_id -eq $Run.RunId } | Sort-Object epoch)
    $Cells = @()
    for ($Epoch = 1; $Epoch -le 8; $Epoch++) {
        $Item = $RunEpochs | Where-Object { $_.epoch -eq $Epoch } | Select-Object -First 1
        $Cells += if ($null -eq $Item) { "-" } else { ([double]$Item.mean_loss).ToString("F4", [Globalization.CultureInfo]::InvariantCulture) }
    }
    $RoundLabel = if ($Run.Round -eq "round1_5epoch") { "5 epoch" } else { "8 epoch" }
    $MarkdownLines.Add("| $RoundLabel | $($Run.LearningRate) | $($Run.Scheduler) | $($Cells -join ' | ') |")
}
$MarkdownLines.Add("")
$MarkdownLines.Add("> Each value is the arithmetic mean of 90 micro-batch losses in that epoch. It describes optimization only and is not an audio-quality, completeness, or speaker-similarity score.")
Set-Content -LiteralPath (Join-Path $ReleaseRoot "metrics\loss_epoch_means.md") -Value $MarkdownLines -Encoding UTF8

Write-Output "Release data assembled successfully."
Write-Output ("Loss records: {0}" -f $LossRecords.Count)
Write-Output ("Epoch summaries: {0}" -f $EpochSummaries.Count)
Write-Output ("Generated WAV files: {0}" -f (($RunInventory | Measure-Object total_wav -Sum).Sum))
