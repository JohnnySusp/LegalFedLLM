param(
    [string]$PythonCommand = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    throw "This setup helper is for native Windows only."
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvRoot = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
$Requirements = Join-Path $RepoRoot "requirements-desktop.txt"
$TorchIndex = "https://download.pytorch.org/whl/cu132"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
    }
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "+ $PythonCommand -m venv $VenvRoot"
    Invoke-Checked $PythonCommand -m venv $VenvRoot
}

Write-Host "+ $VenvPython -m pip install --upgrade pip"
Invoke-Checked $VenvPython -m pip install --upgrade pip

Write-Host "+ $VenvPython -m pip install -r $Requirements"
Invoke-Checked $VenvPython -m pip install -r $Requirements

$TorchProbe = 'import torch, sys; sys.exit(0 if torch.__version__ == "2.13.0+cu132" and torch.cuda.is_available() and torch.version.cuda == "13.2" and torch.cuda.is_bf16_supported() else 1)'
$TorchProbe | & $VenvPython -
if ($LASTEXITCODE -eq 0) {
    Write-Host "+ pinned Windows CUDA Torch is already installed"
} else {
    Write-Host "+ replacing the default CPU Torch wheel with the pinned CUDA 13.2 wheel"
    Invoke-Checked $VenvPython -m pip install --force-reinstall --no-deps torch==2.13.0 --index-url $TorchIndex
}

$Verification = @'
import torch

expected_version = "2.13.0+cu132"
if torch.__version__ != expected_version:
    raise SystemExit(f"expected torch {expected_version}, found {torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to the pinned Windows Torch build")
if torch.version.cuda != "13.2":
    raise SystemExit(f"expected Torch CUDA 13.2, found {torch.version.cuda!r}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("the selected Windows GPU does not report BF16 support")
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("bf16:", torch.cuda.is_bf16_supported())
'@

Write-Host "+ verifying pinned Windows CUDA runtime"
$Verification | & $VenvPython -
if ($LASTEXITCODE -ne 0) {
    throw "Windows CUDA runtime verification failed."
}

Write-Host "LegalFedLLM Windows desktop source environment is ready."
