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
$Verifier = Join-Path $RepoRoot "scripts\verify_windows_desktop.py"
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

$PythonProbe = @'
import struct
import sys

if sys.version_info[:2] != (3, 14):
    raise SystemExit(f"LegalFedLLM requires Python 3.14 x64; found {sys.version_info.major}.{sys.version_info.minor}")
if struct.calcsize("P") * 8 != 64:
    raise SystemExit("LegalFedLLM requires 64-bit Python")
print(sys.executable)
'@

Write-Host "+ verifying system Python 3.14 x64"
$PythonProbe | & $PythonCommand -
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.14 x64 verification failed."
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "+ $PythonCommand -m venv $VenvRoot"
    Invoke-Checked $PythonCommand -m venv $VenvRoot
}

Write-Host "+ $VenvPython -m pip install --upgrade pip"
Invoke-Checked $VenvPython -m pip install --upgrade pip

$TorchProbe = @'
try:
    import torch
except Exception:
    raise SystemExit(1)

ok = (
    torch.__version__ == "2.13.0+cu132"
    and torch.cuda.is_available()
    and torch.version.cuda == "13.2"
    and torch.cuda.is_bf16_supported()
)
raise SystemExit(0 if ok else 1)
'@

$TorchProbe | & $VenvPython -
if ($LASTEXITCODE -eq 0) {
    Write-Host "+ pinned Windows CUDA Torch is already installed"
} else {
    Write-Host "+ installing pinned Windows CUDA Torch 2.13.0+cu132"
    Invoke-Checked $VenvPython -m pip install --no-deps torch==2.13.0+cu132 --index-url $TorchIndex
}

Write-Host "+ $VenvPython -m pip install -r $Requirements"
Invoke-Checked $VenvPython -m pip install -r $Requirements

if (-not (Test-Path $Verifier)) {
    throw "Windows runtime verifier is missing: $Verifier"
}

Write-Host "+ verifying LegalFedLLM Windows runtime"
Invoke-Checked $VenvPython $Verifier

Write-Host "LegalFedLLM Windows desktop environment is ready."
