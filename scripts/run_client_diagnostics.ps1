param(
    [Parameter(Mandatory=$true)][string]$ProfileId,
    [string]$DataRoot = $(if ($env:LEGALFEDLLM_DATA_ROOT) { $env:LEGALFEDLLM_DATA_ROOT } else { Join-Path (Get-Location) "LegalFedLLM-data" })
)

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Root = Split-Path -Parent $PSScriptRoot
$Modes = @("state", "gpu")

if (Get-Command wt.exe -ErrorAction SilentlyContinue) {
    $arguments = @()
    foreach ($mode in $Modes) {
        if ($arguments.Count -gt 0) { $arguments += ";" }
        $arguments += "new-tab"
        $arguments += "--title"
        $arguments += "LegalFedLLM $mode"
        $arguments += "powershell.exe"
        $arguments += "-NoExit"
        $arguments += "-Command"
        $arguments += "Set-Location '$Root'; & '$Python' -m desktop.app --monitor $mode --profile-id '$ProfileId' --data-root '$DataRoot'"
    }
    & wt.exe @arguments
    exit $LASTEXITCODE
}

foreach ($mode in $Modes) {
    Start-Process powershell.exe -ArgumentList @(
        "-NoExit",
        "-Command",
        "Set-Location '$Root'; & '$Python' -m desktop.app --monitor $mode --profile-id '$ProfileId' --data-root '$DataRoot'"
    )
}
