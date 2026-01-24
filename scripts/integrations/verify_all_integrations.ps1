# TensorGuardFlow Integration Verification Script (PowerShell)

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "             TensorGuardFlow Integration Framework - Scorecard" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

# 1. Environment Setup
$env:PYTHONPATH = "src"
$env:TG_SIMULATION = "true"

$corePass = 0
$optionalPass = 0
$totalFailed = 0

function Run-Test {
    param (
        [string]$Name,
        [string]$Command,
        [bool]$Required
    )
    
    Write-Host -NoNewline "Running $Name... "
    $result = Invoke-Expression "$Command 2>&1"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[PASS]" -ForegroundColor Green
        if ($Required) { $script:corePass++ } else { $script:optionalPass++ }
    } else {
        Write-Host "[FAIL]" -ForegroundColor Red
        if ($Required) { $script:totalFailed++ }
        # Write-Host $result
    }
}

# 2. RUN TESTS
Write-Host "`n--- Layer 1: Contract & Schema Tests ---" -ForegroundColor Yellow
Run-Test "Config Schema Validation" "python tests/integration/pipeline/test_config_schemas.py" $true
Run-Test "Connector Protocols" "python tests/integration/pipeline/test_framework_contracts.py" $true

Write-Host "`n--- Layer 2: Local E2E Pipeline ---" -ForegroundColor Yellow
Run-Test "Local RunOnce End-to-End" "python tests/integration/pipeline/test_route_run_once_local_end_to_end.py" $true

Write-Host "`n--- Layer 3: Cloud Smoke Tests (Optional) ---" -ForegroundColor Yellow
if ([string]::IsNullOrEmpty($env:AWS_ACCESS_KEY_ID)) {
    Write-Host "S3 Feed / SageMaker: [SKIP] (Missing AWS Credentials)" -ForegroundColor Gray
}

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "FINAL SCORECARD" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Core Integrations Passed: $corePass / 3"
Write-Host "Optional Integrations Passed: $optionalPass"
Write-Host "Blocking Failures: $totalFailed"

if ($totalFailed -eq 0) {
    Write-Host ">>> SYSTEM STATUS: GREEN <<<" -ForegroundColor Green
} else {
    Write-Host ">>> SYSTEM STATUS: RED <<<" -ForegroundColor Red
    exit 1
}
