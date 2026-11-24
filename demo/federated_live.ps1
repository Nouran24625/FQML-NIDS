# Live federation demo launcher (Windows PowerShell 5). Opens one window for the server and one per site.
# Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File demo\federated_live.ps1
#   powershell -ExecutionPolicy Bypass -File demo\federated_live.ps1 -Sites 5 -Rounds 20 -Aggregator median -Split sitesfam
#   powershell -ExecutionPolicy Bypass -File demo\federated_live.ps1 -Attacker 2 -Attack class_suppression -Target Web
#   powershell -ExecutionPolicy Bypass -File demo\federated_live.ps1 -Scenario attack     (honest, attack, defence,
#                                                                                         defence-cost, quantum)
# Every process runs in one container ("fqml-fed-demo"), so they talk over localhost through the PQ channel.
# Stop: close the windows, then  docker rm -f fqml-fed-demo
param(
    [int]$Sites = 3,
    [int]$Rounds = 30,
    [string]$Model = "mlp_p3000",    # mlp_p3000, mlp_p512, mlp (VQC-sized), vqc, vqc_lin, vqc_noent
    [string]$Split = "sites",        # iid, dir0.5, sites, sitesfam
    [string]$Aggregator = "fedavg",  # fedavg, median, trimmed_mean
    [string]$Config = "default",     # post-quantum settings: default, classical, none
    [int]$Attacker = 0,              # which site attacks (0 = nobody)
    [string]$Attack = "class_suppression",
    [string]$Target = "Web",
    [int]$AttackFrom = 10,           # the attacker behaves honestly until this round
    [double]$RoundDelay = 1.0,       # seconds between rounds, so the audience can follow the dashboard
    [string]$Data = "",              # folder holding ciciot2023_v1/ (default: data\interim of this repo or D:\FQML-NIDS)
    [string]$Scenario = ""           # one-word presets: honest, attack, defence, defence-cost, quantum (see below)
)
$ErrorActionPreference = "Stop"

# Presets for showing it (each sets the options it needs; anything else keeps its value)
switch ($Scenario) {
    ""             { }
    "honest"       { $Attacker = 0 }                                                   # 3 honest sites
    "attack"       { $Attacker = 2; $Attack = "class_suppression"; $Target = "Web"; $AttackFrom = 12 }
    "defence"      { $Attacker = 2; $Attack = "class_suppression"; $Target = "Web"; $AttackFrom = 12; $Aggregator = "median" }
    "defence-cost" { $Attacker = 0; $Aggregator = "median"; $Split = "sitesfam" }      # median with no attacker
    "quantum"      { $Attacker = 0; $Model = "vqc"; $Rounds = 20 }                     # the 64-parameter VQC
    default        { throw "unknown scenario '$Scenario': honest, attack, defence, defence-cost, quantum" }
}
if ($Scenario -ne "") { Write-Host "Scenario: $Scenario" -ForegroundColor Cyan }
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
if ($Data -eq "") {
    if (Test-Path "$repo\data\interim\ciciot2023_v1") { $Data = "$repo\data\interim" } else { $Data = "D:\FQML-NIDS\data\interim" }
}
$Data = $Data -replace '\\', '/'

Write-Host "Starting the demo container (data from $Data) ..." -ForegroundColor Cyan
docker rm -f fqml-fed-demo 2>$null | Out-Null
docker compose -f docker/compose.yml run -d --name fqml-fed-demo -p 8050:8050 -p 9443:9443 -v "${Data}:/workspace/data/interim:ro" dev sleep infinity | Out-Null
docker exec fqml-fed-demo python -m demo.federated_live keys --sites $Sites --config $Config

$server = "docker exec fqml-fed-demo python -m demo.federated_live server --sites $Sites --rounds $Rounds --model $Model --split $Split --aggregator $Aggregator --config $Config --round-delay $RoundDelay --keep-open"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='FED SERVER'; $server"
Start-Sleep -Seconds 3
for ($k = 1; $k -le $Sites; $k++) {
    $name = "site-{0:D2}" -f $k
    $site = "docker exec fqml-fed-demo python -m demo.federated_live site --name $name --config $Config"
    if ($k -eq $Attacker) { $site += " --attack $Attack --target $Target --attack-from $AttackFrom" }
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='$name'; $site"
}
Start-Sleep -Seconds 4
Start-Process "http://localhost:8050"
Write-Host "Opened the dashboard (http://localhost:8050), the server window and $Sites site windows. Stop with: docker rm -f fqml-fed-demo" -ForegroundColor Cyan
