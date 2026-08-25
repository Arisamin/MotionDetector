# Automated 1-click launcher for MotionDetector inside BlueStacks Termux

$adbPath = "C:\Program Files\BlueStacks_nxt\HD-Adb.exe"
$hdPlayer = "C:\Program Files\BlueStacks_nxt\HD-Player.exe"

if (-not (Test-Path $adbPath)) {
    Write-Error "Could not find BlueStacks ADB at: $adbPath"
    exit 1
}

# Auto-detect active BlueStacks emulator instance
$devices = & $adbPath devices | Where-Object { $_ -match "\bdevice\b" -and $_ -notmatch "List of" }

if (-not $devices) {
    Write-Host "[INFO] BlueStacks is not running. Launching BlueStacks Nougat32 instance..." -ForegroundColor Yellow
    if (Test-Path $hdPlayer) {
        Start-Process $hdPlayer -ArgumentList "--instance Nougat32"
        Write-Host "[INFO] Waiting 20 seconds for BlueStacks to boot..." -ForegroundColor Yellow
        Start-Sleep -Seconds 20
    }
    & $adbPath connect 127.0.0.1:5555 | Out-Null
    Start-Sleep -Seconds 3
    $devices = & $adbPath devices | Where-Object { $_ -match "\bdevice\b" -and $_ -notmatch "List of" }
}

$deviceId = ($devices | ForEach-Object { ($_ -split "\s+")[0] } | Select-Object -First 1)

if (-not $deviceId) {
    Write-Host "[INFO] Waiting an additional 10 seconds for ADB device..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    $devices = & $adbPath devices | Where-Object { $_ -match "\bdevice\b" -and $_ -notmatch "List of" }
    $deviceId = ($devices | ForEach-Object { ($_ -split "\s+")[0] } | Select-Object -First 1)
}

if (-not $deviceId) {
    Write-Error "No active BlueStacks device found. Please make sure BlueStacks is started."
    exit 1
}

Write-Host "[OK] Connected to BlueStacks device: $deviceId" -ForegroundColor Green
Write-Host "[INFO] Syncing files to BlueStacks..." -ForegroundColor Cyan
& $adbPath -s $deviceId push C:\MyData\Git\MotionDetector\src /sdcard/MotionDetector/ | Out-Null
& $adbPath -s $deviceId push C:\MyData\Git\MotionDetector\android /sdcard/MotionDetector/ | Out-Null
if (Test-Path "C:\MyData\Git\MotionDetector\config.json") {
    & $adbPath -s $deviceId push C:\MyData\Git\MotionDetector\config.json /sdcard/MotionDetector/ | Out-Null
}

Write-Host "[INFO] Restarting Termux with updated code..." -ForegroundColor Cyan
& $adbPath -s $deviceId shell "am force-stop com.termux" | Out-Null
Start-Sleep -Seconds 1
& $adbPath -s $deviceId shell "am start -n com.termux/.app.TermuxActivity" | Out-Null
Start-Sleep -Seconds 2

Write-Host "[INFO] Launching Orchestrator inside Termux..." -ForegroundColor Cyan

# Send execution command to Termux
& $adbPath -s $deviceId shell "input text bash%s/sdcard/MotionDetector/android/run.sh"
& $adbPath -s $deviceId shell "input keyevent 66"

Start-Sleep -Seconds 3

Write-Host "[INFO] Bringing Reolink app to foreground..." -ForegroundColor Cyan
& $adbPath -s $deviceId shell "am start -n com.mcu.reolink/com.android.bc.login.WelcomeActivity" | Out-Null

Write-Host "[SUCCESS] MotionDetector Orchestrator started with Reolink in foreground!" -ForegroundColor Green
