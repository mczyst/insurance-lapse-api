# ==============================================================================
# SCRIPT OTOMASI MENJALANKAN BACKEND API LOKAL
# ==============================================================================
$Host.UI.RawUI.WindowTitle = "Insurance Lapse Prediction API Server"

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  MENYALAKAN SERVER FASTAPI BACKEND (127.0.0.1:8000) " -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

# 1. Pastikan direktori kerja berada di folder lokasi script ini
$ProjectDir = $PSScriptRoot
Set-Location -Path $ProjectDir

# 2. Path ke eksekusi Python / Uvicorn di Virtual Environment
$UvicornExe = Join-Path $ProjectDir "venv\Scripts\uvicorn.exe"

# Validasi folder venv
if (-not (Test-Path $UvicornExe)) {
    Write-Host "[ERROR] Virtual Environment (venv) tidak ditemukan di: $ProjectDir\venv" -ForegroundColor Red
    Write-Host "Pastikan venv sudah dibuat sebelum menjalankan script ini." -ForegroundColor Yellow
    Pause
    Exit
}

# 3. Buka browser otomatis ke Swagger UI setelah delay 2 detik di background
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 2
    Start-Process "http://127.0.0.1:8000/docs"
} | Out-Null

Write-Host "[INFO] Mengarahkan ke browser: http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "[INFO] Server aktif. Tekan 'Ctrl + C' untuk menghentikan server.`n" -ForegroundColor Gray

# 4. Jalankan Uvicorn Server
& $UvicornExe main:app --reload --host 127.0.0.1 --port 8000