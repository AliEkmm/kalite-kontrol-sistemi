<#
====================================================================
 active.ps1  -  Tek tikla calistirma scripti
====================================================================
 Bu script projeyi sifirdan calisir hale getirir ve web uygulamasini
 baslatir. Eksik olan adimlari otomatik tamamlar, hazir olanlari atlar:

   1. Sanal ortam (.venv) olusturur
   2. Bagimliliklari kurar (requirements.txt + anomalib icin gerekli
      ek paketler + pandas<3 uyumlulugu)
   3. anomalib'in Windows symlink hatasi icin kucuk bir yama uygular
   4. MVTec 'capsule' veri setini indirir/yerlestirir
   5. PatchCore modelini egitir (yoksa)
   6. Flask web uygulamasini baslatir ve tarayicida acar

 KULLANIM:
   Sag tik -> "Run with PowerShell"
   veya bir PowerShell penceresinde:  .\active.ps1

 Calismazsa (execution policy hatasi) sunu deneyin:
   powershell -ExecutionPolicy Bypass -File .\active.ps1

 Parametreler:
   -Category <ad>   Egitilecek/kullanilacak MVTec kategorisi (vars: capsule)
   -Retrain         Model varsa bile yeniden egitir
   -NoBrowser       Tarayiciyi otomatik acma
   -Port <n>        Web sunucusu portu (vars: 5000)
====================================================================
#>

[CmdletBinding()]
param(
    [string]$Category = "capsule",
    [switch]$Retrain,
    [switch]$NoBrowser,
    [int]$Port = 5000
)

$ErrorActionPreference = "Stop"

# --- Yol tanimlari -------------------------------------------------
$Root      = $PSScriptRoot
$VenvPy    = Join-Path $Root ".venv\Scripts\python.exe"
$ReqFile   = Join-Path $Root "requirements.txt"
$DataDir   = Join-Path $Root "data\MVTecAD"
$CatDir    = Join-Path $DataDir $Category
$TrainGood = Join-Path $CatDir "train\good"
$ModelFile = Join-Path $Root ("models\patchcore_{0}.pt" -f $Category)
$SetupMark = Join-Path $Root ".venv\.setup_done"

# Tek-kategori indirme linkleri (otomatik MVTec indirme bozuk oldugu icin
# Hugging Face aynalarini kullaniyoruz).
$CategoryUrls = @{
    "capsule" = "https://huggingface.co/datasets/alexsu52/mvtec_capsule/resolve/main/capsule.tar.xz"
}

function Write-Step($msg)  { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "    [OK] $msg" -ForegroundColor Green }
function Write-Info($msg)  { Write-Host "    $msg" -ForegroundColor Gray }
function Write-Warn2($msg) { Write-Host "    [UYARI] $msg" -ForegroundColor Yellow }

Set-Location $Root
Write-Host "=============================================================" -ForegroundColor DarkCyan
Write-Host " Gorsel Kalite Kontrol Sistemi - Tek tikla baslatma" -ForegroundColor White
Write-Host " Kategori: $Category | Port: $Port" -ForegroundColor White
Write-Host "=============================================================" -ForegroundColor DarkCyan

# --- 0) Sunucu zaten calisiyor mu? --------------------------------
$listening = $null
try { $listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue } catch {}
if ($listening) {
    Write-Step "Sunucu zaten calisiyor (port $Port)."
    $url = "http://127.0.0.1:$Port"
    if (-not $NoBrowser) { Start-Process $url }
    Write-Ok "Tarayicida acin: $url"
    return
}

# --- 1) Sanal ortam ------------------------------------------------
Write-Step "Sanal ortam (.venv) kontrol ediliyor..."
if (-not (Test-Path $VenvPy)) {
    Write-Info "Olusturuluyor (python -m venv .venv)..."
    python -m venv .venv
    & $VenvPy -m pip install --upgrade pip | Out-Null
    Write-Ok "Sanal ortam olusturuldu."
} else {
    Write-Ok "Sanal ortam mevcut."
}

# --- 2) Bagimliliklar ---------------------------------------------
Write-Step "Bagimliliklar kontrol ediliyor..."
if ((-not (Test-Path $SetupMark)) -or $Retrain) {
    Write-Info "requirements.txt kuruluyor (ilk sefer birkac dakika surebilir)..."
    & $VenvPy -m pip install --no-cache-dir -r $ReqFile
    Write-Info "anomalib icin gerekli ek paketler kuruluyor..."
    & $VenvPy -m pip install --no-cache-dir python-dotenv open-clip-torch requests tensorboard "pandas<3"
    if ($LASTEXITCODE -ne 0) { Write-Warn2 "Bazi paketler kurulamadi; yine de devam ediliyor."; }
    New-Item -ItemType File -Force -Path $SetupMark | Out-Null
    Write-Ok "Bagimliliklar hazir."
} else {
    Write-Ok "Bagimliliklar daha once kurulmus (atlandi)."
}

# --- 3) anomalib symlink yamasi (Windows) -------------------------
Write-Step "anomalib Windows symlink yamasi kontrol ediliyor..."
$patchPy = @'
import pathlib
import anomalib
p = pathlib.Path(anomalib.__file__).parent / "utils" / "path.py"
s = p.read_text(encoding="utf-8")
if "ONE_CLICK_SYMLINK_PATCH" in s:
    print("zaten uygulanmis")
else:
    old = "    latest_link_path.symlink_to(new_version_dir, target_is_directory=True)\n\n    return latest_link_path"
    new = (
        "    try:  # ONE_CLICK_SYMLINK_PATCH\n"
        "        latest_link_path.symlink_to(new_version_dir, target_is_directory=True)\n"
        "    except OSError:\n"
        "        return new_version_dir\n\n    return latest_link_path"
    )
    if old in s:
        p.write_text(s.replace(old, new), encoding="utf-8")
        print("uygulandi")
    else:
        print("hedef satir bulunamadi")
'@
$patchOut = & $VenvPy -c $patchPy
Write-Info "Yama durumu: $patchOut"

# --- 4) Veri seti --------------------------------------------------
Write-Step "Veri seti kontrol ediliyor ($Category)..."
$haveData = (Test-Path $TrainGood) -and ((Get-ChildItem $TrainGood -Filter *.png -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0)
if (-not $haveData) {
    if (-not $CategoryUrls.ContainsKey($Category)) {
        Write-Warn2 "'$Category' icin otomatik indirme linki tanimli degil."
        Write-Warn2 "Lutfen veriyi elle '$CatDir' altina yerlestirin (train/good, test/, ground_truth/)."
        throw "Veri seti bulunamadi: $Category"
    }
    $url = $CategoryUrls[$Category]
    $tar = Join-Path $DataDir ("{0}.tar.xz" -f $Category)
    New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
    Write-Info "Indiriliyor: $url"
    curl.exe -L -o $tar $url
    Write-Info "Cikariliyor..."
    tar -xf $tar -C $DataDir
    Remove-Item $tar -Force -ErrorAction SilentlyContinue
    Write-Ok "Veri seti hazir."
} else {
    Write-Ok "Veri seti mevcut."
}

# --- 5) Model ------------------------------------------------------
Write-Step "Model kontrol ediliyor..."
if ((-not (Test-Path $ModelFile)) -or $Retrain) {
    Write-Info "Model egitiliyor (python src/train.py --category $Category)..."
    Write-Info "Bu islem CPU'da birkac dakika surebilir; lutfen bekleyin."
    & $VenvPy (Join-Path $Root "src\train.py") --category $Category
    if (-not (Test-Path $ModelFile)) { throw "Egitim tamamlandi ama model dosyasi olusmadi: $ModelFile" }
    Write-Ok "Model egitildi: $ModelFile"
} else {
    Write-Ok "Egitilmis model mevcut (atlandi)."
}

# --- 6) Web uygulamasini baslat -----------------------------------
Write-Step "Web uygulamasi baslatiliyor..."
$url = "http://127.0.0.1:$Port"

if (-not $NoBrowser) {
    # Sunucu ayaga kalkinca tarayiciyi acmak icin arka planda bekleyen is.
    Start-Job -ScriptBlock {
        param($u, $p)
        for ($i = 0; $i -lt 60; $i++) {
            Start-Sleep -Milliseconds 700
            try {
                $c = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
                if ($c) { Start-Process $u; break }
            } catch {}
        }
    } -ArgumentList $url, $Port | Out-Null
}

Write-Ok "Tarayicida acilacak: $url"
Write-Info "Durdurmak icin bu pencerede Ctrl+C yapin."
$env:PYTHONUNBUFFERED = "1"
& $VenvPy (Join-Path $Root "src\app.py")
