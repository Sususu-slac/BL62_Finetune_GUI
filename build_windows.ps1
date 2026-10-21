$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
python -m PyInstaller --noconfirm --clean --windowed --onedir --name BL62_Finetune_GUI --icon assets\BL62_Finetune.ico --add-data 'assets;assets' --exclude-module torch --exclude-module torchvision --exclude-module matplotlib --exclude-module scipy --exclude-module pandas --exclude-module IPython --exclude-module pytest app.py
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed' }
$iscc = Join-Path $env:LOCALAPPDATA 'BL62-BuildTools\InnoSetup\ISCC.exe'
if (-not (Test-Path $iscc)) { $iscc = 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe' }
if (Test-Path $iscc) {
 & $iscc installer.iss
 if ($LASTEXITCODE -ne 0) { throw 'Installer build failed' }
}
