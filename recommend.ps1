cd $PSScriptRoot
Write-Host "========================================"
Write-Host "  K-Fashion Recommendation System"
Write-Host "========================================"
Write-Host "Put image in: $PSScriptRoot\data\"
Write-Host ""
$IMG    = Read-Host "Image filename (e.g. photo.jpg)"
$HEIGHT = Read-Host "Height (cm)"
$WEIGHT = Read-Host "Weight (kg)"
Write-Host "Fit preference: slim / regular / over"
$FIT    = Read-Host "Fit (default: regular)"
if ($FIT -eq "") { $FIT = "regular" }
Write-Host ""
python run_pipeline.py $IMG $HEIGHT $WEIGHT $FIT
Write-Host ""
Read-Host "Press Enter to exit"