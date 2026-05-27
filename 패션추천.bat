@echo off
call conda activate kfashion

echo ========================================
echo   K-Fashion Recommendation System
echo ========================================
echo.
set /p IMG=Image filename (e.g. photo.jpg):
set /p HEIGHT=Height (cm):
set /p WEIGHT=Weight (kg):
set /p FIT=Fit (slim/regular/over, default=regular):
if "%FIT%"=="" set FIT=regular

echo.
python "%~dp0run_pipeline.py" %IMG% %HEIGHT% %WEIGHT% %FIT%
echo.
pause
