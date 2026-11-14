@echo off
REM Camouflage detection demo. Usage: run_demo.bat [yolo^|baseline] [seed]
set MODE=%1
if "%MODE%"=="" set MODE=yolo
set SEED=%2
if "%SEED%"=="" set SEED=42
wsl -d Ubuntu-24.04 -- bash /mnt/c/ES-proj/camo_ws/scripts/run_demo.sh %MODE% %SEED%
