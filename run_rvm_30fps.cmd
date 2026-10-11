@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE="
if /I "%CONDA_DEFAULT_ENV%"=="RVM_4090_2" if exist "%CONDA_PREFIX%\python.exe" set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\anaconda3\envs\RVM_4090_2\python.exe" set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\RVM_4090_2\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\miniconda3\envs\RVM_4090_2\python.exe" set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\RVM_4090_2\python.exe"
if not defined PYTHON_EXE if exist "%ProgramData%\anaconda3\envs\RVM_4090_2\python.exe" set "PYTHON_EXE=%ProgramData%\anaconda3\envs\RVM_4090_2\python.exe"
if defined PYTHON_EXE goto :launch

echo [ERROR] Cannot find python.exe for the RVM_4090_2 Conda environment.
echo Activate RVM_4090_2 in Anaconda Prompt and run this file again.
pause
exit /b 1

:launch
echo [INFO] Python: %PYTHON_EXE%

rem Input FPS: RVM_FPS env var if set, otherwise ask (Enter = 30).
rem Must match the OBS Video FPS setting (OBS Virtual Camera only offers that rate).
if defined RVM_FPS goto :check_fps
:ask_fps
set "RVM_FPS=30"
set /p "RVM_FPS=Input FPS number (must match OBS: 25 / 30 / 30000/1001 / 60), Enter = 30: "
:check_fps
rem Accept only a number (25, 29.97) or a ratio (30000/1001).
echo(%RVM_FPS%| findstr /r /x "[0-9][0-9]*[.]*[0-9]* [0-9][0-9]*/[0-9][0-9]*" >nul
if errorlevel 1 (
  echo [WARN] "%RVM_FPS%" is not a frame rate. Enter a number such as 30.
  goto :ask_fps
)
echo [INFO] Input FPS: %RVM_FPS%

"%PYTHON_EXE%" "rvm_main.py" ^
  --ffmpeg-in dshow ^
  --dshow-name "OBS Virtual Camera" ^
  --force-fps %RVM_FPS% ^
  --force-size 1920x1080 ^
  --raw-format bgr24 ^
  --preview ^
  --preview-scale 0.5 ^
  --spout-out 1 ^
  --spout-name "RVM_OUTPUT" ^
  --spout-mode matte ^
  --spout-bg 1 ^
  --spout-bg-name "RVM_BG" ^
  --spout-bg-mode raw ^
  --person-distance 1 ^
  --dist-hfov 60 ^
  --energy 1 ^
  --energy-color "0,80,200" ^
  --energy-blend normal ^
  --energy-width 40 ^
  --energy-blur 30 ^
  --energy-gain 8 ^
  --energy-on-bg 1 ^
  --energy-color-lerp-ms 800 %*

set "RVM_EXIT=%ERRORLEVEL%"
if "%RVM_EXIT%"=="0" goto :done
echo.
echo [ERROR] RVM exited with code %RVM_EXIT%.
pause

:done
exit /b %RVM_EXIT%
