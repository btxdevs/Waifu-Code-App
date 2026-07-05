@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ============================================================================
REM Package the app into a standalone Windows onedir build (PyInstaller).
REM
REM   build-app.bat            full build (Vite UI + PyInstaller)
REM   build-app.bat /skipui    reuse the existing dist\ (skip npm run build)
REM
REM Output: build\pyi-dist\WaifuCodeApp\Waifu Code App.exe
REM ============================================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "SKIPUI="
if /I "%~1"=="/skipui" set "SKIPUI=1"
if /I "%~1"=="-skipui" set "SKIPUI=1"

set "VENVPY=%ROOT%\python\.venv\Scripts\python.exe"
if not exist "%VENVPY%" (
  echo venv python not found at "%VENVPY%" ^(create it: see run-app.bat^)
  exit /b 1
)

set "DISTPATH=%ROOT%\build\pyi-dist"
set "WORKPATH=%ROOT%\build\pyi-work"
set "APPDIR=%DISTPATH%\WaifuCodeApp"

REM --- 1. Vite UI ---
if defined SKIPUI (
  echo Skipping UI build ^(/skipui^); reusing existing dist\.
  if not exist "%ROOT%\dist\index.html" ( echo dist\index.html missing -- run without /skipui. & exit /b 1 )
) else (
  echo Building Vite UI ^(npm run build^)...
  pushd "%ROOT%"
  call npm run build
  if errorlevel 1 ( echo npm run build failed & popd & exit /b 1 )
  popd
)

REM --- 2. PyInstaller ---
REM A previously-built app still running would hold app.log / dlls open in the output
REM dir and make PyInstaller's clean step fail. Stop any first.
taskkill /F /IM "Waifu Code App.exe" >nul 2>&1

echo Running PyInstaller (this takes a few minutes)...
pushd "%ROOT%"
"%VENVPY%" -m PyInstaller "app.spec" --noconfirm --distpath "%DISTPATH%" --workpath "%WORKPATH%"
set "PYIERR=%ERRORLEVEL%"
popd
if not "%PYIERR%"=="0" ( echo PyInstaller failed ^(%PYIERR%^) & exit /b 1 )
if not exist "%APPDIR%" ( echo Expected build output not found at "%APPDIR%" & exit /b 1 )

REM --- 3. Stage external resources beside the exe ---
echo Staging resources next to the exe...

REM UI: dist\ -> app\dist\
if exist "%APPDIR%\dist" rmdir /S /Q "%APPDIR%\dist"
robocopy "%ROOT%\dist" "%APPDIR%\dist" /E /NFL /NDL /NJH /NJS /NP >nul
if errorlevel 8 ( echo Failed to stage dist\ & exit /b 1 )

REM Vendored ripgrep: python\vendor -> app\vendor
if exist "%ROOT%\python\vendor" (
  if exist "%APPDIR%\vendor" rmdir /S /Q "%APPDIR%\vendor"
  robocopy "%ROOT%\python\vendor" "%APPDIR%\vendor" /E /NFL /NDL /NJH /NJS /NP >nul
  if errorlevel 8 ( echo Failed to stage vendor\ & exit /b 1 )
)

REM Read-only prompt template. Per-machine *.config.json files are intentionally NOT staged.
if exist "%ROOT%\system_prompt.txt" (
  copy /Y "%ROOT%\system_prompt.txt" "%APPDIR%\system_prompt.txt" >nul
) else (
  echo   ^(skip system_prompt.txt -- not present^)
)
REM Defensively remove any per-machine config a prior build staged (must never ship).
for %%L in (llm.config.json app.config.json) do (
  if exist "%APPDIR%\%%L" ( del /F /Q "%APPDIR%\%%L" & echo   removed staged %%L ^(must not ship^) )
)

REM Refresh the MSVC runtime from System32. onnxruntime needs the current VS2022 runtime;
REM some wheel ships an older 14.29 copy that breaks its DLL init. (Top level of _internal,
REM matching the original script.)
set "INTERNAL=%APPDIR%\_internal"
for %%D in (msvcp140.dll msvcp140_1.dll vcruntime140.dll vcruntime140_1.dll) do (
  if exist "%INTERNAL%\%%D" if exist "%WINDIR%\System32\%%D" (
    copy /Y "%WINDIR%\System32\%%D" "%INTERNAL%\%%D" >nul
    echo   refreshed MSVC runtime %%D
  )
)

set "EXE=%APPDIR%\Waifu Code App.exe"
if exist "%EXE%" (
  echo BUILD SUCCEEDED -^> "%EXE%"
  exit /b 0
) else (
  echo Build finished but exe not found at "%EXE%"
  exit /b 1
)
