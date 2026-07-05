@echo off
REM ---------------------------------------------------------------------------
REM Launch the app using its bundled virtualenv (python/.venv).
REM
REM Run this yourself — before or after Unity, in any order. The app connects to
REM Unity's fixed WebSocket port (app.config.json -> unity.port, default 8770) and
REM (re)connects on its own; until then the chat window shows a "Disconnected from Unity"
REM overlay. To have the app launch the Unity build itself, set unity.autostart=true
REM and unity.exePath in app.config.json.
REM
REM First-time setup (run once):
REM     cd python
REM     python -m venv .venv
REM     .venv\Scripts\pip install -r requirements.txt
REM ---------------------------------------------------------------------------
setlocal
set "HERE=%~dp0"
set "VENV_PY=%HERE%python\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" goto :no_venv

cd /d "%HERE%python"
"%VENV_PY%" app.py %*
goto :end

:no_venv
echo [run-app] Python venv not found at:
echo     "%VENV_PY%"
echo.
echo Create it with:
echo     cd "%HERE%python"
echo     python -m venv .venv
echo     .venv\Scripts\pip install -r requirements.txt
echo.
pause

:end
