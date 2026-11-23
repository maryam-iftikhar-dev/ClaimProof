@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python virtual environment not found. Follow README.md setup steps first.
  pause
  exit /b 1
)
if not exist "node_modules" (
  echo Node dependencies not found. Run npm install first.
  pause
  exit /b 1
)
start "ClaimProof model service" cmd /k ".venv\Scripts\python.exe server\inference_api.py"
timeout /t 3 /nobreak >nul
start "ClaimProof Node server" cmd /k "npm start"
timeout /t 2 /nobreak >nul
start http://127.0.0.1:3000
