@echo off
:: Start Harley Inference Engine
echo Starting Harley Inference Engine...
cd /d "%~dp0"

:: Start llama.cpp backend
echo Starting llama.cpp vision model on :8080...
start "llama.cpp" /MIN cmd /c "C:\HarleysPlace\models\llamacpp\backends\b9967\win-avx-x64\build\bin\llama-server.exe -m "C:\HarleysPlace\models\lmstudio-community\Qwen2.5-VL-3B-Instruct-GGUF\Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf" --mmproj "C:\HarleysPlace\models\lmstudio-community\Qwen2.5-VL-3B-Instruct-GGUF\mmproj-model-f16.gguf" -c 4096 -t 8 --host 0.0.0.0 --port 8080"

echo Waiting for model to load (15s)...
timeout /t 15 /nobreak >nul

:: Start Harley server
echo Starting Harley server on :5051...
start "Harley" /MIN cmd /c "C:\Program Files\Python312\python.exe server.py"

echo.
echo Harley is ready!
echo Chat: http://localhost:5051
echo Model: http://localhost:8080
echo.
pause
