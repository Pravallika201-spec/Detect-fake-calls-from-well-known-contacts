@echo off
echo Activating virtual environment...
call .venv\Scripts\activate

echo.
echo Running inference on all audio files...
python -m backend.pipelines.inference_pipeline

echo.
echo Demo complete. Press any key to exit.
pause >nul
