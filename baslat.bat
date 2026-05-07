@echo off
echo =======================================================
echo     Sosyal Medya Bot Tespiti - AntiGravity GPT
echo =======================================================
echo Streamlit arayuzu baslatiliyor, lutfen bekleyin...

cd /d "%~dp0"

IF EXIST "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    python -m streamlit run app\app.py
) ELSE (
    echo [HATA] venv klasoru bulunamadi!
    echo Lutfen once projenin kurulu oldugundan emin olun.
)
pause
