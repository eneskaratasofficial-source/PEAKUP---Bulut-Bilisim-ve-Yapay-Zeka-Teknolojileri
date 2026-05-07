@echo off
color 0a
title SociRea Bot Detector Sunucusu
echo ==============================================
echo SociRea Uygulamasi Baslatiliyor...
echo Lutfen sunucu penceresini kapatmayiniz.
echo ==============================================

cd "C:\Users\user\Documents\GitHub\socirea"

echo Yapay Zeka cevresel baglantilari kuruluyor...
call "C:\Users\user\Documents\GitHub\PEAKUP---Bulut-Bilisim-ve-Yapay-Zeka-Teknolojileri\venv\Scripts\activate.bat"

streamlit run app\app.py
pause
