@echo off
title EGX30 Multi-Model Forecaster
color 0A
cd /d "%~dp0"

echo.
echo ================================================
echo   EGX30 Multi-Model Stock Forecaster
echo   GBM + GARCH + Jump-Diffusion + ARIMA + Prophet
echo   By: Osman Osama
echo ================================================
echo.

echo [1/2] Installing libraries (first run may take 1-2 min)...
pip install streamlit yfinance plotly scipy pandas numpy arch statsmodels --quiet
echo.
echo Note: Prophet is optional. Install separately if needed:
echo       pip install prophet
echo.

echo [2/2] Launching app...
echo       Browser will open at http://localhost:8501
echo       Press CTRL+C to stop.
echo.

streamlit run egx30_forecaster.py

pause
