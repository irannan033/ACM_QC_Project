@echo off
REM Starts the dashboard so other PCs on the network can open http://THIS-PC-IP:8501
cd /d "%~dp0"
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
