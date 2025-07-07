@echo off
cd /d "SEUCAMINHO*YOURDIRETORY"
call venv\Scripts\activate
start cmd /k "py -m app"
