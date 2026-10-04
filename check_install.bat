@echo off
call "%~dp0install_requirements.bat" --check %*
exit /b %errorlevel%
