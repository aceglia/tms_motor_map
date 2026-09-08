@echo off
echo Activating Conda environment... 
call D:\Documents\miniconda\Scripts\activate.bat tms_map

echo Cleaning previous build...
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

echo.
echo Building application...
pyinstaller --clean app.spec

if errorlevel 1 (
    echo.
    echo ============================
    echo BUILD FAILED!
    echo ============================
    echo.
    pause
    exit /b 1
)

echo.
echo Copying configuration...
copy default_map_options.yaml "dist\TMS motor map app\default_map_options.yaml"

if errorlevel 1 (
    echo.
    echo ============================
    echo CONFIG COPY FAILED!
    echo ============================
    echo.
    pause
    exit /b 1
)

echo.
echo ============================
echo BUILD COMPLETE!
echo ============================
echo.
echo Output:
echo dist\TMS motor map app\
echo.

pause
