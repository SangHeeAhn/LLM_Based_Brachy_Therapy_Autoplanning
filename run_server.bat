@echo off
REM ===================================================================
REM  Brachy RT auto planning - 서버 실행
REM  이 파일을 더블클릭하면 서버가 뜨고 브라우저가 열립니다.
REM
REM  폴더를 바꾸려면 아래 SET 줄을 고치십시오.
REM  이 PC에서만 쓰려면 HOST 를 127.0.0.1 로 바꾸십시오.
REM ===================================================================

SET HOST=0.0.0.0
SET PORT=8765
SET INCOMING=incoming
SET OUTGOING=outgoing

REM 예) 다른 드라이브를 쓸 때
REM SET INCOMING=D:\dicom_in
REM SET OUTGOING=D:\rtplan_out

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
  echo.
  echo  [오류] 가상환경이 없습니다.
  echo         INSTALL.md 의 3장을 따라 먼저 설치하십시오:
  echo             python -m venv venv
  echo             venv\Scripts\activate
  echo             pip install --no-index --find-links=wheels numpy pydicom scipy
  echo.
  pause
  exit /b 1
)

call venv\Scripts\activate.bat

python app_server.py --host %HOST% --port %PORT% ^
    --incoming "%INCOMING%" --outgoing "%OUTGOING%"

echo.
echo  서버가 종료되었습니다.
pause
