#!/usr/bin/env python3
"""설치 자체 점검 — 새 PC에 옮긴 뒤 이것부터 실행한다.

무엇을 확인하는가
  1. 파이썬 버전
  2. 필수/선택 패키지 (numpy, pydicom / scipy)
  3. 프로그램 파일이 빠짐없이 왔는가
  4. 선원 데이터 로드
  5. 계산 엔진이 실제로 도는가 (TG-43 한 점 계산)
  6. 웹 서버 기동 → 화면·API 응답 → 종료
  7. LM Studio 연결 (없어도 됨 — 휴리스틱으로 동작)

사용법
    python check_install.py
    python check_install.py --lmstudio http://localhost:1234/v1
"""

from __future__ import annotations

import argparse
import importlib
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))

NEEDED_FILES = [
    "app_server.py", "run_phase3.py", "export_rtplan.py", "compare_plans.py",
    "webapp/index.html", "webapp/style.css", "webapp/app.js",
    "brachyplan/__init__.py", "brachyplan/data/ir192_mHDR_v2_clrp.json",
    "brachyplan/io/struct_to_mask.py", "brachyplan/dose/tg43.py",
    "brachyplan/optim/optimizer.py", "brachyplan/agent/lm_client.py",
    "tools/mock_lmstudio.py",
]

OK, BAD, WARN = "  [OK]  ", "  [실패]", "  [주의]"
fails: list[str] = []
warns: list[str] = []


def ok(msg):
    print(OK + " " + msg)


def bad(msg, fix=""):
    print(BAD + " " + msg)
    fails.append(msg + (("\n         → " + fix) if fix else ""))


def warn(msg, fix=""):
    print(WARN + " " + msg)
    warns.append(msg + (("\n         → " + fix) if fix else ""))


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmstudio", default="http://localhost:1234/v1")
    args = ap.parse_args()

    print("=" * 68)
    print(" Brachy RT auto planning — 설치 점검")
    print("=" * 68)
    print(f" 폴더   : {ROOT}")
    print(f" 파이썬 : {sys.executable}")

    # 1) 파이썬 -----------------------------------------------------------
    print("\n[1] 파이썬 버전")
    v = sys.version_info
    print(f"      {v.major}.{v.minor}.{v.micro}")
    if (v.major, v.minor) >= (3, 10):
        ok(f"3.10 이상 (현재 {v.major}.{v.minor})")
    else:
        bad(f"3.10 이상이 필요합니다 (현재 {v.major}.{v.minor})",
            "새 파이썬을 설치하고 가상환경을 다시 만드십시오")

    # 2) 패키지 -----------------------------------------------------------
    print("\n[2] 패키지")
    for name, required in (("numpy", True), ("pydicom", True), ("scipy", False)):
        try:
            m = importlib.import_module(name)
            ok(f"{name} {getattr(m, '__version__', '?')}")
        except ImportError as e:
            if required:
                bad(f"{name} 없음 ({e})",
                    f"pip install --no-index --find-links=wheels {name}")
            else:
                warn(f"{name} 없음 — FISTA 백엔드로 자동 전환됩니다 (느림)",
                     "pip install --no-index --find-links=wheels scipy")

    # 2b) wheels 폴더 ------------------------------------------------------
    wheels = os.path.join(ROOT, "wheels")
    if os.path.isdir(wheels):
        print("\n[2b] wheels 폴더 (폐쇄망 설치용)")
        tag = f"cp{v.major}{v.minor}"
        files = [f for f in sorted(os.listdir(wheels)) if f.endswith(".whl")]
        if not files:
            warn("wheels 폴더가 비어 있습니다")
        bad_whl = []
        for f in files:
            # 이름 형식: 패키지-버전-파이썬태그-ABI태그-플랫폼.whl
            parts = f[:-4].split("-")
            pytag = parts[2] if len(parts) >= 5 else "?"
            fits = pytag.startswith("py3") or tag in pytag
            print(f"       {'OK ' if fits else '불일치'}  {f}")
            if not fits:
                bad_whl.append((f, pytag))
        if bad_whl:
            warn(f"이 파이썬({tag})에 맞지 않는 휠 {len(bad_whl)}개 "
                 f"— 그대로 설치하면 'is not a supported wheel' 오류",
                 f"인터넷 되는 PC에서 다시 받으십시오:\n"
                 f"           pip download numpy pydicom scipy -d wheels "
                 f"--python-version {v.major}{v.minor} "
                 f"--only-binary=:all: --platform win_amd64")
        elif files:
            ok(f"{len(files)}개 모두 이 파이썬({tag})에 맞습니다")

    # 3) 파일 -------------------------------------------------------------
    print("\n[3] 프로그램 파일")
    missing = [f for f in NEEDED_FILES if not os.path.isfile(os.path.join(ROOT, *f.split("/")))]
    if missing:
        bad(f"{len(missing)}개 누락: {', '.join(missing[:5])}"
            + (" …" if len(missing) > 5 else ""),
            "폴더를 통째로 다시 복사하십시오")
    else:
        ok(f"{len(NEEDED_FILES)}개 확인")

    if fails:                      # 여기서부터는 import가 필요하므로 중단
        return report()

    sys.path.insert(0, ROOT)

    # 4) 선원 데이터 ---------------------------------------------------------
    print("\n[4] 선원 데이터")
    try:
        from brachyplan.dose.source_data import SourceData
        sd = SourceData.load("ir192_mHDR_v2_clrp")
        ok(f"{getattr(sd, 'name', 'ir192_mHDR_v2_clrp')} 로드")
    except Exception as e:                                   # noqa: BLE001
        bad(f"선원 데이터 로드 실패: {e}",
            "brachyplan/data/ir192_mHDR_v2_clrp.json 이 있는지 확인")
        return report()

    # 5) 계산 엔진 ---------------------------------------------------------
    print("\n[5] 계산 엔진 (TG-43)")
    try:
        import numpy as np
        from brachyplan.dose.tg43 import TG43Calculator
        calc = TG43Calculator(sd)
        pt = np.array([[0.0, 10.0, 0.0]])          # 선원에서 1 cm 옆
        r = float(calc.dose_rate_per_U(pt, np.zeros(3), np.array([0.0, 0.0, 1.0]))[0])
        if r > 0 and r == r:
            ok(f"1 cm 지점 선량률 {r:.4g} cGy/h/U — 계산 정상")
        else:
            bad(f"비정상 값: {r}")
    except Exception as e:                                   # noqa: BLE001
        bad(f"계산 실패: {e}")
        return report()

    # 6) 웹 서버 ----------------------------------------------------------
    print("\n[6] 웹 서버 기동")
    port = free_port()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.join(ROOT, "app_server.py"),
         "--port", str(port), "--token", "SELFTEST", "--no-browser"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace")
    try:
        def get(path, token=None):
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}",
                headers={"X-Auth-Token": token} if token else {})
            try:
                with urllib.request.urlopen(req, timeout=5) as f:
                    return f.status, f.read()
            except urllib.error.HTTPError as e:
                return e.code, e.read()

        up = False
        for _ in range(40):
            time.sleep(0.4)
            if proc.poll() is not None:
                break
            try:
                if get("/healthz")[0] == 200:
                    up = True
                    break
            except Exception:                                # noqa: BLE001
                pass
        if not up:
            out = proc.stdout.read() if proc.stdout else ""
            bad("서버가 뜨지 않았습니다", (out or "").strip()[:400] or "로그 없음")
        else:
            ok(f"기동 성공 (포트 {port})")
            for path, want in (("/style.css", 200), ("/app.js", 200),
                               ("/favicon.svg", 200)):
                code = get(path)[0]        # 토큰 없이 — 브라우저와 동일
                (ok if code == want else bad)(
                    f"{path} → {code} (토큰 없이 접근 가능해야 함)")
            for path in ("/api/patients", "/api/jobs"):
                code = get(path)[0]
                (ok if code == 401 else bad)(f"{path} → {code} (토큰 없으면 401)")
            code, body = get("/", "SELFTEST")
            if code == 200 and b"Brachy RT auto planning" in body:
                ok("/ → 200, 대시보드 HTML 확인")
            else:
                bad(f"/ → {code} (대시보드가 아닙니다)",
                    "webapp/index.html 을 확인하십시오")
            code, _ = get("/api/patients", "SELFTEST")
            (ok if code == 200 else bad)(f"/api/patients (토큰 있음) → {code}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:                    # pragma: no cover
            proc.kill()

    # 7) LM Studio --------------------------------------------------------
    print("\n[7] LM Studio (선택)")
    try:
        from brachyplan.agent.lm_client import LMStudioClient
        models = LMStudioClient(base_url=args.lmstudio, timeout_s=5.0).list_models()
        if models:
            ok(f"연결됨 — 모델 {len(models)}개: {', '.join(models[:3])}"
               + (" …" if len(models) > 3 else ""))
        else:
            warn("연결됐지만 로드된 모델이 없습니다",
                 "LM Studio에서 모델을 로드하십시오")
    except Exception as e:                                   # noqa: BLE001
        warn(f"미연결 ({str(e).splitlines()[0]})",
             "LM Studio ▸ Developer ▸ Start Server. "
             "연결되지 않아도 휴리스틱으로 동작합니다")

    return report()


def report() -> int:
    print("\n" + "=" * 68)
    if fails:
        print(f" 실패 {len(fails)}건 — 해결해야 실행할 수 있습니다")
        for f in fails:
            print("   · " + f)
    else:
        print(" 설치 정상 — 실행할 수 있습니다")
        print("\n   python app_server.py                 (이 PC에서만)")
        print("   python app_server.py --host 0.0.0.0   (내부망 공개)")
    if warns:
        print(f"\n 주의 {len(warns)}건 (실행은 가능)")
        for w in warns:
            print("   · " + w)
    print("=" * 68)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
