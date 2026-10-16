#!/usr/bin/env python3
"""Brachy RT auto planning — 웹 대시보드 서버.

DICOM 수신 폴더 → 환자 선택 → LLM 자동계획 → RTPLAN export 를 브라우저에서
수행한다. LM Studio와 함께 구동되며, 연결 상태·모델을 화면에서 확인하고
바꿀 수 있다.

구성
----
* `webapp/` 정적 파일(index.html / style.css / app.js) + JSON API.
  구동 방식과 화면 디자인은 TG43_Verification/server.py 의 대시보드를 따랐다.
* **표준 라이브러리만 사용한다.** 폐쇄망 PC에 추가 반입할 패키지가 없도록
  http.server 기반이며, 화면도 외부 CDN을 참조하지 않는다.
* **계산은 기존 CLI를 subprocess로 실행한다.** run_phase3.py /
  export_rtplan.py / compare_plans.py 는 이미 검증된 경로다. 서버가 로직을
  복제하면 두 벌을 유지해야 하고 결과가 갈라진다. 서버는 오케스트레이션만 한다.
* **작업은 한 번에 하나씩** 큐로 처리한다. 2 mm 그리드는 kernel만 1 GB를 쓴다.

API
---
  GET  /api/health                  → LM Studio 상태/모델
  POST /api/config                  → LM Studio URL·모델 변경 {base_url, model}
  GET  /api/patients                → 수신 폴더의 환자·검사 목록
  GET  /api/jobs                    → 작업 목록
  POST /api/jobs                    → 실행 {case_id, options}
  GET  /api/jobs/<id>               → 작업 상세
  GET  /api/jobs/<id>/log?offset=   → 로그 증분
  POST /api/jobs/<id>/cancel        → 취소
  GET  /api/jobs/<id>/file/<name>   → 산출물 내려받기

보안
----
* 기본 바인딩은 127.0.0.1. 다른 PC에서 접속하려면 --host 0.0.0.0 을 명시해야
  하고, 그때 경고와 접속 가능한 실제 IP를 출력한다.
* 모든 화면·API가 토큰을 요구한다 (--token 미지정 시 임의 발급).
* TLS가 없다. 환자 정보가 평문으로 오가므로 **신뢰된 내부망 전용**이다.

★ 연구용입니다. 생성되는 RTPLAN은 ApprovalStatus=UNAPPROVED이며, TPS 재계산과
  독립 검증을 거치기 전에는 치료에 사용할 수 없습니다.

실행
----
    python app_server.py
    python app_server.py --host 0.0.0.0 --incoming D:\\dicom_in --outgoing D:\\rtplan_out
    python app_server.py --start-lmstudio       # LM Studio 서버도 함께 기동 시도
"""

from __future__ import annotations

import argparse
import hmac
import json
import mimetypes
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

APP_NAME = "Brachy RT auto planning"
ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "webapp")
sys.path.insert(0, ROOT)

try:
    import pydicom
except ImportError:                                          # pragma: no cover
    print("pydicom이 필요합니다: pip install pydicom", file=sys.stderr)
    raise

from brachyplan.agent.lm_client import DEFAULT_BASE_URL, LMStudioClient  # noqa: E402

SCAN_TAGS = ["Modality", "PatientID", "PatientName", "StudyInstanceUID",
             "StudyDate", "RTPlanLabel", "SOPInstanceUID", "FrameOfReferenceUID",
             "ReferencedStructureSetSequence", "ReferencedFrameOfReferenceSequence"]
CONTENT_TYPES = {".html": "text/html; charset=utf-8",
                 ".css": "text/css; charset=utf-8",
                 ".js": "application/javascript; charset=utf-8",
                 ".json": "application/json; charset=utf-8",
                 ".svg": "image/svg+xml", ".png": "image/png",
                 ".ico": "image/x-icon"}

# 토큰 없이 받을 수 있는 정적 자원.
# 브라우저는 <link>/<script>/favicon 요청에 쿼리스트링이나 커스텀 헤더를
# 붙이지 않는다. 이 파일들에 인증을 걸면 화면이 스타일 없이 깨진다.
# 환자 정보가 전혀 없는 UI 코드이므로 공개해도 안전하다.
# (데이터가 오가는 /api/* 는 그대로 토큰을 요구한다.)
PUBLIC_FILES = {"style.css", "app.js", "favicon.svg", "favicon.ico"}

# 런타임 설정 (설정 모달에서 바뀐다)
CONFIG = {"base_url": DEFAULT_BASE_URL, "model": None}


# ===========================================================================
# 경로 — 네트워크 공유(SMB/UNC) 지원
# ===========================================================================
def is_abs_path(p: str) -> bool:
    r"""절대경로인가. Windows UNC(``\\서버\공유``)도 절대경로로 본다.

    Windows의 os.path.isabs 는 UNC를 이미 True로 판정하지만, 명시해 두면
    다른 OS에서 돌리거나 슬래시를 섞어 써도(``//nas/dicom``) 안전하다.
    """
    return os.path.isabs(p) or p.startswith("\\\\") or p.startswith("//")


def safe_copy(src: str, dst: str) -> str:
    r"""파일 복사. 실패 사유를 사람이 읽을 수 있게 돌려준다.

    네트워크 공유(SMB)에서는 ``copy2`` 가 내용은 복사해 놓고 메타데이터
    (mtime/권한) 복사 단계에서 실패하는 경우가 있다. 그때 내용은 이미
    멀쩡하므로 작업을 실패로 처리하면 안 된다 — 내용만 복사하는 방식으로
    한 번 더 시도한다.

    Returns
    -------
    "" 이면 성공, 아니면 경고 문구.
    """
    try:
        shutil.copy2(src, dst)
        return ""
    except OSError:
        try:
            shutil.copyfile(src, dst)        # 내용만 (메타데이터 생략)
            return (f"{os.path.basename(dst)}: 시간·권한 정보는 복사하지 "
                    f"못했습니다 (내용은 정상). 네트워크 공유에서 흔합니다")
        except OSError as e:
            raise OSError(f"복사 실패 {src} → {dst}: {e}") from e


def check_dir(path: str, need_write: bool) -> str:
    """폴더 접근성 점검. 문제가 있으면 사람이 읽을 수 있는 사유를 돌려준다."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        return f"만들거나 접근할 수 없습니다: {e}"
    if not os.path.isdir(path):
        return "폴더가 아닙니다"
    if need_write:
        probe = os.path.join(path, f".write_test_{uuid.uuid4().hex[:8]}")
        try:
            with open(probe, "w", encoding="utf-8") as fh:
                fh.write("ok")
        except OSError as e:
            return f"쓰기 권한이 없습니다: {e}"
        # 시험 파일 삭제 실패는 통과시킨다.
        # 쓰기는 되지만 삭제 권한이 없는 SMB 공유가 흔한데, 이걸 실패로
        # 처리하면 실제로는 멀쩡히 쓸 수 있는 폴더에서 서버가 뜨지 않는다.
        try:
            os.remove(probe)
        except OSError:
            pass
    else:
        try:
            os.listdir(path)
        except OSError as e:
            return f"읽을 수 없습니다: {e}"
    return ""


# ===========================================================================
# LM Studio
# ===========================================================================
def lm_health() -> dict:
    """LM Studio 연결 상태와 로드된 모델 목록."""
    url = CONFIG["base_url"]
    try:
        client = LMStudioClient(base_url=url, timeout_s=5.0)
        models = client.list_models()
        model = CONFIG["model"] or (models[0] if models else None)
        return {"ok": bool(models), "base_url": url, "models": models,
                "model": model,
                "error": "" if models else "로드된 모델이 없습니다"}
    except Exception as e:                                   # noqa: BLE001
        return {"ok": False, "base_url": url, "models": [], "model": None,
                "error": str(e)}


def start_lmstudio(wait_s: float = 25.0) -> str:
    """LM Studio 로컬 서버 기동을 시도한다 (`lms server start`).

    LM Studio CLI(`lms`)가 PATH에 있어야 한다. 없거나 실패해도 치명적이지
    않다 — 에이전트는 LM Studio 없이 휴리스틱으로 동작하기 때문이다.
    사용자가 이미 GUI에서 서버를 켜 두었다면 그대로 감지된다.
    """
    if lm_health()["ok"]:
        return "이미 실행 중입니다"
    exe = shutil.which("lms") or shutil.which("lms.exe")
    if not exe:
        return ("LM Studio CLI(`lms`)를 찾지 못했습니다. "
                "LM Studio에서 Developer ▸ Start Server 로 직접 켜십시오.")
    try:
        subprocess.Popen([exe, "server", "start"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        return f"실행 실패: {e}"
    deadline = time.time() + wait_s
    while time.time() < deadline:
        time.sleep(1.5)
        if lm_health()["ok"]:
            return "기동 완료"
    return f"{wait_s:.0f}초 안에 응답하지 않았습니다 (모델이 로드되지 않았을 수 있습니다)"


# ===========================================================================
# DICOM 수신 폴더 스캔
# ===========================================================================
class Library:
    """수신 폴더를 훑어 **RTPLAN 하나 = 케이스 하나** 로 묶는다.

    왜 RTPLAN 기준인가
    ------------------
    자동계획이 실제로 읽는 것은 RTPLAN + RTSTRUCT 뿐이다 (선택으로 RTDOSE의
    dose point). CT/MR은 파이프라인 어디에서도 쓰지 않는다 —
    `load_context()` 가 `files["RTSTRUCT"]`, `files["RTPLAN"]`,
    `files["RTDOSE"]` 만 참조한다.

    이전에는 (환자, StudyInstanceUID) 로 묶었는데 두 가지 문제가 있었다.
      * 영상만 있는 study가 "자료 부족" 케이스로 목록에 떴다.
      * RTPLAN과 RTSTRUCT가 서로 다른 study에 있으면 (내보내기·익명화
        과정에서 흔하다) 짝을 찾지 못했다.

    그래서 RTPLAN을 기준으로 삼고, 그 plan이 **실제로 참조하는**
    RTSTRUCT(ReferencedStructureSetSequence → SOPInstanceUID)를 연결한다.
    참조가 끊겨 있으면 같은 FrameOfReference → 같은 study 순으로 보완한다.

    파일 mtime/크기로 헤더를 캐시하므로 재스캔이 빠르다.
    """

    def __init__(self, incoming: str):
        self.incoming = incoming
        self._cache: dict[str, tuple[float, int, dict]] = {}
        self._lock = threading.Lock()

    def _header(self, path: str) -> dict | None:
        try:
            st = os.stat(path)
        except OSError:
            return None
        with self._lock:
            hit = self._cache.get(path)
        if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
            return hit[2]
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True,
                                 specific_tags=SCAN_TAGS)
        except Exception:
            return None
        mod = str(getattr(ds, "Modality", ""))
        if not mod:
            return None
        # RTPLAN이 참조하는 RTSTRUCT의 SOP UID
        ref_ss = ""
        seq = ds.get("ReferencedStructureSetSequence", None)
        if seq is not None and len(seq):
            ref_ss = str(getattr(seq[0], "ReferencedSOPInstanceUID", "") or "")
        # FrameOfReference — RTSTRUCT는 시퀀스 안에 들어 있다
        forr = str(getattr(ds, "FrameOfReferenceUID", "") or "")
        if not forr:
            rf = ds.get("ReferencedFrameOfReferenceSequence", None)
            if rf is not None and len(rf):
                forr = str(getattr(rf[0], "FrameOfReferenceUID", "") or "")

        info = {"path": path, "modality": mod,
                "sop_uid": str(getattr(ds, "SOPInstanceUID", "") or ""),
                "patient_id": str(getattr(ds, "PatientID", "") or "(no ID)"),
                "patient_name": str(getattr(ds, "PatientName", "") or ""),
                "study_uid": str(getattr(ds, "StudyInstanceUID", "") or ""),
                "study_date": str(getattr(ds, "StudyDate", "") or ""),
                "plan_label": str(getattr(ds, "RTPlanLabel", "") or ""),
                "ref_structure_set": ref_ss,
                "frame_uid": forr}
        with self._lock:
            self._cache[path] = (st.st_mtime, st.st_size, info)
        return info

    def _all_headers(self) -> list[dict]:
        out = []
        if not os.path.isdir(self.incoming):
            return out
        for dirpath, _dirs, names in os.walk(self.incoming):
            for nm in names:
                if not nm.lower().endswith((".dcm", ".ima")):
                    continue
                info = self._header(os.path.join(dirpath, nm))
                if info is not None:
                    out.append(info)
        return out

    def scan(self) -> list[dict]:
        """RTPLAN 하나당 케이스 하나. RTSTRUCT는 참조를 따라 붙인다."""
        heads = self._all_headers()
        plans = [h for h in heads if h["modality"] == "RTPLAN"]
        structs = [h for h in heads if h["modality"] == "RTSTRUCT"]
        doses = [h for h in heads if h["modality"] == "RTDOSE"]
        by_sop = {h["sop_uid"]: h for h in structs if h["sop_uid"]}

        cases = []
        for p in plans:
            # 1순위: plan이 명시적으로 참조하는 RTSTRUCT
            rs = by_sop.get(p["ref_structure_set"]) if p["ref_structure_set"] else None
            link = "참조" if rs else ""
            # 2순위: 같은 FrameOfReference
            if rs is None and p["frame_uid"]:
                rs = next((s for s in structs if s["frame_uid"] == p["frame_uid"]), None)
                link = "동일 FrameOfReference" if rs else ""
            # 3순위: 같은 study
            if rs is None:
                rs = next((s for s in structs
                           if s["study_uid"] == p["study_uid"]), None)
                link = "동일 study" if rs else ""

            rd = next((d for d in doses if d["study_uid"] == p["study_uid"]), None)

            files = [p["path"]] + ([rs["path"]] if rs else []) \
                                + ([rd["path"]] if rd else [])
            counts = {"RTPLAN": 1}
            if rs:
                counts["RTSTRUCT"] = 1
            if rd:
                counts["RTDOSE"] = 1

            cases.append({
                "case_id": _case_id(p["patient_id"], p["sop_uid"]),
                "patient_id": p["patient_id"],
                "patient_name": p["patient_name"],
                "study_uid": p["study_uid"],
                "study_date": p["study_date"],
                "plan_label": p["plan_label"],
                "counts": counts,
                "files": files,
                "struct_link": link,
                "ready": rs is not None,
                "reason": "" if rs else
                          "이 RTPLAN이 참조하는 RTSTRUCT를 찾을 수 없습니다. "
                          "구조물 파일을 같은 폴더에 넣어 주십시오.",
            })
        cases.sort(key=lambda x: (x["patient_id"], x["study_date"], x["plan_label"]))
        return cases

    def inventory(self) -> list[dict]:
        """수신 폴더 전체 현황 (데이터 탭용). 영상까지 modality별로 센다."""
        groups: dict[tuple[str, str], dict] = {}
        for h in self._all_headers():
            key = (h["patient_id"], h["study_uid"])
            g = groups.setdefault(key, {
                "patient_id": h["patient_id"], "patient_name": h["patient_name"],
                "study_uid": h["study_uid"], "study_date": h["study_date"],
                "counts": {}})
            g["counts"][h["modality"]] = g["counts"].get(h["modality"], 0) + 1
        out = sorted(groups.values(),
                     key=lambda x: (x["patient_id"], x["study_date"]))
        for g in out:
            g["case_id"] = _case_id(g["patient_id"], g["study_uid"])
        return out

    def find(self, case_id: str) -> dict | None:
        return next((c for c in self.scan() if c["case_id"] == case_id), None)


def _case_id(patient_id: str, study_uid: str) -> str:
    import hashlib
    return hashlib.sha1(f"{patient_id}|{study_uid}".encode()).hexdigest()[:12]


# ===========================================================================
# 작업
# ===========================================================================
class Job:
    def __init__(self, job_id: str, case: dict, opts: dict, work_dir: str):
        self.id = job_id
        self.case = {k: v for k, v in case.items() if k != "files"}
        self.files = list(case["files"])
        self.opts = opts
        self.dir = work_dir
        self.status = "queued"        # queued|running|done|failed|cancelled
        self.step = ""
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.error = ""
        self.warnings: list[str] = []
        self.artifacts: dict[str, str] = {}
        self.summary: dict = {}
        self.proc: subprocess.Popen | None = None
        self.cancel = False

    @property
    def log_path(self) -> str:
        return os.path.join(self.dir, "run.log")

    def to_json(self) -> dict:
        return {"id": self.id, "status": self.status, "step": self.step,
                "case": self.case, "opts": self.opts, "error": self.error,
                "warnings": self.warnings, "created": self.created,
                "started": self.started, "finished": self.finished,
                "artifacts": sorted(self.artifacts), "summary": self.summary}


class Runner(threading.Thread):
    """작업을 하나씩 순서대로 실행한다."""

    def __init__(self, outgoing: str):
        super().__init__(daemon=True)
        self.q: queue.Queue[Job] = queue.Queue()
        self.jobs: dict[str, Job] = {}
        self.outgoing = outgoing
        self.current: Job | None = None
        self._lock = threading.Lock()

    def submit(self, job: Job) -> None:
        with self._lock:
            self.jobs[job.id] = job
        self.q.put(job)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self.jobs.get(job_id)

    def listing(self) -> list[dict]:
        with self._lock:
            js = sorted(self.jobs.values(), key=lambda j: -j.created)
        return [j.to_json() for j in js]

    def cancel(self, job_id: str) -> bool:
        j = self.get(job_id)
        if j is None or j.status in ("done", "failed", "cancelled"):
            return False
        j.cancel = True
        if j.proc and j.proc.poll() is None:
            try:
                j.proc.terminate()
            except Exception:                                # pragma: no cover
                pass
        if j.status == "queued":
            j.status = "cancelled"
        return True

    # ------------------------------------------------------------------
    def run(self) -> None:                                   # pragma: no cover
        while True:
            job = self.q.get()
            if job.status == "cancelled":
                self.q.task_done()
                continue
            self.current = job
            try:
                self._execute(job)
            except Exception:
                job.status = "failed"
                job.error = traceback.format_exc(limit=3)
                self._log(job, "\n[서버 오류]\n" + job.error)
            finally:
                job.finished = time.time()
                self.current = None
                self.q.task_done()

    def _log(self, job: Job, text: str) -> None:
        with open(job.log_path, "a", encoding="utf-8") as fh:
            fh.write(text)

    def _run_cmd(self, job: Job, args: list[str], step: str) -> int:
        job.step = step
        self._log(job, f"\n{'=' * 70}\n[{step}]\n{'=' * 70}\n")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        with open(job.log_path, "a", encoding="utf-8") as fh:
            job.proc = subprocess.Popen(args, cwd=ROOT, env=env,
                                        stdout=fh, stderr=subprocess.STDOUT)
            rc = job.proc.wait()
        job.proc = None
        return rc

    def _execute(self, job: Job) -> None:
        job.status = "running"
        job.started = time.time()
        o = job.opts
        case_dir = os.path.join(job.dir, "case")
        os.makedirs(case_dir, exist_ok=True)

        job.step = "DICOM 준비"
        self._log(job, f"{APP_NAME}\n작업 {job.id} · {job.case['patient_id']} "
                       f"({job.case.get('patient_name', '')})\n"
                       f"시작 {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                       f"필요한 DICOM {len(job.files)}개만 작업공간으로 복사합니다 "
                       f"({', '.join(sorted(job.case['counts']))})\n")
        for i, src in enumerate(job.files):
            note = safe_copy(src, os.path.join(
                case_dir, f"{i:05d}_{os.path.basename(src)}"))
            if note:
                self._log(job, f"  [주의] {note}\n")

        py = [sys.executable, "-u"]
        dwell = os.path.join(job.dir, "dwell.npy")
        trace = os.path.join(job.dir, "trace.json")

        # --- 자동계획 ------------------------------------------------------
        cmd = py + [os.path.join(ROOT, "run_phase3.py"),
                    "--case", case_dir,
                    "--spacing", str(o["spacing"]),
                    "--target", o["target"], "--secondary", o["secondary"],
                    "--max-iter", str(o["max_iter"]),
                    "--goals-source", o["goals_source"],
                    "--objectives", o["objectives"],
                    "--t-max", str(o["t_max"]), "--smooth", str(o["smooth"]),
                    "--seed", str(o["seed"]),
                    "--ebrt-dose", str(o["ebrt_dose"]),
                    "--ebrt-fractions", str(o["ebrt_fractions"]),
                    "--save-dwell", dwell, "--save-trace", trace]
        if o["use_llm"]:
            cmd += ["--base-url", o["base_url"]]
            if o.get("model"):
                cmd += ["--model", o["model"]]
            if o.get("no_schema"):
                cmd.append("--no-schema")
        else:
            cmd.append("--no-llm")
        if o.get("no_reference_probe"):
            cmd.append("--no-reference-probe")

        rc = self._run_cmd(job, cmd, "자동계획 (Phase 3)")
        if job.cancel:
            job.status = "cancelled"
            return
        # 종료코드 2 = plan은 나왔으나 필수 목표 미충족/기준선 대비 미개선.
        # 계산 실패가 아니라 **계획 품질 판정**이므로 멈추지 않는다.
        if rc == 2 and os.path.isfile(dwell):
            job.warnings.append("자동계획: 필수 목표 미충족 또는 기준선 대비 "
                                "미개선 — 로그의 DVH를 확인하십시오")
        elif rc != 0 or not os.path.isfile(dwell):
            job.status = "failed"
            job.error = f"자동계획 실패 (종료코드 {rc})"
            return
        job.artifacts["dwell.npy"] = dwell
        if os.path.isfile(trace):
            job.artifacts["trace.json"] = trace

        # --- RTPLAN 내보내기 -------------------------------------------------
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", job.case["patient_id"])[:24]
        out_rp = os.path.join(job.dir, f"RP_{safe}_{job.id}.dcm")
        cmd = py + [os.path.join(ROOT, "export_rtplan.py"),
                    "--case", case_dir, "--dwell", dwell, "--out", out_rp,
                    "--suffix", o["suffix"],
                    "--time-resolution", str(o["time_resolution"]),
                    "--uid-style", o["uid_style"]]
        if o.get("keep_series_uid"):
            cmd.append("--keep-series-uid")
        if o.get("new_private_uid"):
            cmd.append("--new-private-uid")
        if o.get("skip_dose_check"):
            cmd.append("--no-dose-check")
        else:
            cmd += ["--spacing", str(o["spacing"])]

        rc = self._run_cmd(job, cmd, "RTPLAN 내보내기")
        if job.cancel:
            job.status = "cancelled"
            return
        if not os.path.isfile(out_rp):
            job.status = "failed"
            job.error = f"RTPLAN 내보내기 실패 (종료코드 {rc})"
            return
        if rc != 0:
            job.status = "failed"
            job.error = ("RTPLAN 파일 정합성 검증 실패 — 로그를 확인하십시오. "
                         "이 파일은 사용하지 마십시오.")
            return
        job.artifacts[os.path.basename(out_rp)] = out_rp

        # --- 검증 ----------------------------------------------------------
        cmd = py + [os.path.join(ROOT, "compare_plans.py"),
                    "--case", case_dir, "--ai", out_rp,
                    "--spacing", str(o["spacing"])]
        if o.get("skip_dose_check"):
            cmd.append("--no-dose")
        if self._run_cmd(job, cmd, "검증 (원본 plan과 대조)") == 1:
            job.warnings.append("검증: 원본 plan과 구조적 차이가 보고되었습니다 "
                                "— 로그의 '확인 필요' 항목을 보십시오")

        # --- 출력 폴더 복사 ---------------------------------------------------
        # 출력 폴더 복사가 실패해도 작업은 성공이다 — RTPLAN은 이미 만들어졌고
        # 화면에서 내려받을 수 있다. 네트워크 공유가 잠깐 끊긴 경우가 흔하다.
        if self.outgoing:
            dst = os.path.join(self.outgoing, os.path.basename(out_rp))
            try:
                os.makedirs(self.outgoing, exist_ok=True)
                note = safe_copy(out_rp, dst)
                self._log(job, f"\n[완료] {dst} 로 복사했습니다\n")
                if note:
                    job.warnings.append(note)
            except OSError as e:
                msg = (f"출력 폴더에 복사하지 못했습니다 ({self.outgoing}): {e} "
                       f"— 화면에서 직접 내려받으십시오")
                job.warnings.append(msg)
                self._log(job, f"\n[주의] {msg}\n")

        job.summary = _summarize(job.log_path)
        job.step = "완료"
        job.status = "done"


def _summarize(log_path: str) -> dict:
    """로그에서 최종 판정만 뽑는다 (화면 요약용)."""
    out: dict = {}
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return out
    m = re.search(r"\[최종 plan\] score ([\d.]+)", text)
    if m:
        out["score"] = float(m.group(1))
    m = re.search(r"종합 점수\s+([\d.]+) / 100\s+\((\d+)/(\d+) 충족\)", text)
    if m:
        out["score"] = float(m.group(1))
        out["met"] = f"{m.group(2)}/{m.group(3)}"
    out["hard_ok"] = "필수 목표 준수     : PASS" in text
    m = re.search(r"총 조사시간: ([\d.]+) s \(원본 ([\d.]+) s\)", text)
    if m:
        out["total_time_s"] = float(m.group(1))
        out["total_time_original_s"] = float(m.group(2))
    out["export_ok"] = "파일 정합성 : PASS" in text
    return out


# ===========================================================================
# 옵션 검증
# ===========================================================================
def _num(d: dict, key: str, default, lo, hi, cast=float):
    try:
        v = cast(d.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"{key} 값이 올바르지 않습니다")
    if not (lo <= v <= hi):
        raise ValueError(f"{key}는 {lo}~{hi} 범위여야 합니다 (받은 값 {v})")
    return v


def normalize_opts(d: dict) -> dict:
    name_re = re.compile(r"^[A-Za-z0-9_\-]{1,32}$")
    target = str(d.get("target", "HR_CTV"))
    secondary = str(d.get("secondary", "IR_CTV"))
    for v in (target, secondary):
        if not name_re.match(v):
            raise ValueError(f"구조물 이름이 올바르지 않습니다: {v!r}")
    base_url = str(d.get("base_url") or CONFIG["base_url"])
    if not re.match(r"^https?://[A-Za-z0-9._\-]+(:\d+)?(/[A-Za-z0-9._\-/]*)?$", base_url):
        raise ValueError("LM Studio 주소 형식이 올바르지 않습니다")
    goals_source = str(d.get("goals_source", "embrace2"))
    objectives = str(d.get("objectives", "goals"))
    uid_style = str(d.get("uid_style", "pydicom"))
    if goals_source not in ("embrace2", "clinical"):
        raise ValueError("goals_source가 올바르지 않습니다")
    if objectives not in ("goals", "clinical"):
        raise ValueError("objectives가 올바르지 않습니다")
    if uid_style not in ("pydicom", "source"):
        raise ValueError("uid_style이 올바르지 않습니다")
    suffix = str(d.get("suffix", "_AI"))
    if not re.match(r"^[A-Za-z0-9_\-]{0,8}$", suffix):
        raise ValueError("접미사는 영숫자/밑줄 8자 이내여야 합니다")
    model = d.get("model") or CONFIG["model"]
    if model is not None and not re.match(r"^[A-Za-z0-9._@:/\-]{1,120}$", str(model)):
        raise ValueError("모델 이름이 올바르지 않습니다")
    return {
        "spacing": _num(d, "spacing", 2.0, 1.0, 10.0),
        "max_iter": _num(d, "max_iter", 12, 1, 60, int),
        "t_max": _num(d, "t_max", 200.0, 1.0, 1000.0),
        "smooth": _num(d, "smooth", 0.05, 0.0, 100.0),
        "seed": _num(d, "seed", 0, 0, 2 ** 31 - 1, int),
        "time_resolution": _num(d, "time_resolution", 0.1, 0.0, 10.0),
        "ebrt_dose": _num(d, "ebrt_dose", 45.0, 0.0, 100.0),
        "ebrt_fractions": _num(d, "ebrt_fractions", 25, 1, 60, int),
        "target": target, "secondary": secondary,
        "goals_source": goals_source, "objectives": objectives,
        "uid_style": uid_style, "suffix": suffix,
        "use_llm": bool(d.get("use_llm", True)),
        "no_schema": bool(d.get("no_schema", False)),
        "base_url": base_url, "model": model,
        "keep_series_uid": bool(d.get("keep_series_uid", False)),
        "new_private_uid": bool(d.get("new_private_uid", False)),
        "skip_dose_check": bool(d.get("skip_dose_check", False)),
        "no_reference_probe": bool(d.get("no_reference_probe", False)),
    }


# ===========================================================================
# HTTP
# ===========================================================================
class Handler(BaseHTTPRequestHandler):
    server_version = "BrachyRTAutoPlanning/1.0"
    library: Library
    runner: Runner
    token: str
    paths: dict

    def log_message(self, fmt, *a):
        sys.stderr.write("  [http] " + (fmt % a) + "\n")

    # -- 공통 -------------------------------------------------------------
    def _authed(self, params: dict) -> bool:
        given = (self.headers.get("X-Auth-Token")
                 or (params.get("token") or [""])[0] or "")
        return hmac.compare_digest(given, self.token)

    def _send(self, code: int, body: bytes, ctype: str, extra=None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _err(self, code: int, msg: str) -> None:
        self._json({"error": msg}, code)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 1_000_000:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    def _static(self, rel: str):
        """webapp/ 안의 파일만 제공한다 (경로 이탈 차단)."""
        rel = unquote(rel).lstrip("/") or "index.html"
        path = os.path.realpath(os.path.join(WEB_DIR, rel))
        base = os.path.realpath(WEB_DIR)
        if not (path == base or path.startswith(base + os.sep)) or not os.path.isfile(path):
            return self._err(404, "없는 파일")
        ext = os.path.splitext(path)[1].lower()
        ctype = CONTENT_TYPES.get(ext) or mimetypes.guess_type(path)[0] \
            or "application/octet-stream"
        with open(path, "rb") as fh:
            self._send(200, fh.read(), ctype)

    # -- 라우팅 -----------------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        p, params = u.path, parse_qs(u.query)

        if p == "/healthz":
            return self._json({"ok": True, "app": APP_NAME})

        # 정적 UI 자원은 토큰 없이 제공한다 (위 PUBLIC_FILES 주석 참조).
        # 로그인 화면도 style.css 를 쓰므로 이 처리가 인증보다 앞에 와야 한다.
        if p.lstrip("/") in PUBLIC_FILES:
            return self._static(p)

        if not self._authed(params):
            if p in ("/", "/index.html"):
                return self._send(401, LOGIN_HTML.encode("utf-8"),
                                  "text/html; charset=utf-8")
            return self._err(401, "토큰이 필요합니다")

        if p == "/" or not p.startswith("/api/"):
            return self._static(p if p != "/" else "index.html")

        if p == "/api/health":
            return self._json(lm_health())
        if p == "/api/patients":
            return self._json({
                "server": {k: self.paths[k] for k in ("incoming", "outgoing", "work")},
                "patients": [{k: v for k, v in c.items() if k != "files"}
                             for c in self.library.scan()],
                "inventory": self.library.inventory()})
        if p == "/api/jobs":
            return self._json({"jobs": self.runner.listing()})

        m = re.match(r"^/api/jobs/([0-9a-f]{12})$", p)
        if m:
            j = self.runner.get(m.group(1))
            return self._json(j.to_json()) if j else self._err(404, "없는 작업")

        m = re.match(r"^/api/jobs/([0-9a-f]{12})/log$", p)
        if m:
            j = self.runner.get(m.group(1))
            if not j:
                return self._err(404, "없는 작업")
            off = int((params.get("offset") or ["0"])[0])
            try:
                size = os.path.getsize(j.log_path)
                with open(j.log_path, "rb") as fh:
                    fh.seek(max(0, min(off, size)))
                    chunk = fh.read(200_000)
            except OSError:
                size, chunk = 0, b""
            return self._json({"offset": min(off, size) + len(chunk), "size": size,
                               "text": chunk.decode("utf-8", errors="replace"),
                               "status": j.status, "step": j.step})

        m = re.match(r"^/api/jobs/([0-9a-f]{12})/file/(.+)$", p)
        if m:
            return self._download(m.group(1), unquote(m.group(2)))

        return self._err(404, "없는 경로")

    def do_POST(self):
        u = urlparse(self.path)
        p, params = u.path, parse_qs(u.query)
        if not self._authed(params):
            return self._err(401, "토큰이 필요합니다")

        if p == "/api/config":
            b = self._body()
            url = str(b.get("base_url") or "").strip()
            if not re.match(r"^https?://[A-Za-z0-9._\-]+(:\d+)?(/[A-Za-z0-9._\-/]*)?$", url):
                return self._err(400, "주소 형식이 올바르지 않습니다")
            CONFIG["base_url"] = url
            CONFIG["model"] = (str(b.get("model")).strip() or None) \
                if b.get("model") else None
            return self._json(lm_health())

        if p == "/api/jobs":
            return self._create_job(self._body())

        m = re.match(r"^/api/jobs/([0-9a-f]{12})/cancel$", p)
        if m:
            ok = self.runner.cancel(m.group(1))
            return self._json({"cancelled": ok}, 200 if ok else 409)

        return self._err(404, "없는 경로")

    do_HEAD = do_GET

    # -- 동작 -------------------------------------------------------------
    def _download(self, job_id: str, name: str):
        j = self.runner.get(job_id)
        if not j:
            return self._err(404, "없는 작업")
        path = j.artifacts.get(name)
        if not path:
            return self._err(404, "없는 산출물")
        real, base = os.path.realpath(path), os.path.realpath(j.dir)
        if not real.startswith(base + os.sep) or not os.path.isfile(real):
            return self._err(403, "허용되지 않은 경로")
        with open(real, "rb") as fh:
            data = fh.read()
        self._send(200, data, "application/octet-stream",
                   {"Content-Disposition":
                    f'attachment; filename="{os.path.basename(real)}"'})

    def _create_job(self, body: dict):
        case = self.library.find(str(body.get("case_id", "")))
        if case is None:
            return self._err(404, "환자를 찾을 수 없습니다 (폴더가 바뀌었을 수 있습니다)")
        if not case["ready"]:
            return self._err(400, case["reason"])
        try:
            opts = normalize_opts(body.get("options") or {})
        except ValueError as e:
            return self._err(400, str(e))

        job_id = uuid.uuid4().hex[:12]
        work = os.path.join(self.paths["work"], job_id)
        os.makedirs(work, exist_ok=True)
        job = Job(job_id, case, opts, work)
        open(job.log_path, "w", encoding="utf-8").close()
        self.runner.submit(job)
        with open(os.path.join(self.paths["work"], "jobs.jsonl"), "a",
                  encoding="utf-8") as fh:
            fh.write(json.dumps({"time": datetime.now().isoformat(), "job": job_id,
                                 "patient_id": case["patient_id"],
                                 "study_uid": case["study_uid"],
                                 "client": self.address_string(),
                                 "options": opts}, ensure_ascii=False) + "\n")
        return self._json(job.to_json(), 201)


LOGIN_HTML = """<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Brachy RT auto planning — 인증</title>
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/style.css">
<body><header class="topbar"><div class="brand">
<span class="brand-logo"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="2.6"/>
<path d="M13.9 15.29 L16.75 20.23 A9.5 9.5 0 0 1 7.25 20.23 L10.1 15.29 A3.8 3.8 0 0 0 13.9 15.29 Z"/>
<path d="M8.2 12 L2.5 12 A9.5 9.5 0 0 1 7.25 3.77 L10.1 8.71 A3.8 3.8 0 0 0 8.2 12 Z"/>
<path d="M13.9 8.71 L16.75 3.77 A9.5 9.5 0 0 1 21.5 12 L15.8 12 A3.8 3.8 0 0 0 13.9 8.71 Z"/>
</svg></span><span class="brand-name">Brachy&nbsp;RT&nbsp;<b>auto&nbsp;planning</b></span>
</div></header>
<main class="main"><div class="panel" style="max-width:440px">
<h2>접속 토큰</h2><div class="sub">서버를 실행한 콘솔에 출력된 토큰을 입력하십시오.</div>
<div class="row"><input class="txt wide" id="t" type="password" placeholder="토큰" autofocus>
<button class="btn primary" onclick="go()">접속</button></div></div></main>
<script>
function go(){var v=document.getElementById('t').value.trim();
if(v) location.href='/?token='+encodeURIComponent(v);}
document.getElementById('t').addEventListener('keydown',function(e){
if(e.key==='Enter') go();});
</script></body></html>"""


# ===========================================================================
def local_addresses() -> list[str]:
    """이 PC로 접속할 때 쓸 수 있는 IPv4 주소들.

    0.0.0.0 은 "모든 인터페이스에서 수신"이라는 **바인딩 표기**이지 접속
    주소가 아니다. 그대로 안내하면 브라우저에서 열리지 않으므로 실제 주소를
    찾아 보여준다.
    """
    import socket
    found: list[str] = []

    def add(ip: str) -> None:
        if (ip and ip not in found and not ip.startswith("127.")
                and not ip.startswith("169.254.")):
            found.append(ip)

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))       # 패킷은 나가지 않는다
            add(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            add(ip)
    except OSError:
        pass
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=APP_NAME)
    ap.add_argument("--host", default="127.0.0.1",
                    help="바인딩 주소. 다른 PC에서 접속하려면 0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--incoming", default="incoming", help="DICOM 수신 폴더")
    ap.add_argument("--outgoing", default="outgoing", help="RTPLAN 출력 폴더")
    ap.add_argument("--work", default="work", help="작업 폴더")
    ap.add_argument("--token", default=None, help="미지정 시 임의 발급")
    ap.add_argument("--lmstudio", default=DEFAULT_BASE_URL,
                    help="LM Studio base URL (기본 http://localhost:1234/v1)")
    ap.add_argument("--model", default=None, help="모델 id (미지정 시 자동)")
    ap.add_argument("--start-lmstudio", action="store_true",
                    help="LM Studio 로컬 서버 기동을 시도한다 (`lms server start`)")
    ap.add_argument("--no-browser", action="store_true", help="브라우저 자동 실행 안 함")
    args = ap.parse_args()

    if not os.path.isdir(WEB_DIR):
        print(f"[오류] webapp 폴더가 없습니다: {WEB_DIR}", file=sys.stderr)
        return 2

    def abspath(p):
        # UNC(\\서버\공유)는 그대로, 상대경로만 프로그램 폴더 기준으로 푼다
        return p if is_abs_path(p) else os.path.join(ROOT, p)

    paths = {k: abspath(v) for k, v in
             (("incoming", args.incoming), ("outgoing", args.outgoing),
              ("work", args.work))}

    # 네트워크 공유(SMB)가 끊겨 있으면 여기서 알아채야 한다.
    # 나중에 작업 도중 터지면 원인을 찾기 어렵다.
    problems = []
    for key, need_write in (("incoming", False), ("outgoing", True),
                            ("work", True)):
        why = check_dir(paths[key], need_write)
        if why:
            problems.append((key, paths[key], why))
    if problems:
        print("=" * 72)
        print(f" {APP_NAME} — 폴더를 사용할 수 없습니다")
        print("=" * 72)
        for key, path, why in problems:
            label = {"incoming": "수신", "outgoing": "출력", "work": "작업"}[key]
            print(f" [{label}] {path}\n        {why}")
        print("\n 네트워크 공유(\\\\서버\\공유)라면 확인하십시오:")
        print("   · 탐색기에서 그 경로가 열리는가")
        print("   · 자격 증명이 필요한가  →  net use \\\\서버\\공유 /user:도메인\\계정")
        print("   · 매핑된 드라이브 문자(Z:)를 쓰지 말고 UNC 경로를 그대로 주십시오")
        print("     (드라이브 문자는 로그인 세션에만 존재합니다)")
        print("=" * 72)
        return 2

    CONFIG["base_url"] = args.lmstudio
    CONFIG["model"] = args.model

    token = args.token or secrets.token_urlsafe(18)
    runner = Runner(paths["outgoing"])
    runner.start()

    Handler.library = Library(paths["incoming"])
    Handler.runner = runner
    Handler.token = token
    Handler.paths = paths

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)

    bind_all = args.host in ("0.0.0.0", "::", "")
    if bind_all:
        hosts = ["localhost"] + local_addresses()
    elif args.host in ("127.0.0.1", "localhost"):
        hosts = ["localhost"]
    else:
        hosts = [args.host]

    print("=" * 72)
    print(f" {APP_NAME}")
    print("=" * 72)
    print(f" 수신 폴더 : {paths['incoming']}")
    print(f" 출력 폴더 : {paths['outgoing']}")
    print(f" 작업 폴더 : {paths['work']}")
    print(f" 바인딩    : {args.host}:{args.port}"
          + ("  (모든 인터페이스)" if bind_all else ""))
    print(f" 토큰      : {token}")

    if args.start_lmstudio:
        print("\n LM Studio 기동 시도 중…")
        print(f"   → {start_lmstudio()}")
    h = lm_health()
    print(f"\n LM Studio : {h['base_url']} — "
          + (f"연결됨, 모델 {len(h['models'])}개"
             + (f" (사용: {h['model']})" if h['model'] else "")
             if h["ok"] else f"미연결 ({h['error']})"))
    if not h["ok"]:
        print("   LM Studio ▸ Developer ▸ Start Server 로 켜거나, 화면 우측 상단 ⚙에서")
        print("   주소를 바꾸십시오. 연결되지 않아도 내장 휴리스틱으로 실행됩니다.")

    print("\n 접속 주소")
    for hh in hosts:
        label = " (이 PC에서만)" if hh == "localhost" else " (다른 PC에서)"
        print(f"   http://{hh}:{args.port}/?token={token}{label}")
    if bind_all and len(hosts) == 1:
        print("   [주의] 이 PC의 LAN IP를 찾지 못했습니다. `ipconfig` 로 IPv4 주소를")
        print(f"          확인해 http://<그 주소>:{args.port}/?token=... 로 접속하십시오.")
    if bind_all or args.host not in ("127.0.0.1", "localhost"):
        print("\n [경고] 이 서버는 TLS가 없습니다. 환자 정보가 평문으로 오갑니다.")
        print("        신뢰된 내부망에서만 사용하고 외부망에 노출하지 마십시오.")
        print("        방화벽에서 이 포트를 내부망으로만 열어 두십시오.")
    print("\n ※ 연구용입니다. 생성 RTPLAN은 UNAPPROVED이며 TPS 재계산과 독립 검증")
    print("    없이는 치료에 사용할 수 없습니다.")
    print("\n 종료: Ctrl+C")
    print("=" * 72)

    if not args.no_browser:
        url = f"http://localhost:{args.port}/?token={token}"
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n종료합니다.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
