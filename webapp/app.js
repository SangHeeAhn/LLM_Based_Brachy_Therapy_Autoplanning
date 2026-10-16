/* Brachy RT auto planning — 대시보드 프런트엔드
   외부 라이브러리를 쓰지 않는다 (폐쇄망에 CDN 접근이 없다).
   아이콘은 전부 인라인 SVG로 그린다 — 글리프 문자는 폰트마다 크기가 달라
   글자와 정렬이 어긋난다. */
"use strict";

/* ============================== 아이콘 ============================== */
const P = {
  grid:    '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  terminal:'<path d="M4 17l5-5-5-5"/><path d="M12 19h8"/>',
  database:'<ellipse cx="12" cy="5.5" rx="8" ry="3"/><path d="M4 5.5v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/><path d="M4 11.5v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
  info:    '<circle cx="12" cy="12" r="9"/><path d="M12 16v-5M12 8h.01"/>',
  gear:    '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1A1.6 1.6 0 0 0 9 19.4a1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1A1.6 1.6 0 0 0 4.6 9a1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z"/>',
  search:  '<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.6-3.6"/>',
  refresh: '<path d="M20 11a8 8 0 0 0-13.7-5.7L4 7.5"/><path d="M4 4v4h4"/><path d="M4 13a8 8 0 0 0 13.7 5.7L20 16.5"/><path d="M20 20v-4h-4"/>',
  user:    '<circle cx="12" cy="8" r="4"/><path d="M4.5 20a7.5 7.5 0 0 1 15 0"/>',
  play:    '<path d="M6 4.5l13 7.5-13 7.5z"/>',
  download:'<path d="M12 3v12"/><path d="M7.5 10.5L12 15l4.5-4.5"/><path d="M4 20h16"/>',
  arrow:   '<path d="M4 12h15"/><path d="M13 6l6 6-6 6"/>',
  back:    '<path d="M20 12H5"/><path d="M11 18l-6-6 6-6"/>',
  shield:  '<path d="M12 3l7.5 3v5.5c0 4.8-3.2 8.4-7.5 10-4.3-1.6-7.5-5.2-7.5-10V6z"/>',
  chip:    '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9.5 2.5v3.5M14.5 2.5v3.5M9.5 18v3.5M14.5 18v3.5M2.5 9.5H6M2.5 14.5H6M18 9.5h3.5M18 14.5h3.5"/>',
  list:    '<path d="M4 6h16M4 12h16M4 18h10"/>',
  check:   '<circle cx="12" cy="12" r="9"/><path d="M8.5 12.5l2.5 2.5 4.5-5"/>',
  alert:   '<path d="M10.3 4.3L2.6 17.5A2 2 0 0 0 4.3 20.5h15.4a2 2 0 0 0 1.7-3L13.7 4.3a2 2 0 0 0-3.4 0z"/><path d="M12 9.5v4M12 17h.01"/>',
  x:       '<circle cx="12" cy="12" r="9"/><path d="M15 9l-6 6M9 9l6 6"/>',
  clock:   '<circle cx="12" cy="12" r="9"/><path d="M12 7v5.2l3.2 2"/>',
  file:    '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>',
  atom:    '<circle cx="12" cy="12" r="2.6"/><ellipse cx="12" cy="12" rx="9.5" ry="4" /><ellipse cx="12" cy="12" rx="9.5" ry="4" transform="rotate(60 12 12)"/><ellipse cx="12" cy="12" rx="9.5" ry="4" transform="rotate(120 12 12)"/>',
};
const ico = (n, cls) =>
  `<svg class="i ${cls || ""}" viewBox="0 0 24 24" aria-hidden="true">${P[n] || ""}</svg>`;

/* ============================== 상태 ============================== */
const TOKEN = new URLSearchParams(location.search).get("token") || "";
const H = { "X-Auth-Token": TOKEN, "Content-Type": "application/json" };

const NAV = [
  { id: "plan", icon: "grid",     label: "자동계획" },
  { id: "jobs", icon: "terminal", label: "작업" },
  { id: "data", icon: "database", label: "데이터" },
  { id: "info", icon: "info",     label: "정보" },
];

const S = {
  view: "plan", patients: [], inventory: [], jobs: [], sel: null, selJob: null,
  health: null, server: null, query: "", busy: false,
  // 작업별 로그를 화면이 아니라 **상태에** 모은다.
  // 탭을 옮기면 <pre>가 사라지므로, DOM에만 쌓으면 돌아왔을 때 비어 보인다.
  logs: {},        // { jobId: {text, off} }
};
const logOf = id => (S.logs[id] || (S.logs[id] = { text: "", off: 0 }));

/* ============================== 유틸 ============================== */
async function api(path, opts) {
  const r = await fetch(path, Object.assign({ headers: H }, opts || {}));
  if (!r.ok) {
    let m; try { m = (await r.json()).error; } catch (e) { m = r.statusText; }
    throw new Error(m || "HTTP " + r.status);
  }
  return r.json();
}
const esc = s => (s == null ? "" : String(s)).replace(/[&<>"]/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const $ = id => document.getElementById(id);
const hhmm = t => new Date(t * 1000).toLocaleTimeString("ko-KR",
  { hour: "2-digit", minute: "2-digit", second: "2-digit" });
const fileSummary = c => Object.entries(c).map(([k, v]) => k + " " + v).join(" · ");

/* ============================== 셸 ============================== */
function buildShell() {
  $("brand-mark").innerHTML = ico("atom");
  $("settings-btn").innerHTML = ico("gear");
  $("searchbox").insertAdjacentHTML("afterbegin", ico("search"));
  $("refresh-btn").insertAdjacentHTML("afterbegin", ico("refresh"));
  $("rail").innerHTML = NAV.map(n =>
    `<button class="rail-btn" data-view="${n.id}" title="${n.label}">${ico(n.icon)}</button>`
  ).join("");
  $("catlist").innerHTML = NAV.map(n =>
    `<li class="cat" data-view="${n.id}">${ico(n.icon)}<span>${n.label}</span>` +
    (n.id === "info" ? "" : `<span class="count" id="c-${n.id}">0</span>`) + `</li>`
  ).join("");
  document.querySelectorAll(".rail-btn,.cat").forEach(b => {
    b.onclick = () => { S.view = b.dataset.view; S.sel = null; render(); };
  });
}

function paintTop() {
  const h = S.health || {};
  const lm = $("lm-pill");
  lm.className = "pill " + (h.ok ? "green" : "amber");
  lm.innerHTML = ico("shield") + `<span>${h.ok ? "LM Studio 연결됨" : "LM Studio 미연결"}</span>`;
  lm.title = h.ok ? h.base_url + " · 모델 " + (h.models || []).length + "개"
                  : "연결 실패: " + (h.error || "") + "\n이 상태로도 휴리스틱으로 실행됩니다";
  $("model-box").textContent = h.ok ? (h.model || h.models[0] || "모델 없음") : "모델 미연결";

  const act = S.jobs.filter(j => j.status === "queued" || j.status === "running");
  const q = $("queue-pill");
  q.className = "pill " + (act.length ? "amber" : "");
  q.innerHTML = ico(act.length ? "clock" : "list") +
    `<span>${act.length ? (act.some(j => j.status === "running")
      ? "실행 중 " + act.length + "건" : "대기 " + act.length + "건") : "작업 없음"}</span>`;

  const sv = S.server || {};
  $("side-foot").innerHTML = sv.incoming
    ? `<b>수신</b><br>${esc(sv.incoming)}<br><b>출력</b><br>${esc(sv.outgoing)}`
    : "";
  $("c-plan").textContent = S.patients.filter(p => p.ready).length;
  $("c-data").textContent = S.inventory.length;
  $("c-jobs").textContent = S.jobs.length;
}

/* ============================== 데이터 ============================== */
async function refreshHealth() {
  try { S.health = await api("/api/health"); }
  catch (e) { S.health = { ok: false, error: e.message, models: [] }; }
  paintTop();
}
async function refreshJobs(rerender) {
  try {
    const before = JSON.stringify(S.jobs.map(j => [j.id, j.status, j.step]));
    S.jobs = (await api("/api/jobs")).jobs;
    paintTop();
    if (rerender && S.view === "jobs" &&
        before !== JSON.stringify(S.jobs.map(j => [j.id, j.status, j.step]))) render();
  } catch (e) { /* 무시 */ }
}
async function loadPatients() {
  try {
    const d = await api("/api/patients");
    S.patients = d.patients; S.server = d.server;
    S.inventory = d.inventory || [];
  } catch (e) {
    S.patients = [];
    $("side-foot").textContent = "오류: " + e.message;
  }
  paintTop(); render();
}
const match = (o, q) =>
  ((o.patient_id || "") + " " + (o.patient_name || "") + " " +
   (o.plan_label || "")).toLowerCase().includes(q);
// 자동계획 탭에는 **실행 가능한 plan만** 보인다.
// 영상(MR/CT)만 있는 검사는 애초에 케이스로 잡히지 않고, RTSTRUCT를 못 찾은
// plan은 데이터 탭에서 확인한다.
const filtered = () => {
  const q = S.query.trim().toLowerCase();
  const list = S.patients.filter(p => p.ready);
  return q ? list.filter(p => match(p, q)) : list;
};
const invFiltered = () => {
  const q = S.query.trim().toLowerCase();
  return q ? S.inventory.filter(g => match(g, q)) : S.inventory;
};

/* ============================== 렌더 ============================== */
function render() {
  document.querySelectorAll(".rail-btn").forEach(b =>
    b.classList.toggle("on", b.dataset.view === S.view));
  document.querySelectorAll(".cat").forEach(b =>
    b.classList.toggle("on", b.dataset.view === S.view));
  const v = $("view");
  v.innerHTML = S.view === "plan" ? (S.sel ? viewSetup() : viewPatients())
    : S.view === "jobs" ? viewJobs()
    : S.view === "data" ? viewData() : viewInfo();
  bind();
  v.scrollTop = 0;
}

const NOTICE = `<div class="notice">${ico("alert")}<div><b>연구용입니다.</b>
 생성되는 RTPLAN은 ApprovalStatus = UNAPPROVED이며, TPS로 선량을 재계산하고
 독립 검증을 거치기 전에는 치료에 사용할 수 없습니다.</div></div>`;

/* ---------- 1단계: 환자 카드 ---------- */
function patientCard(p) {
  const link = p.struct_link && p.struct_link !== "참조"
    ? `<span class="tag info">RTSTRUCT: ${esc(p.struct_link)}</span>` : "";
  return `<div class="card click" data-case="${p.case_id}">
    <div class="card-head">
      <span class="card-ic">${ico("user")}</span>
      <div class="card-ttl">
        <h3>${esc(p.patient_id)}</h3>
        ${p.plan_label ? `<span class="tag">${esc(p.plan_label)}</span>` : ""}
        ${link}
      </div>
    </div>
    <div class="card-body">
      <p>${esc(p.patient_name) || "이름 없음"}
        ${p.study_date ? "<br>검사일 " + esc(p.study_date) : ""}
        <br>${esc(fileSummary(p.counts))}</p>
      <div class="card-act">이 plan으로 계획${ico("arrow")}</div>
    </div>
  </div>`;
}

function viewPatients() {
  const list = filtered();
  return `<div class="head"><h1>자동계획</h1>
      <span class="count-pill">환자 ${list.length}명</span></div>
    ${NOTICE}
    ${list.length ? `<div class="grid">${list.map(patientCard).join("")}</div>`
      : `<div class="panel"><div class="empty">실행할 수 있는 RTPLAN이 없습니다.<br>
         자동계획에는 <b>RTPLAN + RTSTRUCT</b> 두 개면 충분합니다 (CT/MR 불필요).<br>
         수신 폴더에 넣고 좌측 <b>환자 목록 새로고침</b>을 누르십시오.<br>
         넣었는데도 안 보이면 <b>데이터</b> 탭에서 무엇이 들어왔는지 확인하십시오.</div></div>`}`;
}

/* ---------- 2단계: 설정 + 실행 ---------- */
function viewSetup() {
  const p = S.sel, lm = S.health && S.health.ok;
  const num = (id, label, val) =>
    `<div class="f"><label for="o-${id}">${label}</label>
     <input id="o-${id}" value="${val}"></div>`;
  const sel = (id, label, opts) =>
    `<div class="f"><label for="o-${id}">${label}</label><select id="o-${id}">${
      opts.map(([v, t]) => `<option value="${v}">${t}</option>`).join("")}</select></div>`;
  const chk = (id, label, on) =>
    `<label class="chk"><input type="checkbox" id="o-${id}"${on ? " checked" : ""}>
     <span>${label}</span></label>`;

  return `<button class="back" id="back-btn">${ico("back")}환자 목록으로</button>
  <div class="head"><h1>${esc(p.patient_id)}</h1>
    <span class="count-pill">${esc(p.patient_name) || "이름 없음"}</span>
    <span class="count-pill">${esc(fileSummary(p.counts))}</span></div>
  ${NOTICE}
  <div class="panel">
    <h2>계획 설정</h2>
    <div class="sub">기본값 그대로 실행해도 됩니다. 그리드를 5~6 mm로 키우면
      시험 실행이 빨라집니다.</div>

    <div class="sect" style="border:0;margin:0;padding:0">
      <h4>기본</h4>
      <div class="fields">
        ${num("spacing", "그리드 [mm]", "2.0")}
        ${num("max_iter", "에이전트 반복", "12")}
        ${num("time_resolution", "시간 분해능 [s]", "0.1")}
        ${num("target", "주표적", "HR_CTV")}
        ${num("secondary", "부표적", "IR_CTV")}
        ${num("seed", "seed", "0")}
      </div>
    </div>

    <div class="sect"><h4>임상 목표</h4>
      <div class="fields">
        ${sel("goals_source", "목표 출처",
          [["embrace2", "EMBRACE-II 프로토콜"], ["clinical", "임상 plan 실측값"]])}
        ${sel("objectives", "최적화 파라미터",
          [["goals", "목표에서 유도 (신환 가능)"], ["clinical", "임상 plan 실측 D2cm3"]])}
        ${num("ebrt_dose", "EBRT 선량 [Gy]", "45")}
        ${num("ebrt_fractions", "EBRT 분할", "25")}
      </div>
    </div>

    <div class="sect"><h4>LLM</h4>
      ${lm ? "" : `<div class="msg warn">${ico("alert")}<div>LM Studio에 연결되어
        있지 않습니다. 이대로 실행하면 내장 휴리스틱으로 진행합니다.</div></div>`}
      <div class="fields">
        <div class="f span2"><label for="o-base_url">LM Studio Base URL</label>
          <input id="o-base_url" value="${esc((S.health && S.health.base_url)
            || "http://localhost:1234/v1")}"></div>
      </div>
      <div class="checks">
        ${chk("use_llm", "LM Studio 사용", lm)}
        ${chk("no_schema", "structured output 미사용 (reasoning 모델 회피)", false)}
      </div>
    </div>

    <div class="sect"><h4>내보내기 · 수신 TPS 호환</h4>
      <div class="fields">
        ${sel("uid_style", "UID 방식",
          [["pydicom", "표준 (64자)"], ["source", "원본 루트·길이 (구형 TPS)"]])}
      </div>
      <div class="checks">
        ${chk("keep_series_uid", "Series UID 원본 유지", false)}
        ${chk("new_private_uid", "벤더 私有 UID 새로 발급", false)}
        ${chk("skip_dose_check", "선량 검증 생략 (빠름)", false)}
      </div>
    </div>

    <div class="actionbar">
      <button class="btn dark" id="run-btn">${ico("play")}자동계획 실행</button>
      <span class="note">Phase 3 → RTPLAN 내보내기 → 원본 대조 검증까지
        한 번에 실행합니다.</span>
    </div>
  </div>`;
}

/* ---------- 작업 ---------- */
const JOB_TAG = { done: ["ok", "check"], failed: ["bad", "x"],
                  cancelled: ["bad", "x"], running: ["run", "clock"],
                  queued: ["", "clock"] };

function viewJobs() {
  const rows = S.jobs.length ? S.jobs.map(j => {
    const [cls, ic] = JOB_TAG[j.status] || ["", "clock"];
    const s = j.summary || {};
    return `<tr class="click${j.id === S.selJob ? " on" : ""}" data-job="${j.id}">
      <td>${hhmm(j.created)}</td>
      <td><b>${esc(j.case.patient_id)}</b></td>
      <td><span class="tag ${cls}">${j.status}</span></td>
      <td style="color:var(--mut)">${esc(j.step)}</td>
      <td class="num">${s.score != null ? s.score.toFixed(1) : "–"}</td>
      <td class="num">${s.met || "–"}</td>
      <td style="text-align:right">${
        (j.status === "queued" || j.status === "running")
          ? `<button class="btn sm" data-cancel="${j.id}">취소</button>` : ""}</td>
    </tr>`;
  }).join("") : `<tr><td colspan="7" class="empty">아직 실행한 작업이 없습니다.</td></tr>`;

  return `<div class="head"><h1>작업</h1>
      <span class="count-pill">${S.jobs.length}건</span></div>
    <div class="panel" style="padding:8px 16px 4px">
      <table class="tbl"><thead><tr><th>시각</th><th>환자</th><th>상태</th>
        <th>단계</th><th class="num">점수</th><th class="num">충족</th><th></th>
        </tr></thead><tbody>${rows}</tbody></table>
    </div>
    ${jobDetail()}`;
}

function jobDetail() {
  const j = S.jobs.find(x => x.id === S.selJob);
  if (!j) {
    return `<div class="panel"><h2>실행 로그</h2>
      <div class="sub">위에서 작업을 선택하면 로그와 산출물이 표시됩니다.</div></div>`;
  }
  const s = j.summary || {};
  const stats = [];
  if (s.score != null) stats.push(["종합 점수", s.score.toFixed(1) + " / 100", ""]);
  if (s.met) stats.push(["목표 충족", s.met, ""]);
  if (s.hard_ok != null) stats.push(["필수 목표", s.hard_ok ? "PASS" : "FAIL",
    s.hard_ok ? "good" : "bad"]);
  if (s.export_ok != null) stats.push(["파일 정합성", s.export_ok ? "PASS" : "FAIL",
    s.export_ok ? "good" : "bad"]);
  if (s.total_time_s != null) stats.push(["총 조사시간", s.total_time_s + " s", ""]);
  if (s.total_time_original_s != null)
    stats.push(["원본 조사시간", s.total_time_original_s + " s", ""]);

  const running = j.status === "running" || j.status === "queued";
  return `<div class="panel">
    <h2>${esc(j.case.patient_id)} <span style="color:var(--mut);font-weight:400">
      · ${esc(j.case.patient_name)}</span></h2>
    <div class="sub">작업 ${j.id} · ${j.status}${j.step ? " · " + esc(j.step) : ""}
      ${running ? ' <span class="spin"></span>' : ""}</div>
    ${j.error ? `<div class="msg err">${ico("x")}<div>${esc(j.error)}</div></div>` : ""}
    ${(j.warnings || []).map(w =>
      `<div class="msg warn">${ico("alert")}<div>${esc(w)}</div></div>`).join("")}
    ${stats.length ? `<div class="stats">${stats.map(([k, v, c]) =>
      `<div class="stat"><div class="k">${k}</div>
       <div class="v ${c}">${esc(v)}</div></div>`).join("")}</div>` : ""}
    ${(j.artifacts || []).length ? `<div class="actionbar">${j.artifacts.map(a =>
      `<button class="btn ${a.endsWith(".dcm") ? "green" : ""}" data-dl="${esc(a)}">
        ${ico("download")}${a.endsWith(".dcm") ? "RTPLAN 내려받기" : esc(a)}</button>`
      ).join("")}</div>` : ""}
    <pre class="log" id="log">${esc(logOf(j.id).text) || "로그를 불러오는 중…"}</pre>
  </div>`;
}

/* ---------- 데이터 ---------- */
function viewData() {
  const list = invFiltered();
  const plansOf = uid => S.patients.filter(p => p.study_uid === uid);
  const cards = list.map(g => {
    const plans = plansOf(g.study_uid);
    const usable = plans.filter(p => p.ready).length;
    return `<div class="card">
    <div class="card-head">
      <span class="card-ic">${ico("file")}</span>
      <div class="card-ttl"><h3>${esc(g.patient_id)}</h3>
        ${usable ? `<span class="tag ok">실행 가능 plan ${usable}개</span>`
                 : `<span class="tag">계획 대상 아님</span>`}
      </div>
    </div>
    <div class="card-body"><p>
      ${esc(g.patient_name) || "이름 없음"}<br>
      검사일 ${esc(g.study_date) || "–"}<br>
      ${Object.entries(g.counts).map(([k, v]) =>
        `${esc(k)} <b>${v}</b>`).join(" · ")}<br>
      ${plans.filter(p => !p.ready).map(p =>
        `<span style="color:#b45309">· ${esc(p.plan_label || "RTPLAN")}: ${esc(p.reason)}</span>`
        ).join("<br>")}
      <span style="color:var(--dim)">Study …${esc((g.study_uid || "").slice(-16))}</span>
    </p></div></div>`;
  }).join("");
  return `<div class="head"><h1>데이터</h1>
      <span class="count-pill">검사 ${list.length}건</span></div>
    <div class="notice">${ico("info")}<div>수신 폴더에 들어온 모든 DICOM입니다.
      자동계획은 <b>RTPLAN + RTSTRUCT</b> 만 사용하며 CT/MR은 읽지 않습니다 —
      영상만 있는 검사는 자동계획 탭에 나타나지 않습니다.</div></div>
    ${cards ? `<div class="grid">${cards}</div>`
      : `<div class="panel"><div class="empty">수신 폴더가 비어 있습니다.</div></div>`}`;
}

/* ---------- 정보 ---------- */
function viewInfo() {
  const sv = S.server || {}, h = S.health || {};
  const row = (k, v) => `<tr><td style="color:var(--mut);width:170px">${k}</td>
    <td>${v}</td></tr>`;
  return `<div class="head"><h1>정보</h1></div>
  ${NOTICE}
  <div class="panel"><h2>환경</h2>
    <div class="sub">서버가 사용하는 폴더와 LM Studio 연결 상태입니다.</div>
    <table class="tbl"><tbody>
      ${row("수신 폴더", esc(sv.incoming || "–"))}
      ${row("출력 폴더", esc(sv.outgoing || "–"))}
      ${row("작업 폴더", esc(sv.work || "–"))}
      ${row("LM Studio", esc(h.base_url || "–") + " — " +
        (h.ok ? "연결됨" : "미연결 (휴리스틱으로 동작)"))}
      ${row("로드된 모델", esc((h.models || []).join(", ") || "–"))}
    </tbody></table>
  </div>
  <div class="panel"><h2>파이프라인</h2>
    <div class="sub">환자 선택 → 자동계획(Phase 3) → RTPLAN 내보내기 →
      원본 plan 대조 검증 → 출력 폴더 복사. 작업은 한 번에 하나씩 처리됩니다.
      계산 자체는 빠릅니다 — 2 mm 그리드에서 kernel 0.2초, 최적화 1회 0.2초,
      LLM을 끄면 전 과정이 10초쯤입니다. <b>오래 걸린다면 그건 LLM 응답
      대기입니다</b> (모델에 따라 호출당 최대 150초). 작업 로그의
      <code>[시간 분해]</code> 표에서 확인하십시오.</div>
  </div>
  <div class="panel"><h2>동작 원리</h2>
    <div class="sub">LLM은 dwell time을 직접 만들지 않습니다. 목적함수의
      가중치·목표선량만 제안하고, 실제 dwell time은 결정론적 볼록 최적화가
      계산합니다. dwell <b>위치</b>는 원본 RTPLAN 그대로이며 이 도구가 바꾸지
      않습니다. 아래는 그 계산의 수식 요약입니다 (전체 설명은 저장소의
      METHOD.md 참고).</div>

    <div class="info-step">
      <h3><span class="n">1</span>결정 변수 — dwell time</h3>
      <p>정지위치가 N개면 우리가 정하는 값은 이 N개의 시간(초)뿐입니다.
        위치는 바꾸지 않습니다.</p>
      <div class="eq"><span class="c">t = (t₁, t₂, …, t_N)</span>,&nbsp;&nbsp;<span class="c">제약</span>  0 ≤ tₖ ≤ t_max  <span class="c">(기본 200 s)</span></div>
    </div>

    <div class="info-step">
      <h3><span class="n">2</span>선량은 시간에 비례 — 영향행렬</h3>
      <p>TG-43 형식체계에서 한 점의 선량은 dwell time에 정비례하므로,
        정지위치별 "초당 선량"을 미리 표(영향행렬 A, 단위 Gy/s)로 계산해
        두면 이후 계산은 곱셈 한 번으로 끝납니다.</p>
      <div class="eq">점 P의 선량 = A₁·t₁ + A₂·t₂ + … + A_N·t_N

<span class="c">행렬로 쓰면:</span>  <b>dose = Aᵀ · t</b>   <span class="c">A : (N개 dwell) × (전체 복셀), 단위 Gy/s</span></div>
    </div>

    <div class="info-step">
      <h3><span class="n">3</span>목적함수 J(t) — "얼마나 나쁜 plan인가"</h3>
      <p>표적은 목표선량 미달을, 정상장기는 상한 초과를 제곱으로 벌점화하고,
        인접 dwell 간 시간 차이에도 벌점을 둡니다. J가 작을수록 좋은 plan.</p>
      <div class="eq">J(t) = Σₛ w_under,ₛ · mean[ max(0, Dmin_s − d)² ]   <span class="c">← 표적 커버리지 부족분</span>
    + Σₛ w_over,ₛ  · mean[ max(0, d − Dmax_s)² ]   <span class="c">← 정상장기 초과분</span>
    + λ · Σₖ (tₖ − tₖ₊₁)²                          <span class="c">← 평활항, λ = lam_smooth (기본 0.05)</span></div>
      <p><span class="c">w</span> = 가중치(LLM이 조정하는 값), <span class="c">d</span> = A ᵀt로 계산된 실제 선량,
        <span class="c">Dmin/Dmax</span> = EMBRACE-II에서 환산한 목표선량.</p>
    </div>

    <div class="info-step">
      <h3><span class="n">4</span>왜 항상 같은 답으로 수렴하는가 — 볼록성</h3>
      <p>dose가 t의 1차식이고 max(0,·)²이 볼록함수이므로, 이를 합한 J(t)도
        볼록합니다 — 골짜기가 하나뿐이라 시작점과 무관하게 유일해로
        수렴합니다.</p>
      <div class="eq"><span class="c">실측(4가지 초기값 비교):</span>  J = 0.0807029383  <span class="c">(4종 모두 동일)</span>
<span class="c">해 벡터 최대 편차:</span>  0.00000018 s</div>
      <p>즉 "최적화가 실패해 이상한 plan이 나오는" 일은 구조적으로
        일어나지 않습니다 — 결과가 마음에 안 들면 그건 가중치 선택의
        문제입니다.</p>
    </div>

    <div class="info-step">
      <h3><span class="n">5</span>LLM의 역할 — 파라미터만 제안</h3>
      <p>LLM은 위 J(t)의 w_under, w_over, Dmin, Dmax만 제안하고, t는 항상
        2~4단계의 결정론적 최적화가 계산합니다.</p>
      <div class="eq">LLM: {"structure":"Bladder","field":"weight_over","value":10.0}
   ↓
볼록 최적화: t = argmin J(t)   0 ≤ t ≤ t_max     <span class="c">← 결정론적</span>
   ↓
DVH 평가 + 점수(0~100) → 개선 시 채택, 악화 시 되돌림</div>
    </div>
  </div>`;
}

/* ============================== 바인딩 ============================== */
function bind() {
  document.querySelectorAll("[data-case]").forEach(el => {
    const p = S.patients.find(x => x.case_id === el.dataset.case);
    if (p && p.ready) el.onclick = () => { S.sel = p; render(); };
  });
  const back = $("back-btn");
  if (back) back.onclick = () => { S.sel = null; render(); };
  const run = $("run-btn");
  if (run) run.onclick = doRun;

  document.querySelectorAll("[data-job]").forEach(tr => {
    tr.onclick = ev => {
      if (ev.target.closest("[data-cancel]")) return;
      S.selJob = tr.dataset.job; render();
    };
  });
  document.querySelectorAll("[data-cancel]").forEach(b => {
    b.onclick = async ev => {
      ev.stopPropagation();
      try { await api("/api/jobs/" + b.dataset.cancel + "/cancel", { method: "POST" }); }
      catch (e) { alert(e.message); }
      refreshJobs(true);
    };
  });
  document.querySelectorAll("[data-dl]").forEach(b => {
    b.onclick = () => {
      location.href = `/api/jobs/${S.selJob}/file/${encodeURIComponent(b.dataset.dl)}`
        + `?token=${encodeURIComponent(TOKEN)}`;
    };
  });
  const el = $("log");
  if (el) {
    el.scrollTop = el.scrollHeight;   // 다시 그린 직후 맨 아래로
    pollLog();
  }
}

function readOpts() {
  const g = id => ($("o-" + id) ? $("o-" + id).value : undefined);
  const c = id => ($("o-" + id) ? $("o-" + id).checked : false);
  return {
    spacing: +g("spacing"), max_iter: +g("max_iter"),
    time_resolution: +g("time_resolution"), target: g("target"),
    secondary: g("secondary"), seed: +g("seed"),
    goals_source: g("goals_source"), objectives: g("objectives"),
    ebrt_dose: +g("ebrt_dose"), ebrt_fractions: +g("ebrt_fractions"),
    use_llm: c("use_llm"), no_schema: c("no_schema"), base_url: g("base_url"),
    uid_style: g("uid_style"), keep_series_uid: c("keep_series_uid"),
    new_private_uid: c("new_private_uid"), skip_dose_check: c("skip_dose_check"),
  };
}

async function doRun() {
  if (!S.sel || S.busy) return;
  S.busy = true;
  const b = $("run-btn");
  b.disabled = true; b.innerHTML = '<span class="spin"></span>제출 중…';
  try {
    const j = await api("/api/jobs", { method: "POST",
      body: JSON.stringify({ case_id: S.sel.case_id, options: readOpts() }) });
    await refreshJobs(false);
    S.selJob = j.id; S.view = "jobs"; S.sel = null;
    render();
  } catch (e) {
    alert("실행 실패: " + e.message);
    b.disabled = false; b.innerHTML = ico("play") + "자동계획 실행";
  } finally { S.busy = false; }
}

async function pollLog() {
  if (!S.selJob) return;
  const st = logOf(S.selJob);
  try {
    const d = await api(`/api/jobs/${S.selJob}/log?offset=${st.off}`);
    if (d.text) {
      st.text += d.text;               // 상태에 누적 — 탭을 옮겨도 남는다
      const el = $("log");
      if (el) {
        const stick = el.scrollTop + el.clientHeight >= el.scrollHeight - 40;
        el.textContent = st.text;
        if (stick) el.scrollTop = el.scrollHeight;
      }
    } else {
      const el = $("log");
      if (el && !st.text && d.size === 0) el.textContent = "(아직 출력이 없습니다)";
    }
    st.off = d.offset;
  } catch (e) { /* 무시 */ }
}

/* ============================== 설정 ============================== */
function openSettings() {
  const h = S.health || {};
  $("cfg-url").value = h.base_url || "http://localhost:1234/v1";
  $("cfg-model").innerHTML = `<option value="">(자동 선택)</option>` +
    (h.models || []).map(m => `<option value="${esc(m)}"${
      m === h.model ? " selected" : ""}>${esc(m)}</option>`).join("");
  $("cfg-hint").textContent = h.ok
    ? "연결됨 · 모델 " + (h.models || []).length + "개"
    : "미연결: " + (h.error || "").split("\n")[0] +
      " — LM Studio ▸ Developer ▸ Start Server 를 확인하십시오.";
  $("cfg-err").textContent = "";
  $("settings-modal").hidden = false;
}
async function saveSettings() {
  $("cfg-err").textContent = "";
  try {
    S.health = await api("/api/config", { method: "POST",
      body: JSON.stringify({ base_url: $("cfg-url").value.trim(),
                             model: $("cfg-model").value || null }) });
    paintTop();
    $("settings-modal").hidden = true;
    render();
  } catch (e) { $("cfg-err").textContent = e.message; }
}

/* ============================== 시작 ============================== */
buildShell();
$("refresh-btn").onclick = loadPatients;
$("search").oninput = e => { S.query = e.target.value; if (!S.sel) render(); };
$("settings-btn").onclick = openSettings;
$("cfg-cancel").onclick = () => { $("settings-modal").hidden = true; };
$("cfg-save").onclick = saveSettings;
$("settings-modal").onclick = e => {
  if (e.target.id === "settings-modal") $("settings-modal").hidden = true;
};
document.addEventListener("keydown", e => {
  if (e.key === "Escape") $("settings-modal").hidden = true;
});

(async function init() {
  await refreshHealth();
  await refreshJobs(false);
  await loadPatients();
  setInterval(refreshHealth, 15000);
  setInterval(() => refreshJobs(true), 3000);
  setInterval(pollLog, 2000);
})();
