#!/usr/bin/env python3
"""dwell time 시각화 — 단일 HTML 리포트 생성.

`dwell_times_optimized.npy`는 (n_dwell,) 실수 배열 하나뿐이라 그 자체로는
의미 있는 그림이 안 나온다. 위치·채널·임상 dwell time은 케이스의 RTPLAN에서,
구조물 윤곽은 RTSTRUCT에서 읽어 함께 그린다.

출력물 (외부 의존성 없는 단일 HTML, 데이터 내장)
  1. dwell time 막대그래프 — 채널별, 임상 vs 최적화
  2. 3D applicator 뷰 — 마우스 드래그 회전, 휠 확대.
     dwell을 시간에 비례한 색/크기로 표시. 구조물 윤곽 토글 가능.

사용법:
    python visualize_dwell.py
    python visualize_dwell.py --dwell dwell_times_optimized.npy --out report.html
    python visualize_dwell.py --structures HR_CTV Bladder Rectum Sigmoid
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brachyplan.io import find_files, load_plan  # noqa: E402

DEFAULT_STRUCTURES = ["HR_CTV", "IR_CTV", "Bladder", "Rectum", "Sigmoid"]
STRUCT_COLORS = {
    "GTV": "#ff2d55", "HR_CTV": "#ff375f", "IR_CTV": "#ff9f0a",
    "Bladder": "#ffd60a", "Rectum": "#8b5a2b", "Sigmoid": "#bf5af2",
    "Bowel": "#30d158", "Urethra": "#64d2ff", "Anus": "#a2845e",
}


# ---------------------------------------------------------------------------
def collect_data(case_dir: str, dwell_path: str,
                 structures: list[str], max_pts: int) -> dict:
    files = find_files(case_dir)
    if not files["RTPLAN"]:
        raise FileNotFoundError(f"RTPLAN을 찾을 수 없습니다: {case_dir}")
    rd = files["RTDOSE"][0] if files["RTDOSE"] else None
    plan = load_plan(files["RTPLAN"][0], rd)

    t_clin = np.array([d.time_s for d in plan.dwells], dtype=float)

    if not os.path.isfile(dwell_path):
        raise FileNotFoundError(
            f"{dwell_path} 가 없습니다.\n먼저 최적화를 실행하십시오:  python run_phase2.py"
        )
    t_opt = np.load(dwell_path).astype(float).ravel()
    if t_opt.shape != t_clin.shape:
        raise ValueError(
            f"dwell 수 불일치: {dwell_path}는 {t_opt.shape[0]}개인데 "
            f"RTPLAN은 {t_clin.shape[0]}개입니다. 같은 케이스인지 확인하십시오."
        )

    dwells = [
        {
            "i": i,
            "ch": int(d.channel),
            "p": [round(float(v), 3) for v in d.position],
            "tc": round(float(t_clin[i]), 3),
            "to": round(float(t_opt[i]), 3),
        }
        for i, d in enumerate(plan.dwells)
    ]

    # 채널 폴리라인 (dwell 순서대로)
    channels: dict[str, list] = {}
    for d in dwells:
        channels.setdefault(str(d["ch"]), []).append(d["p"])

    # 구조물 윤곽 (점 수를 줄여서 내장)
    structs = {}
    if files["RTSTRUCT"]:
        import pydicom
        rs = pydicom.dcmread(files["RTSTRUCT"][0])
        from brachyplan.io.struct_to_mask import canonical_roi_name
        name_of = {int(s.ROINumber): canonical_roi_name(s.ROIName)
                   for s in rs.StructureSetROISequence}
        want = {canonical_roi_name(n) for n in structures}
        for roi in rs.get("ROIContourSequence", []):
            name = name_of.get(int(roi.ReferencedROINumber))
            if name not in want:
                continue
            loops = []
            for c in roi.get("ContourSequence", []):
                pts = np.asarray(c.ContourData, dtype=float).reshape(-1, 3)
                if len(pts) > max_pts:                       # 균등 간격으로 솎아냄
                    pts = pts[np.linspace(0, len(pts) - 1, max_pts).astype(int)]
                loops.append([[round(float(v), 2) for v in p] for p in pts])
            if loops:
                structs[name] = {"loops": loops,
                                 "color": STRUCT_COLORS.get(name, "#8e8e93")}

    def stats(t):
        return {
            "active": int(np.sum(t > 0.5)),
            "n": int(len(t)),
            "cv": round(float(np.std(t) / max(np.mean(t), 1e-9)), 3),
            "max": round(float(t.max()), 2),
            "total": round(float(t.sum()), 1),
        }

    return {
        "case": os.path.basename(os.path.abspath(case_dir)),
        "dwellFile": os.path.basename(dwell_path),
        "isotope": plan.isotope,
        "nFx": plan.n_fractions,
        "rx": plan.fraction_dose_gy,
        "sk": round(plan.sk_ref, 1),
        "dwells": dwells,
        "channels": channels,
        "structures": structs,
        "statsClin": stats(t_clin),
        "statsOpt": stats(t_opt),
    }


# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dwell time 시각화 — __CASE__</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#272b35;--fg:#e6e8ee;--dim:#9aa0ae;
      --clin:#5ac8fa;--opt:#ff9f0a;--accent:#30d158}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI","Malgun Gothic",sans-serif}
header{padding:20px 24px 12px;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:19px;font-weight:650}
.sub{color:var(--dim);font-size:13px}
.wrap{padding:20px 24px 40px;max-width:1280px;margin:0 auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
      padding:18px;margin-bottom:20px}
h2{margin:0 0 14px;font-size:15px;font-weight:600}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:6px}
.stat{background:#1e222b;border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.stat .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.stat .v{font-size:17px;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums}
.stat .d{font-size:11px;color:var(--dim);margin-top:1px}
.legend{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin-bottom:10px;font-size:12px;color:var(--dim)}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
button,select{background:#232833;color:var(--fg);border:1px solid var(--line);
       border-radius:7px;padding:6px 11px;font-size:12.5px;cursor:pointer;font-family:inherit}
button:hover,select:hover{background:#2b313e}
button.on{background:var(--accent);color:#062a12;border-color:var(--accent);font-weight:600}
.chip{display:inline-flex;align-items:center;gap:5px;background:#232833;border:1px solid var(--line);
      border-radius:20px;padding:4px 11px;font-size:12px;cursor:pointer;user-select:none}
.chip.off{opacity:.35}
canvas{display:block;width:100%;border-radius:8px;background:#0b0d11;cursor:grab;touch-action:none}
canvas.drag{cursor:grabbing}
svg{display:block;width:100%;height:auto}
.hint{color:var(--dim);font-size:12px;margin-top:8px}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--dim);font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.pos{color:var(--accent)}.neg{color:#ff453a}
#tip{position:fixed;pointer-events:none;background:#000c;border:1px solid var(--line);
     border-radius:7px;padding:7px 10px;font-size:12px;display:none;z-index:9;white-space:pre}
.grid2{display:grid;grid-template-columns:1fr 300px;gap:18px}
@media(max-width:860px){.grid2{grid-template-columns:1fr}}
</style></head><body>
<header>
  <h1>Dwell time 시각화</h1>
  <div class="sub" id="hdr"></div>
</header>
<div class="wrap">
  <div class="card">
    <h2>요약</h2>
    <div class="stats" id="stats"></div>
  </div>

  <div class="card">
    <h2>Dwell time — 임상 vs 최적화</h2>
    <div class="legend">
      <span><i class="sw" style="background:var(--clin)"></i>임상 plan</span>
      <span><i class="sw" style="background:var(--opt)"></i>최적화 plan</span>
      <span style="margin-left:auto">막대 위에 마우스를 올리면 값이 표시됩니다</span>
    </div>
    <div id="bars"></div>
  </div>

  <div class="card">
    <h2>3D applicator 기하</h2>
    <div class="controls">
      <button id="bClin">임상</button>
      <button id="bOpt" class="on">최적화</button>
      <button id="bDiff">차이</button>
      <span style="width:12px"></span>
      <button id="bReset">시점 초기화</button>
      <label style="color:var(--dim);font-size:12px">
        구조물 <span id="chips"></span>
      </label>
    </div>
    <div class="grid2">
      <div>
        <canvas id="cv"></canvas>
        <div class="hint">드래그: 회전 · 휠: 확대/축소 · Shift+드래그: 이동</div>
      </div>
      <div>
        <div id="scale"></div>
        <table id="tbl"></table>
      </div>
    </div>
  </div>
</div>
<div id="tip"></div>
<script>
const D = __DATA__;
const tip = document.getElementById('tip');
const fmt = (v,n=2)=>Number(v).toFixed(n);
function showTip(e,txt){tip.textContent=txt;tip.style.display='block';
  tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px';}
function hideTip(){tip.style.display='none';}

/* ---------- 헤더 & 요약 ---------- */
document.getElementById('hdr').textContent =
  `${D.case} · ${D.isotope} · ${D.nFx} fx × ${D.rx} Gy · S_K ${D.sk} U · ${D.dwellFile}`;

const sc=D.statsClin, so=D.statsOpt;
const delta=(a,b,unit='',inv=false)=>{
  const d=b-a, p=a?100*d/a:0, good=inv?d<0:d>0;
  return `<div class="d ${Math.abs(p)<0.5?'':(good?'pos':'neg')}">임상 ${fmt(a,2)}${unit} → ${p>=0?'+':''}${fmt(p,1)}%</div>`;
};
document.getElementById('stats').innerHTML = `
 <div class="stat"><div class="k">활성 dwell</div><div class="v">${so.active}/${so.n}</div>
   <div class="d">임상 ${sc.active}/${sc.n}</div></div>
 <div class="stat"><div class="k">총 조사시간</div><div class="v">${fmt(so.total,1)} s</div>
   ${delta(sc.total,so.total,' s',true)}</div>
 <div class="stat"><div class="k">최대 dwell</div><div class="v">${fmt(so.max,1)} s</div>
   ${delta(sc.max,so.max,' s',true)}</div>
 <div class="stat"><div class="k">변동계수</div><div class="v">${fmt(so.cv,2)}</div>
   ${delta(sc.cv,so.cv,'',true)}</div>`;

/* ---------- 1) 막대그래프 ---------- */
(function bars(){
  const W=1200, padL=52, padR=16, padT=14, padB=40, rowH=210;
  const chs=[...new Set(D.dwells.map(d=>d.ch))].sort((a,b)=>a-b);
  const tmax=Math.max(...D.dwells.map(d=>Math.max(d.tc,d.to)))*1.12;
  const H=padT+chs.length*rowH+padB;
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img">`;
  chs.forEach((ch,ci)=>{
    const list=D.dwells.filter(d=>d.ch===ch);
    const y0=padT+ci*rowH, h=rowH-46;
    const bw=(W-padL-padR)/list.length;
    s+=`<text x="${padL}" y="${y0+13}" fill="#e6e8ee" font-size="13" font-weight="600">Channel ${ch}`
      +` <tspan fill="#9aa0ae" font-weight="400">(${list.length} dwell)</tspan></text>`;
    // y축 눈금
    for(let g=0;g<=4;g++){
      const v=tmax*g/4, yy=y0+24+h-(h*g/4);
      s+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="#272b35"/>`
        +`<text x="${padL-7}" y="${yy+4}" fill="#9aa0ae" font-size="10" text-anchor="end">${fmt(v,0)}</text>`;
    }
    list.forEach((d,k)=>{
      const x=padL+k*bw, w=bw*0.36;
      const hc=h*d.tc/tmax, ho=h*d.to/tmax;
      const yb=y0+24+h;
      s+=`<rect class="bar" x="${x+bw*0.10}" y="${yb-hc}" width="${w}" height="${Math.max(hc,0.5)}" fill="#5ac8fa" rx="2"`
        +` data-t="dwell #${d.i} (ch${d.ch})&#10;임상 ${fmt(d.tc)} s"/>`;
      s+=`<rect class="bar" x="${x+bw*0.50}" y="${yb-ho}" width="${w}" height="${Math.max(ho,0.5)}" fill="#ff9f0a" rx="2"`
        +` data-t="dwell #${d.i} (ch${d.ch})&#10;최적화 ${fmt(d.to)} s&#10;임상 대비 ${d.tc?((d.to-d.tc)/d.tc*100>=0?'+':'')+fmt((d.to-d.tc)/d.tc*100,0)+'%':'n/a'}"/>`;
      s+=`<text x="${x+bw/2}" y="${yb+13}" fill="#9aa0ae" font-size="9.5" text-anchor="middle">${d.i}</text>`;
    });
    s+=`<line x1="${padL}" y1="${y0+24+h}" x2="${W-padR}" y2="${y0+24+h}" stroke="#4a505e"/>`;
    s+=`<text x="${padL-7}" y="${y0+24+h+26}" fill="#9aa0ae" font-size="10" text-anchor="end">dwell #</text>`;
  });
  s+=`<text transform="translate(13,${H/2}) rotate(-90)" fill="#9aa0ae" font-size="11" text-anchor="middle">dwell time (s)</text>`;
  s+='</svg>';
  document.getElementById('bars').innerHTML=s;
  document.querySelectorAll('.bar').forEach(el=>{
    el.addEventListener('mousemove',e=>showTip(e,el.dataset.t.replace(/&#10;/g,'\n')));
    el.addEventListener('mouseleave',hideTip);
  });
})();

/* ---------- 2) 3D 뷰 ---------- */
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
let mode='opt';
const active={}; Object.keys(D.structures).forEach((k,i)=>active[k]=i<2);

// 색상 스케일 (파랑 -> 노랑 -> 빨강)
function ramp(u){
  u=Math.max(0,Math.min(1,u));
  const stops=[[40,90,190],[60,190,190],[240,200,60],[235,90,50]];
  const x=u*(stops.length-1), i=Math.min(Math.floor(x),stops.length-2), f=x-i;
  const c=stops[i].map((v,k)=>Math.round(v+(stops[i+1][k]-v)*f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function val(d){return mode==='clin'?d.tc:mode==='opt'?d.to:d.to-d.tc;}
function range(){
  const vs=D.dwells.map(val);
  return mode==='diff'
    ? (m=>[-m,m])(Math.max(1e-6,...vs.map(Math.abs)))
    : [0,Math.max(1e-6,...vs)];
}

// 카메라 — 중심은 applicator에 고정하고, 보이는 구조물까지 담기도록 범위만 넓힌다
const allPts=D.dwells.map(d=>d.p);
const c0=[0,1,2].map(k=>allPts.reduce((s,p)=>s+p[k],0)/allPts.length);
let yaw=-0.6, pitch=-0.35, zoom=1, panX=0, panY=0;
const dist=p=>Math.hypot(p[0]-c0[0],p[1]-c0[1],p[2]-c0[2]);
let span=1;
function computeSpan(){
  let r=Math.max(...allPts.map(dist));
  for(const [name,st] of Object.entries(D.structures)){
    if(!active[name]) continue;
    for(const loop of st.loops) for(const p of loop) r=Math.max(r,dist(p));
  }
  span=r*2.4;
}
computeSpan();

function project(p,w,h){
  let x=p[0]-c0[0], y=p[1]-c0[1], z=p[2]-c0[2];
  let X=x*Math.cos(yaw)-y*Math.sin(yaw), Y=x*Math.sin(yaw)+y*Math.cos(yaw);
  let Z=z;
  let Y2=Y*Math.cos(pitch)-Z*Math.sin(pitch), Z2=Y*Math.sin(pitch)+Z*Math.cos(pitch);
  const s=Math.min(w,h)/span*zoom;
  const persp=1/(1+Z2/(span*3));
  return [w/2+panX+X*s*persp, h/2+panY-Y2*s*persp, Z2, persp];
}

function draw(){
  const w=cv.clientWidth, h=Math.round(w*0.72);
  if(cv.width!==w*devicePixelRatio||cv.height!==h*devicePixelRatio){
    cv.width=w*devicePixelRatio; cv.height=h*devicePixelRatio;
    cv.style.height=h+'px';
  }
  ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle='#0b0d11'; ctx.fillRect(0,0,w,h);

  // 구조물 윤곽
  for(const [name,st] of Object.entries(D.structures)){
    if(!active[name]) continue;
    ctx.strokeStyle=st.color; ctx.globalAlpha=0.30; ctx.lineWidth=1;
    for(const loop of st.loops){
      ctx.beginPath();
      loop.forEach((p,i)=>{const q=project(p,w,h); i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});
      ctx.closePath(); ctx.stroke();
    }
  }
  ctx.globalAlpha=1;

  // 채널 경로
  ctx.strokeStyle='#5b6272'; ctx.lineWidth=2;
  for(const pts of Object.values(D.channels)){
    ctx.beginPath();
    pts.forEach((p,i)=>{const q=project(p,w,h); i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});
    ctx.stroke();
  }

  // dwell (뒤에서 앞으로 그리기)
  const [lo,hi]=range();
  const items=D.dwells.map(d=>({d,q:project(d.p,w,h)})).sort((a,b)=>b.q[2]-a.q[2]);
  const rmax=Math.min(w,h)*0.030;
  for(const {d,q} of items){
    const v=val(d), u=(v-lo)/(hi-lo||1);
    const mag=mode==='diff'?Math.abs(v)/Math.max(hi,1e-6):(hi?v/hi:0);
    const r=Math.max(2.5,(3+rmax*Math.sqrt(mag))*q[3]);
    ctx.beginPath(); ctx.arc(q[0],q[1],r,0,6.284);
    ctx.fillStyle=ramp(u); ctx.fill();
    ctx.strokeStyle='#0b0d11'; ctx.lineWidth=1.2; ctx.stroke();
    d._s=[q[0],q[1],r];
  }

  // 축 표시
  ctx.fillStyle='#9aa0ae'; ctx.font='11px sans-serif';
  ctx.fillText(mode==='clin'?'임상 plan':mode==='opt'?'최적화 plan':'차이 (최적화 − 임상)',12,18);
}

/* 색 스케일 + 표 */
function sidebar(){
  const [lo,hi]=range();
  let g='<div style="font-size:11px;color:var(--dim);margin-bottom:6px">'
       +(mode==='diff'?'차이 (s)':'dwell time (s)')+'</div>'
       +'<div style="height:14px;border-radius:4px;margin-bottom:4px;background:linear-gradient(90deg,'
       +[0,.25,.5,.75,1].map(u=>ramp(u)).join(',')+')"></div>'
       +`<div style="display:flex;justify-content:space-between;font-size:11px;color:var(--dim);margin-bottom:14px">`
       +`<span>${fmt(lo,1)}</span><span>${fmt((lo+hi)/2,1)}</span><span>${fmt(hi,1)}</span></div>`;
  document.getElementById('scale').innerHTML=g;

  let t='<tr><th>#</th><th>ch</th><th>임상</th><th>최적화</th><th>Δ%</th></tr>';
  for(const d of D.dwells){
    const p=d.tc?100*(d.to-d.tc)/d.tc:null;
    const cls=p===null?'':(Math.abs(p)<1?'':(p>0?'neg':'pos'));
    t+=`<tr><td>${d.i}</td><td>${d.ch}</td><td>${fmt(d.tc)}</td><td>${fmt(d.to)}</td>`
      +`<td class="${cls}">${p===null?'—':(p>=0?'+':'')+fmt(p,0)}</td></tr>`;
  }
  document.getElementById('tbl').innerHTML=t;
}

/* 구조물 토글 칩 */
document.getElementById('chips').innerHTML=Object.entries(D.structures).map(([n,s])=>
  `<span class="chip ${active[n]?'':'off'}" data-n="${n}">`
  +`<i class="sw" style="background:${s.color}"></i>${n}</span>`).join(' ');
document.querySelectorAll('.chip').forEach(el=>el.onclick=()=>{
  const n=el.dataset.n; active[n]=!active[n]; el.classList.toggle('off',!active[n]);
  computeSpan(); draw();   // 켠 구조물이 화면 밖으로 잘리지 않도록 범위 갱신
});

/* 모드 버튼 */
const btns={clin:document.getElementById('bClin'),opt:document.getElementById('bOpt'),
            diff:document.getElementById('bDiff')};
for(const [k,b] of Object.entries(btns)) b.onclick=()=>{
  mode=k; for(const bb of Object.values(btns)) bb.classList.remove('on');
  b.classList.add('on'); sidebar(); draw();
};
document.getElementById('bReset').onclick=()=>{
  yaw=-0.6;pitch=-0.35;zoom=1;panX=0;panY=0;computeSpan();draw();
};

/* 마우스 상호작용 */
let drag=null;
cv.addEventListener('pointerdown',e=>{
  drag={x:e.clientX,y:e.clientY,shift:e.shiftKey};
  cv.setPointerCapture(e.pointerId); cv.classList.add('drag');
});
cv.addEventListener('pointerup',e=>{drag=null;cv.classList.remove('drag');});
cv.addEventListener('pointerleave',hideTip);
cv.addEventListener('pointermove',e=>{
  if(drag){
    const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
    if(drag.shift){panX+=dx;panY+=dy;}
    else{yaw+=dx*0.008; pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*0.008));}
    drag.x=e.clientX; drag.y=e.clientY; draw(); return;
  }
  const r=cv.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  let hit=null;
  for(const d of D.dwells){
    if(!d._s) continue;
    if(Math.hypot(mx-d._s[0],my-d._s[1])<=d._s[2]+3) hit=d;
  }
  if(hit) showTip(e,`dwell #${hit.i} · channel ${hit.ch}\n임상 ${fmt(hit.tc)} s\n최적화 ${fmt(hit.to)} s\n`
                    +`(${fmt(hit.p[0],1)}, ${fmt(hit.p[1],1)}, ${fmt(hit.p[2],1)}) mm`);
  else hideTip();
});
cv.addEventListener('wheel',e=>{
  e.preventDefault(); zoom=Math.max(0.25,Math.min(8,zoom*(e.deltaY<0?1.12:0.89))); draw();
},{passive:false});
window.addEventListener('resize',draw);

sidebar(); draw();
</script></body></html>
"""


def build_html(data: dict) -> str:
    return (HTML
            .replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__CASE__", data["case"]))


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="Geneva_Case")
    ap.add_argument("--dwell", default="dwell_times_optimized.npy")
    ap.add_argument("--out", default="dwell_report.html")
    ap.add_argument("--structures", nargs="*", default=DEFAULT_STRUCTURES)
    ap.add_argument("--max-contour-points", type=int, default=48,
                    help="윤곽 하나당 내장할 최대 점 수 (파일 크기 조절)")
    ap.add_argument("--open", action="store_true", help="생성 후 브라우저로 열기")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    case_dir = args.case if os.path.isabs(args.case) else os.path.join(root, args.case)
    dwell = args.dwell if os.path.isabs(args.dwell) else os.path.join(root, args.dwell)
    out = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)

    data = collect_data(case_dir, dwell, args.structures, args.max_contour_points)
    html = build_html(data)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)

    sc, so = data["statsClin"], data["statsOpt"]
    print(f"생성 완료: {out}  ({len(html) / 1024:.0f} KB)")
    print(f"  dwell {so['n']}개 · 구조물 {len(data['structures'])}개 내장")
    print(f"  임상   : 활성 {sc['active']}/{sc['n']}, 총 {sc['total']}s, "
          f"최대 {sc['max']}s, CV {sc['cv']}")
    print(f"  최적화 : 활성 {so['active']}/{so['n']}, 총 {so['total']}s, "
          f"최대 {so['max']}s, CV {so['cv']}")
    if args.open:
        webbrowser.open("file://" + out.replace("\\", "/"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
