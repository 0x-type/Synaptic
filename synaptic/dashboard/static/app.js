"use strict";
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const SVGNS = "http://www.w3.org/2000/svg";
const STAGE_ORDER = ["sweep", "classify", "census", "deep_dive", "report"];
const STAGE_NAME = { sweep: "Sweep", classify: "Classify", census: "Census", deep_dive: "Deep dives", report: "Reports" };
const VERDICT_NAME = { new_signal: "New signal", known_phenomenon: "Known phenomenon", artifact: "Artifact", not_robust: "Not robust" };
const ROLE_NAME = { worker: "Worker", supervisor: "Supervisor", curator: "Curator", harness: "Harness", editor: "Editor", judge: "Judge", triage: "Triage", planner: "Planner" };

const S = { follow: true, cid: null, data: null, events: [], lastEvent: 0, sig: "", selected: null, report: null, tab: "discoveries", benchDrawn: "" };

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

/* ------------------------------------------------------------------ polling */
async function loadCampaigns(selectNewest = false) {
  const list = await getJSON("/api/campaigns");
  const sel = $("#campaign");
  sel.innerHTML = list.map((c) => `<option value="${c.id}">${esc(c.name)} (${c.mode}, ${new Date(c.started * 1000).toLocaleString()})</option>`).join("")
    || "<option>No campaigns yet</option>";
  if (list.length && (selectNewest || !S.cid)) S.cid = list[0].id;
  if (S.cid) sel.value = S.cid;
}

async function poll() {
  if (!S.cid) { await loadCampaigns(); if (!S.cid) return; }
  try {
    const d = await getJSON(`/api/state?cid=${S.cid}&since=${S.lastEvent}`);
    if (d.empty) return;
    S.data = d;
    if (d.events.length) {
      S.events = d.events.concat(S.events).slice(0, 500);
      S.lastEvent = Math.max(S.lastEvent, d.last_event || 0);
    }
    render();
    const live = d.tasks.filter((t) => ["running", "review", "revising"].includes(t.status)).sort((a, b) => (b.started || 0) - (a.started || 0));
    if (S.follow && live.length && live[0].id !== S.selected) openTask(live[0].id, true, true);
    else if (S.selected && d.tasks.find((t) => t.id === S.selected && ["running", "review", "revising"].includes(t.status))) openTask(S.selected, true, S.follow);
  } catch (e) { console.warn(e); }
}

/* ------------------------------------------------------------------ header + vitals */
function render() {
  const d = S.data, c = d.campaign, b = c.brief;
  $("#title").textContent = c.name;
  $("#question").textContent = (b.question || "").trim();
  const mode = $("#mode");
  const worker = d.models.find((m) => m.role === "worker");
  if (c.mode === "interactive") {
    mode.className = "mode";
    mode.innerHTML = "<strong>Interactive investigation.</strong> A scientist working with Claude followed up the agents' leads, as in the paper's Claude Science sessions.";
  } else if (c.mode === "mock") {
    mode.className = "mode mock";
    mode.innerHTML = "<strong>Offline demo.</strong> Scripted agents run the real pipeline on real NASA data; no model reasoning is shown.";
  } else {
    mode.className = "mode";
    mode.innerHTML = `<strong>Live.</strong> Agents: ${esc(worker ? worker.model : "starting")}. Status: ${esc(c.status)}${d.stop_requested ? ", stop requested" : ""}.`;
  }
  $("#stop").disabled = c.status !== "running" || d.stop_requested;

  const counts = {};
  d.tasks.forEach((t) => (counts[t.stage] = (counts[t.stage] || 0) + 1));
  $("#stages").innerHTML = d.stages.map((s) => `<li class="${s.status}"><span class="dot"></span>${esc(s.name)}${counts[s.key] ? ` <span class="n">${counts[s.key]}</span>` : ""}</li>`).join("");

  const st = d.stats, tasks = d.tasks;
  const fu = tasks.filter((t) => t.origin !== "mission").length;
  const cand = d.candidates, passed = cand.filter((k) => k.passed).length, prom = cand.filter((k) => k.promoted).length;
  const v = (o) => cand.filter((k) => k.outcome === o).length;
  const tok = (st.tokens_in || 0) + (st.tokens_out || 0);
  $("#counters").innerHTML = `<b>${tasks.length}</b> tasks, <b>${fu}</b> opened by the agents themselves; <b>${st.sessions}</b> agent sessions over <b>${st.agent_hours}</b> agent-hours and <b>${fmt(tok)}</b> tokens. `
    + `The sweep found <b>${cand.length}</b> anomaly families, <b>${passed}</b> passed the pre-registered filters, <b>${prom}</b> went to deep dives. `
    + `Verdicts: <b>${v("new_signal")}</b> new signal, <b>${v("known_phenomenon")}</b> known, <b>${v("artifact")}</b> artifact, <b>${v("not_robust")}</b> not robust.`;

  const sig = tasks.map((t) => t.id + t.status + (t.verdict || "")).join("|") + (S.selected || "");
  if (sig !== S.sig) { S.sig = sig; drawNetwork(); }
  if (S.tab === "discoveries") drawReports();
  if (S.tab === "map") drawMap();
  if (S.tab === "kb") drawKB();
  if (S.tab === "log") drawLog();
}
const fmt = (n) => n >= 1e6 ? (n / 1e6).toFixed(1) + " M" : n >= 1e3 ? (n / 1e3).toFixed(0) + " k" : String(n);

/* ------------------------------------------------------------------ task network (drainage tree) */
function nodeState(t) {
  if (["running", "review", "revising", "starting"].includes(t.status)) return "running";
  if (["queued", "triage"].includes(t.status)) return "waiting";
  if (t.status === "rejected") return t.origin === "mission" ? "failed" : "declined";
  if (["failed", "stalled"].includes(t.status)) return "failed";
  return "done";
}

function drawNetwork() {
  const tasks = S.data.tasks, byId = Object.fromEntries(tasks.map((t) => [t.id, { ...t, kids: [] }]));
  const roots = [];
  Object.values(byId).forEach((t) => {
    if (t.origin !== "mission" && t.parent_id && byId[t.parent_id]) byId[t.parent_id].kids.push(t);
    else roots.push(t);
  });
  roots.sort((a, b) => STAGE_ORDER.indexOf(a.stage) - STAGE_ORDER.indexOf(b.stage) || a.created - b.created);
  const UX = 40, UY = 62, TOP = 66, PAD = 28;
  const width = (t) => (t.w = Math.max(1, t.kids.reduce((s, k) => s + width(k), 0)));
  let x = PAD, lastStage = null, maxDepth = 0, groupStart = PAD, groupLabel = "";
  const labels = [], placed = [];
  const place = (t, left, depth) => {
    t.y = TOP + depth * UY; maxDepth = Math.max(maxDepth, depth);
    let cx = left;
    t.kids.sort((a, b) => a.created - b.created).forEach((k) => { place(k, cx, depth + 1); cx += k.w * UX; });
    t.x = depth === 0 || !t.kids.length ? left + UX / 2 : (t.kids[0].x + t.kids[t.kids.length - 1].x) / 2;
    placed.push(t);
  };
  const closeGroup = () => { if (groupLabel) x = Math.max(x, groupStart + groupLabel.length * 7.4 + 18); };
  roots.forEach((r) => {
    width(r);
    if (r.stage !== lastStage) {
      closeGroup();
      if (lastStage) x += 22;
      groupStart = x; groupLabel = STAGE_NAME[r.stage] || r.stage;
      labels.push({ x, text: groupLabel }); lastStage = r.stage;
    }
    place(r, x, 0);
    x += r.w * UX;
  });
  closeGroup();
  const W = Math.max(x + PAD, 320), H = TOP + maxDepth * UY + 40;
  let svg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Task network with ${tasks.length} tasks">`;
  if (roots.length) svg += `<path class="trunk" d="M${roots[0].x} ${TOP} L${roots[roots.length - 1].x} ${TOP}"/>`;
  labels.forEach((l) => (svg += `<text class="stage-label" x="${l.x + 6}" y="${TOP - 30}">${esc(l.text)}</text>`));
  placed.forEach((t) => t.kids.forEach((k) => {
    const my = (t.y + k.y) / 2;
    svg += `<path class="edge ${k.origin === "worker" ? "worker" : "followup"}" d="M${t.x} ${t.y} C${t.x} ${my} ${k.x} ${my} ${k.x} ${k.y}"/>`;
  }));
  placed.forEach((t) => {
    const st = nodeState(t), r = t.origin === "mission" ? 9 : 7;
    const vx = t.x + r + 2, vy = t.y - r - 2;
    svg += `<g class="node ${st} o-${t.origin}${S.selected === t.id ? " selected" : ""}" data-id="${t.id}" tabindex="0" role="button" aria-label="${esc(t.title)}, ${esc(t.status)}">`
      + `<title>${esc(t.title)} (${esc(t.status)}${t.verdict ? ", " + esc(VERDICT_NAME[t.verdict] || t.verdict) : ""})</title>`
      + (st === "running" ? `<circle class="ring-running" cx="${t.x}" cy="${t.y}" r="${r + 4}"/>` : "")
      + `<circle class="body n-${t.origin}" cx="${t.x}" cy="${t.y}" r="${r}"/>`
      + (t.verdict ? `<rect class="verdict-mark v-${t.verdict}" x="${vx - 4}" y="${vy - 4}" width="8" height="8" transform="rotate(45 ${vx} ${vy})"/>` : "")
      + `</g>`;
  });
  svg += "</svg>";
  const net = $("#network");
  net.innerHTML = tasks.length ? svg : `<p class="hint" style="padding:18px">The planner is writing the campaign plan. Tasks appear here as soon as they are queued.</p>`;
  net.querySelectorAll(".node").forEach((n) => {
    n.addEventListener("click", () => openTask(n.dataset.id));
    n.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openTask(n.dataset.id); } });
  });
}

/* ------------------------------------------------------------------ inspector */
async function openTask(id, quiet = false, auto = false) {
  if (!auto) S.follow = false;
  const changed = S.selected !== id;
  S.selected = id;
  if (changed) { S.sig = ""; drawNetwork(); }
  const d = await getJSON(`/api/task/${id}`);
  const t = d.task, insp = $("#inspector");
  const keepScroll = quiet ? insp.scrollTop : 0;
  const chips = [`<span class="chip">${esc(t.kind.replace("_", " "))}</span>`, `<span class="chip">${esc(t.status)}</span>`,
    `<span class="chip">from ${esc(t.origin === "mission" ? "the brief" : t.origin)}</span>`]
    .concat(t.revisions ? [`<span class="chip">${t.revisions} revision${t.revisions > 1 ? "s" : ""}</span>`] : [])
    .concat(t.verdict ? [`<span class="chip v-${t.verdict}" style="background:var(--${{ new_signal: "vegetation", known_phenomenon: "sediment", artifact: "falsecolor", not_robust: "mute" }[t.verdict]})">${esc(VERDICT_NAME[t.verdict])}</span>`] : []);
  const items = d.events.slice(-300).map(eventItem).join("");
  const follow = S.follow ? `<p class="insp-meta following">Following the latest running task. Select any node to stop following.</p>` : "";
  insp.innerHTML = `${follow}<p class="insp-meta">Task ${esc(t.id)}${t.parent_id ? `, branch of <a href="#" data-open="${t.parent_id}">${esc(t.parent_id)}</a>` : ""}</p>
    <h2 class="insp-title">${esc(t.title)}</h2><div>${chips.join("")}</div>
    ${t.triage_note ? `<p class="insp-meta">Triage: ${esc(t.triage_note)}</p>` : ""}
    <details ${t.brief.length < 500 ? "open" : ""}><summary class="hint">Brief</summary><div class="brief">${esc(t.brief)}</div></details>
    ${t.summary ? `<details open><summary class="hint">Result</summary><div class="brief">${esc(t.summary.slice(0, 3000))}</div></details>` : ""}
    ${d.children.length ? `<p class="insp-meta">Follow-ups opened from here: ${d.children.map((c) => `<a href="#" data-open="${c.id}">${esc(c.title)}</a> (${esc(c.status)})`).join("; ")}</p>` : ""}
    <ol class="timeline">${items || '<li><span class="hint">No activity yet.</span></li>'}</ol>`;
  insp.querySelectorAll("[data-open]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); openTask(a.dataset.open); }));
  insp.querySelectorAll("img.looked").forEach((im) => im.addEventListener("click", () => lightbox(im.src)));
  insp.scrollTop = S.follow ? insp.scrollHeight : keepScroll;
  if (!quiet && window.matchMedia("(max-width: 900px)").matches) insp.focus();
}

function eventItem(e) {
  const who = ROLE_NAME[e.role] || e.role;
  const time = new Date(e.ts * 1000).toLocaleTimeString();
  const head = `<div class="who">${esc(who)}, ${time}</div>`;
  switch (e.type) {
    case "thought": return `<li class="t-thought">${head}<div class="thought">${esc(e.content.slice(-1500))}</div></li>`;
    case "action": return `<li class="t-action">${head}<div class="action"><code>${esc(e.content.slice(0, 900))}</code></div></li>`;
    case "observation": {
      const first = e.content.split("\n")[0].slice(0, 140);
      return `<li class="t-observation">${head}<details class="obs"><summary>${esc(first)}</summary><pre>${esc(e.content)}</pre></details></li>`;
    }
    case "image": return `<li class="t-image">${head}<div class="body">Looked at this image</div>${e.src ? `<img class="looked" src="${e.src}" alt="Figure the agent looked at" loading="lazy">` : ""}</li>`;
    case "plan_review": return `<li class="t-plan_review">${head}<div class="body">Plan review: ${esc(e.content)}</div></li>`;
    case "review": return `<li class="t-review">${head}<div class="body">Result review: ${esc(e.content)}</div></li>`;
    case "gate": return `<li class="t-gate">${head}<div class="body">Finish refused: ${esc(e.content)}</div></li>`;
    case "error": return `<li class="t-error">${head}<div class="body">${esc(e.content.slice(-800))}</div></li>`;
    case "verdict": return `<li class="t-review">${head}<details class="obs"><summary>${esc(e.content.slice(0, 120))}</summary><pre>${esc(e.content)}</pre></details></li>`;
    default: return `<li>${head}<div class="body">${esc(e.content.slice(0, 600))}</div></li>`;
  }
}

function lightbox(src) { const d = $("#lightbox"); $("img", d).src = src; d.showModal(); }

/* ------------------------------------------------------------------ discoveries */
function drawReports() {
  const reps = S.data.reports, ol = $("#reports");
  if (!reps.length) {
    ol.innerHTML = `<li class="hint">No reports filed yet. They are written after deep dives return a verdict worth documenting.</li>`;
    $("#report").innerHTML = `<p class="muted">Ranked reports appear here once deep dives finish and the editor files them.</p>`;
    S.report = null;
    return;
  }
  if (S.report && !reps.find((r) => r.id === S.report)) S.report = null;
  const html = reps.map((r, i) => `<li><button type="button" data-id="${r.id}" aria-current="${S.report === r.id}">
      <span class="rank">${r.rank || i + 1}</span><span class="rt">${esc(r.title)}</span>
      <span class="rs">${esc(VERDICT_NAME[r.outcome] || "")}${r.played ? `, won ${r.wins} of ${r.played} comparisons` : ""}</span></button></li>`).join("");
  if (ol.innerHTML !== html) {
    ol.innerHTML = html;
    ol.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => openReport(b.dataset.id)));
  }
  if (!S.report && reps.length) openReport(reps[0].id);
}

async function openReport(id) {
  S.report = id;
  const r = await getJSON(`/api/report/${id}`);
  $("#report").innerHTML = r.html;
  $("#report").querySelectorAll("img").forEach((im) => im.addEventListener("click", () => lightbox(im.src)));
  $("#reports").querySelectorAll("button").forEach((b) => b.setAttribute("aria-current", b.dataset.id === id));
}

/* ------------------------------------------------------------------ anomaly map */
const OUTCOME_VAR = { new_signal: "--vegetation", known_phenomenon: "--sediment", artifact: "--falsecolor", not_robust: "--mute" };
function drawMap() {
  const d = S.data, reg = d.campaign.brief.region, el = $("#map");
  const key = d.candidates.map((k) => k.id + k.passed + k.promoted + (k.outcome || "")).join("|");
  if (el.dataset.key === key) return;
  el.dataset.key = key;
  const W = 1000, H = Math.round(W * (reg.lat_max - reg.lat_min) / (reg.lon_max - reg.lon_min));
  const X = (lo) => ((lo - reg.lon_min) / (reg.lon_max - reg.lon_min)) * W;
  const Y = (la) => ((reg.lat_max - la) / (reg.lat_max - reg.lat_min)) * H;
  const cw = (0.625 / (reg.lon_max - reg.lon_min)) * W, ch = (0.5 / (reg.lat_max - reg.lat_min)) * H;
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Map of anomaly families"><image href="/api/basemap/${S.cid}" x="0" y="0" width="${W}" height="${H}" preserveAspectRatio="none"/>`;
  const fams = [...d.candidates].sort((a, b) => b.n_cells - a.n_cells);
  fams.forEach((k) => {
    if (!k.promoted) return;
    const col = `var(${OUTCOME_VAR[k.outcome] || "--river"})`;
    svg += `<g class="cells" data-f="${esc(k.id)}" opacity=".0">` + k.cells.map(([la, lo]) =>
      `<rect class="cell" x="${X(lo) - cw / 2}" y="${Y(la) - ch / 2}" width="${cw}" height="${ch}" fill="${col}" fill-opacity=".45" stroke="${col}"/>`).join("") + `</g>`;
  });
  fams.forEach((k) => {
    const r = 5 + Math.sqrt(k.n_cells) * 1.6;
    const col = k.promoted ? `var(${OUTCOME_VAR[k.outcome] || "--river"})` : "transparent";
    const stroke = k.passed ? "var(--paper)" : "rgba(255,255,255,.55)";
    svg += `<circle class="fam" data-f="${esc(k.id)}" cx="${X(k.lon)}" cy="${Y(k.lat)}" r="${r}" fill="${col}" fill-opacity="${k.promoted ? .85 : 0}" stroke="${stroke}" stroke-width="${k.passed ? 2.5 : 1}" stroke-dasharray="${k.passed ? "" : "3 3"}"/>`;
  });
  svg += "</svg><div class='tooltip' hidden></div>";
  el.innerHTML = svg;
  const tip = $(".tooltip", el);
  el.querySelectorAll(".fam").forEach((c) => {
    const k = d.candidates.find((x) => x.id === c.dataset.f);
    c.addEventListener("mouseenter", () => {
      el.querySelectorAll(`.cells[data-f="${CSS.escape(k.id)}"]`).forEach((g) => g.setAttribute("opacity", "1"));
      tip.hidden = false;
      tip.innerHTML = `<b>${esc(k.id)}</b><br>${k.n_cells} cells${k.onset ? ", onset " + k.onset : ""}, effect ${(+k.effect).toFixed(2)}, q ${(+k.qvalue).toExponential(1)}<br>`
        + `${k.passed ? "Passed filters" : "Did not pass filters"}${k.filters && k.filters.artifact_flag ? "<br>Artifact flag: " + esc(k.filters.artifact_flag) : ""}${k.outcome ? "<br>Verdict: " + esc(VERDICT_NAME[k.outcome]) : ""}`;
      const box = el.getBoundingClientRect(), cb = c.getBoundingClientRect();
      tip.style.left = Math.min(cb.right - box.left + 8, box.width - 290) + "px";
      tip.style.top = (cb.top - box.top) + "px";
    });
    c.addEventListener("mouseleave", () => { tip.hidden = true; el.querySelectorAll(".cells").forEach((g) => g.setAttribute("opacity", "0")); });
    c.addEventListener("click", () => {
      const t = S.data.tasks.find((x) => x.family === k.id && x.kind === "deep_dive");
      if (t) { openTask(t.id); window.scrollTo({ top: 0, behavior: "smooth" }); }
    });
  });
  $("#map-legend").innerHTML = Object.entries(OUTCOME_VAR).map(([o, v]) => `<li><span class="swatch" style="background:var(${v})"></span>${VERDICT_NAME[o]}</li>`).join("")
    + `<li><span class="swatch" style="background:var(--river)"></span>Deep dive in progress</li><li><span class="swatch" style="border:2px dashed var(--mute)"></span>Did not pass the filters</li>`;
  $("#thresholds").innerHTML = d.thresholds.map((t) => `<dt>${esc(t.key)} = ${+t.value}</dt><dd>${esc(t.rationale)}</dd>`).join("");
}

/* ------------------------------------------------------------------ knowledge base + log */
function drawKB() {
  $("#kb").innerHTML = S.data.kb.map((k) => `<li><span class="k ${esc(k.kind)}">${esc(k.kind)}</span><h4>${esc(k.title)}</h4><p>${esc(k.body)}</p></li>`).join("")
    || `<li class="hint">The curator files findings, rejections and methods here after each task.</li>`;
}
function drawLog() {
  $("#log").innerHTML = S.events.slice(0, 300).map((e) => `<li><span class="ts">${new Date(e.ts * 1000).toLocaleTimeString()}</span><span class="r">${esc(ROLE_NAME[e.role] || e.role)}</span><span class="c">${esc(e.type)}: ${esc(e.content.replace(/\s+/g, " ").slice(0, 220))}</span></li>`).join("");
}

/* ------------------------------------------------------------------ benchmark */
async function drawBench() {
  const d = await getJSON("/api/bench");
  const el = $("#bench");
  if (d.empty) { el.innerHTML = `<p class="hint">Run <code>python -m synaptic bench</code> to measure how well each model finds anomalies planted in real NASA data.</p>`; return; }
  if (S.benchDrawn === d.id + d.rows.length) return;
  S.benchDrawn = d.id + d.rows.length;
  const rows = d.rows, models = [...new Set(rows.map((r) => r.model))], levels = [...new Set(rows.map((r) => r.level))].sort();
  const mean = (a, k) => a.length ? a.reduce((s, r) => s + r[k], 0) / a.length : null;
  const palette = ["--river", "--sediment", "--vegetation", "--falsecolor", "--ink", "--mute"];
  // line chart: detection by level
  const W = 460, H = 240, L = 40, B = 34, T = 12, R = 120;
  const px = (i) => L + (i / Math.max(1, levels.length - 1)) * (W - L - R), py = (v) => T + (1 - v) * (H - T - B);
  let line = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Detection rate by information level">`;
  [0, .25, .5, .75, 1].forEach((v) => (line += `<line class="axis" x1="${L}" x2="${W - R}" y1="${py(v)}" y2="${py(v)}"/><text x="${L - 6}" y="${py(v) + 4}" text-anchor="end">${v * 100}%</text>`));
  levels.forEach((lv, i) => (line += `<text x="${px(i)}" y="${H - 14}" text-anchor="middle">Level ${lv}</text>`));
  models.forEach((m, mi) => {
    const pts = levels.map((lv, i) => [px(i), py(mean(rows.filter((r) => r.model === m && r.level === lv), "detected") ?? 0)]);
    const col = `var(${palette[mi % palette.length]})`;
    line += `<polyline fill="none" stroke="${col}" stroke-width="2.2" points="${pts.map((p) => p.join(",")).join(" ")}"/>`
      + pts.map((p) => `<circle cx="${p[0]}" cy="${p[1]}" r="3.5" fill="${col}"/>`).join("")
      + `<text x="${pts[pts.length - 1][0] + 8}" y="${pts[pts.length - 1][1] + 4}" style="fill:${col}">${esc(m.split("/").pop())}</text>`;
  });
  line += "</svg>";
  // bars: detection by raw data read (levels 3-5)
  const bins = [[0, 1, "none"], [1, 200, "<200"], [200, 600, "200-599"], [600, 2000, "600-1999"], [2000, 1e12, "2000+"]];
  const tool = rows.filter((r) => r.level >= 3);
  const BW = 460, BH = 240;
  let bars = `<svg viewBox="0 0 ${BW} ${BH}" role="img" aria-label="Detection rate by raw data read">`;
  [0, .5, 1].forEach((v) => (bars += `<line class="axis" x1="${L}" x2="${BW - 10}" y1="${py(v)}" y2="${py(v)}"/><text x="${L - 6}" y="${py(v) + 4}" text-anchor="end">${v * 100}%</text>`));
  const bw = (BW - L - 20) / bins.length;
  bins.forEach(([lo, hi, lab], i) => {
    const sub = tool.filter((r) => r.longest_run >= lo && r.longest_run < hi), v = mean(sub, "detected");
    const x0 = L + i * bw + 8;
    if (v !== null) bars += `<rect x="${x0}" y="${py(v)}" width="${bw - 16}" height="${py(0) - py(v)}" fill="var(--river)" rx="3"/><text x="${x0 + (bw - 16) / 2}" y="${py(v) - 5}" text-anchor="middle">n=${sub.length}</text>`;
    bars += `<text x="${x0 + (bw - 16) / 2}" y="${BH - 14}" text-anchor="middle">${lab}</text>`;
  });
  bars += "</svg>";
  const types = [...new Set(rows.map((r) => r.case_id.split(":")[0]))];
  const table = `<table><thead><tr><th>Model</th>${levels.map((l) => `<th>L${l}</th>`).join("")}</tr></thead><tbody>${models.map((m) =>
    `<tr><td>${esc(m.split("/").pop())}</td>${levels.map((l) => { const v = mean(rows.filter((r) => r.model === m && r.level === l), "score"); return `<td>${v === null ? "" : v.toFixed(1)}</td>`; }).join("")}</tr>`).join("")}</tbody></table>`;
  const ttable = `<table><thead><tr><th>Planted anomaly</th><th>Detected</th><th>Right type</th><th>Artifact call right</th><th>n</th></tr></thead><tbody>${types.map((t) => {
    const s = rows.filter((r) => r.case_id.startsWith(t + ":") && r.level >= 3);
    return `<tr><td>${esc(t.replace("_", " "))}</td><td>${pct(mean(s, "detected"))}</td><td>${pct(mean(s, "type_ok"))}</td><td>${pct(mean(s, "artifact_ok"))}</td><td>${s.length}</td></tr>`;
  }).join("")}</tbody></table>`;
  el.innerHTML = `<figure><h3>Detection rate by information level</h3>${line}<figcaption>Levels 1-2 put data in the prompt without tools; levels 3-5 give the full dataset as a file with tools, then figures, then literature and imagery. Run ${esc(d.id)}${models.includes("mock") ? " (rule-based baseline, not a language model)" : ""}.</figcaption></figure>
    <figure><h3>Detection by raw data read into context</h3>${bars}<figcaption>Longest contiguous stretch of raw numbers each attempt read, levels 3-5. The reference study found recognition rose with the amount of raw sequence read.</figcaption></figure>
    <figure><h3>Mean score out of 5</h3>${table}<figcaption>Detection, type, onset within 2 years, location, and the physical-versus-artifact call.</figcaption></figure>
    <figure><h3>By kind of planted anomaly (levels 3-5)</h3>${ttable}</figure>`;
}
const pct = (v) => (v === null ? "" : Math.round(v * 100) + "%");

/* ------------------------------------------------------------------ tabs, controls */
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  S.tab = b.dataset.tab;
  document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", x === b));
  document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== "tab-" + S.tab));
  if (S.tab === "bench") drawBench();
  else if (S.data) render();
}));
$("#campaign").addEventListener("change", (e) => {
  S.cid = e.target.value; S.follow = true; S.events = []; S.lastEvent = 0; S.sig = ""; S.selected = null; S.report = null;
  $("#map").dataset.key = ""; $("#inspector").innerHTML = `<div class="empty-state"><h2>Inspector</h2><p>Select a task in the network.</p></div>`;
  poll();
});
$("#stop").addEventListener("click", async () => {
  if (!S.cid || !confirm("Stop this campaign after the running tasks finish?")) return;
  await getJSON(`/api/stop/${S.cid}`, { method: "POST" });
  poll();
});
$("#launch-open").addEventListener("click", async () => {
  const briefs = await getJSON("/api/briefs");
  $("#brief").innerHTML = briefs.map((b) => `<option>${esc(b)}</option>`).join("");
  $("#launch").showModal();
});
$("#launch-form").addEventListener("submit", async (e) => {
  if (e.submitter && e.submitter.value === "go") {
    await getJSON("/api/launch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ brief: $("#brief").value, mock: $("#mock").checked }) });
    setTimeout(async () => { await loadCampaigns(true); S.events = []; S.lastEvent = 0; S.sig = ""; poll(); }, 2500);
  }
});

loadCampaigns().then(poll);
setInterval(poll, 2500);
