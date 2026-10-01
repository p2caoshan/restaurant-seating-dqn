import { RestaurantEnv, QNetwork, fcfs, greedy, sanity, checkProbe } from "./sim.js";

const ACTION_NAME = ["Wait", "Turn away", "Table 1", "Table 2", "Table 3", "Table 4", "Table 5"];
const state = {
  config: null,
  net: null,
  study: null,
  env: null,
  policy: "dqn",
  seed: 7,
  playing: false,
  speed: 160,
  peek: false,
  trail: [],
  log: [],
  timer: null,
  userMetrics: null,
};

const $ = (id) => document.getElementById(id);

function money(n) {
  return `$${Math.round(n).toLocaleString("en-US")}`;
}
function moneyExact(n) {
  return `$${Number(n).toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}`;
}
function pct(n) {
  return `${(100 * n).toFixed(1)}%`;
}
function clock(t) {
  const mins = 17 * 60 + t;
  const h = Math.floor(mins / 60) % 24;
  const m = mins % 60;
  const suffix = h >= 12 ? "PM" : "AM";
  const h12 = ((h + 11) % 12) + 1;
  return `${h12}:${String(m).padStart(2, "0")} ${suffix}`;
}

function choose(env, policy) {
  if (policy === "fcfs") return fcfs(env);
  if (policy === "greedy") return greedy(env);
  return state.net.act(env.observe(), env.actionMask());
}

function explain(env, action) {
  if (!env.queue.length) return "The door is clear.";
  const party = env.queue[0];
  if (action === 1 && party.size <= 2 && env.minSlack(party) >= 2 && env.phaseName() !== "Late") {
    const booked = party.reservation ? " They hold a reservation, and the table is still kept back." : "";
    return `Strategic deferral. Seating ${party.size} would tie up a large table while ${env.phaseName().toLowerCase()} service can still bring a bigger check.${booked}`;
  }
  if (action >= 2) {
    const table = action - 2;
    const slack = env.config.tables[table] - party.size;
    if (slack === 0) return `Exact fit at Table ${table + 1}. The check is ${money(party.revenue)}.`;
    if (slack >= 3) return `This puts ${party.size} guests at a ${env.config.tables[table]}-top, leaving ${slack} seats empty.`;
    return `Seat them at Table ${table + 1}. The check is ${money(party.revenue)}.`;
  }
  if (action === 0) return "No legal table. They keep their place in line.";
  return "The party is turned away so the table stays free.";
}

function pushLog(events) {
  const lines = [];
  for (const event of events) {
    if (event.type === "seat") lines.push(`Seated ${event.party.size} at Table ${event.table + 1} for ${money(event.party.revenue)}.`);
    else if (event.type === "reject") lines.push(`Turned away a party of ${event.party.size}${event.party.reservation ? " with a reservation" : ""}.`);
    else if (event.type === "renege") lines.push(`A party of ${event.party.size} left after waiting.`);
    else if (event.type === "lost") lines.push("The line was full, so an arrival walked on.");
    else if (event.type === "free") lines.push(`Table ${event.table + 1} is clear.`);
  }
  if (!lines.length) return;
  state.log.unshift(`${clock(state.env.t)} · ${lines.join(" ")}`);
  state.log = state.log.slice(0, 8);
}

function commit(action) {
  const result = state.env.step(action);
  state.trail.push({ t: state.env.t, revenue: state.env.grossRevenue });
  pushLog(result.events);
  if (state.env.done) {
    stop();
    if (state.policy === "host") state.userMetrics = state.env.metrics();
  }
  return result;
}

function startNight(seed) {
  stop();
  state.seed = seed;
  state.env = new RestaurantEnv(state.config);
  state.env.reset(seed);
  state.trail = [{ t: state.env.t, revenue: 0 }];
  state.log = [`${clock(state.env.t)} · The doors open.`];
  state.userMetrics = null;
  const url = new URL(location.href);
  url.searchParams.set("seed", String(seed));
  url.searchParams.set("policy", state.policy);
  history.replaceState(null, "", url);
  render();
}

function stop() {
  state.playing = false;
  if (state.timer) clearTimeout(state.timer);
  state.timer = null;
}

function play() {
  if (state.policy === "host" || state.env.done) return;
  state.playing = true;
  const loop = () => {
    if (!state.playing || state.env.done) return;
    commit(choose(state.env, state.policy));
    render();
    if (!state.env.done) state.timer = setTimeout(loop, state.speed);
  };
  loop();
}

function rollout(policy, seed) {
  const env = new RestaurantEnv(state.config);
  env.reset(seed);
  let guard = 0;
  while (!env.done && guard < 5000) {
    env.step(choose(env, policy));
    guard += 1;
  }
  return env.metrics();
}

async function compareNights() {
  const button = $("compare-nights");
  button.disabled = true;
  const seeds = Array.from({ length: 30 }, (_, i) => i + 1);
  const totals = { fcfs: [], greedy: [], dqn: [] };
  for (let i = 0; i < seeds.length; i += 1) {
    totals.fcfs.push(rollout("fcfs", seeds[i]));
    totals.greedy.push(rollout("greedy", seeds[i]));
    totals.dqn.push(rollout("dqn", seeds[i]));
    if (i % 5 === 0) {
      $("bench").textContent = `Resimulating nights in this browser… ${i + 1} / 30`;
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }
  const mean = (rows) => rows.reduce((s, r) => s + r.grossRevenue, 0) / rows.length;
  const fcfsM = mean(totals.fcfs);
  const greedyM = mean(totals.greedy);
  const dqnM = mean(totals.dqn);
  $("bench").textContent = `30 nights in this browser. FCFS ${money(fcfsM)}, greedy ${money(greedyM)} (${pct((greedyM - fcfsM) / fcfsM)} vs FCFS), Double DQN ${money(dqnM)} (${pct((dqnM - fcfsM) / fcfsM)} vs FCFS). These draws use the browser random generator, so they will not match the published test set exactly.`;
  button.disabled = false;
}

function renderTables() {
  const mask = state.env.queue.length ? state.env.actionMask() : [];
  document.querySelectorAll(".table").forEach((button) => {
    const i = Number(button.dataset.table);
    const party = state.env.tableParty[i];
    const cap = state.config.tables[i];
    button.classList.toggle("occupied", Boolean(party));
    button.classList.toggle("can-seat", state.policy === "host" && !state.env.done && Boolean(mask[i + 2]));
    button.classList.toggle("dim", state.policy === "host" && !state.env.done && !mask[i + 2]);
    button.disabled = state.policy !== "host" || state.env.done || !mask[i + 2];
    button.innerHTML = party
      ? `<strong>Table ${i + 1}</strong><span>${party.size} seated · ${state.env.tableRem[i]} min</span><span>${money(party.revenue)}</span>`
      : `<strong>Table ${i + 1}</strong><span>${cap} seats</span><span>Open</span>`;
  });
}

function renderParty() {
  const env = state.env;
  if (!env.queue.length) {
    $("party").innerHTML = `<div class="party"><p>${env.done ? "Service is over." : "Waiting for the next arrival."}</p></div>`;
    $("queue").innerHTML = "";
    return;
  }
  const [head, ...rest] = env.queue;
  $("party").innerHTML = partyCard(head, true);
  $("queue").innerHTML = rest.map((party) => partyCard(party, false)).join("");
}

function partyCard(party, now) {
  return `<article class="party${now ? " now" : ""}">
    <div class="party-top"><span class="kicker">${now ? "Deciding now" : "Waiting"}</span><span class="money">${money(party.revenue)}</span></div>
    <div class="row"><strong>${party.size} ${party.size === 1 ? "guest" : "guests"}</strong>${party.reservation ? '<span class="tag">Reserved</span>' : ""}</div>
    <p class="kicker">${party.duration} min table · waited ${party.wait} / ${party.patience}</p>
  </article>`;
}

function renderActions() {
  const env = state.env;
  const box = $("actions");
  const bars = $("qbars");
  const explainEl = $("explain");
  if (env.done || !env.queue.length) {
    box.innerHTML = "";
    bars.innerHTML = "";
    explainEl.textContent = env.done ? "The kitchen is closed. Start another night to host again." : "";
    return;
  }
  const mask = env.actionMask();
  const showQ = state.policy === "dqn" || state.peek;
  const q = showQ ? state.net.q(env.observe()) : null;
  const suggested = q ? state.net.act(env.observe(), mask) : null;
  if (state.policy === "host") {
    const buttons = [
      [0, "Ask them to wait", ""],
      [1, "Turn away", "danger"],
    ];
    for (let i = 0; i < 5; i += 1) buttons.push([i + 2, `Seat Table ${i + 1}`, ""]);
    box.innerHTML = buttons.map(([action, label, cls]) =>
      `<button type="button" data-action="${action}" class="${cls}" ${mask[action] ? "" : "disabled"}>${label}</button>`
    ).join("");
    explainEl.textContent = showQ ? `DQN would choose ${ACTION_NAME[suggested]}. ${explain(env, suggested)}` : "Seat them at a highlighted table, ask them to wait, or turn them away.";
  } else {
    const action = choose(env, state.policy);
    box.innerHTML = `<button type="button" class="primary" id="play-inline">${state.playing ? "Pause" : "Play this choice"}</button>`;
    explainEl.textContent = explain(env, action);
    $("play-inline").addEventListener("click", () => (state.playing ? (stop(), render()) : play()));
  }
  if (!q) {
    bars.innerHTML = "";
    return;
  }
  const validQ = q.filter((_, i) => mask[i]);
  const lo = Math.min(...validQ);
  const hi = Math.max(...validQ);
  bars.innerHTML = q.map((value, i) => {
    const width = mask[i] && hi > lo ? ((value - lo) / (hi - lo)) * 100 : 0;
    return `<div class="qrow${i === suggested ? " best" : ""}${mask[i] ? "" : " bad"}">
      <span>${ACTION_NAME[i]}</span>
      <div class="qtrack"><div class="qfill" style="width:${width}%"></div></div>
      <span>${mask[i] ? (value * state.net.rewardScale).toFixed(0) : "—"}</span>
    </div>`;
  }).join("");
}

function renderMetrics() {
  const m = state.env.metrics();
  const phase = state.env.done ? "Closed" : state.env.phaseName();
  $("clock").textContent = `${clock(state.env.t)} · ${phase}`;
  const cards = [
    [money(m.grossRevenue), "Gross revenue"],
    [String(m.seated), "Parties seated"],
    [pct(m.utilization), "Table time used"],
    [m.avgWait.toFixed(1), "Average wait, minutes"],
  ];
  $("metrics").innerHTML = cards.map(([value, label]) => `<article class="metric"><b>${value}</b><span>${label}</span></article>`).join("");
  $("log").innerHTML = state.log.map((line) => `<li>${line}</li>`).join("");
  $("live").textContent = state.log[0] || "";
  drawSpark();
  if (state.env.done) drawScore();
  else $("score").innerHTML = "";
}

function drawSpark() {
  const svg = $("spark");
  const pts = state.trail;
  if (pts.length < 2) {
    svg.innerHTML = "";
    return;
  }
  const maxR = Math.max(...pts.map((p) => p.revenue), 1);
  const d = pts.map((p, i) => {
    const x = (p.t / state.config.shift) * 640;
    const y = 68 - (p.revenue / maxR) * 60;
    return `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(" ");
  svg.innerHTML = `<path d="${d}" fill="none" stroke="#e4b15a" stroke-width="2"/>`;
}

function drawScore() {
  const rows = [
    ["FCFS", rollout("fcfs", state.seed)],
    ["Greedy", rollout("greedy", state.seed)],
    ["Double DQN", rollout("dqn", state.seed)],
  ];
  if (state.userMetrics) rows.unshift(["Your service", state.userMetrics]);
  const maxR = Math.max(...rows.map(([, m]) => m.grossRevenue), 1);
  $("score").innerHTML = rows.map(([name, m]) => `
    <article class="${name === "Your service" ? "you" : ""}">
      <p class="kicker">${name}</p>
      <b class="money">${money(m.grossRevenue)}</b>
      <p class="note">${pct(m.utilization)} full · wait ${m.avgWait.toFixed(1)} min · ${m.looseSeats} loose seats</p>
      <div class="barline"><span style="width:${(100 * m.grossRevenue) / maxR}%"></span></div>
    </article>`).join("");
}

function renderStudy() {
  const study = state.study;
  const hist = study.history;
  const vals = hist.map((row) => row.val_revenue);
  const fcfs = study.baselines_val.fcfs.gross_revenue.mean;
  const greedyMean = study.baselines_val.greedy.gross_revenue.mean;
  const lo = Math.min(...vals, fcfs, greedyMean) * 0.96;
  const hi = Math.max(...vals, fcfs, greedyMean) * 1.04;
  const x = (i) => 36 + (i / Math.max(1, hist.length - 1)) * 580;
  const y = (v) => 20 + ((hi - v) / (hi - lo)) * 190;
  const d = hist.map((row, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(row.val_revenue).toFixed(1)}`).join(" ");
  $("curve").innerHTML = `
    <line x1="36" y1="${y(fcfs)}" x2="616" y2="${y(fcfs)}" stroke="#d86a45" stroke-dasharray="4 4"/>
    <line x1="36" y1="${y(greedyMean)}" x2="616" y2="${y(greedyMean)}" stroke="#9cba78" stroke-dasharray="4 4"/>
    <path d="${d}" fill="none" stroke="#e4b15a" stroke-width="2.5"/>
    <text x="40" y="16" fill="#b7aa9b" font-size="12">DQN validation · best checkpoint episode ${study.best_episode}</text>`;

  const test = study.test;
  const published = study.report_table;
  $("study").innerHTML = `
    <article>
      <h3>This reconstruction</h3>
      <p class="note">50 held-out nights, after training.</p>
      ${metricTable(test)}
    </article>
    <article>
      <h3>Published in the report</h3>
      <p class="note">${published.note}</p>
      ${reportTable(published)}
    </article>`;
  $("deferral").disabled = !study.demo;
}

function metricTable(test) {
  const row = (label, key, fmt) => `<tr><td>${label}</td>${["fcfs", "greedy", "dqn"].map((name) => `<td>${fmt(test[name][key])}</td>`).join("")}</tr>`;
  return `<table>
    <tr><th></th><th>FCFS</th><th>Greedy</th><th>DQN</th></tr>
    ${row("Revenue", "gross_revenue", (v) => `${money(v.mean)} ± ${Math.round(v.std)}`)}
    ${row("Utilisation", "utilization", (v) => pct(v.mean))}
    ${row("Wait", "avg_wait", (v) => v.mean.toFixed(1))}
    <tr><td>Vs FCFS</td><td>—</td><td>${pct(test.greedy.uplift_vs_fcfs)}</td><td>${pct(test.dqn.uplift_vs_fcfs)}</td></tr>
  </table>`;
}

function reportTable(published) {
  const cell = (name, key, digits = 1) => Number(published[name][key]).toFixed(digits);
  return `<table>
    <tr><th></th><th>FCFS</th><th>Greedy</th><th>DQN</th></tr>
    <tr><td>Revenue</td><td>${moneyExact(published.fcfs.gross_revenue)}</td><td>${moneyExact(published.greedy.gross_revenue)}</td><td>${moneyExact(published.dqn.gross_revenue)}</td></tr>
    <tr><td>Utilisation</td><td>${pct(published.fcfs.utilization)}</td><td>${pct(published.greedy.utilization)}</td><td>${pct(published.dqn.utilization)}</td></tr>
    <tr><td>Wait</td><td>${cell("fcfs", "avg_wait")}</td><td>${cell("greedy", "avg_wait")}</td><td>${cell("dqn", "avg_wait")}</td></tr>
    <tr><td>Vs FCFS</td><td>—</td><td>${pct(published.greedy.uplift)}</td><td>${pct(published.dqn.uplift)}</td></tr>
  </table>`;
}

function render() {
  renderTables();
  renderParty();
  renderActions();
  renderMetrics();
  document.querySelectorAll("#policies button").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.policy === state.policy));
  });
  $("play").textContent = state.playing ? "Pause" : "Play service";
  $("play").disabled = state.policy === "host" || state.env.done;
  $("seed").value = String(state.seed);
}

function findDeferral() {
  for (let seed = 1; seed <= 120; seed += 1) {
    const env = new RestaurantEnv(state.config);
    env.reset(seed);
    let step = 0;
    while (!env.done && step < 500) {
      const party = env.queue[0];
      const mask = env.actionMask();
      const action = state.net.act(env.observe(), mask);
      const open = [];
      env.config.tables.forEach((cap, i) => {
        if (mask[i + 2]) open.push(cap);
      });
      if (action === 1 && party.size <= 2 && open.length && Math.min(...open) >= 6 && env.phaseName() !== "Late") {
        return { seed, decisionIndex: step };
      }
      env.step(action);
      step += 1;
    }
  }
  return null;
}

function watchDeferral() {
  $("deferral").disabled = true;
  $("explain").textContent = "Looking through nights for a moment when only a large table is free…";
  window.setTimeout(() => {
    const demo = findDeferral();
    $("deferral").disabled = false;
    if (!demo) {
      $("explain").textContent = "No clear deferral turned up in the first 120 nights. Try again.";
      return;
    }
    state.policy = "dqn";
    startNight(demo.seed);
    for (let i = 0; i < demo.decisionIndex; i += 1) {
      if (state.env.done) break;
      commit(choose(state.env, "dqn"));
    }
    state.log.unshift("Paused where the agent turns a small party away rather than fill the large table.");
    render();
  }, 30);
}

function bind() {
  $("policies").addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    stop();
    state.policy = button.dataset.policy;
    render();
  });
  $("play").addEventListener("click", () => (state.playing ? (stop(), render()) : play()));
  $("step").addEventListener("click", () => {
    if (state.env.done || state.policy === "host") return;
    stop();
    commit(choose(state.env, state.policy));
    render();
  });
  $("speeds").addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    state.speed = Number(button.dataset.speed);
    document.querySelectorAll("#speeds button").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
  });
  $("fresh").addEventListener("click", () => startNight(1 + Math.floor(Math.random() * 99998)));
  $("seed").addEventListener("change", () => startNight(Math.max(1, Number($("seed").value) || 1)));
  $("deferral").addEventListener("click", watchDeferral);
  $("peek").addEventListener("change", () => {
    state.peek = $("peek").checked;
    render();
  });
  $("compare-nights").addEventListener("click", compareNights);
  $("actions").addEventListener("click", (event) => {
    const button = event.target.closest("[data-action]");
    if (!button || state.policy !== "host" || state.env.done) return;
    commit(Number(button.dataset.action));
    render();
  });
  $("room").addEventListener("click", (event) => {
    const button = event.target.closest("[data-table]");
    if (!button || state.policy !== "host" || state.env.done) return;
    const action = Number(button.dataset.table) + 2;
    if (!state.env.actionMask()[action]) return;
    commit(action);
    render();
  });
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input")) return;
    if (event.code === "Space") {
      event.preventDefault();
      if (state.policy !== "host") state.playing ? (stop(), render()) : play();
    }
    if (state.policy !== "host" || state.env.done) return;
    const key = event.key.toLowerCase();
    const map = { w: 0, x: 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6 };
    if (map[key] === undefined) return;
    if (!state.env.actionMask()[map[key]]) return;
    commit(map[key]);
    render();
  });
}

async function main() {
  try {
    const [config, weights, study] = await Promise.all([
      fetch("config.json").then((res) => res.json()),
      fetch("weights.json").then((res) => res.json()),
      fetch("study.json").then((res) => res.json()),
    ]);
    sanity(config);
    state.config = config;
    state.net = new QNetwork(weights);
    state.study = study;
    const gap = checkProbe(state.net, study.probe);
    if (gap > 1e-4) throw new Error(`weight check failed (${gap.toExponential(2)})`);
    const params = new URLSearchParams(location.search);
    state.policy = params.get("policy") || "dqn";
    if (!["host", "dqn", "greedy", "fcfs"].includes(state.policy)) state.policy = "dqn";
    $("boot").hidden = true;
    $("app").hidden = false;
    bind();
    renderStudy();
    startNight(Math.max(1, Number(params.get("seed")) || (study.demo ? study.demo.seed : 7)));
  } catch (error) {
    $("boot").textContent = `The dining room did not load. ${error.message}`;
  }
}

main();
