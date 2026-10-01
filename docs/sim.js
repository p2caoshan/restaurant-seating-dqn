/** Browser port of RestaurantEnv and the Double DQN policy. */

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function rng() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function poisson(lam, rng) {
  if (lam <= 0) return 0;
  const L = Math.exp(-lam);
  let k = 0;
  let p = 1;
  do {
    k += 1;
    let u = rng();
    if (u <= 0) u = 1e-12;
    p *= u;
  } while (p > L);
  return k - 1;
}

function maxDuration(config) {
  return config.duration_base + config.duration_per_guest * 6;
}

function maxRevenue(config) {
  return config.revenue_per_guest * 6;
}

export class RestaurantEnv {
  constructor(config) {
    this.config = config;
    this.rng = mulberry32(1);
    this.reset(1);
  }

  reset(seed) {
    this.rng = mulberry32(seed >>> 0);
    this.t = 0;
    this.tableRem = [0, 0, 0, 0, 0];
    this.tableParty = [null, null, null, null, null];
    this.queue = [];
    this.nextId = 1;
    this.done = false;
    this.grossRevenue = 0;
    this.shaped = 0;
    this.occupiedMinutes = 0;
    this.seated = 0;
    this.rejected = 0;
    this.reneged = 0;
    this.lostFull = 0;
    this.unservedClose = 0;
    this.waitSum = 0;
    this.badSeats = 0;
    this.looseSeats = 0;
    this.strategicRejects = 0;
    this.decisions = 0;
    this._arrive();
    while (this.queue.length === 0 && this.t < this.config.shift) this._tick();
    if (this.t >= this.config.shift) this.done = true;
    return this.observe();
  }

  phaseIndex() {
    if (this.t < 80) return 0;
    if (this.t < 160) return 1;
    return 2;
  }

  phaseName() {
    return this.config.phases[this.phaseIndex()].name;
  }

  lam() {
    return this.config.phases[this.phaseIndex()].lambda;
  }

  actionMask() {
    const mask = [true, true, false, false, false, false, false];
    if (!this.queue.length) return mask;
    const size = this.queue[0].size;
    this.config.tables.forEach((cap, i) => {
      mask[i + 2] = this.tableRem[i] === 0 && cap >= size;
    });
    return mask;
  }

  observe() {
    if (!this.queue.length) return new Array(12).fill(0);
    const party = this.queue[0];
    const dur = maxDuration(this.config);
    const rev = maxRevenue(this.config);
    const phase = [0, 0.5, 1][this.phaseIndex()];
    return [
      (this.config.shift - this.t) / this.config.shift,
      ...this.tableRem.map((rem) => rem / dur),
      this.queue.length / this.config.max_queue,
      party.size / 6,
      party.duration / dur,
      party.revenue / rev,
      party.reservation ? 1 : 0,
      phase,
    ];
  }

  minSlack(party) {
    let best = 99;
    this.config.tables.forEach((cap, i) => {
      if (this.tableRem[i] === 0 && cap >= party.size) best = Math.min(best, cap - party.size);
    });
    return best;
  }

  seatReward(party, table) {
    const r = this.config.reward;
    const slack = this.config.tables[table] - party.size;
    const bfit = slack === 0 ? r.fit_exact : slack === 1 ? r.fit_near : 0;
    const bres = party.reservation ? r.reservation : 0;
    const bpeak = this.phaseName() === "Peak" && slack <= 1 ? r.peak : 0;
    const bprompt = Math.max(0, r.prompt_base - r.prompt_per_min * party.wait);
    const wasteK = this.phaseName() === "Late" ? r.waste_per_slack_late : r.waste_per_slack;
    return party.revenue + bfit + bres + bpeak + bprompt - slack * wasteK;
  }

  _sampleParty() {
    const probs = this.config.size_probs;
    let u = this.rng();
    let size = 6;
    let acc = 0;
    for (let i = 0; i < probs.length; i += 1) {
      acc += probs[i];
      if (u <= acc) {
        size = i + 1;
        break;
      }
    }
    const party = {
      id: this.nextId,
      size,
      duration: this.config.duration_base + this.config.duration_per_guest * size,
      revenue: this.config.revenue_per_guest * size,
      reservation: this.rng() < this.config.reservation_prob,
      patience: this.config.patience_base + this.config.patience_per_guest * size,
      wait: 0,
    };
    this.nextId += 1;
    return party;
  }

  _arrive() {
    const events = [];
    const n = poisson(this.lam(), this.rng);
    for (let i = 0; i < n; i += 1) {
      if (this.queue.length >= this.config.max_queue) {
        this.lostFull += 1;
        events.push({ type: "lost" });
        continue;
      }
      const party = this._sampleParty();
      this.queue.push(party);
      events.push({ type: "arrive", party: { ...party } });
    }
    return events;
  }

  _tick() {
    const events = [];
    let penalty = 0;
    this.occupiedMinutes += this.tableRem.filter((rem) => rem > 0).length;
    for (let i = 0; i < 5; i += 1) {
      if (this.tableRem[i] > 0) {
        this.tableRem[i] -= 1;
        if (this.tableRem[i] === 0) {
          this.tableParty[i] = null;
          events.push({ type: "free", table: i });
        }
      }
    }
    this.t += 1;
    const still = [];
    for (const party of this.queue) {
      party.wait += 1;
      if (party.wait >= party.patience) {
        this.reneged += 1;
        penalty += this.config.reward.renege;
        events.push({ type: "renege", party: { ...party } });
      } else still.push(party);
    }
    this.queue = still;
    if (this.t < this.config.shift) events.push(...this._arrive());
    return { events, penalty };
  }

  step(action) {
    if (this.done) throw new Error("episode is finished");
    if (!this.queue.length) throw new Error("no party to decide on");
    const party = this.queue[0];
    const mask = this.actionMask();
    const events = [];
    let reward = 0;
    let decision = { type: "wait", party: { ...party }, table: null };

    if (action >= 2 && !mask[action]) {
      reward = -this.config.reward.invalid;
      decision = { type: "invalid", party: { ...party }, table: null };
    } else if (action === 0) {
      reward = -this.config.reward.wait;
    } else if (action === 1) {
      const feasible = mask.slice(2).some(Boolean);
      const loose = this.minSlack(party) >= 2;
      this.queue.shift();
      this.rejected += 1;
      if (party.size <= 2 && feasible && loose && this.phaseName() !== "Late") this.strategicRejects += 1;
      const pen = party.reservation ? this.config.reward.reject_reservation : this.config.reward.reject_other;
      reward = -pen;
      decision = { type: "reject", party: { ...party }, table: null };
    } else {
      const ti = action - 2;
      this.queue.shift();
      this.tableRem[ti] = party.duration;
      this.tableParty[ti] = party;
      this.seated += 1;
      this.grossRevenue += party.revenue;
      this.waitSum += party.wait;
      const slack = this.config.tables[ti] - party.size;
      if (slack >= 3) this.badSeats += 1;
      if (slack >= 2) this.looseSeats += 1;
      reward = this.seatReward(party, ti);
      decision = { type: "seat", party: { ...party }, table: ti };
    }

    this.decisions += 1;
    events.push(decision);
    while (true) {
      const tick = this._tick();
      events.push(...tick.events);
      reward -= tick.penalty;
      if (this.t >= this.config.shift || this.queue.length) break;
    }

    if (this.t >= this.config.shift) {
      this.unservedClose = this.queue.length;
      reward -= this.config.reward.close_per_party * this.unservedClose;
      this.queue = [];
      this.done = true;
    }
    this.shaped += reward;
    return { reward, done: this.done, events };
  }

  metrics() {
    return {
      grossRevenue: this.grossRevenue,
      utilization: this.occupiedMinutes / (this.config.shift * 5),
      avgWait: this.seated ? this.waitSum / this.seated : 0,
      seated: this.seated,
      rejected: this.rejected,
      reneged: this.reneged,
      lostFull: this.lostFull,
      unservedClose: this.unservedClose,
      looseSeats: this.looseSeats,
      strategicRejects: this.strategicRejects,
      decisions: this.decisions,
    };
  }
}

export class QNetwork {
  constructor(payload) {
    this.layers = payload.layers;
    this.rewardScale = payload.reward_scale;
  }

  q(state) {
    let h = state;
    this.layers.forEach((layer, idx) => {
      const out = new Array(layer.b.length);
      for (let i = 0; i < layer.b.length; i += 1) {
        let s = layer.b[i];
        const row = layer.w[i];
        for (let j = 0; j < h.length; j += 1) s += row[j] * h[j];
        out[i] = idx === this.layers.length - 1 ? s : Math.max(0, s);
      }
      h = out;
    });
    return h;
  }

  act(state, mask) {
    const q = this.q(state);
    let best = -1;
    let bestQ = -1e18;
    for (let i = 0; i < q.length; i += 1) {
      if (mask[i] && q[i] > bestQ) {
        bestQ = q[i];
        best = i;
      }
    }
    return best;
  }
}

export function fcfs(env) {
  const mask = env.actionMask();
  for (let i = 0; i < 5; i += 1) if (mask[i + 2]) return i + 2;
  return 0;
}

export function greedy(env) {
  const mask = env.actionMask();
  const party = env.queue[0];
  const candidates = [];
  for (let i = 0; i < 5; i += 1) if (mask[i + 2]) candidates.push(i);
  if (!candidates.length) return 0;
  candidates.sort((a, b) => {
    const sa = env.config.tables[a] - party.size;
    const sb = env.config.tables[b] - party.size;
    if (sa !== sb) return sa - sb;
    if (env.config.tables[a] !== env.config.tables[b]) return env.config.tables[a] - env.config.tables[b];
    return a - b;
  });
  const best = candidates[0];
  const slack = env.config.tables[best] - party.size;
  if (party.reservation || slack <= 1 || env.phaseName() === "Late") return best + 2;
  if (party.size <= 2 && slack >= 3) return 1;
  return best + 2;
}

export function sanity(config) {
  const env = new RestaurantEnv(config);
  env.t = 100;
  env.tableRem = [0, 0, 0, 0, 0];
  env.queue = [{ id: 1, size: 6, duration: 50, revenue: 120, reservation: false, patience: 40, wait: 0 }];
  const mask = env.actionMask();
  if (mask.join(",") !== "true,true,false,false,false,false,true") {
    throw new Error(`unexpected mask ${mask.join(",")}`);
  }
  env.queue[0] = { id: 1, size: 2, duration: 30, revenue: 40, reservation: false, patience: 26, wait: 0 };
  const loose = env.seatReward(env.queue[0], 4);
  const fit = env.seatReward(env.queue[0], 1);
  if (Math.abs(loose - 18) > 1e-6 || Math.abs(fit - 74) > 1e-6) {
    throw new Error(`reward mismatch loose=${loose} fit=${fit}`);
  }
}

export function checkProbe(net, probe) {
  const q = net.q(probe.state);
  let max = 0;
  for (let i = 0; i < q.length; i += 1) max = Math.max(max, Math.abs(q[i] - probe.q[i]));
  return max;
}
