"""RestaurantEnv: a 240-minute table-assignment MDP.

Reconstructed from the project report "Deep Q-Learning for Dynamic Restaurant
Revenue Management". Coefficients that the report leaves unspecified live in
CONFIG and are written to docs/config.json so the public frontend uses the
same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TABLES = (4, 2, 2, 4, 6)
SHIFT = 240
# 0 defer (party waits), 1 reject (party leaves), 2..6 seat at tables 1..5.
# The report's "7 discrete actions" are defer/reject plus one seat action per table.
N_ACTIONS = 7
STATE_DIM = 12

# Report lambdas (0.6, 1.8, 0.4) are kept in ratio. They are treated as
# expected arrivals per 10 minutes, then scaled so the peak hour is congested
# enough that holding a large table can matter. Equal phase thirds.
LAM_SCALE = 2.5

CONFIG = {
    "tables": list(TABLES),
    "shift": SHIFT,
    "phases": [
        {"name": "Early", "start": 0, "end": 80, "lambda": 0.6 / 10.0 * LAM_SCALE},
        {"name": "Peak", "start": 80, "end": 160, "lambda": 1.8 / 10.0 * LAM_SCALE},
        {"name": "Late", "start": 160, "end": 240, "lambda": 0.4 / 10.0 * LAM_SCALE},
    ],
    "size_probs": [0.08, 0.32, 0.22, 0.18, 0.08, 0.12],
    "duration_base": 20,
    "duration_per_guest": 5,
    "revenue_per_guest": 20,
    "reservation_prob": 0.28,
    "patience_base": 18,
    "patience_per_guest": 4,
    "max_queue": 8,
    "reward": {
        "fit_exact": 20.0,
        "fit_near": 10.0,
        "reservation": 15.0,
        "peak": 4.0,
        "prompt_base": 10.0,
        "prompt_per_min": 0.4,
        "waste_per_slack": 8.0,
        "waste_per_slack_late": 3.0,
        "reject_reservation": 8.0,
        "reject_other": 1.0,
        "wait": 0.2,
        "renege": 6.0,
        "close_per_party": 5.0,
        "invalid": 25.0,
        "train_divisor": 50.0,
    },
}


def max_duration() -> int:
    return CONFIG["duration_base"] + CONFIG["duration_per_guest"] * 6


def max_revenue() -> int:
    return CONFIG["revenue_per_guest"] * 6


@dataclass
class Party:
    id: int
    size: int
    duration: int
    revenue: float
    reservation: bool
    patience: int
    wait: int = 0


class RestaurantEnv:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.table_rem = [0] * 5
        self.table_party: list[Party | None] = [None] * 5
        self.queue: list[Party] = []
        self.next_id = 1
        self.done = False
        self._reset_metrics()

    def _reset_metrics(self) -> None:
        self.gross_revenue = 0.0
        self.shaped = 0.0
        self.occupied_minutes = 0
        self.seated = 0
        self.rejected = 0
        self.reneged = 0
        self.lost_full = 0
        self.unserved_close = 0
        self.wait_sum = 0
        self.bad_seats = 0
        self.loose_seats = 0
        self.strategic_rejects = 0
        self.invalids = 0
        self.decision_count = 0

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.table_rem = [0] * 5
        self.table_party = [None] * 5
        self.queue = []
        self.next_id = 1
        self.done = False
        self._reset_metrics()
        self._arrive()
        while not self.queue and self.t < SHIFT:
            self._tick()
        if self.t >= SHIFT:
            self.done = True
        return self.observe(), self.action_mask()

    def phase_index(self) -> int:
        if self.t < 80:
            return 0
        if self.t < 160:
            return 1
        return 2

    def phase_name(self) -> str:
        return CONFIG["phases"][self.phase_index()]["name"]

    def lam(self) -> float:
        return CONFIG["phases"][self.phase_index()]["lambda"]

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[0] = True
        mask[1] = True
        if not self.queue:
            return mask
        size = self.queue[0].size
        for i, cap in enumerate(TABLES):
            mask[i + 2] = self.table_rem[i] == 0 and cap >= size
        return mask

    def observe(self) -> np.ndarray:
        if not self.queue:
            return np.zeros(STATE_DIM, dtype=np.float32)
        party = self.queue[0]
        dur = max_duration()
        rev = max_revenue()
        phase = (0.0, 0.5, 1.0)[self.phase_index()]
        obs = np.array(
            [
                (SHIFT - self.t) / SHIFT,
                *[rem / dur for rem in self.table_rem],
                len(self.queue) / CONFIG["max_queue"],
                party.size / 6.0,
                party.duration / dur,
                party.revenue / rev,
                1.0 if party.reservation else 0.0,
                phase,
            ],
            dtype=np.float32,
        )
        return obs

    def _sample_party(self) -> Party:
        size = int(self.rng.choice(np.arange(1, 7), p=np.array(CONFIG["size_probs"])))
        duration = CONFIG["duration_base"] + CONFIG["duration_per_guest"] * size
        revenue = float(CONFIG["revenue_per_guest"] * size)
        reservation = bool(self.rng.random() < CONFIG["reservation_prob"])
        patience = CONFIG["patience_base"] + CONFIG["patience_per_guest"] * size
        party = Party(self.next_id, size, duration, revenue, reservation, patience, 0)
        self.next_id += 1
        return party

    def _arrive(self) -> list[dict]:
        events = []
        n = int(self.rng.poisson(self.lam()))
        for _ in range(n):
            if len(self.queue) >= CONFIG["max_queue"]:
                self.lost_full += 1
                events.append({"type": "lost"})
                continue
            party = self._sample_party()
            self.queue.append(party)
            events.append({"type": "arrive", "party": _party_dict(party)})
        return events

    def _tick(self) -> tuple[list[dict], float]:
        events: list[dict] = []
        penalty = 0.0
        self.occupied_minutes += sum(1 for rem in self.table_rem if rem > 0)
        for i in range(5):
            if self.table_rem[i] > 0:
                self.table_rem[i] -= 1
                if self.table_rem[i] == 0:
                    self.table_party[i] = None
                    events.append({"type": "free", "table": i})
        self.t += 1
        still: list[Party] = []
        for party in self.queue:
            party.wait += 1
            if party.wait >= party.patience:
                self.reneged += 1
                penalty += CONFIG["reward"]["renege"]
                events.append({"type": "renege", "party": _party_dict(party)})
            else:
                still.append(party)
        self.queue = still
        if self.t < SHIFT:
            events.extend(self._arrive())
        return events, penalty

    def _min_slack(self, party: Party) -> int:
        slacks = [
            TABLES[i] - party.size
            for i in range(5)
            if self.table_rem[i] == 0 and TABLES[i] >= party.size
        ]
        return min(slacks) if slacks else 99

    def _seat_reward(self, party: Party, table: int) -> float:
        r = CONFIG["reward"]
        slack = TABLES[table] - party.size
        if slack == 0:
            bfit = r["fit_exact"]
        elif slack == 1:
            bfit = r["fit_near"]
        else:
            bfit = 0.0
        bres = r["reservation"] if party.reservation else 0.0
        bpeak = r["peak"] if self.phase_name() == "Peak" and slack <= 1 else 0.0
        bprompt = max(0.0, r["prompt_base"] - r["prompt_per_min"] * party.wait)
        waste_k = r["waste_per_slack_late"] if self.phase_name() == "Late" else r["waste_per_slack"]
        pwaste = slack * waste_k
        return party.revenue + bfit + bres + bpeak + bprompt - pwaste

    def step(self, action: int):
        if self.done:
            raise RuntimeError("episode is finished")
        if not self.queue:
            raise RuntimeError("no party to decide on")

        party = self.queue[0]
        mask = self.action_mask()
        action = int(action)
        events: list[dict] = []
        decision = {"type": "wait", "party": _party_dict(party), "table": None}

        if action >= 2 and not mask[action]:
            self.invalids += 1
            reward = -CONFIG["reward"]["invalid"]
            decision["type"] = "invalid"
        elif action == 0:
            reward = -CONFIG["reward"]["wait"]
            decision["type"] = "wait"
        elif action == 1:
            feasible = bool(mask[2:].any())
            loose = self._min_slack(party) >= 2
            self.queue.pop(0)
            self.rejected += 1
            if party.size <= 2 and feasible and loose and self.phase_name() != "Late":
                self.strategic_rejects += 1
            pen = (
                CONFIG["reward"]["reject_reservation"]
                if party.reservation
                else CONFIG["reward"]["reject_other"]
            )
            reward = -pen
            decision["type"] = "reject"
        else:
            ti = action - 2
            self.queue.pop(0)
            self.table_rem[ti] = party.duration
            self.table_party[ti] = party
            self.seated += 1
            self.gross_revenue += party.revenue
            self.wait_sum += party.wait
            if TABLES[ti] - party.size >= 3:
                self.bad_seats += 1
            if TABLES[ti] - party.size >= 2:
                self.loose_seats += 1
            reward = self._seat_reward(party, ti)
            decision = {"type": "seat", "party": _party_dict(party), "table": ti}

        self.decision_count += 1
        events.append(decision)

        while True:
            tick_events, pen = self._tick()
            events.extend(tick_events)
            reward -= pen
            if self.t >= SHIFT or self.queue:
                break

        done = self.t >= SHIFT
        if done:
            self.unserved_close = len(self.queue)
            reward -= CONFIG["reward"]["close_per_party"] * self.unserved_close
            self.queue.clear()
            self.done = True
            obs = np.zeros(STATE_DIM, dtype=np.float32)
            mask = np.zeros(N_ACTIONS, dtype=bool)
            mask[0] = True
        else:
            obs = self.observe()
            mask = self.action_mask()

        self.shaped += reward
        info = {
            "events": events,
            "metrics": self.metrics() if done else None,
        }
        return obs, float(reward), done, info, mask

    def metrics(self) -> dict:
        util = self.occupied_minutes / float(SHIFT * 5)
        avg_wait = (self.wait_sum / self.seated) if self.seated else 0.0
        return {
            "gross_revenue": self.gross_revenue,
            "utilization": util,
            "avg_wait": avg_wait,
            "seated": self.seated,
            "rejected": self.rejected,
            "reneged": self.reneged,
            "lost_full": self.lost_full,
            "unserved_close": self.unserved_close,
            "shaped_return": self.shaped,
            "bad_seats": self.bad_seats,
            "loose_seats": self.loose_seats,
            "strategic_rejects": self.strategic_rejects,
            "invalids": self.invalids,
            "decisions": self.decision_count,
        }


def _party_dict(party: Party) -> dict:
    return {
        "id": party.id,
        "size": party.size,
        "duration": party.duration,
        "revenue": party.revenue,
        "reservation": party.reservation,
        "patience": party.patience,
        "wait": party.wait,
    }


def self_check() -> None:
    env = RestaurantEnv(0)
    obs, mask = env.reset(0)
    assert obs.shape == (12,)
    assert mask.shape == (7,) and mask[0]
    # A party of 6 cannot use a 2-top.
    env.queue.clear()
    env.table_rem = [0, 0, 0, 0, 0]
    env.queue.append(Party(1, 6, 50, 120.0, False, 40, 0))
    mask = env.action_mask()
    assert mask.tolist() == [True, True, False, False, False, False, True]
    env.table_rem[4] = 10
    mask = env.action_mask()
    assert not mask[2:].any()
    # Prices quoted in the report.
    assert CONFIG["revenue_per_guest"] * 2 == 40
    assert CONFIG["revenue_per_guest"] * 6 == 120
    env = RestaurantEnv(1)
    obs, mask = env.reset(1)
    done = False
    guard = 0
    while not done:
        valid = np.flatnonzero(mask)
        action = int(valid[-1]) if mask[2:].any() else 0
        _, _, done, _, mask = env.step(action)
        guard += 1
        if guard > 5000:
            raise AssertionError("episode did not finish")
    m = env.metrics()
    assert env.t == SHIFT
    assert 0.0 <= m["utilization"] <= 1.0
    print("self_check ok", {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})


if __name__ == "__main__":
    self_check()
