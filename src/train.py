"""Train a Double DQN and export weights for the public frontend.

Architecture from the report: MLP 12 -> 64 -> 64 -> 7, target network,
gamma 0.98, valid-action masking, 2,000 episodes with early stopping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from env import CONFIG, N_ACTIONS, SHIFT, STATE_DIM, RestaurantEnv, self_check
from policies import fcfs, greedy, masked_argmax

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MODELS = ROOT / "models"

LAYERS = (STATE_DIM, 64, 64, N_ACTIONS)
GAMMA = 0.98
REWARD_SCALE = CONFIG["reward"]["train_divisor"]


class MLP:
    def __init__(self, rng: np.random.Generator):
        self.W1 = rng.normal(0, np.sqrt(2 / LAYERS[0]), size=(LAYERS[1], LAYERS[0])).astype(np.float64)
        self.b1 = np.zeros(LAYERS[1], dtype=np.float64)
        self.W2 = rng.normal(0, np.sqrt(2 / LAYERS[1]), size=(LAYERS[2], LAYERS[1])).astype(np.float64)
        self.b2 = np.zeros(LAYERS[2], dtype=np.float64)
        self.W3 = rng.normal(0, np.sqrt(2 / LAYERS[2]), size=(LAYERS[3], LAYERS[2])).astype(np.float64)
        self.b3 = np.zeros(LAYERS[3], dtype=np.float64)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def copy_from(self, other: "MLP") -> None:
        for dst, src in zip(self.params(), other.params()):
            dst[...] = src

    def forward(self, x: np.ndarray):
        z1 = x @ self.W1.T + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.W2.T + self.b2
        h2 = np.maximum(z2, 0.0)
        q = h2 @ self.W3.T + self.b3
        return q, (x, z1, h1, z2, h2)

    def q(self, x: np.ndarray) -> np.ndarray:
        q, _ = self.forward(x)
        return q


class Replay:
    def __init__(self, cap: int):
        self.cap = cap
        self.s = np.zeros((cap, STATE_DIM), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.s2 = np.zeros((cap, STATE_DIM), np.float32)
        self.d = np.zeros(cap, np.float32)
        self.m2 = np.zeros((cap, N_ACTIONS), np.bool_)
        self.i = 0
        self.n = 0

    def add(self, s, a, r, s2, d, m2) -> None:
        self.s[self.i] = s
        self.a[self.i] = a
        self.r[self.i] = r
        self.s2[self.i] = s2
        self.d[self.i] = d
        self.m2[self.i] = m2
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)


class Adam:
    def __init__(self, params, lr: float):
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads) -> None:
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * (g * g)
            mhat = m / (1 - b1**self.t)
            vhat = v / (1 - b2**self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + eps)


def run_episode(env: RestaurantEnv, choose, seed: int) -> dict:
    _, mask = env.reset(seed)
    done = False
    while not done:
        action = choose(env)
        _, _, done, info, mask = env.step(action)
    return info["metrics"]


def summarize(rows: list[dict]) -> dict:
    def col(key):
        arr = np.array([r[key] for r in rows], dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0)}

    keys = (
        "gross_revenue",
        "utilization",
        "avg_wait",
        "seated",
        "rejected",
        "bad_seats",
        "loose_seats",
        "strategic_rejects",
    )
    return {k: col(k) for k in keys}


def evaluate(choose, seeds: list[int]) -> dict:
    env = RestaurantEnv()
    rows = [run_episode(env, choose, s) for s in seeds]
    return summarize(rows)


def dqn_policy(net: MLP):
    def choose(env):
        q = net.q(env.observe()[None, :])[0]
        return masked_argmax(q, env.action_mask())

    return choose


def learn(online: MLP, target: MLP, opt: Adam, replay: Replay, rng: np.random.Generator, batch: int) -> float:
    idx = rng.integers(0, replay.n, size=batch)
    s = replay.s[idx].astype(np.float64)
    a = replay.a[idx]
    r = replay.r[idx].astype(np.float64)
    s2 = replay.s2[idx].astype(np.float64)
    d = replay.d[idx].astype(np.float64)
    m2 = replay.m2[idx]

    q2_online = online.q(s2)
    q2_target = target.q(s2)
    a2 = np.argmax(np.where(m2, q2_online, -1e9), axis=1)
    y = r + GAMMA * q2_target[np.arange(batch), a2] * (1.0 - d)

    q, (x, z1, h1, z2, h2) = online.forward(s)
    chosen = q[np.arange(batch), a]
    diff = (chosen - y) / batch
    loss = float(np.mean((chosen - y) ** 2))

    dQ = np.zeros_like(q)
    dQ[np.arange(batch), a] = diff
    dW3 = dQ.T @ h2
    db3 = dQ.sum(axis=0)
    dh2 = dQ @ online.W3
    dz2 = dh2 * (z2 > 0)
    dW2 = dz2.T @ h1
    db2 = dz2.sum(axis=0)
    dh1 = dz2 @ online.W2
    dz1 = dh1 * (z1 > 0)
    dW1 = dz1.T @ x
    db1 = dz1.sum(axis=0)

    grads = [dW1, db1, dW2, db2, dW3, db3]
    norm = float(np.sqrt(sum(np.square(g).sum() for g in grads)))
    if norm > 5.0:
        grads = [g * (5.0 / norm) for g in grads]
    opt.step(online.params(), grads)
    return loss


def export_weights(net: MLP, path: Path) -> None:
    payload = {
        "architecture": list(LAYERS),
        "gamma": GAMMA,
        "reward_scale": REWARD_SCALE,
        "layers": [
            {"w": net.W1.tolist(), "b": net.b1.tolist()},
            {"w": net.W2.tolist(), "b": net.b2.tolist()},
            {"w": net.W3.tolist(), "b": net.b3.tolist()},
        ],
    }
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def verify_export(net: MLP, path: Path) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    x = np.linspace(-0.2, 1.0, STATE_DIM, dtype=np.float64)[None, :]
    q = net.q(x)[0]
    h = x
    for layer in raw["layers"]:
        w = np.array(layer["w"], dtype=np.float64)
        b = np.array(layer["b"], dtype=np.float64)
        h = h @ w.T + b
        if layer is not raw["layers"][-1]:
            h = np.maximum(h, 0.0)
    if np.max(np.abs(h[0] - q)) > 1e-6:
        raise AssertionError("exported weights do not match the network")


def find_demo(net: MLP, seeds: list[int]) -> dict | None:
    """First early/peak decision where DQN turns away a small party at a loose table."""
    env = RestaurantEnv()
    choose = dqn_policy(net)
    for seed in seeds:
        _, mask = env.reset(seed)
        done = False
        step_i = 0
        while not done:
            party = env.queue[0]
            tables = list(env.table_rem)
            phase = env.phase_name()
            t = env.t
            action = choose(env)
            loose = party.size <= 2 and mask[2:].any() and env._min_slack(party) >= 2
            if action == 1 and loose and phase != "Late":
                return {
                    "seed": seed,
                    "decision_index": step_i,
                    "minute": t,
                    "phase": phase,
                    "party_size": party.size,
                    "reservation": party.reservation,
                    "table_remaining": tables,
                }
            _, _, done, _, mask = env.step(action)
            step_i += 1
    return None


def train(episodes: int, seed: int) -> None:
    self_check()
    rng = np.random.default_rng(seed)
    online = MLP(rng)
    target = MLP(rng)
    target.copy_from(online)
    opt = Adam(online.params(), lr=7e-4)
    replay = Replay(50_000)
    env = RestaurantEnv()

    val_seeds = list(range(100, 130))
    fcfs_val = evaluate(fcfs, val_seeds)
    greedy_val = evaluate(greedy, val_seeds)
    print(
        "val baselines",
        round(fcfs_val["gross_revenue"]["mean"], 1),
        round(greedy_val["gross_revenue"]["mean"], 1),
    )

    best_value = -1e18
    best_params = [p.copy() for p in online.params()]
    best_ep = 0
    stale = 0
    history = []
    updates = 0
    recent: list[float] = []

    for ep in range(1, episodes + 1):
        frac = min(1.0, ep / 1500.0)
        eps = 1.0 + frac * (0.05 - 1.0)
        obs, mask = env.reset(int(rng.integers(0, 1_000_000_000)))
        done = False
        ep_shaped = 0.0
        while not done:
            if rng.random() < eps:
                action = int(rng.choice(np.flatnonzero(mask)))
            else:
                q = online.q(obs[None, :])[0]
                action = masked_argmax(q, mask)
            nxt, reward, done, _, nxt_mask = env.step(action)
            replay.add(obs, action, reward / REWARD_SCALE, nxt, float(done), nxt_mask)
            ep_shaped += reward
            obs, mask = nxt, nxt_mask
            if replay.n >= 2_000:
                learn(online, target, opt, replay, rng, 128)
                updates += 1
                if updates % 250 == 0:
                    target.copy_from(online)
        recent.append(ep_shaped)
        if len(recent) > 50:
            recent.pop(0)

        if ep % 100 == 0 or ep == episodes:
            metrics = evaluate(dqn_policy(online), val_seeds)
            value = metrics["gross_revenue"]["mean"]
            row = {
                "episode": ep,
                "epsilon": eps,
                "train_shaped": float(np.mean(recent)),
                "val_revenue": value,
                "val_util": metrics["utilization"]["mean"],
                "val_wait": metrics["avg_wait"]["mean"],
            }
            history.append(row)
            print(
                f"ep {ep:4d}  eps {eps:.2f}  train {row['train_shaped']:.0f}  "
                f"val ${value:.0f}  util {100*row['val_util']:.1f}%  wait {row['val_wait']:.1f}"
            )
            if value > best_value + 5:
                best_value = value
                best_params = [p.copy() for p in online.params()]
                best_ep = ep
                stale = 0
            else:
                stale += 1
            if ep >= 800 and stale >= 5:
                print(f"early stop at {ep}, best ep {best_ep}")
                break

    for p, src in zip(online.params(), best_params):
        p[...] = src
    target.copy_from(online)

    test_seeds = list(range(2000, 2050))
    results = {
        "fcfs": evaluate(fcfs, test_seeds),
        "greedy": evaluate(greedy, test_seeds),
        "dqn": evaluate(dqn_policy(online), test_seeds),
    }
    base = results["fcfs"]["gross_revenue"]["mean"]
    for name in results:
        mean = results[name]["gross_revenue"]["mean"]
        results[name]["uplift_vs_fcfs"] = 0.0 if name == "fcfs" else (mean - base) / base

    demo = find_demo(online, list(range(1, 250)))
    probe_env = RestaurantEnv(7)
    probe_obs, probe_mask = probe_env.reset(7)
    probe_q = online.q(probe_obs[None, :])[0]

    DOCS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    (DOCS / "config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
    export_weights(online, DOCS / "weights.json")
    export_weights(online, MODELS / "dqn_weights.json")
    verify_export(online, DOCS / "weights.json")

    study = {
        "episodes_requested": episodes,
        "best_episode": best_ep,
        "gamma": GAMMA,
        "architecture": list(LAYERS),
        "reward_scale": REWARD_SCALE,
        "val_seeds": val_seeds,
        "test_seeds": test_seeds,
        "history": history,
        "baselines_val": {"fcfs": fcfs_val, "greedy": greedy_val},
        "test": results,
        "demo": demo,
        "probe": {
            "seed": 7,
            "state": probe_obs.tolist(),
            "mask": probe_mask.astype(int).tolist(),
            "q": probe_q.tolist(),
        },
        "report_table": {
            "note": "Figures printed in the project report. They are not this reconstruction's measurements.",
            "fcfs": {"gross_revenue": 2497.5, "gross_sd": 85, "utilization": 0.684, "avg_wait": 12.4, "uplift": 0.0},
            "greedy": {"gross_revenue": 2664.4, "gross_sd": 72, "utilization": 0.741, "avg_wait": 14.8, "uplift": 0.067},
            "dqn": {"gross_revenue": 2933.2, "gross_sd": 45, "utilization": 0.823, "avg_wait": 9.2, "uplift": 0.174},
        },
    }
    (DOCS / "study.json").write_text(json.dumps(study, indent=2), encoding="utf-8")
    np.savez(MODELS / "dqn_weights.npz", W1=online.W1, b1=online.b1, W2=online.W2, b2=online.b2, W3=online.W3, b3=online.b3)

    print("TEST")
    for name, row in results.items():
        print(
            f"{name:7}  ${row['gross_revenue']['mean']:.1f} ± {row['gross_revenue']['std']:.1f}  "
            f"util {100*row['utilization']['mean']:.1f}%  wait {row['avg_wait']['mean']:.1f}  "
            f"uplift {100*row['uplift_vs_fcfs']:.1f}%"
        )
    print("demo", demo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.episodes, args.seed)


if __name__ == "__main__":
    main()
