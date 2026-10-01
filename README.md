# Five Tables

A public seating desk for the project **Deep Q-Learning for Dynamic Restaurant Revenue Management** (Javiera Bao, Shan Cao, and Tamara Dokic, ESSEC & CentraleSupélec).

The dining room has five tables, capacities 4, 2, 2, 4, and 6. Over a 240-minute service, parties arrive in a slow period, a peak, and a late lull. The host can ask the party at the door to wait, turn them away, or seat them at a specific free table. A Double DQN learns when a small party should be refused so a large table is still open for a larger check.

The site runs entirely in the browser. Nothing is uploaded.

**Live desk:** https://p2caoshan.github.io/restaurant-seating-dqn/

## Host a service

- **You host** — click a table, or press `1`–`5`. `W` asks them to wait. `X` turns them away.
- **Double DQN** — press Play. The bars are action values; masked seats are illegal.
- **Show a deferral** jumps to a night where the agent turns a small party away rather than waste a large table.
- At the end of the night the same arrivals are replayed with FCFS, the greedy host, and the DQN.

## Train

```bash
pip install -r requirements.txt
python src/train.py
```

Training writes `docs/config.json`, `docs/weights.json`, and `docs/study.json`. Serve the site locally with:

```bash
python -m http.server -d docs 8000
```

## What follows the report

- Markov decision process, 12-dimensional state, and 7 actions: wait, reject, and one seat action per table.
- Double DQN, two hidden layers of 64 units, target network, γ = 0.98, valid-action masking.
- Reward `R = Revenue + Bfit + Bres + Bpeak + Bprompt − Pwaste`.
- Baselines: first-come-first-served (first feasible table) and a greedy host that protects large tables and honors reservations.
- 2,000 training episodes with early stopping. The published comparison uses held-out nights.

## What the report left unspecified

These choices are fixed in `src/env.py` and copied to `docs/config.json`:

- The service is split into three equal phases of 80 minutes.
- The report's arrival rates 0.6, 1.8, and 0.4 are treated as expected arrivals per 10 minutes, then multiplied by 2.5 so the peak is congested. The ratio is unchanged.
- Bills are $20 per guest, so a two-top is $40 and a six-top is $120, as in the report.
- A meal lasts `20 + 5 × party size` minutes.
- Occupancy in the state is remaining service time on each table. The 12th feature is the phase of service, which the listed features did not number but the policy depends on.
- Dollar totals in the report's Table 1 are shown on the site as published figures. The reconstruction measures its own nights and does not copy those totals.

## Layout

- `src/env.py` — simulation
- `src/policies.py` — FCFS and greedy hosts
- `src/train.py` — Double DQN and export
- `docs/` — the public site
- `RL___Project_Report.pdf` — the project report
