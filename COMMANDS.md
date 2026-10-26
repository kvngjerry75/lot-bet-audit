# astro-latest: commands

Everything runs from inside the `astro-latest` folder. The data folders hold the latest scrape (NLA to 24 Oct 2026,
Afriluck to 26 Oct 2026 Anopa), so every tool runs straight away. The audit of October 2026 is in
`reports/AUDIT_2026-10.md` and the re-audit in `reports/REAUDIT_2026-10.md`: read their verdicts first.

## What is in the folder

| Item | What it is |
|---|---|
| `main_beast.py` | **The one runner.** Every model on one game (Afriluck or DayWA), the backtest table (with Perm-2 GH₵ back per GH₵1 on Afriluck), the next-draw core and the money verdict. Default combiner since the audit: chance + sharp + colour (`--combiner all` = the old 15-model average). `--fast` skips the original stage 4. `--save f.npz` keeps every forecast of every component |
| `bet_lab.py` | NEW. Every Afriluck bet type (Perm-2 on 3-10 numbers, Direct-1, Banker, Direct-3, Direct-4, machine-number Perm-2) replayed walk-forward in GH₵: back per GH₵1 with 95% interval, net, deepest fall, longest losing run, random-pick replays, fair-machine controls, multiple-testing correction, and the ceiling (best possible return if the pink bias were known exactly) |
| `dependence.py` | NEW. Looks for an edge beyond colour: pairs that come out together, ball-set changes over time, the machine numbers. Each test calibrated by simulation and proven on a planted effect first |
| `forward_test.py` | NEW. Log the pick BEFORE each draw, score it after, with a fixed stop rule (`plan` prints how many draws a test needs) |
| `test_audit.py` | NEW. The audit's checks: no leakage, loader, payout maths, positive and negative controls |
| `tuning_lab.py` | NEW (re-audit). Can more tuning raise the hit rate? 124 variants chosen on older draws, scored on the newest 750. Answer: no |
| `audit_extras.py` | NEW. The audit's remaining experiments: combiners (nested), the machine-number "early ball" model, old vs new in GH₵ |
| `profit_history.py` | Draw-by-draw Perm-2 profit and loss (`--core 3` for the top 3 only, `--component sharp` for one model) |
| `sharp.py` | The best single predictor (self-tuning colour / low-high model) |
| `forecaster.py` | Stage 5 forecaster (laws, precedents, partners, analogues, groups; `--brain` adds the ML brain). Any game |
| `afriluck_scraper.py`, `nla_scraper.py`, `clean_data.py` | Data: scrape Afriluck, scrape NLA, clean NLA into `nla_audited/` |
| `afriluck_lab.py`, `when_to_stake.py`, `exit_order.py`, `monitor.py`, `randomness_battery.py` | Earlier labs (unchanged) |
| `eval_astro2_v3/` | The full v3 system (stage 4 ensemble, rule engine, tracker, tests) |
| `reports/` | `AUDIT_2026-10.md` (this audit) and the outputs it quotes, plus the earlier reports |

## A. Google Colab (how you work)

### Once per session: set up
Upload `astro-latest.zip` to `MyDrive/`, open a new notebook and run this cell:
```
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/astro-latest.zip /content/
!unzip -o -q /content/astro-latest.zip -d /content/
%cd /content/astro-latest
!pip -q install -r requirements.txt
!pip -q install lightgbm torch
!mkdir -p /content/drive/MyDrive/astro-latest
!cp /content/drive/MyDrive/astro-latest/forward_log.csv . 2>/dev/null || echo "no forward log yet"
```
(If you saved newer data to Drive before, also restore it:
`!cp -r /content/drive/MyDrive/astro-latest/afriluck_output /content/drive/MyDrive/astro-latest/nla_audited . 2>/dev/null`)

### 1. Check everything works (2 minutes)
```
!python test_audit.py
```
Expect `ALL AUDIT TESTS PASSED`. The original tests: `%cd eval_astro2_v3`, `!python test_v3.py`, `!python test_pipeline.py`,
`%cd ..` (in Colab the two TFT tests run too, because torch is installed).

### 2. Update the data
```
!python afriluck_scraper.py
!python nla_scraper.py --start 2017-12
!python clean_data.py --from-scraper nla_output --out-dir nla_audited
```
The Afriluck update takes seconds; the NLA scrape about 10 minutes (only needed for DayWA).

### 3. The forecast and the money verdict
```
!python main_beast.py --game Afriluck
!python main_beast.py --game DayWA
```
Read the end: `FORECASTING` (which draw), `Forecast` (top 6 and 10-ball core) and `MONEY` (what any bet returns per
GH₵1 at best). With the TFT army on a Colab CPU this takes several minutes; Runtime > Change runtime type > T4 GPU is
faster. `--fast` (about 30 seconds) gives the SAME core, because the default BEAST uses only chance + sharp + colour.

### 4. Every bet in GH₵ (the main audit table)
```
!python main_beast.py --game Afriluck --save afl.npz
!python bet_lab.py --forecasts afl.npz --controls 200
!python profit_history.py afl.npz
!python profit_history.py afl.npz --core 3
```
`bet_lab.py` takes about 10 minutes with 200 controls (`--controls 0` for 1 minute).

### 5. Look for an edge beyond colour
```
!python dependence.py --sims 300
!python when_to_stake.py --afriluck-dir afriluck_output
!python sharp.py --game Afriluck --controls 200
!python sharp.py --game DayWA --nla-dir nla_audited --controls 200
```

### 5b. Reproduce the audit's old-vs-new and combiner tables
```
!python main_beast.py --game Afriluck --save afl.npz
!python main_beast.py --game Afriluck --combiner all --no-long-run --save afl_old.npz
!python audit_extras.py --new afl.npz --old afl_old.npz
```

### 5c. Check whether more tuning helps (re-audit)
```
!python tuning_lab.py --game Afriluck
!python tuning_lab.py --game DayWA
```

### 6. Test a hidden draw
```
!python main_beast.py --game Afriluck --holdout 1
!python main_beast.py --game DayWA --holdout 1
```

### 7. The forward test (paper only: no stake)
Before each draw you want to test (Anopa 10:00, Midday 13:30, evening 19:45 Ghana time):
```
!python afriluck_scraper.py
!python forward_test.py log
!cp forward_log.csv /content/drive/MyDrive/astro-latest/
```
After the draw (any time later, e.g. once a day):
```
!python afriluck_scraper.py
!python forward_test.py score
```
`!python forward_test.py plan` prints how many draws each bet needs. The default bet is Perm-2 on the top 10
(`--bet perm2_3` for the top 3; log each bet you want to test separately). Picks logged after the draw started are
marked LATE and never counted. Copy `forward_log.csv` to Drive after every `log`, or Colab will delete it.

### 8. Old behaviour, for comparison
```
!python main_beast.py --game Afriluck --combiner all --no-long-run
```

### Keep your data
Colab deletes its files when the session ends:
```
!cp -r nla_output nla_audited afriluck_output reports forward_log.csv /content/drive/MyDrive/astro-latest/
```

## B. On a PC (Windows, Mac or Linux)

Python 3.10+, then in a terminal in the `astro-latest` folder: `pip install -r requirements.txt` (and
`pip install lightgbm torch` for the full stage 4). The commands are the same as above without the `!`, e.g.
`python main_beast.py --game Afriluck`.

## What the numbers mean
- **10-ball core**: the 10 numbers the system rates highest. Perm-2 on it = 45 pairs = GH₵45 a draw.
- **2+ in core**: how often 2 or more winning numbers fell inside it. Chance: Afriluck 28.1%, DayWA 38.1%.
- **Back per GH₵1**: money returned for every GH₵1 staked over many draws. Below 1.00 = a loss over time.
  Break-even for every Afriluck bet is 1.00; a fair machine gives 0.56 (Perm-2); the pink bias lifts that to about
  0.62-0.64. Nothing tested gets near 1.00.
- **Latest backtests (draws 561-2060)**: MAIN BEAST 31.5% 2+ in core, 0.644 back on Perm-2 top 10 (-GH₵24,060 over
  1,500 draws at GH₵45 a draw). DayWA 40.3% vs 38.1% by chance (not significant). Main, Noonrush, VAG at chance.
