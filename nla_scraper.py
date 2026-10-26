#!/usr/bin/env python3
"""
Spool NLA past results from ghanayello.com into separate CSV files (v2).

New in v2
  * MACHINE NUMBERS are kept ("5 Last Numbers" / "Machine Numbers" column) as Machine1..Machine5.
  * The EVENT number is kept when the site shows one. It orders same-day draws of the main games only:
    on the site, VAG and NOONRUSH event numbers repeat across different days, so they are not trusted.
  * DELAYED DRAWS are put back where they belong. A Monday Special drawn on Tuesday stays a Monday Special:
    its row keeps the real draw date (Date = the day the balls were drawn) and also records
    ScheduledDate (the Monday), WasShifted = 1 and ShiftDays = 1. The Monday then gets NO holiday row,
    because its draw was not cancelled, only moved.
    For the daily games (DayWA, VAG, Noonrush) a postponed draw shows up as two draws on one date with
    the previous draw day empty. The earlier of the two (lower event number, or the one listed lower on the
    site) is marked as the make-up draw for that empty day.
  * Rows are written in true drawing order: by real draw date, then by event number (or site order) within
    the day, newest first like before.

Setup:
    pip install requests beautifulsoup4

Usage:
    python nla_scraper.py                                  # current month back 5 years
    python nla_scraper.py --start 2017-12                  # everything the site has (back to Dec 2017)
    python nla_scraper.py --start 2026-04 --end 2026-10    # explicit range
    python nla_scraper.py --inspect 2026-10                # save one month's page and show its columns (debugging)
    python nla_scraper.py --url-template "<url with {month}>"   # fallback if the form is not found

Output (folder nla_output, change with --outdir), newest first:
    main_draw_results.csv  vag_results.csv  daywa_results.csv  noonrush_results.csv
    Date,Promotionday,Number1..Number5,RealDay,WasShifted,ScheduledDate,ShiftDays,Event,Machine1..Machine5

    Date           the day the draw actually took place
    Promotionday   the day the draw belongs to (Monday for Monday Special), or Holiday
    RealDay        weekday of Date
    WasShifted     1 if the draw took place on a later (or earlier) day than scheduled
    ScheduledDate  the day it was scheduled for
    ShiftDays      Date minus ScheduledDate in days (0 for a normal draw)
    Event          the site's event number, if shown
    Machine1..5    the machine numbers, if shown

Days with no draw at all are written as Holiday rows (between a file's first and last result only, never inside
months that failed to download). Days a game never runs (Noonrush on Sundays, any weekday a single game never
uses) are skipped. Nothing is dropped silently: everything suspicious is listed at the end.
"""

import argparse
import csv
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

URL = "https://www.ghanayello.com/lottery/results/history"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
DATE_RE = re.compile(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})")
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

MAIN_GAMES = {  # normalised game name -> scheduled weekday (Monday = 0)
    "mondayspecial": 0,
    "luckytuesday": 1,
    "midweek": 2,
    "fortunethursday": 3,
    "fridaybonanza": 4,
    "nationalweekly": 5,
    "sundayaseda": 6,
}
SINGLE_GAMES = {"vag": 90, "daywa": 39, "noonrush": 90}   # name prefix -> pool size
FILES = {
    "main": "main_draw_results.csv",
    "vag": "vag_results.csv",
    "daywa": "daywa_results.csv",
    "noonrush": "noonrush_results.csv",
}
NO_DRAW_WEEKDAYS = {"noonrush": {6}}  # NOONRUSH: no Sunday draw
HEADER = (["Date", "Promotionday"] + [f"Number{i}" for i in range(1, 6)]
          + ["RealDay", "WasShifted", "ScheduledDate", "ShiftDays", "Event"]
          + [f"Machine{i}" for i in range(1, 6)])
MAX_SHIFT = 3   # a main draw found up to 3 days after its weekday is a delayed draw


# ------------------------------------------------------------------ small helpers
def month_range(start, end):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    months = []
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return months


def month_key(text):
    text = (text or "").strip()
    for fmt in ("%Y-%m", "%B %Y", "%b %Y", "%m/%Y", "%Y/%m", "%Y%m"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m")
        except ValueError:
            continue
    return None


def parse_date(day, month_name, year):
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(f"{day} {month_name} {year}", fmt).date()
        except ValueError:
            continue
    return None


def norm(game):
    return re.sub(r"[^a-z0-9]", "", game.lower())


def classify(game):
    """('main', weekday) | (single key, None) | (None, None)"""
    key = norm(game)
    for name, wd in MAIN_GAMES.items():
        if key.startswith(name):
            return "main", wd
    for name in SINGLE_GAMES:
        if key.startswith(name):
            return name, None
    return None, None


# ---------------------------------------------------------------- form handling
def find_form(soup):
    for sel in soup.find_all("select"):
        texts = [o.get_text(strip=True) for o in sel.find_all("option")]
        if "Last 30 draws" in texts:
            return sel.find_parent("form"), sel
    return None, None


def option_value(option):
    return option.get("value", option.get_text(strip=True))


def build_month_values(month_select):
    out = {}
    for o in month_select.find_all("option"):
        key = month_key(o.get_text(strip=True)) or month_key(option_value(o))
        if key:
            out[key] = option_value(o)
    return out


def build_payload(form, month_select, month_value):
    payload = {}
    for el in form.find_all(["input", "select", "textarea"]):
        name = el.get("name")
        if not name:
            continue
        if el.name == "select":
            opts = el.find_all("option")
            if not opts:
                continue
            if el is not month_select:
                chosen = next(
                    (o for o in opts if o.get_text(strip=True).lower() == "all games"),
                    next((o for o in opts if o.has_attr("selected")), opts[0]),
                )
                payload[name] = option_value(chosen)
        elif el.name == "input":
            t = (el.get("type") or "text").lower()
            if t in ("checkbox", "radio") and not el.has_attr("checked"):
                continue
            if t in ("button", "image", "reset"):
                continue
            payload[name] = el.get("value", "")
        else:
            payload[name] = el.get_text()
    payload[month_select["name"]] = month_value
    return payload


def get_html(session, method, url, payload=None, retries=3):
    for attempt in range(1, retries + 1):
        try:
            if method == "post":
                r = session.post(url, data=payload, timeout=30)
            else:
                r = session.get(url, params=payload, timeout=30)
            r.raise_for_status()
            return r.text
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            client_error = status is not None and 400 <= status < 500 and status != 429
            if attempt == retries or client_error:
                raise
            print(f"  retry {attempt}/{retries - 1} after error: {exc}")
            time.sleep(3 * attempt)


# --------------------------------------------------------------------- parsing
def find_table(soup):
    for t in soup.find_all("table"):
        first = t.find("tr")
        if first and "draw date" in first.get_text(" ", strip=True).lower():
            return t
    return None


def header_index(headers):
    """Locate the columns by header text. Machine numbers and event are optional."""
    def find(*needles, exclude=()):
        for i, h in enumerate(headers):
            if any(n in h for n in needles) and not any(x in h for x in exclude):
                return i
        return None
    idx = dict(date=find("draw date"), game=find("lotto game", "game"),
               win=find("winning"),
               machine=find("machine", "last number", "last 5", "5 last"),
               event=find("event", "draw no", "draw #", "draw number", exclude=("date",)))
    if idx["machine"] == idx["win"]:
        idx["machine"] = None
    missing = [k for k in ("date", "game", "win") if idx[k] is None]
    if missing:
        raise ValueError(f"column(s) {missing} not found in headers {headers}")
    return idx


def parse_page(html, month):
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text(" ", strip=True)
    m = re.search(r"List of ([\d,]+) Ghana Lotto Draws", page_text)
    declared = int(m.group(1).replace(",", "")) if m else None
    table = find_table(soup)
    if table is None:
        return [], declared, [f"{month}: results table not found"], {}
    headers = [c.get_text(" ", strip=True).lower() for c in table.find("tr").find_all(["th", "td"])]
    idx = header_index(headers)

    rows, warnings = [], []
    for pos, tr in enumerate(table.find_all("tr")[1:]):
        cells = tr.find_all("td")
        need = max(v for v in idx.values() if v is not None)
        if len(cells) <= max(idx["date"], idx["game"], idx["win"]):
            continue
        raw_date = cells[idx["date"]].get_text(" ", strip=True)
        dm = DATE_RE.search(raw_date)
        draw_date = parse_date(*dm.groups()) if dm else None
        if draw_date is None:
            warnings.append(f"{month}: unreadable date {raw_date!r} (row skipped)")
            continue
        game = re.sub(r"\s+Results?$", "", cells[idx["game"]].get_text(" ", strip=True), flags=re.I)
        nums = [int(n) for n in re.findall(r"\d+", cells[idx["win"]].get_text(" ", strip=True))]
        machine = []
        if idx["machine"] is not None and len(cells) > idx["machine"]:
            machine = [int(n) for n in re.findall(r"\d+", cells[idx["machine"]].get_text(" ", strip=True))]
        event = None
        if idx["event"] is not None and len(cells) > idx["event"]:
            em = re.search(r"\d+", cells[idx["event"]].get_text(" ", strip=True))
            event = int(em.group()) if em else None
        if draw_date.strftime("%Y-%m") != month:
            warnings.append(f"{month}: {draw_date} {game} is outside the requested month (filter may not have applied)")
        rows.append(dict(date=draw_date, game=game, nums=nums, machine=machine, event=event, site_pos=pos,
                         month=month))
    if declared is not None and declared != len(rows):
        warnings.append(f"{month}: site says {declared} draws but {len(rows)} rows were parsed")
    info = dict(headers=headers, machine_col=idx["machine"] is not None, event_col=idx["event"] is not None)
    _ = need
    return rows, declared, warnings, info


def validate(r, pool, warnings):
    nums, mach = r["nums"], r["machine"]
    ok = len(nums) == 5 and len(set(nums)) == 5 and all(1 <= n <= pool for n in nums)
    if not ok:
        warnings.append(f"{r['date']} {r['game']}: odd winning numbers {nums} (pool 1-{pool})")
    if mach:
        if len(mach) != 5 or len(set(mach)) != 5 or not all(1 <= n <= pool for n in mach):
            warnings.append(f"{r['date']} {r['game']}: odd machine numbers {mach} (kept as given)")
    return ok


# ----------------------------------------------------------- schedule / delays
def within_day_key(r):
    """Order of draws inside one date: event number for the main games (their event numbers run in sequence),
    else the site order (the site lists newest first, so a LOWER position on the page is an EARLIER draw).
    VAG and NOONRUSH event numbers repeat across different days on the site, so they are never used for ordering."""
    if r.get("use_event") and r["event"] is not None:
        return (0, r["event"])
    return (1, -r["site_pos"])


def schedule_main(rows, warnings):
    for r in rows:
        actual = r["date"].weekday()
        shift = (actual - r["wd"]) % 7
        if shift == 0:
            r["sched"] = r["date"]
        elif shift <= MAX_SHIFT:
            r["sched"] = r["date"] - timedelta(days=shift)
            warnings.append(f"{r['date']} {r['game']} drawn {shift} day(s) late (scheduled {r['sched']}): "
                            f"kept as {DAYS[r['wd']]}, WasShifted=1")
        else:
            back = 7 - shift
            r["sched"] = r["date"] + timedelta(days=back)
            warnings.append(f"{r['date']} {r['game']} drawn {back} day(s) EARLY (scheduled {r['sched']}): "
                            f"kept as {DAYS[r['wd']]}, WasShifted=1, check this one")
        r["label"] = DAYS[r["wd"]]


def schedule_single(key, rows, notes, warnings):
    """Two draws on one date + an empty draw day before it -> the earlier draw is the make-up draw."""
    by_date = defaultdict(list)
    for r in rows:
        by_date[r["date"]].append(r)
    wd_count = Counter(d.weekday() for d in by_date)
    typical = sorted(wd_count.values())[len(wd_count) // 2] if wd_count else 0
    # a weekday the game (almost) never uses is not a draw day: no holidays there, no make-ups assigned to it
    rare = {w for w in range(7) if wd_count.get(w, 0) < 0.2 * typical} if len(by_date) > 56 else set()
    skip = set(NO_DRAW_WEEKDAYS.get(key, set())) | rare
    taken = set(by_date)
    for d in sorted(by_date):
        day = sorted(by_date[d], key=within_day_key)
        for r in day:
            r["sched"], r["label"] = d, DAYS[d.weekday()]
        extra = day[:-1]                       # all but the last draw of the day are candidates for make-ups
        if not extra:
            continue
        free, back = [], d - timedelta(days=1)
        while len(free) < len(extra) and (d - back).days <= MAX_SHIFT:
            if back.weekday() not in skip and back not in taken:
                free.append(back)
            back -= timedelta(days=1)
        free.sort()
        for r, s in zip(extra, free):
            r["sched"], r["label"] = s, DAYS[s.weekday()]
            taken.add(s)
            warnings.append(f"{FILES[key]}: {d} has {len(day)} draws; the earlier one is the make-up draw for "
                            f"{s} ({DAYS[s.weekday()]}), WasShifted=1")
        if len(free) < len(extra):
            warnings.append(f"{FILES[key]}: {d} has {len(day)} draws and no empty draw day in the {MAX_SHIFT} days "
                            f"before it: "
                            f"left as two draws on {d}")
    if skip - set(NO_DRAW_WEEKDAYS.get(key, set())):
        notes.append(f"{FILES[key]}: (almost) never draws on " + ", ".join(DAYS[i] + "s" for i in sorted(skip))
                     + " (no holiday rows or make-up draws there)")
    return skip


# ---------------------------------------------------------------------- output
def build_file(key, rows, notes, warnings, missing_months):
    if key == "main":
        schedule_main(rows, warnings)
        skip = set()
    else:
        skip = schedule_single(key, rows, notes, warnings)
    seen = Counter((r["sched"], r["game"]) for r in rows)
    for (s, g), c in seen.items():
        if c > 1:
            warnings.append(f"{FILES[key]}: {g} has {c} draws for {s}")
    rows.sort(key=lambda r: (r["date"], r["sched"], within_day_key(r)))
    first = min(r["sched"] for r in rows)
    last = max(r["date"] for r in rows)
    covered = {r["sched"] for r in rows}
    holidays = []
    d = first
    while d <= last:
        if d not in covered and d.weekday() not in skip and d.strftime("%Y-%m") not in missing_months:
            holidays.append(d)
        d += timedelta(days=1)
    out = [("draw", r["date"], r) for r in rows] + [("holiday", h, None) for h in holidays]
    order = {id(r): i for i, r in enumerate(rows)}
    out.sort(key=lambda x: (x[1], 0 if x[0] == "holiday" else 1, order.get(id(x[2]), -1)))
    return out[::-1], holidays, first, last


def write_csv(path, items):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for kind, d, r in items:
            if kind == "holiday":
                w.writerow([d.isoformat(), "Holiday"] + [""] * 5 + [DAYS[d.weekday()], 0, d.isoformat(), 0, ""]
                           + [""] * 5)
                continue
            shift = (r["date"] - r["sched"]).days
            w.writerow([r["date"].isoformat(), r["label"]] + (r["nums"] + [""] * 5)[:5]
                       + [DAYS[r["date"].weekday()], int(shift != 0), r["sched"].isoformat(), shift,
                          "" if r["event"] is None else r["event"]]
                       + (r["machine"] + [""] * 5)[:5])


def build_outputs(all_rows, outdir, warnings, missing_months=()):
    os.makedirs(outdir, exist_ok=True)
    missing_months = frozenset(missing_months)
    buckets, other = defaultdict(list), []
    for r in all_rows:
        kind, wd = classify(r["game"])
        if kind is None:
            other.append(r)
            continue
        pool = 90 if kind == "main" else SINGLE_GAMES[kind]
        if not validate(r, pool, warnings):
            continue
        r["wd"] = wd
        r["use_event"] = kind == "main"
        buckets[kind].append(r)
    for kind, rows in buckets.items():
        ev = Counter(r["event"] for r in rows if r["event"] is not None)
        rep = sum(1 for c in ev.values() if c > 1)
        if rep:
            warnings.append(f"{FILES.get(kind, kind)}: {rep} event number(s) appear on more than one draw "
                            f"(the site's own numbering); kept as published, not used for ordering"
                            + ("" if kind != "main" else " where they repeat"))

    notes, holiday_sets, result_dates = [], {}, {}
    print()
    for key, fname in FILES.items():
        rows = buckets.get(key, [])
        if not rows:
            warnings.append(f"{fname}: no results found, file not written")
            continue
        items, holidays, first, last = build_file(key, rows, notes, warnings, missing_months)
        write_csv(os.path.join(outdir, fname), items)
        holiday_sets[key] = set(holidays)
        result_dates[key] = {r["sched"] for r in rows}
        n_shift = sum(1 for r in rows if r["sched"] != r["date"])
        n_mach = sum(1 for r in rows if len(r["machine"]) == 5)
        print(f"{fname}: {len(rows)} draws ({n_shift} delayed, {n_mach} with machine numbers) + "
              f"{len(holidays)} holiday rows, {first} to {last}")

    if other:
        path = os.path.join(outdir, "other_games_results.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Date", "Game"] + [f"Number{i}" for i in range(1, 6)] + ["Event"]
                       + [f"Machine{i}" for i in range(1, 6)])
            for r in sorted(other, key=lambda r: r["date"], reverse=True):
                w.writerow([r["date"].isoformat(), r["game"]] + (r["nums"] + [""] * 5)[:5]
                           + ["" if r["event"] is None else r["event"]] + (r["machine"] + [""] * 5)[:5])
        games = sorted({r["game"] for r in other})
        warnings.append(f"{len(other)} rows from games not in your four files ({', '.join(games)}) "
                        "were written to other_games_results.csv")

    for n in notes:
        print("NOTE:", n)
    if missing_months:
        print("NOTE: no holiday rows were created inside missing months: " + ", ".join(sorted(missing_months)))
    partial = []
    all_hol = set().union(*holiday_sets.values()) if holiday_sets else set()
    for d in sorted(all_hol, reverse=True):
        missing = [k for k, s in holiday_sets.items() if d in s]
        present = [k for k, s in result_dates.items() if d in s]
        if present:
            partial.append(f"{d}: holiday in {', '.join(missing)} but results exist in {', '.join(present)}")
    if partial:
        print(f"\n{len(partial)} holiday row(s) are NOT shared by every file. A real public holiday usually hits "
              "all games, so these may be gaps on the site:")
        for line in partial[:25]:
            print("  -", line)
        if len(partial) > 25:
            print(f"  ... and {len(partial) - 25} more")


# ------------------------------------------------------------------------ main
def inspect(session, args, form, month_select, method, action, month_values):
    month = args.inspect
    if args.url_template:
        html = get_html(session, "get", args.url_template.format(month=month))
    else:
        html = get_html(session, method, action, build_payload(form, month_select, month_values[month]))
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"inspect_{month}.html"
    p.write_text(html, encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    t = find_table(soup)
    print(f"Saved {p}")
    if t is None:
        print("No results table found on this page.")
        return
    trs = t.find_all("tr")
    print("HEADERS:", [c.get_text(" ", strip=True) for c in trs[0].find_all(["th", "td"])])
    for tr in trs[1:6]:
        print("ROW:", [c.get_text(" ", strip=True) for c in tr.find_all("td")])
    rows, declared, warnings, info = parse_page(html, month)
    print(f"Parsed {len(rows)} rows | machine-number column found: {info.get('machine_col')} | "
          f"event column found: {info.get('event_col')}")


def main():
    ap = argparse.ArgumentParser(description="Spool NLA past results to CSV files")
    ap.add_argument("--years", type=int, default=5, help="years back (default 5)")
    ap.add_argument("--start", help="first month YYYY-MM (overrides --years)")
    ap.add_argument("--end", help="last month YYYY-MM (default: current month)")
    ap.add_argument("--outdir", default="nla_output", help="folder for the CSV files")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    ap.add_argument("--url-template", help="URL containing {month}, bypasses form discovery")
    ap.add_argument("--inspect", metavar="YYYY-MM", help="save one month's page, print its columns, and stop")
    ap.add_argument("--from-html", nargs="+", metavar="FILE", help="parse saved pages instead of downloading (testing)")
    args = ap.parse_args()

    if args.from_html:
        all_rows, warns = [], []
        for f in args.from_html:
            html = Path(f).read_text(encoding="utf-8")
            m = re.search(r"(\d{4}-\d{2})", Path(f).name)
            rows, _, w, info = parse_page(html, m.group(1) if m else "file")
            all_rows += rows
            warns += [x for x in w if "outside the requested month" not in x]
            print(f"{f}: {len(rows)} rows, machine column {info.get('machine_col')}, event column {info.get('event_col')}")
        build_outputs(all_rows, args.outdir, warns)
        for x in warns:
            print("  -", x)
        return

    today = date.today()
    end = args.end or today.strftime("%Y-%m")
    start = args.start or f"{today.year - args.years:04d}-{today.month:02d}"
    months = month_range(start, end)[::-1]

    session = requests.Session()
    session.headers.update(HEADERS)
    form = month_select = None
    method, action, month_values = "get", URL, {}
    if not args.url_template:
        landing = BeautifulSoup(get_html(session, "get", URL), "html.parser")
        form, month_select = find_form(landing)
        if form is None or not month_select.get("name"):
            sys.exit("Could not find the filter form. The site may load results with JavaScript. Open DevTools > "
                     "Network, click SEARCH NUMBERS, copy the request URL and re-run with --url-template "
                     "(replace the month with {month}).")
        method = (form.get("method") or "get").lower()
        action = urljoin(URL, form.get("action") or URL)
        month_values = build_month_values(month_select)
        print(f"Form found: method={method.upper()} action={action} month field='{month_select['name']}' "
              f"({len(month_values)} months offered)")
        if not month_values:
            sys.exit("The month dropdown was found but none of its options look like months.")

    if args.inspect:
        return inspect(session, args, form, month_select, method, action, month_values)

    print(f"Months to fetch: {months[-1]} to {months[0]} ({len(months)} months)")
    all_rows, all_warnings, missing_months = [], [], set()
    machine_seen = False
    for i, month in enumerate(months):
        try:
            if args.url_template:
                html = get_html(session, "get", args.url_template.format(month=month))
            else:
                if month not in month_values:
                    all_warnings.append(f"{month}: not offered in the dropdown, skipped")
                    missing_months.add(month)
                    continue
                html = get_html(session, method, action, build_payload(form, month_select, month_values[month]))
        except requests.RequestException as exc:
            all_warnings.append(f"{month}: FAILED after retries ({exc}), month is missing")
            missing_months.add(month)
            print(f"{month}: FAILED")
            continue
        try:
            rows, declared, warnings, info = parse_page(html, month)
        except ValueError as exc:
            all_warnings.append(f"{month}: {exc}")
            missing_months.add(month)
            continue
        machine_seen |= bool(info.get("machine_col"))
        if not rows:
            missing_months.add(month)
        all_rows.extend(rows)
        all_warnings.extend(warnings)
        print(f"{month}: {len(rows)} rows parsed (site says {declared})")
        if i < len(months) - 1:
            time.sleep(args.delay)

    if not machine_seen:
        all_warnings.append("No machine-number column was recognised on any page. Run with --inspect YYYY-MM and "
                            "send the HEADERS line it prints.")
    build_outputs(all_rows, args.outdir, all_warnings, missing_months)
    print(f"\nFiles written to ./{args.outdir}/")
    if all_warnings:
        print(f"\n{len(all_warnings)} WARNING(S) / NOTE(S), check these before trusting the data:")
        for wmsg in all_warnings:
            print("  -", wmsg)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
