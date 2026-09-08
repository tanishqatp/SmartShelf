#!/usr/bin/env python3
"""
seed_full_week_history.py — seeds THREE WEEKS of history across all 7 days
for three flavors (Cheetos, Lays, Popcorners), so the demo has a real
pattern to show regardless of which day someone actually tests it on.
Run this once before any demo session.

Design: each flavor has a different "problem day" so the demo can show
distinct patterns depending on what day it's run:
  - Lays: reliably runs low Wednesday evenings (~17:30-18:00)
  - Cheetos: reliably runs low Friday evenings (~18:00-18:30)
  - Popcorners: stays healthy all week (proves the system doesn't cry wolf
    on every flavor — only flags real recurring patterns)
"""
import subprocess
import time

SHELF_ID = "A3"

EVENTS = [
    # --- Lays: Wednesday evening pattern, 3 weeks ---
    {"flavor": "Lays", "remaining": 2, "day_time": "Wednesday 16:15"},
    {"flavor": "Lays", "remaining": 1, "day_time": "Wednesday 17:55"},
    {"flavor": "Lays", "remaining": 2, "day_time": "Wednesday 15:40"},
    {"flavor": "Lays", "remaining": 1, "day_time": "Wednesday 17:48"},
    {"flavor": "Lays", "remaining": 2, "day_time": "Wednesday 16:30"},
    {"flavor": "Lays", "remaining": 0, "day_time": "Wednesday 18:05"},

    # --- Cheetos: Friday evening pattern, 3 weeks ---
    {"flavor": "Cheetos", "remaining": 2, "day_time": "Friday 17:00"},
    {"flavor": "Cheetos", "remaining": 1, "day_time": "Friday 18:10"},
    {"flavor": "Cheetos", "remaining": 2, "day_time": "Friday 16:45"},
    {"flavor": "Cheetos", "remaining": 0, "day_time": "Friday 18:25"},
    {"flavor": "Cheetos", "remaining": 2, "day_time": "Friday 17:15"},
    {"flavor": "Cheetos", "remaining": 1, "day_time": "Friday 18:00"},

    # --- Popcorners: healthy/routine readings all week, no real pattern ---
    {"flavor": "Popcorners", "remaining": 3, "day_time": "Monday 10:00"},
    {"flavor": "Popcorners", "remaining": 2, "day_time": "Tuesday 14:00"},
    {"flavor": "Popcorners", "remaining": 3, "day_time": "Wednesday 09:30"},
    {"flavor": "Popcorners", "remaining": 2, "day_time": "Thursday 12:00"},
    {"flavor": "Popcorners", "remaining": 3, "day_time": "Friday 10:00"},
    {"flavor": "Popcorners", "remaining": 2, "day_time": "Saturday 11:00"},

    # --- A few extra healthy readings for Lays/Cheetos on off-days, for contrast ---
    {"flavor": "Lays", "remaining": 3, "day_time": "Monday 10:05"},
    {"flavor": "Cheetos", "remaining": 3, "day_time": "Tuesday 14:05"},
    {"flavor": "Lays", "remaining": 3, "day_time": "Thursday 12:05"},
    {"flavor": "Cheetos", "remaining": 2, "day_time": "Sunday 09:00"},
]


def ask_hermes(flavor, remaining, day_time):
    prompt = (
        f"Shelf inventory event. Shelf: {SHELF_ID}. Flavor removed: {flavor}. "
        f"Remaining count: {remaining}. Day/time: {day_time}. "
        f"Log this to memory with the day and time. Check memory for whether this "
        f"flavor has hit low stock (1 or fewer remaining) at a similar day/time "
        f"before. If a repeating pattern is emerging, note it explicitly in your reply."
    )
    result = subprocess.run(
        ["hermes", "chat", "--quiet", "-q", prompt],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"  error: {result.stderr[:300]}")
        return ""
    return result.stdout.strip()


def main():
    for i, event in enumerate(EVENTS, 1):
        print(f"[{i}/{len(EVENTS)}] Sending: {event['flavor']} -> {event['remaining']} @ {event['day_time']}")
        response = ask_hermes(event["flavor"], event["remaining"], event["day_time"])
        print(f"  Hermes: {response[:250]}\n")
        time.sleep(2)
    print("Full week seeded (Cheetos, Lays, Popcorners). Reset inventory_state.json before live camera demos.")


if __name__ == "__main__":
    main()