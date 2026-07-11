"""Stateful ingestion step: snapshot -> events, via a persistent vessel cache.

Solves two problems a naive pair-diff has:

1. Static data (vessel type, name, destination) is broadcast only every
   ~6 min, so any single window misses most of it. The cache keeps the best
   known identity per MMSI across runs ("sticky" fields).
2. Coverage flicker: a vessel not heard in one window has not left the gulf.
   Each vessel is diffed against its own last observation, whenever that
   was, and the anchorage census counts recently-seen vessels rather than
   only this window's.

State and the rolling event log are plain JSON, committed as artifacts by
the Actions worker (same pattern as the indexes).
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from digest import (STATIONARY_KN, census_events, load_places,
                    location_phrase, place_at, vessel_events, vessel_label)

STATE_MAX_AGE = timedelta(hours=48)     # forget vessels not heard for this
CENSUS_FRESH = timedelta(minutes=45)    # census counts vessels heard within
EVENTS_MAX_AGE = timedelta(hours=48)    # rolling live-layer window
STICKY_FIELDS = ("type_code", "category", "destination", "name")


def parse_iso(ts):
    return datetime.fromisoformat(ts)


def empty_state():
    return {"updated_utc": None, "vessels": {}, "last_census": {}}


def load_json(path, default):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    return default


def merge_record(cached, fresh, seen_utc):
    """Fresh observation wins, but sticky identity fields survive gaps."""
    merged = dict(fresh)
    if cached:
        for field in STICKY_FIELDS:
            if not merged.get(field) or merged[field] == "unknown":
                if cached.get(field):
                    merged[field] = cached[field]
    merged["last_seen"] = seen_utc
    return merged


def state_census(state, places, now):
    counts = {a["name"]: 0 for a in places["anchorages"]}
    for v in state["vessels"].values():
        if now - parse_iso(v["last_seen"]) > CENSUS_FRESH:
            continue
        if (v.get("sog") or 0) >= STATIONARY_KN:
            continue
        spot = place_at(v["lat"], v["lon"], places["anchorages"])
        if spot:
            counts[spot["name"]] += 1
    return counts


def run_snapshot(state, snapshot, places=None):
    """Apply one snapshot to the state; return the new events."""
    places = places or load_places()
    when = snapshot["generated_utc"]
    now = parse_iso(when)
    events = []

    for fresh in snapshot["vessels"]:
        mmsi = str(fresh["mmsi"])
        cached = state["vessels"].get(mmsi)
        merged = merge_record(cached, fresh, when)
        if cached:
            events.extend(vessel_events(cached, merged, when, places))
        state["vessels"][mmsi] = merged

    state["vessels"] = {
        mmsi: v for mmsi, v in state["vessels"].items()
        if now - parse_iso(v["last_seen"]) <= STATE_MAX_AGE
    }

    counts = state_census(state, places, now)
    events.extend(census_events(counts, state.get("last_census", {}), when))
    state["last_census"] = counts
    state["updated_utc"] = when
    return events


COMPASS = ["north", "northeast", "east", "southeast",
           "south", "southwest", "west", "northwest"]


def compass(cog):
    return COMPASS[int((cog + 22.5) % 360 // 45)]


def status_chunks(state, places, now):
    """One 'right now' card per recently-heard vessel. Texts are written to
    be STABLE while the vessel's situation is unchanged (no timestamps, no
    raw coordinates for stationary vessels) so the embedding cache hits."""
    chunks = []
    for v in state["vessels"].values():
        if now - parse_iso(v["last_seen"]) > CENSUS_FRESH:
            continue
        label = vessel_label(v)
        sog = v.get("sog")
        loc = location_phrase(v["lat"], v["lon"], places)
        if sog is None:
            text = f"{label} was last heard near {loc}."
        elif sog < STATIONARY_KN:
            nav = v.get("nav_status")
            posture = ("at anchor" if nav == 1
                       else "moored" if nav == 5 else "stationary")
            text = f"{label} is {posture} at {loc}."
        else:
            text = (f"{label} is underway at {round(sog)} knots near {loc}, "
                    f"heading {compass(v['cog'])}." if v.get("cog") is not None
                    else f"{label} is underway at {round(sog)} knots "
                         f"near {loc}.")
        if v.get("destination"):
            text += f" Reported destination: {v['destination']}."
        chunks.append({
            "id": f"status-{v['mmsi']}",
            "type": "vessel_status",
            "mmsi": v["mmsi"],
            "name": v.get("name") or "",
            "lat": v["lat"],
            "lon": v["lon"],
            "time_utc": v["last_seen"],
            "text": text,
        })
    return chunks


def trim_events(events, now):
    cutoff = now - EVENTS_MAX_AGE
    return [e for e in events if parse_iso(e["time_utc"]) > cutoff]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", nargs="+", type=Path,
                        help="snapshot JSON file(s), oldest first")
    parser.add_argument("--state", type=Path, default=Path("data/state.json"))
    parser.add_argument("--events", type=Path,
                        default=Path("data/events.json"))
    parser.add_argument("--status", type=Path,
                        default=Path("data/status.json"))
    args = parser.parse_args()

    state = load_json(args.state, empty_state())
    events = load_json(args.events, [])
    places = load_places()

    fresh_total = 0
    for snap_path in args.snapshots:
        snapshot = json.loads(snap_path.read_text())
        fresh = run_snapshot(state, snapshot, places)
        fresh_total += len(fresh)
        events.extend(fresh)
        for e in fresh:
            print(e["text"])

    now = parse_iso(state["updated_utc"])
    events = trim_events(events, now)

    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state, indent=1))
    args.events.parent.mkdir(parents=True, exist_ok=True)
    args.events.write_text(json.dumps(events, indent=1))

    status = status_chunks(state, places, now)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    args.status.write_text(json.dumps(status, indent=1))

    known_types = sum(1 for v in state["vessels"].values()
                      if v.get("type_code"))
    print(f"-- {fresh_total} new events; state: "
          f"{len(state['vessels'])} vessels "
          f"({known_types} with known type); "
          f"{len(events)} events in rolling window; "
          f"{len(status)} status cards")


if __name__ == "__main__":
    main()
