"""Diff two snapshots into event digests — the RAG corpus generator.

Events are short, self-contained English sentences with metadata, designed
to be embedded and retrieved: departures, arrivals, anchorings, weighed
anchors, speed anomalies, and anchorage census (with delta vs previous
snapshot).

Rules corroborate speed + position and never trust nav_status alone
(operators leave it stale), per the step-1 data audit.
"""

import argparse
import json
import math
import sys
from pathlib import Path

STATIONARY_KN = 0.5   # below this a vessel is "not moving"
UNDERWAY_KN = 2.0     # above this a vessel is clearly making way
ANOMALY_KN = 30.0     # suspicious for anything but fast ferries
FAST_CATEGORIES = {"high-speed craft", "passenger"}
NEARBY_KM = 8.0       # "off <place>" labelling range
NAV_ANCHORED = 1

PLACES_PATH = Path(__file__).parent / "places.json"


def load_places(path=PLACES_PATH):
    return json.loads(Path(path).read_text())


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest(lat, lon, spots):
    """Closest place record and its distance in km."""
    best, best_km = None, float("inf")
    for spot in spots:
        km = haversine_km(lat, lon, spot["lat"], spot["lon"])
        if km < best_km:
            best, best_km = spot, km
    return best, best_km


def place_at(lat, lon, spots):
    """The place whose radius contains the point, or None."""
    spot, km = nearest(lat, lon, spots)
    if spot and km <= spot["radius_km"]:
        return spot
    return None


def vessel_label(v):
    name = v.get("name") or f"MMSI {v['mmsi']}"
    category = v.get("category")
    if category and category not in ("unknown", "other"):
        return f"{name} ({category})"
    return name


def location_phrase(lat, lon, places):
    port = place_at(lat, lon, places["ports"])
    if port:
        return port["name"]
    anchorage = place_at(lat, lon, places["anchorages"])
    if anchorage:
        return anchorage["name"]
    spot, km = nearest(lat, lon, places["ports"])
    if spot and km <= NEARBY_KM:
        return f"{km:.0f} km off {spot['name']}"
    return f"open water ({lat:.3f}, {lon:.3f})"


def short_time(generated_utc):
    # "2026-07-11T19:52:33+00:00" -> "19:52 UTC"
    return f"{generated_utc[11:16]} UTC"


def _event(kind, v, time_utc, text):
    return {
        "id": f"{time_utc}-{v['mmsi']}-{kind}",
        "type": kind,
        "mmsi": v["mmsi"],
        "name": v.get("name") or "",
        "lat": v["lat"],
        "lon": v["lon"],
        "time_utc": time_utc,
        "text": text,
    }


def vessel_events(p, c, when, places):
    """Event digests implied by one vessel's previous -> current record."""
    clock = short_time(when)
    label = vessel_label(c)
    psog, csog = p.get("sog"), c.get("sog")
    speeds_known = psog is not None and csog is not None
    events = []

    if speeds_known and psog < STATIONARY_KN and csog > UNDERWAY_KN:
        origin = place_at(p["lat"], p["lon"], places["ports"])
        if origin:
            events.append(_event("departure", c, when,
                f"{label} departed {origin['name']} at {clock}, "
                f"making {csog:.1f} knots."))
        elif (p.get("nav_status") == NAV_ANCHORED
              or place_at(p["lat"], p["lon"], places["anchorages"])):
            events.append(_event("weighed_anchor", c, when,
                f"{label} weighed anchor at "
                f"{location_phrase(p['lat'], p['lon'], places)} "
                f"at {clock} and is underway at {csog:.1f} knots."))

    if speeds_known and psog > UNDERWAY_KN and csog < STATIONARY_KN:
        dest = place_at(c["lat"], c["lon"], places["ports"])
        if dest:
            events.append(_event("arrival", c, when,
                f"{label} arrived at {dest['name']} at {clock}."))

    if (c.get("nav_status") == NAV_ANCHORED
            and p.get("nav_status") != NAV_ANCHORED
            and (csog is None or csog < STATIONARY_KN)):
        events.append(_event("anchoring", c, when,
            f"{label} anchored at "
            f"{location_phrase(c['lat'], c['lon'], places)} at {clock}."))

    if (csog is not None and csog >= ANOMALY_KN
            and c.get("category") not in FAST_CATEGORIES
            and (psog is None or psog < ANOMALY_KN)):
        events.append(_event("speed_anomaly", c, when,
            f"Speed anomaly: {label} recorded at {csog:.1f} knots near "
            f"{location_phrase(c['lat'], c['lon'], places)} at {clock}."))

    return events


def diff_snapshots(prev, curr, places=None):
    """Return the list of event digests implied by prev -> curr."""
    places = places or load_places()
    when = curr["generated_utc"]
    prev_by = {v["mmsi"]: v for v in prev["vessels"]}
    events = []
    for c in curr["vessels"]:
        p = prev_by.get(c["mmsi"])
        if p is None:
            continue  # newly heard, not necessarily newly arrived
        events.extend(vessel_events(p, c, when, places))
    events.extend(census_events(
        _census_counts(curr, places), _census_counts(prev, places), when))
    return events


def _census_counts(snap, places):
    counts = {a["name"]: 0 for a in places["anchorages"]}
    for v in snap["vessels"]:
        if (v.get("sog") or 0) >= STATIONARY_KN:
            continue
        spot = place_at(v["lat"], v["lon"], places["anchorages"])
        if spot:
            counts[spot["name"]] += 1
    return counts


def census_events(now, before, when):
    clock = short_time(when)
    events = []
    for name, count in now.items():
        delta = count - before.get(name, 0)
        if delta == 0:
            continue  # only speak on change; steady state is corpus spam
        if delta > 0:
            trend = f"up {delta} since the previous snapshot"
        else:
            trend = f"down {abs(delta)} since the previous snapshot"
        events.append({
            "id": f"{when}-census-{name}",
            "type": "anchorage_census",
            "mmsi": None,
            "name": name,
            "lat": None,
            "lon": None,
            "time_utc": when,
            "text": f"{count} vessels lying at {name} at {clock} ({trend}).",
        })
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prev", type=Path)
    parser.add_argument("curr", type=Path)
    parser.add_argument("--out", type=Path, help="write events JSON here")
    args = parser.parse_args()

    prev = json.loads(args.prev.read_text())
    curr = json.loads(args.curr.read_text())
    events = diff_snapshots(prev, curr)

    for e in events:
        print(e["text"])
    print(f"-- {len(events)} events "
          f"({prev['generated_utc']} -> {curr['generated_utc']})",
          file=sys.stderr)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(events, indent=1))


if __name__ == "__main__":
    main()
