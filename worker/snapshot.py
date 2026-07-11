"""Snapshot vessel positions in the Saronic Gulf from aisstream.io.

Connects to the aisstream.io websocket, subscribes to the project bounding
box, listens for a fixed window, and keeps the latest position per MMSI
(Class A and Class B). AIS "not available" sentinels are stored as null.
Output is a single JSON snapshot; a summary is printed for eyeballing
against MarineTraffic's public map (step-1 feed validation gate).
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import websockets

AISSTREAM_URL = "wss://stream.aisstream.io/v0/stream"

# Full Saronic + Argosaronic (CLAUDE.md scope, tuned at step 1):
# lat 37.15-38.1 N, lon 22.9-24.1 E — includes Hydra/Spetses (ferry-delay
# routes) and the Corinth Canal east entrance.
# aisstream wants [[lat, lon] SW corner, [lat, lon] NE corner].
BOUNDING_BOX = [[[37.15, 22.9], [38.1, 24.1]]]

POSITION_TYPES = {
    "PositionReport",                  # Class A (AIS msg 1-3)
    "StandardClassBPositionReport",    # Class B (msg 18)
    "ExtendedClassBPositionReport",    # Class B (msg 19, carries Name/Type)
}
STATIC_TYPES = {
    "ShipStaticData",                  # Class A (msg 5)
    "StaticDataReport",                # Class B (msg 24)
}

# AIS ship-type code ranges -> coarse category (ITU-R M.1371).
SHIP_TYPE_CATEGORIES = [
    (20, 29, "wing-in-ground"),
    (30, 30, "fishing"),
    (31, 32, "tug/tow"),
    (33, 34, "special craft"),
    (35, 35, "military"),
    (36, 36, "sailing"),
    (37, 37, "pleasure"),
    (40, 49, "high-speed craft"),
    (50, 59, "special craft"),
    (60, 69, "passenger"),
    (70, 79, "cargo"),
    (80, 89, "tanker"),
]

TIME_UTC_RE = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:\.(\d+))?")


def ship_category(type_code):
    if not type_code:  # None or 0 = "not available"
        return "unknown"
    for lo, hi, name in SHIP_TYPE_CATEGORIES:
        if lo <= type_code <= hi:
            return name
    return "other"


def parse_time_utc(raw):
    """aisstream time_utc is Go's time.Time string, nanosecond precision:
    '2026-07-11 19:23:16.923975164 +0000 UTC'. Returns aware datetime."""
    m = TIME_UTC_RE.match(raw or "")
    if not m:
        return None
    dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    fraction = (m.group(2) or "")[:6].ljust(6, "0")
    return dt.replace(microsecond=int(fraction), tzinfo=timezone.utc)


def clean_sog(sog):
    return None if sog is None or sog >= 102.3 else sog  # 102.3 = n/a


def clean_cog(cog):
    return None if cog is None or cog >= 360 else cog  # 360 = n/a


def clean_heading(heading):
    return None if heading in (None, 511) else heading  # 511 = n/a


def valid_mmsi(mmsi):
    # Ship MID-prefixed range; excludes 0, base stations, ATON, SAR aircraft
    # and truncated values from misconfigured transponders.
    return isinstance(mmsi, int) and 200_000_000 <= mmsi <= 799_999_999


def handle_position(mtype, report, meta, mmsi, vessels, static):
    lat = report.get("Latitude")
    lon = report.get("Longitude")
    if lat is None or lon is None or abs(lat) > 90 or abs(lon) > 180:
        return
    seen = parse_time_utc(meta.get("time_utc"))
    previous = vessels.get(mmsi)
    # Volunteer receivers overlap; frames are not timestamp-ordered.
    if previous and seen and previous["_seen"] and seen < previous["_seen"]:
        return
    vessels[mmsi] = {
        "mmsi": mmsi,
        "name": (meta.get("ShipName") or "").strip(),
        "lat": lat,
        "lon": lon,
        "sog": clean_sog(report.get("Sog")),
        "cog": clean_cog(report.get("Cog")),
        "heading": clean_heading(report.get("TrueHeading")),
        "nav_status": report.get("NavigationalStatus"),  # Class B: absent
        "time_utc": seen.isoformat(timespec="seconds") if seen else None,
        "_seen": seen,
    }
    if mtype == "ExtendedClassBPositionReport" and report.get("Type"):
        static.setdefault(mmsi, {})["type_code"] = report["Type"]


def handle_static(mtype, data, mmsi, static):
    entry = static.setdefault(mmsi, {})
    if mtype == "ShipStaticData":
        if data.get("Type"):
            entry["type_code"] = data["Type"]
        if data.get("Destination"):
            entry["destination"] = data["Destination"].strip()
    else:  # StaticDataReport (Class B msg 24, part B carries the type)
        part_b = data.get("ReportB") or {}
        type_code = part_b.get("ShipType") or part_b.get("Type")
        if part_b.get("Valid") and type_code:
            entry["type_code"] = type_code


async def collect(api_key: str, duration: float) -> dict:
    """Listen for `duration` seconds; return latest vessel state per MMSI."""
    vessels: dict[int, dict] = {}
    static: dict[int, dict] = {}
    message_count = 0

    async with websockets.connect(AISSTREAM_URL) as ws:
        await ws.send(json.dumps({
            "APIKey": api_key,
            "BoundingBoxes": BOUNDING_BOX,
            "FilterMessageTypes": sorted(POSITION_TYPES | STATIC_TYPES),
        }))
        # Start the clock after the handshake so setup latency doesn't
        # eat into the listen window.
        deadline = asyncio.get_event_loop().time() + duration

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            except websockets.ConnectionClosed:
                print("warning: stream closed early, writing partial "
                      "snapshot", file=sys.stderr)
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if "error" in msg:
                sys.exit(f"aisstream error: {msg['error']}")
            message_count += 1

            meta = msg.get("MetaData") or {}
            mmsi = meta.get("MMSI")
            mtype = msg.get("MessageType")
            if not valid_mmsi(mmsi):
                continue
            try:
                if mtype in POSITION_TYPES:
                    handle_position(mtype, msg["Message"][mtype], meta,
                                    mmsi, vessels, static)
                elif mtype in STATIC_TYPES:
                    handle_static(mtype, msg["Message"][mtype], mmsi, static)
            except (KeyError, TypeError):
                continue

    for mmsi, record in vessels.items():
        extra = static.get(mmsi, {})
        record["type_code"] = extra.get("type_code")
        record["category"] = ship_category(extra.get("type_code"))
        record["destination"] = extra.get("destination", "")
        del record["_seen"]

    return {"vessels": vessels, "message_count": message_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=180,
                        help="listen window in seconds (default 180)")
    parser.add_argument("--out", type=Path, default=Path("data/snapshot.json"))
    args = parser.parse_args()

    api_key = os.environ.get("AISSTREAM_API_KEY")
    if not api_key:
        sys.exit("AISSTREAM_API_KEY not set")

    started = datetime.now(timezone.utc)
    result = asyncio.run(collect(api_key, args.duration))
    vessels = result["vessels"]

    snapshot = {
        "generated_utc": started.isoformat(timespec="seconds"),
        "window_seconds": args.duration,
        "bounding_box": BOUNDING_BOX,
        "vessel_count": len(vessels),
        "vessels": sorted(vessels.values(), key=lambda v: v["mmsi"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=1))

    by_category: dict[str, int] = {}
    moving = 0
    for v in vessels.values():
        by_category[v["category"]] = by_category.get(v["category"], 0) + 1
        if (v["sog"] or 0) > 0.5:
            moving += 1

    print(f"Snapshot {started:%Y-%m-%d %H:%M UTC} — "
          f"{len(vessels)} vessels from {result['message_count']} messages "
          f"in {args.duration:.0f}s")
    print(f"  moving (>0.5 kn): {moving}, stationary: {len(vessels) - moving}")
    for category, count in sorted(by_category.items(), key=lambda x: -x[1]):
        print(f"  {category}: {count}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
