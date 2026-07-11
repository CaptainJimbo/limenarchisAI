import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "worker"))

from digest import diff_snapshots, haversine_km, load_places  # noqa: E402

PLACES = load_places()

# Handy coordinates
PIRAEUS = (37.942, 23.636)
AEGINA = (37.746, 23.428)
ANCHORAGE = (37.900, 23.570)   # Piraeus anchorage centre
OPEN_WATER = (37.60, 23.80)


def vessel(mmsi=240000001, name="TEST SHIP", lat=37.9, lon=23.6, sog=0.0,
           nav_status=0, category="passenger", **extra):
    return {
        "mmsi": mmsi, "name": name, "lat": lat, "lon": lon, "sog": sog,
        "cog": 90.0, "heading": 90, "nav_status": nav_status,
        "time_utc": "2026-07-12T08:00:00+00:00", "type_code": 60,
        "category": category, "destination": "", **extra,
    }


def snap(vessels, when="2026-07-12T08:00:00+00:00"):
    return {
        "generated_utc": when,
        "window_seconds": 120.0,
        "vessel_count": len(vessels),
        "vessels": vessels,
    }


def events_of(prev_vessels, curr_vessels):
    return diff_snapshots(
        snap(prev_vessels, "2026-07-12T08:00:00+00:00"),
        snap(curr_vessels, "2026-07-12T08:20:00+00:00"),
        PLACES,
    )


def by_type(events, kind):
    return [e for e in events if e["type"] == kind]


def test_departure_from_piraeus():
    before = vessel(lat=PIRAEUS[0], lon=PIRAEUS[1], sog=0.0, nav_status=5)
    after = vessel(lat=37.93, lon=23.62, sog=12.0, nav_status=0)
    events = by_type(events_of([before], [after]), "departure")
    assert len(events) == 1
    assert "departed Piraeus" in events[0]["text"]
    assert "TEST SHIP (passenger)" in events[0]["text"]
    assert events[0]["time_utc"] == "2026-07-12T08:20:00+00:00"


def test_arrival_at_aegina():
    before = vessel(lat=37.80, lon=23.50, sog=14.0)
    after = vessel(lat=AEGINA[0], lon=AEGINA[1], sog=0.1, nav_status=5)
    events = by_type(events_of([before], [after]), "arrival")
    assert len(events) == 1
    assert "arrived at Aegina" in events[0]["text"]


def test_anchoring_at_piraeus_anchorage():
    before = vessel(lat=37.88, lon=23.55, sog=6.0, nav_status=0,
                    category="tanker", type_code=80)
    after = vessel(lat=ANCHORAGE[0], lon=ANCHORAGE[1], sog=0.0, nav_status=1,
                   category="tanker", type_code=80)
    events = by_type(events_of([before], [after]), "anchoring")
    assert len(events) == 1
    assert "anchored at Piraeus anchorage" in events[0]["text"]
    assert "(tanker)" in events[0]["text"]


def test_anchoring_not_trusted_when_still_moving():
    # nav_status says anchored but the ship is doing 7 knots (stale status,
    # observed in real data) — must NOT fire.
    before = vessel(sog=7.0, nav_status=0)
    after = vessel(sog=7.4, nav_status=1)
    assert by_type(events_of([before], [after]), "anchoring") == []


def test_weighed_anchor():
    before = vessel(lat=ANCHORAGE[0], lon=ANCHORAGE[1], sog=0.0, nav_status=1,
                    category="tanker")
    after = vessel(lat=37.89, lon=23.58, sog=8.0, nav_status=0,
                   category="tanker")
    events = by_type(events_of([before], [after]), "weighed_anchor")
    assert len(events) == 1
    assert "weighed anchor" in events[0]["text"]


def test_no_events_when_nothing_changes():
    moored = vessel(lat=PIRAEUS[0], lon=PIRAEUS[1], sog=0.0, nav_status=5)
    steaming = vessel(mmsi=240000002, lat=OPEN_WATER[0], lon=OPEN_WATER[1],
                      sog=14.0)
    events = events_of([moored, steaming], [moored, steaming])
    # census of an occupied anchorage may appear; nothing vessel-specific may
    assert [e for e in events if e["mmsi"] is not None] == []


def test_new_mmsi_is_not_an_arrival():
    newcomer = vessel(lat=AEGINA[0], lon=AEGINA[1], sog=0.0)
    events = events_of([], [newcomer])
    assert by_type(events, "arrival") == []
    assert by_type(events, "departure") == []


def test_none_sog_never_fires_movement_events():
    before = vessel(sog=None, lat=PIRAEUS[0], lon=PIRAEUS[1])
    after = vessel(sog=None, lat=37.80, lon=23.50)
    events = events_of([before], [after])
    assert by_type(events, "departure") == []
    assert by_type(events, "arrival") == []


def test_speed_anomaly_for_slow_category_only():
    before_t = vessel(sog=10.0, category="tanker", type_code=80,
                      lat=OPEN_WATER[0], lon=OPEN_WATER[1])
    after_t = vessel(sog=33.0, category="tanker", type_code=80,
                     lat=OPEN_WATER[0], lon=OPEN_WATER[1])
    assert len(by_type(events_of([before_t], [after_t]), "speed_anomaly")) == 1

    before_f = vessel(mmsi=240000003, sog=10.0, category="high-speed craft")
    after_f = vessel(mmsi=240000003, sog=33.0, category="high-speed craft")
    assert by_type(events_of([before_f], [after_f]), "speed_anomaly") == []


def test_census_reports_count_and_delta():
    tanker = lambda i, lat, lon: vessel(  # noqa: E731
        mmsi=240000010 + i, lat=lat, lon=lon, sog=0.0, nav_status=1,
        category="tanker")
    before = [tanker(i, 37.90 + i * 0.005, 23.57) for i in range(2)]
    after = before + [tanker(9, 37.895, 23.575)]
    events = by_type(events_of(before, after), "anchorage_census")
    piraeus = [e for e in events if e["name"] == "Piraeus anchorage"]
    assert len(piraeus) == 1
    assert "3 vessels lying at Piraeus anchorage" in piraeus[0]["text"]
    assert "up 1" in piraeus[0]["text"]


def test_census_silent_for_empty_unchanged_anchorage():
    steaming = vessel(lat=OPEN_WATER[0], lon=OPEN_WATER[1], sog=14.0)
    events = by_type(events_of([steaming], [steaming]), "anchorage_census")
    assert events == []


def test_unnamed_vessel_uses_mmsi_label():
    before = vessel(name="", lat=PIRAEUS[0], lon=PIRAEUS[1], sog=0.0,
                    category="unknown", type_code=None)
    after = vessel(name="", lat=37.93, lon=23.62, sog=12.0,
                   category="unknown", type_code=None)
    events = by_type(events_of([before], [after]), "departure")
    assert len(events) == 1
    assert "MMSI 240000001" in events[0]["text"]
    assert "(unknown)" not in events[0]["text"]


def test_haversine_sanity():
    # Piraeus to Aegina is about 28 km
    km = haversine_km(*PIRAEUS, *AEGINA)
    assert 25 < km < 31
