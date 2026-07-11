import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "worker"))

from digest import load_places  # noqa: E402
from pipeline import empty_state, run_snapshot, trim_events, parse_iso  # noqa: E402

from test_digest import vessel, snap  # noqa: E402

PLACES = load_places()

T0 = "2026-07-12T08:00:00+00:00"
T1 = "2026-07-12T08:20:00+00:00"
T2 = "2026-07-12T08:40:00+00:00"
PIRAEUS = (37.942, 23.636)
ANCHORAGE = (37.900, 23.570)


def test_type_is_sticky_across_runs():
    state = empty_state()
    typed = vessel(sog=0.0, lat=PIRAEUS[0], lon=PIRAEUS[1],
                   category="passenger", type_code=60, nav_status=5)
    run_snapshot(state, snap([typed], T0), PLACES)

    untyped = vessel(sog=12.0, lat=37.93, lon=23.62,
                     category="unknown", type_code=None, nav_status=0)
    events = run_snapshot(state, snap([untyped], T1), PLACES)

    cached = state["vessels"][str(untyped["mmsi"])]
    assert cached["type_code"] == 60
    assert cached["category"] == "passenger"
    departures = [e for e in events if e["type"] == "departure"]
    assert len(departures) == 1
    assert "(passenger)" in departures[0]["text"]


def test_flicker_does_not_lose_the_vessel():
    state = empty_state()
    anchored = vessel(lat=ANCHORAGE[0], lon=ANCHORAGE[1], sog=0.0,
                      nav_status=1, category="tanker", type_code=80)
    run_snapshot(state, snap([anchored], T0), PLACES)
    # window 2: vessel not heard at all
    run_snapshot(state, snap([], T1), PLACES)
    # window 3: heard again, still anchored — no events, census stable
    events = run_snapshot(state, snap([anchored], T2), PLACES)

    vessel_specific = [e for e in events if e["mmsi"] is not None]
    assert vessel_specific == []
    census = [e for e in events if e["type"] == "anchorage_census"]
    assert census == []  # count unchanged -> silent


def test_census_survives_one_missed_window():
    state = empty_state()
    anchored = vessel(lat=ANCHORAGE[0], lon=ANCHORAGE[1], sog=0.0,
                      nav_status=1, category="tanker", type_code=80)
    events0 = run_snapshot(state, snap([anchored], T0), PLACES)
    census0 = [e for e in events0 if e["type"] == "anchorage_census"]
    assert len(census0) == 1
    assert "1 vessels lying" in census0[0]["text"]

    # vessel missed in the next window: census must NOT report "down 1"
    events1 = run_snapshot(state, snap([], T1), PLACES)
    assert [e for e in events1 if e["type"] == "anchorage_census"] == []


def test_event_fires_across_a_gap():
    state = empty_state()
    moored = vessel(lat=PIRAEUS[0], lon=PIRAEUS[1], sog=0.0, nav_status=5)
    run_snapshot(state, snap([moored], T0), PLACES)
    run_snapshot(state, snap([], T1), PLACES)  # missed window
    underway = vessel(lat=37.93, lon=23.62, sog=12.0, nav_status=0)
    events = run_snapshot(state, snap([underway], T2), PLACES)
    departures = [e for e in events if e["type"] == "departure"]
    assert len(departures) == 1  # diffed vs its own last observation at T0


def test_state_expires_after_48h():
    state = empty_state()
    run_snapshot(state, snap([vessel()], T0), PLACES)
    assert len(state["vessels"]) == 1
    later = vessel(mmsi=240999999, name="OTHER")
    run_snapshot(state, snap([later], "2026-07-15T09:00:00+00:00"), PLACES)
    assert str(vessel()["mmsi"]) not in state["vessels"]
    assert "240999999" in state["vessels"]


def test_trim_events_rolls_48h_window():
    old = {"time_utc": T0, "text": "old"}
    fresh = {"time_utc": "2026-07-14T07:00:00+00:00", "text": "fresh"}
    kept = trim_events([old, fresh], parse_iso("2026-07-14T08:00:00+00:00"))
    assert kept == [fresh]


def test_no_events_on_first_sighting():
    state = empty_state()
    events = run_snapshot(
        state, snap([vessel(sog=14.0, lat=37.7, lon=23.6)], T0), PLACES)
    assert [e for e in events if e["mmsi"] is not None] == []


def test_status_card_text_is_stable_and_timeless():
    from pipeline import status_chunks
    state = empty_state()
    anchored = vessel(lat=ANCHORAGE[0], lon=ANCHORAGE[1], sog=0.0,
                      nav_status=1, category="tanker", type_code=80,
                      name="HATHOR")
    run_snapshot(state, snap([anchored], T0), PLACES)
    cards0 = status_chunks(state, PLACES, parse_iso(T0))
    run_snapshot(state, snap([anchored], T1), PLACES)
    cards1 = status_chunks(state, PLACES, parse_iso(T1))
    assert cards0[0]["text"] == cards1[0]["text"]  # embedding cache hit
    assert "HATHOR (tanker) is at anchor at Piraeus anchorage." == \
        cards0[0]["text"]
    assert "08:" not in cards0[0]["text"]  # no timestamps in text


def test_status_card_moving_vessel():
    from pipeline import status_chunks
    state = empty_state()
    moving = vessel(sog=18.4, lat=37.80, lon=23.50, nav_status=0)
    run_snapshot(state, snap([moving], T0), PLACES)
    cards = status_chunks(state, PLACES, parse_iso(T0))
    assert len(cards) == 1
    assert "underway at 18 knots" in cards[0]["text"]
    assert "heading east" in cards[0]["text"]


def test_status_card_excludes_stale_vessels():
    from pipeline import status_chunks
    state = empty_state()
    run_snapshot(state, snap([vessel()], T0), PLACES)
    late = parse_iso("2026-07-12T09:30:00+00:00")  # 90 min later
    assert status_chunks(state, PLACES, late) == []
