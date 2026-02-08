"""
Denver Metro Trip Synthetic Data Generator (GUI)

Controls:
- Number of rows
- Start date and duration (days)
- Status ratios: No Show %, Cancel % (Completed auto)
- OTP on-time % (Performed only)
- Pickup window minutes (SchLate = SchTime + window)
- Number of clients (ClientId pool size)
- Number of drivers
- Number of vehicles
- Number of routes/runs
- Fare types mix: ADA / NEMT / CITY (must sum to 100)

OTP rule:
- On-time if Pickup-ActualArriveTime <= SchLate
- Early arrivals are on-time

Build exe (Windows):
  pip install pyinstaller
  pyinstaller --onefile --windowed denver_trip_synth_gui.py
"""

from __future__ import annotations

import math
import os
import random
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime, date, timedelta, time
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

def clamp_int(x: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, x)))

def seconds_since_midnight(dt: datetime) -> int:
    return dt.hour * 3600 + dt.minute * 60 + dt.second

def seconds_to_time_str(sec: int) -> str:
    if sec is None or sec < 0:
        return ""
    sec = sec % 86400
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12
    if h12 == 0:
        h12 = 12
    return f"{h12}:{m:02d}:{s:02d}{suffix}"

def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    R = 3958.7613
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def weighted_choice(items, weights, rng: random.Random):
    total = float(sum(weights))
    if total <= 0:
        return items[-1]
    r = rng.random() * total
    upto = 0.0
    for item, w in zip(items, weights):
        upto += float(w)
        if upto >= r:
            return item
    return items[-1]

def default_desktop_path(filename: str) -> str:
    try:
        desktop = Path.home() / "Desktop"
        return str(desktop / filename)
    except Exception:
        return filename


# -----------------------------
# Denver metro location pool
# -----------------------------

@dataclass(frozen=True)
class Place:
    name: str
    address: str
    city: str
    state: str
    zip: str
    lat: float
    lon: float
    kind: str


DENVER_PLACES: list[Place] = [
    Place("Denver Health Medical Center", "777 Bannock St", "Denver", "CO", "80204", 39.7322, -104.9916, "hospital"),
    Place("UCHealth University of Colorado Hospital", "12605 E 16th Ave", "Aurora", "CO", "80045", 39.7421, -104.8372, "hospital"),
    Place("Children's Hospital Colorado", "13123 E 16th Ave", "Aurora", "CO", "80045", 39.7426, -104.8360, "hospital"),
    Place("Rose Medical Center", "4567 E 9th Ave", "Denver", "CO", "80220", 39.7318, -104.9339, "hospital"),
    Place("Saint Joseph Hospital", "1375 E 19th Ave", "Denver", "CO", "80218", 39.7462, -104.9709, "hospital"),
    Place("Porter Adventist Hospital", "2525 S Downing St", "Denver", "CO", "80210", 39.6714, -104.9737, "hospital"),
    Place("Swedish Medical Center", "501 E Hampden Ave", "Englewood", "CO", "80113", 39.6530, -104.9817, "hospital"),
    Place("Medical Center of Aurora", "1501 S Potomac St", "Aurora", "CO", "80012", 39.6909, -104.8423, "hospital"),
    Place("Denver Union Station", "1701 Wynkoop St", "Denver", "CO", "80202", 39.7527, -104.9992, "transit"),
    Place("RTD Civic Center Station", "1010 14th St", "Denver", "CO", "80202", 39.7426, -104.9910, "transit"),
    Place("Aurora Metro Center Station", "14200 E Exposition Ave", "Aurora", "CO", "80012", 39.7033, -104.8204, "transit"),
    Place("Nine Mile Station", "3100 S Parker Rd", "Aurora", "CO", "80014", 39.6637, -104.8657, "transit"),
    Place("Lakewood - Wadsworth Station", "1390 Wadsworth Blvd", "Lakewood", "CO", "80214", 39.7372, -105.0835, "transit"),
    Place("Arvada Ridge Station", "5600 Wadsworth Bypass", "Arvada", "CO", "80002", 39.7979, -105.0815, "transit"),
    Place("Westminster Station", "6995 W 73rd Ave", "Westminster", "CO", "80030", 39.8302, -105.0180, "transit"),
    Place("Littleton Downtown Station", "5400 S Prince St", "Littleton", "CO", "80120", 39.6165, -105.0181, "transit"),
    Place("Golden Senior Center", "1010 10th St", "Golden", "CO", "80401", 39.7579, -105.2222, "senior"),
    Place("Malley Senior Recreation Center", "3380 Lincoln St", "Englewood", "CO", "80113", 39.6547, -104.9871, "senior"),
    Place("Aurora Center for Active Adults", "30 Del Mar Cir", "Aurora", "CO", "80011", 39.7405, -104.8438, "senior"),
    Place("Wheat Ridge Recreation Center", "4005 Kipling St", "Wheat Ridge", "CO", "80033", 39.7756, -105.1098, "senior"),
    Place("Denver International Airport", "8500 Peña Blvd", "Denver", "CO", "80249", 39.8561, -104.6737, "airport"),
]

CITY_CENTERS = [
    ("Denver", "CO", "80202", 39.7392, -104.9903),
    ("Aurora", "CO", "80012", 39.7294, -104.8319),
    ("Lakewood", "CO", "80226", 39.7047, -105.0814),
    ("Arvada", "CO", "80002", 39.8028, -105.0875),
    ("Westminster", "CO", "80030", 39.8367, -105.0372),
    ("Thornton", "CO", "80229", 39.8680, -104.9719),
    ("Centennial", "CO", "80112", 39.5807, -104.8772),
    ("Littleton", "CO", "80120", 39.6133, -105.0166),
    ("Englewood", "CO", "80110", 39.6478, -104.9878),
    ("Broomfield", "CO", "80020", 39.9205, -105.0867),
    ("Commerce City", "CO", "80022", 39.8083, -104.9339),
    ("Parker", "CO", "80134", 39.5186, -104.7614),
    ("Golden", "CO", "80401", 39.7555, -105.2211),
    ("Highlands Ranch", "CO", "80126", 39.5539, -104.9694),
]

STREET_NAMES = [
    "Colfax Ave", "Broadway", "Speer Blvd", "Federal Blvd", "Colorado Blvd", "Wadsworth Blvd",
    "Hampden Ave", "Alameda Ave", "Mississippi Ave", "Parker Rd", "Peoria St", "Havana St",
    "Quebec St", "Kipling St", "Sheridan Blvd", "Santa Fe Dr", "University Blvd", "Downing St",
    "Evans Ave", "6th Ave", "1st Ave", "Iliff Ave", "Belleview Ave", "Arapahoe Rd", "Leetsdale Dr"
]
UNIT_SUFFIX = ["", "", "", " Apt 2", " Apt 7", " Unit B", " Unit 12", " Ste 200", " Ste 310"]

PURPOSES = [
    ("Dialysis", 0.18),
    ("Medical Appointment", 0.28),
    ("Physical Therapy", 0.10),
    ("Adult Day Program", 0.10),
    ("Pharmacy", 0.08),
    ("Grocery", 0.06),
    ("Work", 0.10),
    ("Functional Assessment", 0.10),
]

PROVIDERS = [
    ("Denver Metro Mobility", 0.40),
    ("Front Range Transit Assist", 0.25),
    ("Mile High Paratransit", 0.20),
    ("Rocky Mountain Mobility", 0.15),
]

SPACE_TYPES = [
    ("WC", 0.20),
    ("AM", 0.75),
    ("ST", 0.05),
]


def make_residential_place(rng: random.Random) -> Place:
    city, state, zc, clat, clon = rng.choice(CITY_CENTERS)
    lat = clat + rng.gauss(0, 0.020)
    lon = clon + rng.gauss(0, 0.025)

    lat = max(39.35, min(40.20, lat))
    lon = max(-105.35, min(-104.60, lon))

    num = rng.randint(100, 9999)
    street = rng.choice(STREET_NAMES)
    unit = rng.choice(UNIT_SUFFIX)
    address = f"{num} {street}{unit}"
    return Place("Residential", address, city, state, zc, lat, lon, "residential")


def pick_place(rng: random.Random, kind_weights: dict[str, float]) -> Place:
    use_poi = rng.random() < 0.65
    if not use_poi:
        return make_residential_place(rng)

    kinds = list(kind_weights.keys())
    weights = list(kind_weights.values())
    chosen_kind = weighted_choice(kinds, weights, rng)

    if chosen_kind == "residential":
        return make_residential_place(rng)

    candidates = [p for p in DENVER_PLACES if p.kind == chosen_kind]
    if not candidates:
        candidates = DENVER_PLACES
    return rng.choice(candidates)


# -----------------------------
# Time and status models
# -----------------------------

def sample_scheduled_time(rng: random.Random) -> time:
    buckets = [
        (time(6, 0), time(9, 30), 0.35),
        (time(9, 30), time(12, 30), 0.25),
        (time(12, 30), time(16, 30), 0.30),
        (time(16, 30), time(19, 30), 0.10),
    ]
    starts, ends, ws = zip(*buckets)
    idx = np.searchsorted(np.cumsum(ws), rng.random(), side="right")
    idx = min(int(idx), len(buckets) - 1)

    st = starts[idx]
    en = ends[idx]

    st_min = st.hour * 60 + st.minute
    en_min = en.hour * 60 + en.minute
    minute = rng.randrange(st_min, en_min + 1, 5)
    return time(minute // 60, minute % 60, 0)


def traffic_multiplier(sched: time) -> float:
    h = sched.hour + sched.minute / 60
    if 6.5 <= h <= 9.0:
        return 1.25
    if 15.5 <= h <= 18.5:
        return 1.20
    return 1.00


def late_group_label(late_minutes: int) -> str:
    if late_minutes <= 0:
        return ""
    if 1 <= late_minutes <= 5:
        return "1 - 5 Mins"
    if 6 <= late_minutes <= 10:
        return "6 - 10 Mins"
    if 11 <= late_minutes <= 15:
        return "11 - 15 Mins"
    if 16 <= late_minutes <= 30:
        return "16 - 30 Mins"
    return "31+ Mins"


def random_dob(rng: random.Random) -> str:
    start = date(1940, 1, 1)
    end = date(2005, 12, 31)
    days = (end - start).days
    d = start + timedelta(days=rng.randint(0, days))
    return d.strftime("%m/%d/%Y") + " A-E"


def generate_driver_names(n: int, rng: random.Random) -> list[str]:
    firsts = ["Alex", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Cameron", "Avery", "Drew", "Skyler",
              "Logan", "Peyton", "Quinn", "Reese", "Hayden", "Blake", "Devon", "Emerson", "Kai", "Jules"]
    lasts = ["Martinez", "Kim", "Johnson", "Brown", "Rivera", "Nguyen", "Lee", "Patel", "Sanchez", "Thompson",
             "Davis", "Walker", "Turner", "Cooper", "Clark", "Miller", "Garcia", "Scott", "Baker", "Howard"]
    names = []
    used = set()
    while len(names) < n:
        nm = f"{rng.choice(firsts)} {rng.choice(lasts)}"
        if nm not in used:
            used.add(nm)
            names.append(nm)
    return names


def generate_vehicle_ids(n: int) -> list[str]:
    base = 101
    return [f"V{base + i:03d}" for i in range(n)]


def generate_client_ids(n: int, rng: random.Random) -> list[int]:
    # Fixed pool so clients repeat across many trips
    # Keep them in a realistic numeric band
    start = rng.randint(210000, 230000)
    return list(range(start, start + n))


# -----------------------------
# Generator core
# -----------------------------

@dataclass
class GeneratorConfig:
    n_rows: int
    start_date: date
    n_days: int
    no_show_pct: float
    cancel_pct: float
    otp_on_time_pct: float
    pickup_window_mins: int
    seed: int
    n_clients: int
    n_drivers: int
    n_vehicles: int
    n_routes: int
    fare_ada_pct: float
    fare_nemt_pct: float
    fare_city_pct: float


class TripSynth:
    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.clients = generate_client_ids(cfg.n_clients, self.rng)
        self.drivers = generate_driver_names(cfg.n_drivers, self.rng)
        self.vehicles = generate_vehicle_ids(cfg.n_vehicles)
        self.routes = list(range(100, 100 + cfg.n_routes))

    def _status_for_row(self) -> str:
        no_show = self.cfg.no_show_pct / 100.0
        cancel = self.cfg.cancel_pct / 100.0
        performed = max(0.0, 1.0 - no_show - cancel)

        r = self.rng.random()
        if r < performed:
            return "Performed"
        if r < performed + no_show:
            return "NoShow"
        return "Cancel Same Day" if self.rng.random() < 0.55 else "Cancel Advance"

    def _otp_is_on_time_target(self) -> bool:
        return self.rng.random() < (self.cfg.otp_on_time_pct / 100.0)

    def _fare_type(self) -> str:
        items = ["ADA", "NEMT", "CITY"]
        weights = [self.cfg.fare_ada_pct, self.cfg.fare_nemt_pct, self.cfg.fare_city_pct]
        return weighted_choice(items, weights, self.rng)

    def _subtype_for_fare(self, fare: str) -> str:
        if fare == "ADA":
            return weighted_choice(["ADA", "FNC", "ADA"], [0.55, 0.35, 0.10], self.rng)
        if fare == "NEMT":
            return weighted_choice(["NEMT", "FNC", "NEMT"], [0.65, 0.25, 0.10], self.rng)
        return weighted_choice(["FNC", "ADA", "NEMT"], [0.70, 0.15, 0.15], self.rng)

    def _make_trip(self) -> dict:
        day_offset = self.rng.randrange(0, max(1, self.cfg.n_days))
        trip_date = self.cfg.start_date + timedelta(days=int(day_offset))

        client_id = self.rng.choice(self.clients)
        booking_id = self.rng.randint(18000000, 18999999)
        route = self.rng.choice(self.routes)
        ev_str_name = self.rng.randint(1000, 1299)

        fare_type = self._fare_type()
        subtype = self._subtype_for_fare(fare_type)

        space_type = weighted_choice([x for x, _ in SPACE_TYPES], [w for _, w in SPACE_TYPES], self.rng)
        purpose = weighted_choice([x for x, _ in PURPOSES], [w for _, w in PURPOSES], self.rng)
        provider = weighted_choice([x for x, _ in PROVIDERS], [w for _, w in PROVIDERS], self.rng)

        driver = self.rng.choice(self.drivers)
        vehicle = self.rng.choice(self.vehicles)
        pass_count = int(self.rng.choice([1, 1, 1, 2, 2, 3]))

        pickup = pick_place(self.rng, {"residential": 0.05, "transit": 0.30, "senior": 0.20, "hospital": 0.45})
        dropoff = pick_place(self.rng, {"residential": 0.20, "transit": 0.20, "senior": 0.15, "hospital": 0.45})

        if pickup.address == dropoff.address and self.rng.random() < 0.85:
            dropoff = pick_place(self.rng, {"residential": 0.20, "transit": 0.20, "senior": 0.15, "hospital": 0.45})

        dist = haversine_miles(pickup.lat, pickup.lon, dropoff.lat, dropoff.lon)
        dist = max(0.5, dist)

        sched_t = sample_scheduled_time(self.rng)
        sch_time = sched_t.hour * 3600 + sched_t.minute * 60 + sched_t.second
        sch_late = sch_time + int(self.cfg.pickup_window_mins * 60)

        status = self._status_for_row()

        mph = self.rng.uniform(18, 32)
        est_minutes = (dist / mph) * 60.0
        est_minutes *= traffic_multiplier(sched_t)
        est_minutes += self.rng.uniform(2, 8)
        est_travel_seconds = int(max(120, est_minutes * 60))

        if fare_type == "CITY":
            base = 0.0
            per_mile = 0.0
        elif fare_type == "ADA":
            base = 3.25
            per_mile = 1.05
        else:
            base = 4.50
            per_mile = 1.35

        fare_amount = round(max(0.0, base + per_mile * dist + self.rng.uniform(-0.75, 1.25)), 2)

        p_arr = -1
        p_dep = -1
        d_arr = -1
        d_dep = -1
        otp = ""
        late_group = ""
        neg_time = ""
        travel_time = 0
        fare_collected = 0.0

        if status == "Performed":
            on_time_target = self._otp_is_on_time_target()

            if on_time_target:
                if self.rng.random() < 0.55:
                    early_sec = int(self.rng.uniform(0, 20) * 60)
                    p_arr = sch_time - early_sec
                else:
                    within_sec = int(self.rng.uniform(0, max(1, self.cfg.pickup_window_mins)) * 60)
                    p_arr = sch_time + within_sec
                p_arr = min(p_arr, sch_late)
            else:
                delay_sec = int(self.rng.uniform(1, 45) * 60)
                p_arr = sch_late + delay_sec

            p_arr = clamp_int(p_arr, 0, 86399)
            dwell_pickup = int(self.rng.uniform(1, 6) * 60)
            p_dep = clamp_int(p_arr + dwell_pickup, 0, 86399)

            noise = self.rng.uniform(0.85, 1.35)
            travel_time = int(est_travel_seconds * noise)
            d_arr = clamp_int(p_dep + travel_time, 0, 86399)

            dwell_drop = int(self.rng.uniform(1, 5) * 60)
            d_dep = clamp_int(d_arr + dwell_drop, 0, 86399)

            is_on_time = (p_arr <= sch_late)
            otp = 1 if is_on_time else 0

            late_min = max(0, int(math.ceil((p_arr - sch_late) / 60)))
            late_group = late_group_label(late_min)

            neg_time = int(p_arr - sch_time)
            fare_collected = 0.0 if fare_type == "CITY" else fare_amount

        elif status == "NoShow":
            if self.rng.random() < 0.85:
                early_sec = int(self.rng.uniform(0, 15) * 60)
                p_arr = clamp_int(sch_time - early_sec, 0, 86399)
            else:
                p_arr = clamp_int(sch_late + int(self.rng.uniform(1, 20) * 60), 0, 86399)

            wait_sec = int(self.rng.uniform(5, 12) * 60)
            p_dep = clamp_int(p_arr + wait_sec, 0, 86399)
            neg_time = int(p_arr - sch_time)
            fare_collected = 0.0

        else:
            fare_collected = 0.0

        row = {
            "Date": trip_date.strftime("%m/%d/%Y"),
            "ClientId": client_id,
            "SubtypeAbbr": subtype,
            "CreBy": "tripgen",
            "PickupLat": round(pickup.lat, 6),
            "PickupLon": round(pickup.lon, 6),
            "DropoffLat": round(dropoff.lat, 6),
            "DropoffLon": round(dropoff.lon, 6),
            "DirectDistance": round(dist, 2),
            "EvStrName": ev_str_name,
            "SpaceType": space_type,
            "SchTime": sch_time,
            "SchLate": sch_late,
            "Pickup-ActualArriveTime": p_arr,
            "Pickup-ActualDepartTime": p_dep,
            "Dropoff-ActualArriveTime": d_arr,
            "Dropoff-ActualDepartTime": d_dep,
            "Purpose": purpose,
            "DoB": random_dob(self.rng),
            "FareTypeAbbr": fare_type,
            "ProviderName": provider,
            "NegTime": neg_time if neg_time != "" else "",
            "PassCount": pass_count,
            "Driver": driver,
            "VehicleNumber": vehicle,
            "SchedStatusName": status,
            "TravelTime": travel_time,
            "BookingId": booking_id,
            "Violation": "",
            "Route": route,
            "PickupTime": seconds_to_time_str(sch_time),
            "OTP": otp,
            "LateGroup": late_group,
            "FareAmount": fare_amount,
            "FareCollected": round(fare_collected, 2),
            "PickupAddress": f"{pickup.address}, {pickup.city}, {pickup.state} {pickup.zip}",
            "DropoffAddress": f"{dropoff.address}, {dropoff.city}, {dropoff.state} {dropoff.zip}",
        }
        return row

    def generate(self) -> pd.DataFrame:
        rows = [self._make_trip() for _ in range(self.cfg.n_rows)]
        df = pd.DataFrame(rows)

        col_order = [
            "Date","ClientId","SubtypeAbbr","CreBy",
            "PickupLat","PickupLon","DropoffLat","DropoffLon",
            "DirectDistance","EvStrName","SpaceType",
            "SchTime","SchLate",
            "Pickup-ActualArriveTime","Pickup-ActualDepartTime",
            "Dropoff-ActualArriveTime","Dropoff-ActualDepartTime",
            "Purpose","DoB",
            "FareTypeAbbr","ProviderName",
            "NegTime","PassCount","Driver","VehicleNumber",
            "SchedStatusName","TravelTime","BookingId","Violation",
            "Route","PickupTime","OTP","LateGroup",
            "FareAmount","FareCollected",
            "PickupAddress","DropoffAddress"
        ]
        return df[col_order]


# -----------------------------
# GUI
# -----------------------------

class App(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=12)
        self.master = master
        self.master.title("Denver Trip Synthetic Data Generator")
        self.master.geometry("1180x800")

        self.var_rows = tk.StringVar(value="5000")
        self.var_start_date = tk.StringVar(value=date.today().strftime("%Y-%m-%d"))
        self.var_days = tk.StringVar(value="30")

        self.var_no_show = tk.DoubleVar(value=8.0)
        self.var_cancel = tk.DoubleVar(value=12.0)
        self.var_otp = tk.DoubleVar(value=85.0)
        self.var_window = tk.IntVar(value=15)

        self.var_seed = tk.StringVar(value="42")

        self.var_clients = tk.StringVar(value="1200")
        self.var_drivers = tk.StringVar(value="25")
        self.var_vehicles = tk.StringVar(value="35")
        self.var_routes = tk.StringVar(value="60")

        self.var_fare_ada = tk.DoubleVar(value=45.0)
        self.var_fare_nemt = tk.DoubleVar(value=45.0)
        self.var_fare_city = tk.DoubleVar(value=10.0)

        self.var_out = tk.StringVar(value=default_desktop_path("denver_trips_synth.csv"))

        self._build()

    def _build(self):
        self.grid(sticky="nsew")
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)

        top = ttk.LabelFrame(self, text="Controls", padding=10)
        top.grid(row=0, column=0, sticky="ew")
        for c in range(8):
            top.columnconfigure(c, weight=1 if c in (3, 7) else 0)

        ttk.Label(top, text="Rows").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.var_rows, width=10).grid(row=0, column=1, sticky="w", padx=(6, 14))

        ttk.Label(top, text="Start date (YYYY-MM-DD)").grid(row=0, column=2, sticky="w")
        ttk.Entry(top, textvariable=self.var_start_date, width=14).grid(row=0, column=3, sticky="w", padx=(6, 14))

        ttk.Label(top, text="Days").grid(row=0, column=4, sticky="w")
        ttk.Entry(top, textvariable=self.var_days, width=8).grid(row=0, column=5, sticky="w", padx=(6, 14))

        ttk.Label(top, text="Seed").grid(row=0, column=6, sticky="w")
        ttk.Entry(top, textvariable=self.var_seed, width=10).grid(row=0, column=7, sticky="w", padx=(6, 0))

        ops = ttk.LabelFrame(top, text="Clients, fleet, and routing", padding=10)
        ops.grid(row=1, column=0, columnspan=8, sticky="ew", pady=(10, 0))

        ttk.Label(ops, text="Clients").grid(row=0, column=0, sticky="w")
        ttk.Entry(ops, textvariable=self.var_clients, width=10).grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Label(ops, text="Drivers").grid(row=0, column=2, sticky="w")
        ttk.Entry(ops, textvariable=self.var_drivers, width=10).grid(row=0, column=3, sticky="w", padx=(6, 18))

        ttk.Label(ops, text="Vehicles").grid(row=0, column=4, sticky="w")
        ttk.Entry(ops, textvariable=self.var_vehicles, width=10).grid(row=0, column=5, sticky="w", padx=(6, 18))

        ttk.Label(ops, text="Routes/Runs").grid(row=0, column=6, sticky="w")
        ttk.Entry(ops, textvariable=self.var_routes, width=10).grid(row=0, column=7, sticky="w", padx=(6, 0))

        ratios = ttk.LabelFrame(top, text="Status mix (Completed auto)", padding=10)
        ratios.grid(row=2, column=0, columnspan=8, sticky="ew", pady=(10, 0))
        ratios.columnconfigure(7, weight=1)

        ttk.Label(ratios, text="No show %").grid(row=0, column=0, sticky="w")
        ttk.Scale(
            ratios, from_=0, to=60, variable=self.var_no_show, orient="horizontal", length=220,
            command=lambda _=None: self._update_status_labels()
        ).grid(row=0, column=1, sticky="w", padx=(6, 14))
        self.lbl_no_show = ttk.Label(ratios, text="8.0%")
        self.lbl_no_show.grid(row=0, column=2, sticky="w")

        ttk.Label(ratios, text="Cancel %").grid(row=0, column=3, sticky="w")
        ttk.Scale(
            ratios, from_=0, to=60, variable=self.var_cancel, orient="horizontal", length=220,
            command=lambda _=None: self._update_status_labels()
        ).grid(row=0, column=4, sticky="w", padx=(6, 14))
        self.lbl_cancel = ttk.Label(ratios, text="12.0%")
        self.lbl_cancel.grid(row=0, column=5, sticky="w")

        self.lbl_completed = ttk.Label(ratios, text="Completed: 80.0%")
        self.lbl_completed.grid(row=0, column=6, columnspan=2, sticky="w")

        otpbox = ttk.LabelFrame(top, text="OTP logic", padding=10)
        otpbox.grid(row=3, column=0, columnspan=8, sticky="ew", pady=(10, 0))
        otpbox.columnconfigure(7, weight=1)

        ttk.Label(otpbox, text="Target On-Time % (Performed only)").grid(row=0, column=0, sticky="w")
        ttk.Scale(
            otpbox, from_=0, to=100, variable=self.var_otp, orient="horizontal", length=220,
            command=lambda _=None: self._update_otp_window_labels()
        ).grid(row=0, column=1, sticky="w", padx=(6, 14))
        self.lbl_otp = ttk.Label(otpbox, text="85.0%")
        self.lbl_otp.grid(row=0, column=2, sticky="w")

        ttk.Label(otpbox, text="Pickup window minutes (SchLate = SchTime + window)").grid(row=0, column=3, sticky="w")
        ttk.Scale(
            otpbox, from_=0, to=60, variable=self.var_window, orient="horizontal", length=220,
            command=lambda _=None: self._update_otp_window_labels()
        ).grid(row=0, column=4, sticky="w", padx=(6, 14))
        self.lbl_window = ttk.Label(otpbox, text="15 mins")
        self.lbl_window.grid(row=0, column=5, sticky="w")

        farebox = ttk.LabelFrame(top, text="Fare type mix (must sum to 100)", padding=10)
        farebox.grid(row=4, column=0, columnspan=8, sticky="ew", pady=(10, 0))

        ttk.Label(farebox, text="ADA %").grid(row=0, column=0, sticky="w")
        ttk.Scale(
            farebox, from_=0, to=100, variable=self.var_fare_ada, orient="horizontal", length=180,
            command=lambda _=None: self._update_fare_labels()
        ).grid(row=0, column=1, sticky="w", padx=(6, 10))
        self.lbl_fare_ada = ttk.Label(farebox, text="45.0%")
        self.lbl_fare_ada.grid(row=0, column=2, sticky="w", padx=(0, 18))

        ttk.Label(farebox, text="NEMT %").grid(row=0, column=3, sticky="w")
        ttk.Scale(
            farebox, from_=0, to=100, variable=self.var_fare_nemt, orient="horizontal", length=180,
            command=lambda _=None: self._update_fare_labels()
        ).grid(row=0, column=4, sticky="w", padx=(6, 10))
        self.lbl_fare_nemt = ttk.Label(farebox, text="45.0%")
        self.lbl_fare_nemt.grid(row=0, column=5, sticky="w", padx=(0, 18))

        ttk.Label(farebox, text="CITY %").grid(row=0, column=6, sticky="w")
        ttk.Scale(
            farebox, from_=0, to=100, variable=self.var_fare_city, orient="horizontal", length=180,
            command=lambda _=None: self._update_fare_labels()
        ).grid(row=0, column=7, sticky="w", padx=(6, 10))
        self.lbl_fare_city = ttk.Label(farebox, text="10.0%")
        self.lbl_fare_city.grid(row=0, column=8, sticky="w")

        self.lbl_fare_sum = ttk.Label(farebox, text="Sum: 100.0%")
        self.lbl_fare_sum.grid(row=1, column=0, columnspan=9, sticky="w", pady=(6, 0))

        outbox = ttk.LabelFrame(top, text="Output", padding=10)
        outbox.grid(row=5, column=0, columnspan=8, sticky="ew", pady=(10, 0))
        outbox.columnconfigure(1, weight=1)

        ttk.Label(outbox, text="CSV path").grid(row=0, column=0, sticky="w")
        ttk.Entry(outbox, textvariable=self.var_out).grid(row=0, column=1, sticky="ew", padx=(6, 10))
        ttk.Button(outbox, text="Browse", command=self._browse).grid(row=0, column=2, sticky="w")
        ttk.Button(outbox, text="Generate CSV", command=self._generate).grid(row=0, column=3, sticky="e")

        self.progress = ttk.Progressbar(self, orient="horizontal", mode="indeterminate")
        self.progress.grid(row=1, column=0, sticky="ew", pady=(10, 6))

        preview = ttk.LabelFrame(self, text="Preview and stats", padding=10)
        preview.grid(row=2, column=0, sticky="nsew")
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)

        self.txt = tk.Text(preview, wrap="none", height=18)
        self.txt.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(preview, orient="vertical", command=self.txt.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(preview, orient="horizontal", command=self.txt.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        self.txt.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        self._update_status_labels()
        self._update_otp_window_labels()
        self._update_fare_labels()

    def _update_status_labels(self):
        no_show = float(self.var_no_show.get())
        cancel = float(self.var_cancel.get())
        completed = 100.0 - no_show - cancel
        self.lbl_no_show.config(text=f"{no_show:.1f}%")
        self.lbl_cancel.config(text=f"{cancel:.1f}%")
        if completed < 0:
            self.lbl_completed.config(text=f"Completed: {completed:.1f}% (fix ratios)", foreground="red")
        else:
            self.lbl_completed.config(text=f"Completed: {completed:.1f}%", foreground="")

    def _update_otp_window_labels(self):
        self.lbl_otp.config(text=f"{float(self.var_otp.get()):.1f}%")
        self.lbl_window.config(text=f"{int(self.var_window.get())} mins")

    def _update_fare_labels(self):
        a = float(self.var_fare_ada.get())
        n = float(self.var_fare_nemt.get())
        c = float(self.var_fare_city.get())
        s = a + n + c
        self.lbl_fare_ada.config(text=f"{a:.1f}%")
        self.lbl_fare_nemt.config(text=f"{n:.1f}%")
        self.lbl_fare_city.config(text=f"{c:.1f}%")
        if abs(s - 100.0) > 0.01:
            self.lbl_fare_sum.config(text=f"Sum: {s:.1f}% (must be 100)", foreground="red")
        else:
            self.lbl_fare_sum.config(text=f"Sum: {s:.1f}%", foreground="")

    def _browse(self):
        p = filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="denver_trips_synth.csv",
        )
        if p:
            self.var_out.set(p)

    def _parse_int(self, val: str, field: str, lo: int, hi: int) -> int:
        try:
            x = int(str(val).strip())
        except Exception:
            raise ValueError(f"{field} must be an integer.")
        if x < lo or x > hi:
            raise ValueError(f"{field} must be between {lo} and {hi}.")
        return x

    def _parse_inputs(self) -> GeneratorConfig:
        n_rows = self._parse_int(self.var_rows.get(), "Rows", 1, 5_000_000)

        try:
            start = datetime.strptime(self.var_start_date.get().strip(), "%Y-%m-%d").date()
        except Exception:
            raise ValueError("Start date must be YYYY-MM-DD.")

        n_days = self._parse_int(self.var_days.get(), "Days", 1, 3650)

        no_show = float(self.var_no_show.get())
        cancel = float(self.var_cancel.get())
        if no_show < 0 or cancel < 0 or (no_show + cancel) > 100:
            raise ValueError("No show % + Cancel % must be between 0 and 100.")

        otp = float(self.var_otp.get())
        if otp < 0 or otp > 100:
            raise ValueError("OTP must be between 0 and 100.")

        window = self._parse_int(self.var_window.get(), "Pickup window minutes", 0, 240)
        seed = self._parse_int(self.var_seed.get(), "Seed", -2_147_483_648, 2_147_483_647)

        n_clients = self._parse_int(self.var_clients.get(), "Clients", 1, 5_000_000)
        n_drivers = self._parse_int(self.var_drivers.get(), "Drivers", 1, 5000)
        n_vehicles = self._parse_int(self.var_vehicles.get(), "Vehicles", 1, 5000)
        n_routes = self._parse_int(self.var_routes.get(), "Routes/Runs", 1, 50000)

        fare_ada = float(self.var_fare_ada.get())
        fare_nemt = float(self.var_fare_nemt.get())
        fare_city = float(self.var_fare_city.get())
        s = fare_ada + fare_nemt + fare_city
        if abs(s - 100.0) > 0.01:
            raise ValueError(f"Fare type mix must sum to 100. Current sum is {s:.2f}.")

        return GeneratorConfig(
            n_rows=n_rows,
            start_date=start,
            n_days=n_days,
            no_show_pct=no_show,
            cancel_pct=cancel,
            otp_on_time_pct=otp,
            pickup_window_mins=window,
            seed=seed,
            n_clients=n_clients,
            n_drivers=n_drivers,
            n_vehicles=n_vehicles,
            n_routes=n_routes,
            fare_ada_pct=fare_ada,
            fare_nemt_pct=fare_nemt,
            fare_city_pct=fare_city,
        )

    def _generate(self):
        try:
            cfg = self._parse_inputs()
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return

        out_path = self.var_out.get().strip()
        if not out_path:
            self._browse()
            out_path = self.var_out.get().strip()
            if not out_path:
                return

        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            messagebox.showerror("Output error", "Output folder does not exist.")
            return

        self.progress.start(12)
        self.master.update_idletasks()

        try:
            synth = TripSynth(cfg)
            df = synth.generate()
            df.to_csv(out_path, index=False)

            performed = df[df["SchedStatusName"] == "Performed"]
            on_time_rate = (performed["OTP"] == 1).mean() if len(performed) else float("nan")
            counts = df["SchedStatusName"].value_counts()
            fare_counts = df["FareTypeAbbr"].value_counts()
            unique_clients = df["ClientId"].nunique()

            self.txt.delete("1.0", tk.END)
            self.txt.insert(tk.END, f"Saved: {out_path}\n\n")
            self.txt.insert(tk.END, "Status counts:\n")
            self.txt.insert(tk.END, counts.to_string() + "\n\n")
            self.txt.insert(tk.END, "Fare type counts:\n")
            self.txt.insert(tk.END, fare_counts.to_string() + "\n\n")
            self.txt.insert(tk.END, f"Unique clients in output: {unique_clients} (pool size: {cfg.n_clients})\n")
            self.txt.insert(tk.END, f"Performed OTP on-time rate: {on_time_rate:.3f}\n")
            self.txt.insert(tk.END, f"Drivers: {cfg.n_drivers}, Vehicles: {cfg.n_vehicles}, Routes: {cfg.n_routes}\n")
            self.txt.insert(tk.END, f"Pickup window mins: {cfg.pickup_window_mins}\n\n")
            self.txt.insert(tk.END, "Preview (first 25 rows):\n")
            self.txt.insert(tk.END, df.head(25).to_string(index=False))

            messagebox.showinfo("Done", f"Generated {len(df)} rows.\nSaved to:\n{out_path}")

        except Exception as e:
            messagebox.showerror("Generation error", str(e))
        finally:
            self.progress.stop()


def main():
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        style.theme_use("clam")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
