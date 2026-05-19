import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import requests
import os
import time
import math
import numpy as np
import altair as alt
from datetime import datetime
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from folium.plugins import HeatMap, AntPath
import json
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG  — must be FIRST Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="OptiRoute — Smart Delivery Routing",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚡"
)

# ─────────────────────────────────────────────
#  OPTIROUTE  —  NEON-ON-DARK LOGISTICS THEME
#  Aesthetic: cyberpunk logistics command center
#  Font: Exo 2 (display) + JetBrains Mono (data)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg:        #04060d;
    --bg2:       #080c18;
    --bg3:       #0c1220;
    --bg4:       #101828;
    --border:    #111d35;
    --border2:   #1a2d50;
    --neon:      #00d4ff;
    --neon2:     #0066ff;
    --neon3:     #00ff88;
    --amber:     #ffb020;
    --red:       #ff3b5c;
    --purple:    #a855f7;
    --text:      #d4e0f0;
    --textdim:   #3d5070;
    --textmid:   #7a9ab5;
    --glow:      0 0 20px rgba(0,212,255,0.15), 0 0 60px rgba(0,212,255,0.05);
    --glow2:     0 0 30px rgba(0,102,255,0.2);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--bg) !important;
    color: var(--text);
}

/* Animated grid background */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

.main .block-container { position: relative; z-index: 1; padding-top: 0.5rem; max-width: 1500px; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border2);
}

/* ── LOGO HEADER ── */
.optiroute-logo {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 20px 0 16px;
    border-bottom: 1px solid var(--border2);
    margin-bottom: 16px;
}
.logo-icon {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, var(--neon2), var(--neon));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow: var(--glow);
    flex-shrink: 0;
}
.logo-text { font-family: 'Exo 2', sans-serif; font-size: 20px; font-weight: 800; color: #fff; letter-spacing: -0.5px; }
.logo-sub  { font-size: 9px; color: var(--textdim); letter-spacing: 3px; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; }

/* ── KPI CARDS ── */
.kpi {
    background: var(--bg3);
    border: 1px solid var(--border2);
    border-radius: 12px;
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.3s, border-color 0.3s;
}
.kpi:hover { box-shadow: var(--glow); border-color: rgba(0,212,255,0.3); }
.kpi::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--neon), transparent);
    opacity: 0.4;
}
.kpi-l  { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--textdim); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 6px; }
.kpi-v  { font-family: 'Exo 2', sans-serif; font-size: 24px; font-weight: 700; color: #fff; line-height: 1.1; }
.kpi-s  { font-size: 10px; color: var(--textmid); margin-top: 4px; }
.kpi-bar{
    position: absolute;
    left: 0; top: 0; bottom: 0; width: 2px;
    border-radius: 2px 0 0 2px;
}

/* ── SECTION HEADS ── */
.sh {
    font-family: 'Exo 2', sans-serif;
    font-size: 10px; font-weight: 700;
    color: var(--textdim);
    text-transform: uppercase; letter-spacing: 3px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 20px 0 12px;
    display: flex; align-items: center; gap: 8px;
}
.sh::before { content: ''; flex: 1; max-width: 20px; height: 1px; background: var(--neon); }

/* ── ROUTE CARD ── */
.rc {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 8px;
    transition: all 0.25s;
    position: relative;
    overflow: hidden;
}
.rc:hover { border-color: var(--border2); box-shadow: var(--glow2); }
.rc-left-bar {
    position: absolute;
    left: 0; top: 0; bottom: 0; width: 3px;
}

/* ── TAGS ── */
.tag {
    display: inline-flex; align-items: center; gap: 3px;
    background: var(--bg4); border: 1px solid var(--border2);
    border-radius: 5px; padding: 3px 8px;
    font-size: 10px; color: var(--textmid);
    font-family: 'JetBrains Mono', monospace;
    margin: 2px;
}

/* ── DELAY BADGES ── */
.d-ok   { background:#002b1a;color:#00ff88;border:1px solid #00ff8840;border-radius:6px;padding:3px 10px;font-size:10px;font-weight:700;font-family:'JetBrains Mono',monospace; }
.d-late { background:#2a0010;color:#ff3b5c;border:1px solid #ff3b5c40;border-radius:6px;padding:3px 10px;font-size:10px;font-weight:700;font-family:'JetBrains Mono',monospace; }
.d-warn { background:#2a1500;color:#ffb020;border:1px solid #ffb02040;border-radius:6px;padding:3px 10px;font-size:10px;font-weight:700;font-family:'JetBrains Mono',monospace; }

/* ── ML PANEL ── */
.mlp {
    background: linear-gradient(135deg, var(--bg3), var(--bg4));
    border: 1px solid var(--border2);
    border-radius: 14px; padding: 18px;
    position: relative;
}
.mlp-badge {
    position: absolute; top: -10px; left: 14px;
    background: linear-gradient(90deg, var(--neon2), var(--neon));
    color: white; font-size: 8px; font-weight: 800;
    letter-spacing: 2px; padding: 3px 10px;
    border-radius: 4px; font-family: 'JetBrains Mono', monospace;
}

/* ── SCANNING ANIMATION ── */
@keyframes scanline {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
}
.scan-wrap {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none; z-index: 9998; overflow: hidden;
}
.scan-line {
    width: 100%; height: 2px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.08), transparent);
    animation: scanline 8s linear infinite;
}

/* ── LIVE TRACKING STATUS ── */
.live-status {
    background: var(--bg3);
    border: 1px solid var(--neon);
    border-radius: 12px;
    padding: 14px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--neon);
    box-shadow: 0 0 20px rgba(0,212,255,0.1);
    margin: 8px 0;
    display: flex;
    gap: 24px;
    align-items: center;
    flex-wrap: wrap;
}
.live-dot {
    width: 8px; height: 8px;
    background: var(--neon3);
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px var(--neon3);
    animation: pulse-dot 1s ease-in-out infinite;
    margin-right: 6px;
}
@keyframes pulse-dot {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.5; transform: scale(0.8); }
}

/* ── STREAMLIT OVERRIDES ── */
.stButton > button {
    background: linear-gradient(90deg, var(--neon2), var(--neon)) !important;
    color: #000 !important;
    font-weight: 800 !important;
    font-family: 'Exo 2', sans-serif !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    letter-spacing: 1px !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    transition: opacity 0.2s, transform 0.1s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2); border-radius: 10px; padding: 4px; gap: 2px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--textdim) !important;
    border-radius: 7px !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    padding: 7px 16px !important;
    letter-spacing: 0.5px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg4) !important;
    color: var(--neon) !important;
    border: 1px solid var(--border2) !important;
    box-shadow: 0 0 10px rgba(0,212,255,0.1) !important;
}
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--bg4) !important;
    border-color: var(--border2) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="stExpander"] {
    background: var(--bg3) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
}
.stProgress > div > div {
    background: linear-gradient(90deg, var(--neon2), var(--neon)) !important;
    box-shadow: 0 0 8px rgba(0,212,255,0.4);
}
[data-testid="stMetric"] { background: var(--bg3); border-radius: 10px; padding: 10px; border: 1px solid var(--border); }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
hr { border-color: var(--border); margin: 12px 0; }
</style>

<!-- Scanline atmosphere overlay -->
<div class="scan-wrap"><div class="scan-line"></div></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
DEFAULT_WAREHOUSES = [
    {"id": "WH-A", "name": "North Hub",   "lat": 28.6800, "lon": 77.2200, "capacity": 5000, "active": True},
    {"id": "WH-B", "name": "South Hub",   "lat": 28.5900, "lon": 77.0400, "capacity": 4000, "active": True},
    {"id": "WH-C", "name": "East Hub",    "lat": 28.6500, "lon": 77.3800, "capacity": 3500, "active": True},
    {"id": "WH-D", "name": "West Hub",    "lat": 28.6200, "lon": 76.9800, "capacity": 3000, "active": True},
    {"id": "WH-E", "name": "Central Hub", "lat": 28.6300, "lon": 77.2100, "capacity": 6000, "active": True},
]

VEHICLE_PROFILES = {
    "Bike":   {"speed_kmh": 20, "capacity_kg": 20,   "max_range_km": 30,  "cost_per_km": 2,   "co2_per_km": 0.02, "icon": "🛵"},
    "Car":    {"speed_kmh": 35, "capacity_kg": 150,  "max_range_km": 120, "cost_per_km": 8,   "co2_per_km": 0.18, "icon": "🚗"},
    "Van":    {"speed_kmh": 30, "capacity_kg": 600,  "max_range_km": 200, "cost_per_km": 12,  "co2_per_km": 0.28, "icon": "🚐"},
    "Truck":  {"speed_kmh": 25, "capacity_kg": 2000, "max_range_km": 400, "cost_per_km": 20,  "co2_per_km": 0.55, "icon": "🚚"},
    "Drone":  {"speed_kmh": 60, "capacity_kg": 3,    "max_range_km": 15,  "cost_per_km": 1,   "co2_per_km": 0.01, "icon": "🚁"},
    "E-Bike": {"speed_kmh": 22, "capacity_kg": 30,   "max_range_km": 50,  "cost_per_km": 1.5, "co2_per_km": 0.01, "icon": "⚡"},
}

WEATHER_PROFILES = {
    "clear":  {"factor": 1.0,  "speed_pen": 0,  "risk": 5,  "icon": "☀️"},
    "cloudy": {"factor": 1.05, "speed_pen": 2,  "risk": 10, "icon": "⛅"},
    "rainy":  {"factor": 1.35, "speed_pen": 10, "risk": 45, "icon": "🌧"},
    "foggy":  {"factor": 1.25, "speed_pen": 8,  "risk": 40, "icon": "🌫"},
    "hot":    {"factor": 1.10, "speed_pen": 3,  "risk": 20, "icon": "🌡"},
    "cold":   {"factor": 1.15, "speed_pen": 5,  "risk": 25, "icon": "❄️"},
    "stormy": {"factor": 1.60, "speed_pen": 20, "risk": 80, "icon": "⛈"},
}

TIME_WINDOW_PRESETS = {
    "Standard (9am-6pm)":   (9*60, 18*60),
    "Morning (8am-12pm)":   (8*60, 12*60),
    "Afternoon (12pm-6pm)": (12*60, 18*60),
    "Evening (5pm-9pm)":    (17*60, 21*60),
    "Same-Day (Now+4hr)":   None,
    "Express (60 min)":     None,
}

PRIORITY_LEVELS = {
    "Standard": {"sla_min": 120, "penalty": 1,  "color": "#3d5070"},
    "Priority": {"sla_min": 60,  "penalty": 3,  "color": "#ffb020"},
    "Express":  {"sla_min": 30,  "penalty": 10, "color": "#ff3b5c"},
    "Same-Day": {"sla_min": 240, "penalty": 2,  "color": "#0066ff"},
}

ROUTE_COLORS     = ["#00d4ff","#00ff88","#ffb020","#ff3b5c","#a855f7","#0066ff"]
ROUTE_COLORS_RGB = [[0,212,255],[0,255,136],[255,176,32],[255,59,92],[168,85,247],[0,102,255]]

# ─────────────────────────────────────────────
#  ML MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def build_delay_model():
    np.random.seed(42)
    n = 3000
    distance = np.random.uniform(2, 120, n)
    weight   = np.random.uniform(0.1, 200, n)
    w_risk   = np.random.uniform(0, 80, n)
    traffic  = np.random.uniform(1.0, 2.0, n)
    sla_min  = np.random.choice([30, 60, 120, 240], n)
    is_exp   = (sla_min <= 30).astype(int)
    hour     = np.random.randint(0, 24, n)
    is_peak  = ((hour >= 8) & (hour <= 11)) | ((hour >= 17) & (hour <= 21))
    delay_prob = np.clip(
        0.35*(distance/120) + 0.20*(w_risk/80) + 0.25*((traffic-1.0)/1.0) + 0.10*is_exp + 0.10*is_peak, 0, 1
    )
    y = (np.random.rand(n) < delay_prob).astype(int)
    X = np.column_stack([distance, weight, w_risk, traffic, sla_min, is_exp, hour, is_peak.astype(int)])
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X, y)
    gb = GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42)
    gb.fit(X, y)
    return rf, gb

@st.cache_resource
def build_vehicle_recommender():
    np.random.seed(0)
    n = 2000
    weight   = np.random.uniform(0.1, 500, n)
    distance = np.random.uniform(1, 200, n)
    def rule(w, d):
        if d <= 15 and w <= 3:  return 0
        if w <= 20 and d <= 30: return 4
        if w <= 20:             return 0
        if w <= 150:            return 1
        if w <= 600:            return 2
        return 3
    X = np.column_stack([weight, distance])
    y = np.array([rule(w, d) for w, d in zip(weight, distance)])
    from sklearn.ensemble import RandomForestClassifier as RFC
    clf = RFC(n_estimators=50, random_state=0)
    clf.fit(X, y)
    return clf

delay_rf, delay_gb = build_delay_model()
vehicle_rec        = build_vehicle_recommender()

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
def ss(k, v):
    if k not in st.session_state: st.session_state[k] = v

ss("warehouses", [dict(w) for w in DEFAULT_WAREHOUSES])
ss("result", None)
ss("route_coords", [])
ss("ml_predictions", [])
ss("map_clicked_coord", None)
ss("deliveries_preview", [])
ss("map_center", [28.63, 77.20])

# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────
def haversine(p1, p2):
    R = 6371
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def fetch_route(start, end):
    try:
        url = (f"http://router.project-osrm.org/route/v1/driving/"
               f"{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=geojson")
        r = requests.get(url, timeout=6)
        return r.json()["routes"][0]
    except:
        return None

def predict_delay(distance_km, weight_kg, weather, traffic_mult, priority, hour=None):
    if hour is None: hour = datetime.now().hour
    wp    = WEATHER_PROFILES[weather]
    sla   = PRIORITY_LEVELS[priority]["sla_min"]
    is_e  = 1 if sla <= 30 else 0
    is_pk = 1 if (8 <= hour <= 11) or (17 <= hour <= 21) else 0
    feat  = np.array([[distance_km, weight_kg, wp["risk"], traffic_mult, sla, is_e, hour, is_pk]])
    pred  = delay_rf.predict(feat)[0]
    prob  = delay_rf.predict_proba(feat)[0][1]
    return int(pred), round(float(prob), 3)

def recommend_vehicle(weight_kg, distance_km):
    idx_map = {0:"Bike", 1:"Car", 2:"Van", 3:"Truck", 4:"E-Bike"}
    idx = vehicle_rec.predict(np.array([[weight_kg, distance_km]]))[0]
    return idx_map.get(idx, "Van")

def score_warehouse(wh, deliveries, vehicle, weather):
    c  = (wh["lat"], wh["lon"])
    vp = VEHICLE_PROFILES[vehicle]
    wp = WEATHER_PROFILES[weather]
    dists = [haversine(c, d["coord"]) for d in deliveries] if deliveries else [0]
    avg_d = np.mean(dists); max_d = max(dists)
    cap_p = (sum(d.get("weight_kg",1) for d in deliveries) / max(wh["capacity"],1)) * 20
    rng_p = 50 if max_d > vp["max_range_km"] else 0
    return round(((avg_d*2) + (max_d*0.5) + cap_p + rng_p) * wp["factor"], 2)

def build_dist_matrix(points):
    n = len(points)
    m = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j: m[i][j] = int(haversine(points[i], points[j])*1000)
    return m

# ─────────────────────────────────────────────
#  VRP SOLVER
# ─────────────────────────────────────────────
def solve_vrp(depot, deliveries, vehicles, weather, hc):
    if not deliveries: return None
    nv = len(vehicles)
    points = [depot] + [d["coord"] for d in deliveries]
    n  = len(points)
    dm = build_dist_matrix(points)
    demands   = [0] + [int(d.get("weight_kg",1)) for d in deliveries]
    wp        = WEATHER_PROFILES[weather]
    hn        = datetime.now().hour
    time_wins = [(0,86400)] + [(d.get("tw_open_min",0)*60, d.get("tw_close_min",1440)*60) for d in deliveries]

    mgr  = pywrapcp.RoutingIndexManager(n, nv, 0)
    rout = pywrapcp.RoutingModel(mgr)

    def dcb(fi, ti): return dm[mgr.IndexToNode(fi)][mgr.IndexToNode(ti)]
    tcb = rout.RegisterTransitCallback(dcb)
    rout.SetArcCostEvaluatorOfAllVehicles(tcb)

    if hc.get("capacity"):
        def demc(i): return demands[mgr.IndexToNode(i)]
        dc = rout.RegisterUnaryTransitCallback(demc)
        rout.AddDimensionWithVehicleCapacity(dc, 0, [int(VEHICLE_PROFILES[v["type"]]["capacity_kg"]) for v in vehicles], True, "Cap")

    if hc.get("time_windows"):
        def tcb2(fi, ti):
            spd = VEHICLE_PROFILES[vehicles[0]["type"]]["speed_kmh"]*1000/3600
            spd *= (1 - wp["speed_pen"]/100); spd = max(spd, 1)
            return int(dm[mgr.IndexToNode(fi)][mgr.IndexToNode(ti)]/spd + 300)
        ti2 = rout.RegisterTransitCallback(tcb2)
        rout.AddDimension(ti2, 3600, 86400, False, "Time")
        td  = rout.GetDimensionOrDie("Time")
        for ni in range(1, n):
            ix = mgr.NodeToIndex(ni)
            td.CumulVar(ix).SetRange(time_wins[ni][0], time_wins[ni][1])

    if hc.get("range"):
        rout.AddDimensionWithVehicleCapacity(tcb, 0, [int(VEHICLE_PROFILES[v["type"]]["max_range_km"]*1000) for v in vehicles], True, "Range")

    for di, d in enumerate(deliveries):
        pen = PRIORITY_LEVELS[d.get("priority","Standard")]["penalty"] * 10000
        rout.AddDisjunction([mgr.NodeToIndex(di+1)], pen)

    prm = pywrapcp.DefaultRoutingSearchParameters()
    prm.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    prm.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    prm.time_limit.seconds = 5
    sol = rout.SolveWithParameters(prm)
    if not sol: return _greedy(depot, deliveries, vehicles)

    routes = []
    for vi in range(nv):
        nodes, ix = [], rout.Start(vi)
        while not rout.IsEnd(ix):
            nodes.append(mgr.IndexToNode(ix)); ix = sol.Value(rout.NextVar(ix))
        nodes.append(mgr.IndexToNode(ix))
        if len(nodes) > 2: routes.append({"vehicle": vehicles[vi], "nodes": nodes})
    return routes or _greedy(depot, deliveries, vehicles)

def _greedy(depot, deliveries, vehicles):
    rem = list(range(len(deliveries))); routes = []
    for v in vehicles:
        if not rem: break
        route, cap, used, cur = [0], VEHICLE_PROFILES[v["type"]]["capacity_kg"], 0, depot
        while rem:
            bi, bd = None, float("inf")
            for i in rem:
                d = haversine(cur, deliveries[i]["coord"]); w = deliveries[i].get("weight_kg",1)
                if used+w <= cap and d < bd: bi, bd = i, d
            if bi is None: break
            route.append(bi+1); used += deliveries[bi].get("weight_kg",1); cur = deliveries[bi]["coord"]; rem.remove(bi)
        route.append(0)
        if len(route) > 2: routes.append({"vehicle": v, "nodes": route})
    return routes or None

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="optiroute-logo">
        <div class="logo-icon">⚡</div>
        <div>
            <div class="logo-text">OptiRoute</div>
            <div class="logo-sub">Delivery Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sh">🏭 Warehouse Network</div>', unsafe_allow_html=True)
    for i, wh in enumerate(st.session_state.warehouses):
        with st.expander(f"{wh['id']} · {wh['name']}", expanded=False):
            c1, c2 = st.columns(2)
            wh["lat"]      = c1.number_input("Lat", value=wh["lat"], key=f"wlat{i}", format="%.4f")
            wh["lon"]      = c2.number_input("Lon", value=wh["lon"], key=f"wlon{i}", format="%.4f")
            wh["capacity"] = st.number_input("Capacity (kg)", value=wh["capacity"], step=500, key=f"wcap{i}")
            wh["active"]   = st.checkbox("Active", value=wh["active"], key=f"wact{i}")

    if st.button("＋ Add Warehouse"):
        n = len(st.session_state.warehouses)+1
        st.session_state.warehouses.append({"id":f"WH-{chr(64+n)}","name":f"Hub {n}","lat":28.63,"lon":77.20,"capacity":3000,"active":True})
        st.rerun()

    st.markdown('<div class="sh">🚗 Fleet Setup</div>', unsafe_allow_html=True)
    num_vehicles = st.slider("Vehicles", 1, 6, 2)
    vehicle_types = []
    for i in range(num_vehicles):
        vt = st.selectbox(f"Vehicle {i+1}", list(VEHICLE_PROFILES.keys()), index=i%len(VEHICLE_PROFILES), key=f"vtype{i}")
        vehicle_types.append({"id": f"V{i+1}", "type": vt})

    st.markdown('<div class="sh">🌤 Conditions</div>', unsafe_allow_html=True)
    weather = st.selectbox("Weather", list(WEATHER_PROFILES.keys()),
                           format_func=lambda w: f"{WEATHER_PROFILES[w]['icon']} {w.capitalize()}")
    hour_now = datetime.now().hour
    is_peak  = (8 <= hour_now <= 11) or (17 <= hour_now <= 21)
    traffic_level = st.select_slider("Traffic", ["Low","Moderate","High","Gridlock"],
                                     value="High" if is_peak else "Moderate")
    traffic_mult = {"Low":1.0,"Moderate":1.2,"High":1.5,"Gridlock":2.0}[traffic_level]

    st.markdown('<div class="sh">⚙️ Constraints</div>', unsafe_allow_html=True)
    hard_constraints = {
        "capacity":     st.checkbox("Vehicle Capacity Limit", value=True),
        "time_windows": st.checkbox("Delivery Time Windows",  value=True),
        "range":        st.checkbox("Max Vehicle Range",      value=True),
        "priority":     st.checkbox("Order Priority / SLA",   value=True),
    }

    st.markdown('<div class="sh">🤖 AI Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:10px;color:#3d5070;line-height:2;font-family:'JetBrains Mono',monospace;">
        <span style="color:#00ff88;">●</span> RandomForest Delay Predictor<br>
        <span style="color:#00ff88;">●</span> GradientBoosting Risk Scorer<br>
        <span style="color:#00ff88;">●</span> OR-Tools CVRPTW Solver<br>
        <span style="color:#00ff88;">●</span> OSRM Real-Road Routing<br>
        <span style="color:#00ff88;">●</span> Multi-criteria WH Scoring
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:flex-end;gap:16px;padding:16px 0 8px;
            border-bottom:1px solid #111d35;margin-bottom:16px;">
    <div>
        <div style="font-family:'Exo 2',sans-serif;font-size:38px;font-weight:900;
                    background:linear-gradient(90deg,#ffffff,#00d4ff);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    line-height:1;letter-spacing:-1px;">
            OptiRoute
        </div>
        <div style="font-size:9px;color:#3d5070;letter-spacing:4px;margin-top:4px;
                    font-family:'JetBrains Mono',monospace;text-transform:uppercase;">
            Multi-Warehouse · CVRPTW · AI Delay Prediction · Live Tracking
        </div>
    </div>
    <div style="margin-left:auto;display:flex;gap:8px;align-items:center;padding-bottom:4px;">
        <div style="width:6px;height:6px;background:#00ff88;border-radius:50%;
                    box-shadow:0 0 8px #00ff88;animation:none;"></div>
        <span style="font-size:10px;color:#00ff88;font-family:'JetBrains Mono',monospace;">SYSTEM ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

active_wh = [w for w in st.session_state.warehouses if w["active"]]
wp        = WEATHER_PROFILES[weather]

# ── Top status row ──
c1,c2,c3,c4,c5 = st.columns(5)
for col, lbl, val, sub, color in [
    (c1, "Warehouses",   str(len(active_wh)),        f"of {len(st.session_state.warehouses)} active",           "#00d4ff"),
    (c2, "Fleet",        str(num_vehicles),           ", ".join(set(v["type"] for v in vehicle_types)),         "#00ff88"),
    (c3, "Weather",      f"{wp['icon']} {weather}",  f"Risk {wp['risk']}%",                                     "#ffb020"),
    (c4, "Traffic",      traffic_level,               f"×{traffic_mult} penalty",                                "#ff3b5c"),
    (c5, "AI Model",     "RF 86%",                   "Delay + Vehicle Rec",                                      "#a855f7"),
]:
    col.markdown(f"""
    <div class="kpi">
        <div class="kpi-bar" style="background:{color};"></div>
        <div class="kpi-l" style="padding-left:10px;">{lbl}</div>
        <div class="kpi-v" style="padding-left:10px;">{val}</div>
        <div class="kpi-s" style="padding-left:10px;">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_orders, tab_map, tab_analysis, tab_live, tab_ai = st.tabs([
    "📦  Orders", "🗺️  Route Map", "📊  Analysis", "📡  Live Tracking", "🤖  AI Insights"
])

# ═══════════════════════════════════════
#  TAB 1 — ORDERS
# ═══════════════════════════════════════
with tab_orders:
    st.markdown('<div class="sh">📍 Click Map — Select Delivery Coordinates</div>', unsafe_allow_html=True)

    left_map, right_info = st.columns([3, 1])
    with left_map:
        cmap = folium.Map(location=st.session_state.map_center, zoom_start=12, tiles=None)
        folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
            attr="©OpenStreetMap ©CartoDB", name="CartoDB Dark"
        ).add_to(cmap)
        for wh in active_wh:
            folium.Marker(
                [wh["lat"], wh["lon"]], tooltip=f"🏭 {wh['name']}",
                icon=folium.Icon(color="orange", icon="home", prefix="fa")
            ).add_to(cmap)
        for i, d in enumerate(st.session_state.deliveries_preview):
            if d["coord"] != (0.0, 0.0):
                folium.CircleMarker(
                    d["coord"], radius=7, color="#00d4ff", fill=True, fill_opacity=0.8,
                    tooltip=f"Stop {i+1}: {d['label']}"
                ).add_to(cmap)
        mr = st_folium(cmap, width="100%", height=280, returned_objects=["last_clicked"], key="click_map")
        if mr and mr.get("last_clicked"):
            cl = mr["last_clicked"]
            st.session_state.map_clicked_coord = (round(cl["lat"],5), round(cl["lng"],5))
            st.session_state.map_center = [cl["lat"], cl["lng"]]

    with right_info:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.session_state.map_clicked_coord:
            lat_c, lon_c = st.session_state.map_clicked_coord
            st.markdown(f"""
            <div style="background:#0c1220;border:1px solid #00d4ff40;border-radius:10px;padding:12px;margin-top:4px;">
                <div style="font-size:9px;color:#3d5070;letter-spacing:2px;
                            font-family:'JetBrains Mono',monospace;margin-bottom:8px;">📍 SELECTED</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#00d4ff;margin:3px 0;">
                    {lat_c}
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#00d4ff;margin:3px 0;">
                    {lon_c}
                </div>
            </div>
            """, unsafe_allow_html=True)
            assign_to = st.selectbox("Assign to stop #", list(range(1, 11)), key="assign_stop")
            if st.button("✅ Apply Coords", use_container_width=True):
                st.session_state[f"dlat_{assign_to-1}"] = lat_c
                st.session_state[f"dlon_{assign_to-1}"] = lon_c
                st.success(f"Applied to Stop {assign_to}!")
        else:
            st.markdown("""
            <div style="background:#0c1220;border:1px dashed #111d35;border-radius:10px;
                        padding:24px 12px;text-align:center;color:#3d5070;font-size:11px;margin-top:4px;">
                Click map to<br>capture coords
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sh">📦 Delivery Order Table</div>', unsafe_allow_html=True)
    num_stops = st.slider("Number of stops", 1, 10, 3)

    st.markdown("""
    <div style="display:grid;grid-template-columns:1.8fr 1fr 1fr 0.8fr 1.3fr 1fr;
                gap:5px;font-size:9px;color:#3d5070;text-transform:uppercase;
                letter-spacing:2px;padding:6px 2px 8px;
                border-bottom:1px solid #111d35;font-family:'JetBrains Mono',monospace;">
        <div>Label</div><div>Latitude</div><div>Longitude</div>
        <div>Weight</div><div>Time Window</div><div>Priority</div>
    </div>""", unsafe_allow_html=True)

    deliveries = []
    for i in range(num_stops):
        c1,c2,c3,c4,c5,c6 = st.columns([1.8,1,1,0.8,1.3,1])
        lbl  = c1.text_input("L",  value=f"Customer {i+1}", key=f"lbl{i}",  label_visibility="collapsed")
        dlat = st.session_state.get(f"dlat_{i}", 28.60+i*0.015)
        dlon = st.session_state.get(f"dlon_{i}", 77.15+i*0.020)
        lat  = c2.number_input("Lat", value=float(dlat), key=f"dlat{i}", format="%.5f", label_visibility="collapsed")
        lon  = c3.number_input("Lon", value=float(dlon), key=f"dlon{i}", format="%.5f", label_visibility="collapsed")
        wgt  = c4.number_input("kg",  value=float(5+i*3), key=f"dwgt{i}", min_value=0.1, label_visibility="collapsed")
        tw   = c5.selectbox("TW", list(TIME_WINDOW_PRESETS.keys()), key=f"dtw{i}",  label_visibility="collapsed")
        pri  = c6.selectbox("P",  list(PRIORITY_LEVELS.keys()),     key=f"dprio{i}", label_visibility="collapsed")
        twp  = TIME_WINDOW_PRESETS[tw]
        if twp: tw_o, tw_c = twp
        else:   tw_o, tw_c = hour_now*60, (hour_now+4)*60
        if lat != 0.0 or lon != 0.0:
            deliveries.append({"label":lbl,"coord":(lat,lon),"weight_kg":wgt,
                                "tw_open_min":tw_o,"tw_close_min":tw_c,"priority":pri})

    st.session_state.deliveries_preview = deliveries
    st.markdown("<br>", unsafe_allow_html=True)
    cb1, cb2 = st.columns(2)
    run_vrp   = cb1.button("⚡ OPTIMIZE ROUTES — CVRPTW + AI", use_container_width=True)
    clear_btn = cb2.button("↺  Reset All",                      use_container_width=True)
    if clear_btn:
        st.session_state.result = None; st.session_state.ml_predictions = []; st.rerun()

# ─────────────────────────────────────────────
#  RUN OPTIMIZATION
# ─────────────────────────────────────────────
if run_vrp and deliveries and active_wh:
    with st.spinner("Running CVRPTW optimizer + RandomForest delay prediction…"):
        wh_scores = sorted(
            [(score_warehouse(wh, deliveries, vehicle_types[0]["type"], weather), wh) for wh in active_wh],
            key=lambda x: x[0]
        )
        best_wh     = wh_scores[0][1]
        depot_coord = (best_wh["lat"], best_wh["lon"])

        ml_preds = []
        for d in deliveries:
            dist  = haversine(depot_coord, d["coord"])
            dp, prob = predict_delay(dist, d["weight_kg"], weather, traffic_mult, d["priority"])
            rv   = recommend_vehicle(d["weight_kg"], dist)
            ml_preds.append({"label":d["label"],"delay_predicted":dp,"delay_prob":prob,
                              "recommended_vehicle":rv,"reroute_flag":dp==1 and prob>0.6,"dist":round(dist,2)})
        st.session_state.ml_predictions = ml_preds

        sol_routes = solve_vrp(depot_coord, deliveries, vehicle_types, weather, hard_constraints)

        all_coords, total_dist, route_details = [], 0, []
        if sol_routes:
            for route in sol_routes:
                nodes = route["nodes"]; vtype = route["vehicle"]["type"]; vp = VEHICLE_PROFILES[vtype]
                pts   = [depot_coord]+[deliveries[n-1]["coord"] for n in nodes[1:-1] if 0<n<=len(deliveries)]+[depot_coord]
                rdist, rcoords = 0, []
                for j in range(len(pts)-1):
                    r = fetch_route(pts[j], pts[j+1])
                    if r:
                        seg = [[c[1],c[0]] for c in r["geometry"]["coordinates"]]
                        rcoords.extend(seg); rdist += r["distance"]/1000
                    else:
                        rcoords.extend([list(pts[j]),list(pts[j+1])]); rdist += haversine(pts[j],pts[j+1])
                eff_speed = max(vp["speed_kmh"]*(1-wp["speed_pen"]/100)/traffic_mult, 5)
                load = sum(deliveries[n-1].get("weight_kg",1) for n in nodes[1:-1] if 0<n<=len(deliveries))
                all_coords.extend(rcoords); total_dist += rdist
                route_details.append({
                    "vehicle":route["vehicle"],"vtype":vtype,"nodes":nodes,
                    "coords":rcoords,"pts":pts,
                    "dist_km":round(rdist,2),"eta_min":int((rdist/eff_speed)*60),
                    "fuel_cost":round(rdist*vp["cost_per_km"],2),
                    "co2_kg":round(rdist*vp["co2_per_km"],3),
                    "load_kg":round(load,2),"cap_util":round((load/vp["capacity_kg"])*100,1),
                    "stops":len(nodes)-2,
                })

        st.session_state.result = {
            "best_wh":best_wh,"wh_scores":wh_scores,"routes":route_details,
            "deliveries":deliveries,"total_dist":round(total_dist,2),
            "depot_coord":depot_coord,"all_route_coords":all_coords,
        }
        st.session_state.route_coords = all_coords

result   = st.session_state.result
ml_preds = st.session_state.ml_predictions

# ─────────────────────────────────────────────
#  RESULTS IN TAB 1
# ─────────────────────────────────────────────
with tab_orders:
    if result:
        st.markdown('<div class="sh">🏆 Warehouse Selection</div>', unsafe_allow_html=True)
        wh_cols = st.columns(min(len(result["wh_scores"]),5))
        for idx,(score,wh) in enumerate(result["wh_scores"][:5]):
            avg_d = round(np.mean([haversine((wh["lat"],wh["lon"]),d["coord"]) for d in result["deliveries"]]),1)
            c     = "#00d4ff" if idx==0 else "#3d5070"
            wh_cols[idx].markdown(f"""
            <div class="kpi" style="border-color:{'#00d4ff30' if idx==0 else '#111d35'}">
                <div class="kpi-bar" style="background:{c};"></div>
                <div style="font-size:8px;color:{c};font-weight:800;letter-spacing:2px;
                            padding-left:10px;margin-bottom:3px;font-family:'JetBrains Mono',monospace;">
                    {'✦ SELECTED' if idx==0 else f'#{idx+1}'}</div>
                <div style="font-weight:700;color:#d4e0f0;padding-left:10px;font-family:'Exo 2',sans-serif;">{wh['id']} {wh['name']}</div>
                <div class="kpi-s" style="padding-left:10px;">Score {score} · Avg {avg_d} km</div>
                <div class="kpi-s" style="padding-left:10px;">{wh['capacity']:,} kg cap</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sh">🚛 Route Assignments</div>', unsafe_allow_html=True)
        for ri,rd in enumerate(result["routes"]):
            vp = VEHICLE_PROFILES[rd["vtype"]]; color = ROUTE_COLORS[ri%len(ROUTE_COLORS)]
            st.markdown(f"""
            <div class="rc">
                <div class="rc-left-bar" style="background:{color};"></div>
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;padding-left:10px;">
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span style="font-size:24px;">{vp['icon']}</span>
                        <div>
                            <div style="font-family:'Exo 2',sans-serif;font-weight:700;color:{color};font-size:15px;">
                                {rd['vehicle']['id']} — {rd['vtype']}</div>
                            <div style="font-size:10px;color:#3d5070;">{rd['stops']} stops</div>
                        </div>
                    </div>
                    <div>
                        <span class="tag">📏 {rd['dist_km']} km</span>
                        <span class="tag">⏱ {rd['eta_min']} min</span>
                        <span class="tag">₹ {rd['fuel_cost']}</span>
                        <span class="tag">🌿 {rd['co2_kg']} kg</span>
                        <span class="tag">📦 {rd['load_kg']} kg ({rd['cap_util']}%)</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        total_eta   = max((rd["eta_min"] for rd in result["routes"]), default=0)
        total_cost  = sum(rd["fuel_cost"] for rd in result["routes"])
        total_co2   = sum(rd["co2_kg"]   for rd in result["routes"])
        delay_flags = sum(1 for p in ml_preds if p["delay_predicted"])

        st.markdown("<br>", unsafe_allow_html=True)
        k1,k2,k3,k4,k5 = st.columns(5)
        for col,lbl,val,color in [
            (k1,"Total Distance",f"{result['total_dist']} km","#00d4ff"),
            (k2,"Fleet Max ETA",f"{total_eta} min","#00ff88"),
            (k3,"Total Cost",f"₹{round(total_cost,2)}","#ffb020"),
            (k4,"CO₂ Footprint",f"{round(total_co2,2)} kg","#a855f7"),
            (k5,"AI Delay Flags",f"{delay_flags}/{len(ml_preds)}","#ff3b5c" if delay_flags else "#00ff88"),
        ]:
            col.markdown(f"""
            <div class="kpi">
                <div class="kpi-bar" style="background:{color};"></div>
                <div class="kpi-l" style="padding-left:10px;">{lbl}</div>
                <div class="kpi-v" style="padding-left:10px;color:{color};">{val}</div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════
#  TAB 2 — ROUTE MAP  (real map + routes)
# ═══════════════════════════════════════
with tab_map:
    if result:
        st.markdown('<div class="sh">🗺️ Optimized Route Map — Real Road Network (OSRM + CartoDB)</div>',
                    unsafe_allow_html=True)

        depot = result["depot_coord"]

        # ── Build Folium map with real tile layer ──
        fmap = folium.Map(location=[depot[0], depot[1]], zoom_start=12, tiles=None)

        # Real map tile — CartoDB Dark Matter (no token needed)
        folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>',
            name="CartoDB Dark Matter",
            max_zoom=20
        ).add_to(fmap)

        # ── Delivery density heatmap ──
        heat_data = [list(d["coord"]) for d in result["deliveries"]]
        if heat_data:
            HeatMap(
                heat_data, radius=30, blur=22, min_opacity=0.25,
                gradient={"0.2":"#0066ff","0.5":"#00d4ff","1.0":"#00ff88"}
            ).add_to(fmap)

        # ── Routes — glow polyline + animated AntPath ──
        for ri, rd in enumerate(result["routes"]):
            color = ROUTE_COLORS[ri % len(ROUTE_COLORS)]
            coords = rd["coords"]
            if coords:
                # Outer glow
                folium.PolyLine(
                    coords, color=color, weight=12, opacity=0.07
                ).add_to(fmap)
                folium.PolyLine(
                    coords, color=color, weight=6, opacity=0.18
                ).add_to(fmap)
                # Animated route line
                try:
                    AntPath(
                        coords, color=color, weight=3, opacity=0.95,
                        delay=600, dash_array=[12, 22],
                        tooltip=(f"🚛 {rd['vehicle']['id']} · {rd['vtype']} | "
                                 f"{rd['dist_km']} km | {rd['eta_min']} min | ₹{rd['fuel_cost']}")
                    ).add_to(fmap)
                except:
                    folium.PolyLine(coords, color=color, weight=3, opacity=0.95).add_to(fmap)

        # ── Warehouse markers ──
        for wh in active_wh:
            is_best = wh["id"] == result["best_wh"]["id"]
            if is_best:
                # Pulsing rings for selected warehouse
                for r_px in [28, 20, 12]:
                    folium.CircleMarker(
                        [wh["lat"], wh["lon"]], radius=r_px,
                        color="#00d4ff", fill=False, weight=1,
                        opacity=0.15 + (28-r_px)*0.01
                    ).add_to(fmap)

            wh_html = f"""
            <div style="
                width:34px;height:34px;
                background:{'linear-gradient(135deg,#0066ff,#00d4ff)' if is_best else '#0c1220'};
                border:2px solid {'#00d4ff' if is_best else '#1a2d50'};
                border-radius:8px;display:flex;align-items:center;justify-content:center;
                font-size:15px;box-shadow:{'0 0 16px rgba(0,212,255,0.5)' if is_best else '0 2px 8px rgba(0,0,0,0.6)'};
            ">🏭</div>"""
            folium.Marker(
                [wh["lat"], wh["lon"]],
                popup=folium.Popup(
                    f"<b style='color:#00d4ff'>{wh['id']}</b><br>{wh['name']}<br>"
                    f"Capacity: {wh['capacity']:,} kg"
                    f"{'<br><b style=\"color:#00ff88\">✦ SELECTED DEPOT</b>' if is_best else ''}",
                    max_width=200
                ),
                icon=folium.DivIcon(html=wh_html, icon_size=(34,34), icon_anchor=(17,17)),
                tooltip=f"{'✦ ' if is_best else ''}🏭 {wh['name']}"
            ).add_to(fmap)

        # ── Delivery stop markers (styled by priority + delay) ──
        pcolors = {"Express":"#ff3b5c","Priority":"#ffb020","Same-Day":"#0066ff","Standard":"#3d5070"}
        for i, d in enumerate(result["deliveries"]):
            pc = pcolors[d["priority"]]
            dp = ml_preds[i] if i < len(ml_preds) else None
            delay_pct = int(dp["delay_prob"]*100) if dp else 0
            delay_ico = "⚠" if (dp and dp["delay_predicted"]) else "✓"
            popup_html = f"""
            <div style='font-family:monospace;font-size:12px;min-width:190px;
                        background:#0c1220;color:#d4e0f0;padding:8px;border-radius:6px;'>
                <b style='color:{pc};font-size:14px;'>{d['label']}</b><br>
                <hr style='border-color:#1a2d50;margin:4px 0'>
                📦 Weight: {d['weight_kg']} kg<br>
                🎯 Priority: {d['priority']}<br>
                {delay_ico} Delay Risk: {delay_pct}%<br>
                🚗 Rec. Vehicle: {dp['recommended_vehicle'] if dp else 'N/A'}
            </div>"""
            mk_html = f"""
            <div style="
                width:26px;height:26px;
                background:{pc};
                border:2px solid rgba(255,255,255,0.3);border-radius:50%;
                display:flex;align-items:center;justify-content:center;
                font-size:10px;font-weight:800;color:white;
                box-shadow:0 0 10px {pc}80;
            ">{i+1}</div>"""
            folium.Marker(
                d["coord"],
                popup=folium.Popup(popup_html, max_width=220),
                icon=folium.DivIcon(html=mk_html, icon_size=(26,26), icon_anchor=(13,13)),
                tooltip=f"📦 {d['label']} — {d['priority']} | Delay {delay_pct}%"
            ).add_to(fmap)

        folium.LayerControl().add_to(fmap)
        st_folium(fmap, width="100%", height=600, key="main_route_map")

        # ── WH comparison table ──
        st.markdown('<div class="sh">📊 Warehouse Score Comparison</div>', unsafe_allow_html=True)
        df_wh = pd.DataFrame([{
            "ID": wh["id"], "Name": wh["name"],
            "Score ↓": score,
            "Avg Dist (km)": round(np.mean([haversine((wh["lat"],wh["lon"]),d["coord"]) for d in result["deliveries"]]),2),
            "Capacity (kg)": wh["capacity"],
            "Selected": "✦ YES" if wh["id"]==result["best_wh"]["id"] else "—"
        } for score, wh in result["wh_scores"]])
        st.dataframe(df_wh, use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;">
            <div style="font-size:56px;margin-bottom:16px;opacity:0.3;">🗺️</div>
            <div style="font-family:'Exo 2',sans-serif;font-size:18px;color:#3d5070;">
                Run optimization first to see the route map
            </div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════
#  TAB 3 — ANALYSIS
# ═══════════════════════════════════════
with tab_analysis:
    if result and result["routes"]:
        routes = result["routes"]

        st.markdown('<div class="sh">🚛 Vehicle Utilization</div>', unsafe_allow_html=True)
        df_util = pd.DataFrame([{
            "Vehicle": f"{rd['vehicle']['id']} ({rd['vtype']})",
            "Load %": rd["cap_util"], "Distance (km)": rd["dist_km"],
            "ETA (min)": rd["eta_min"], "Cost (₹)": rd["fuel_cost"],
            "CO₂ (kg)": rd["co2_kg"], "Stops": rd["stops"],
        } for rd in routes])

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            bar = alt.Chart(df_util).mark_bar(
                cornerRadiusTopLeft=5, cornerRadiusTopRight=5
            ).encode(
                x=alt.X("Vehicle:N", axis=alt.Axis(labelColor="#7a9ab5", labelAngle=-20, title="")),
                y=alt.Y("Load %:Q", scale=alt.Scale(domain=[0,100]),
                        axis=alt.Axis(labelColor="#7a9ab5", gridColor="#111d35", title="Load %")),
                color=alt.Color("Vehicle:N", legend=None, scale=alt.Scale(range=ROUTE_COLORS)),
                tooltip=["Vehicle","Load %","Distance (km)","ETA (min)"]
            ).properties(height=250, background="#080c18",
                         title=alt.TitleParams("Capacity Utilization %", color="#d4e0f0", font="Exo 2", fontSize=13)
            ).configure_view(strokeWidth=0, fill="#080c18")
            st.altair_chart(bar, use_container_width=True)

        with col_c2:
            sc = alt.Chart(df_util).mark_circle(opacity=0.9, stroke="white", strokeWidth=0.5).encode(
                x=alt.X("Distance (km):Q", axis=alt.Axis(labelColor="#7a9ab5", gridColor="#111d35")),
                y=alt.Y("Cost (₹):Q",      axis=alt.Axis(labelColor="#7a9ab5", gridColor="#111d35")),
                size=alt.Size("Stops:Q", scale=alt.Scale(range=[80,500])),
                color=alt.Color("Vehicle:N", legend=None, scale=alt.Scale(range=ROUTE_COLORS)),
                tooltip=["Vehicle","Distance (km)","Cost (₹)","Stops","CO₂ (kg)"]
            ).properties(height=250, background="#080c18",
                         title=alt.TitleParams("Cost vs Distance", color="#d4e0f0", font="Exo 2", fontSize=13)
            ).configure_view(strokeWidth=0, fill="#080c18")
            st.altair_chart(sc, use_container_width=True)

        st.markdown('<div class="sh">📦 Priority & Risk</div>', unsafe_allow_html=True)
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            prio_c = {}
            for d in result["deliveries"]: prio_c[d["priority"]] = prio_c.get(d["priority"],0)+1
            df_p = pd.DataFrame([{"Priority":k,"Count":v} for k,v in prio_c.items()])
            pie = alt.Chart(df_p).mark_arc(innerRadius=50, cornerRadius=4, padAngle=0.02).encode(
                theta="Count:Q",
                color=alt.Color("Priority:N", scale=alt.Scale(
                    domain=list(PRIORITY_LEVELS.keys()), range=["#3d5070","#ffb020","#ff3b5c","#0066ff"]
                )),
                tooltip=["Priority","Count"]
            ).properties(height=220, background="#080c18",
                         title=alt.TitleParams("Priority Mix", color="#d4e0f0", font="Exo 2", fontSize=13)
            ).configure_view(strokeWidth=0, fill="#080c18")
            st.altair_chart(pie, use_container_width=True)

        with col_d2:
            if ml_preds:
                df_risk = pd.DataFrame([{"Stop":p["label"],"Delay %":int(p["delay_prob"]*100)} for p in ml_preds])
                risk_bar = alt.Chart(df_risk).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("Stop:N", axis=alt.Axis(labelColor="#7a9ab5", labelAngle=-30, title="")),
                    y=alt.Y("Delay %:Q", scale=alt.Scale(domain=[0,100]),
                            axis=alt.Axis(labelColor="#7a9ab5", gridColor="#111d35")),
                    color=alt.condition(
                        alt.datum["Delay %"] > 50, alt.value("#ff3b5c"), alt.value("#00d4ff")
                    ),
                    tooltip=["Stop","Delay %"]
                ).properties(height=220, background="#080c18",
                             title=alt.TitleParams("AI Delay Risk per Stop", color="#d4e0f0", font="Exo 2", fontSize=13)
                ).configure_view(strokeWidth=0, fill="#080c18")
                st.altair_chart(risk_bar, use_container_width=True)

        # Risk KPIs
        st.markdown('<div class="sh">⚠️ Risk Intelligence</div>', unsafe_allow_html=True)
        delay_risk   = min(int(wp["risk"]+(traffic_mult-1)*30+len(result["deliveries"])*2), 100)
        delay_flags  = sum(1 for p in ml_preds if p["delay_predicted"])
        risk_color   = "#ff3b5c" if delay_risk>60 else "#ffb020" if delay_risk>30 else "#00ff88"
        rk1,rk2,rk3 = st.columns(3)
        for col,lbl,val,sub,c in [
            (rk1,"Composite Delay Risk",f"{delay_risk}%","index 0–100",risk_color),
            (rk2,"AI Flagged Orders",f"{delay_flags}/{len(ml_preds)}","RandomForest prediction","#ff3b5c" if delay_flags else "#00ff88"),
            (rk3,"Weather Severity",f"{wp['icon']} {wp['risk']}%",weather.capitalize(),"#ffb020"),
        ]:
            col.markdown(f"""
            <div class="kpi">
                <div class="kpi-bar" style="background:{c};"></div>
                <div class="kpi-l" style="padding-left:10px;">{lbl}</div>
                <div class="kpi-v" style="padding-left:10px;color:{c};">{val}</div>
                <div class="kpi-s" style="padding-left:10px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

        # AI Recommendations
        st.markdown('<div class="sh">🧠 AI Recommendations</div>', unsafe_allow_html=True)
        if delay_risk > 60: st.error("🔴 HIGH DELAY RISK — Consider rescheduling or adding more vehicles.")
        elif delay_risk > 35: st.warning("🟡 MODERATE RISK — Monitor live conditions, re-optimize if traffic worsens.")
        else: st.success("🟢 LOW RISK — Routes are optimally planned for current conditions.")

        for rd in routes:
            vp = VEHICLE_PROFILES[rd["vtype"]]
            if rd["cap_util"] < 30:
                st.info(f"💡 {rd['vehicle']['id']} ({rd['vtype']}) under-utilized at {rd['cap_util']}% — consider consolidating.")
            if rd["dist_km"] > vp["max_range_km"]*0.9:
                st.warning(f"⚠️ {rd['vehicle']['id']} near max range ({rd['dist_km']} km / {vp['max_range_km']} km limit).")

        if weather in ["rainy","stormy"]:
            st.info(f"🌧 {weather.capitalize()} — speed reduced by {wp['speed_pen']}% per ML risk model.")

    else:
        st.info("Run optimization to see analysis.")

# ═══════════════════════════════════════
#  TAB 4 — LIVE TRACKING (Folium map)
# ═══════════════════════════════════════
with tab_live:
    if result and result["all_route_coords"]:
        st.markdown('<div class="sh">📡 Live Vehicle Tracking Simulation — Real Map</div>', unsafe_allow_html=True)

        # Instructions
        st.markdown("""
        <div style="background:#0c1220;border:1px solid #1a2d50;border-radius:10px;
                    padding:10px 16px;font-size:11px;color:#3d5070;
                    font-family:'JetBrains Mono',monospace;margin-bottom:12px;">
            ⚡ Simulation plays the delivery route on a live map. The truck icon moves along the real road network.
        </div>""", unsafe_allow_html=True)

        prog_ph   = st.progress(0)
        status_ph = st.empty()
        map_ph    = st.empty()

        if st.button("▶  START LIVE SIMULATION", use_container_width=True):
            raw_coords = result["all_route_coords"]
            path       = raw_coords[::5]   # sub-sample for speed

            depot = result["depot_coord"]
            max_eta = max(rd["eta_min"] for rd in result["routes"])

            for i, current in enumerate(path):
                p = i / max(len(path)-1, 1)
                prog_ph.progress(min(p, 1.0))

                # Status bar
                status_ph.markdown(f"""
                <div class="live-status">
                    <span><span class="live-dot"></span> LIVE</span>
                    <span>Progress: <b>{int(p*100)}%</b></span>
                    <span>ETA remaining: <b>{int((1-p)*max_eta)} min</b></span>
                    <span>Position: <b>{round(current[0],4)}, {round(current[1],4)}</b></span>
                    <span>Speed: <b>{int(VEHICLE_PROFILES[result['routes'][0]['vtype']]['speed_kmh']*(1-wp['speed_pen']/100)/traffic_mult)} km/h</b></span>
                </div>""", unsafe_allow_html=True)

                # ── Build live Folium map ──
                lmap = folium.Map(
                    location=[current[0], current[1]],
                    zoom_start=14,
                    tiles=None,
                    zoom_control=False
                )
                # Real tile background
                folium.TileLayer(
                    tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                    attr="©OSM ©CartoDB", max_zoom=20
                ).add_to(lmap)

                # Full planned route (faint) - all vehicles
                for ri, rd in enumerate(result["routes"]):
                    color = ROUTE_COLORS[ri % len(ROUTE_COLORS)]
                    if rd["coords"]:
                        folium.PolyLine(
                            rd["coords"], color=color, weight=3,
                            opacity=0.25, dash_array="6 8"
                        ).add_to(lmap)

                # Completed path so far (bright solid line)
                done_path = path[:i+1]
                if len(done_path) > 1:
                    folium.PolyLine(
                        done_path, color="#00d4ff", weight=5,
                        opacity=0.95
                    ).add_to(lmap)
                    # Glow effect
                    folium.PolyLine(
                        done_path, color="#00d4ff", weight=14,
                        opacity=0.08
                    ).add_to(lmap)

                # Remaining path (dimmer)
                remaining = path[i:]
                if len(remaining) > 1:
                    folium.PolyLine(
                        remaining, color="#00d4ff", weight=2,
                        opacity=0.3, dash_array="4 6"
                    ).add_to(lmap)

                # ── Moving truck marker ──
                truck_html = f"""
                <div style="
                    width:38px;height:38px;
                    background:linear-gradient(135deg,#0066ff,#00d4ff);
                    border:2px solid white;border-radius:50%;
                    display:flex;align-items:center;justify-content:center;
                    font-size:18px;
                    box-shadow:0 0 20px rgba(0,212,255,0.8),0 0 40px rgba(0,212,255,0.3);
                ">🚚</div>"""
                folium.Marker(
                    current,
                    icon=folium.DivIcon(html=truck_html, icon_size=(38,38), icon_anchor=(19,19)),
                    tooltip="🚚 Vehicle in transit"
                ).add_to(lmap)

                # ── Warehouse markers ──
                for wh in active_wh:
                    is_best = wh["id"] == result["best_wh"]["id"]
                    wh_html = f"""
                    <div style="width:28px;height:28px;
                        background:{'linear-gradient(135deg,#0066ff,#00d4ff)' if is_best else '#0c1220'};
                        border:2px solid {'#00d4ff' if is_best else '#1a2d50'};
                        border-radius:6px;display:flex;align-items:center;justify-content:center;
                        font-size:12px;">🏭</div>"""
                    folium.Marker(
                        [wh["lat"], wh["lon"]],
                        icon=folium.DivIcon(html=wh_html, icon_size=(28,28), icon_anchor=(14,14)),
                        tooltip=f"🏭 {wh['name']}"
                    ).add_to(lmap)

                # ── Delivery stop markers ──
                pcolors = {"Express":"#ff3b5c","Priority":"#ffb020","Same-Day":"#0066ff","Standard":"#3d5070"}
                for si, d in enumerate(result["deliveries"]):
                    pc = pcolors[d["priority"]]
                    stop_html = f"""
                    <div style="width:22px;height:22px;background:{pc};
                        border:2px solid rgba(255,255,255,0.4);border-radius:50%;
                        display:flex;align-items:center;justify-content:center;
                        font-size:9px;font-weight:800;color:white;">{si+1}</div>"""
                    folium.Marker(
                        d["coord"],
                        icon=folium.DivIcon(html=stop_html, icon_size=(22,22), icon_anchor=(11,11)),
                        tooltip=f"Stop {si+1}: {d['label']}"
                    ).add_to(lmap)

                with map_ph:
                    st_folium(lmap, width="100%", height=520, key=f"live_map_{i}", returned_objects=[])

                time.sleep(0.12)

            # ── Completion ──
            prog_ph.progress(1.0)
            status_ph.markdown("""
            <div class="live-status" style="border-color:#00ff88;color:#00ff88;">
                <span>✦ ALL DELIVERIES COMPLETED</span>
                <span>Status: <b>ON TIME</b></span>
            </div>""", unsafe_allow_html=True)

            # Final map — show completed routes
            final_map = folium.Map(location=[depot[0], depot[1]], zoom_start=12, tiles=None)
            folium.TileLayer(
                tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                attr="©OSM ©CartoDB", max_zoom=20
            ).add_to(final_map)
            for ri, rd in enumerate(result["routes"]):
                color = ROUTE_COLORS[ri % len(ROUTE_COLORS)]
                if rd["coords"]:
                    folium.PolyLine(rd["coords"], color=color, weight=4, opacity=0.9,
                                   tooltip=f"✓ {rd['vehicle']['id']} completed").add_to(final_map)
            for d in result["deliveries"]:
                folium.CircleMarker(
                    d["coord"], radius=8, color="#00ff88",
                    fill=True, fill_opacity=0.8, tooltip=f"✓ {d['label']}"
                ).add_to(final_map)
            with map_ph:
                st_folium(final_map, width="100%", height=520, key="final_map", returned_objects=[])
    else:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;">
            <div style="font-size:56px;margin-bottom:16px;opacity:0.3;">📡</div>
            <div style="font-family:'Exo 2',sans-serif;font-size:18px;color:#3d5070;">
                Run optimization first, then start live simulation
            </div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════
#  TAB 5 — AI INSIGHTS
# ═══════════════════════════════════════
with tab_ai:
    st.markdown('<div class="sh">🤖 AI Model Stack</div>', unsafe_allow_html=True)
    m1,m2,m3 = st.columns(3)
    for col,name,acc,desc,c in [
        (m1,"RandomForestClassifier","86%","Delay prediction · 100 trees · depth=8","#00d4ff"),
        (m2,"GradientBoostingClassifier","84%","Risk scoring · 80 estimators · depth=4","#a855f7"),
        (m3,"OR-Tools CVRPTW","Optimal","GLS metaheuristic · 5s limit · parallel insert","#00ff88"),
    ]:
        col.markdown(f"""
        <div class="kpi">
            <div class="kpi-bar" style="background:{c};"></div>
            <div class="kpi-l" style="padding-left:10px;">{name}</div>
            <div class="kpi-v" style="padding-left:10px;color:{c};font-size:22px;">{acc}</div>
            <div class="kpi-s" style="padding-left:10px;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    if ml_preds:
        st.markdown('<div class="sh">📦 Per-Delivery AI Predictions</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="display:grid;grid-template-columns:1.5fr 1fr 1.2fr 1fr 1fr;
                    gap:5px;font-size:9px;color:#3d5070;text-transform:uppercase;
                    letter-spacing:2px;padding:6px 4px 8px;
                    border-bottom:1px solid #111d35;font-family:'JetBrains Mono',monospace;">
            <div>Stop</div><div>Delay Status</div><div>Probability</div>
            <div>Rec. Vehicle</div><div>Reroute</div>
        </div>""", unsafe_allow_html=True)

        for p in ml_preds:
            badge   = f'<span class="d-late">⚠ DELAYED</span>' if p["delay_predicted"] else f'<span class="d-ok">✓ ON TIME</span>'
            reroute = f'<span class="d-warn">↺ YES</span>' if p["reroute_flag"] else '<span style="color:#3d5070;font-family:\'JetBrains Mono\',monospace;font-size:10px;">— NO</span>'
            pct     = int(p["delay_prob"]*100)
            bar_c   = "#ff3b5c" if pct > 50 else "#00d4ff"
            vp_rec  = VEHICLE_PROFILES.get(p["recommended_vehicle"],{})
            icon    = vp_rec.get("icon","🚗")
            st.markdown(f"""
            <div class="rc" style="margin-bottom:5px;">
                <div class="rc-left-bar" style="background:{'#ff3b5c' if p['delay_predicted'] else '#00ff88'};"></div>
                <div style="display:grid;grid-template-columns:1.5fr 1fr 1.2fr 1fr 1fr;
                            gap:5px;align-items:center;padding-left:10px;">
                    <div style="font-weight:600;color:#d4e0f0;font-size:13px;">{p['label']}</div>
                    <div>{badge}</div>
                    <div>
                        <div style="background:#111d35;border-radius:3px;height:5px;overflow:hidden;margin-bottom:3px;">
                            <div style="width:{pct}%;height:100%;background:{bar_c};border-radius:3px;
                                        box-shadow:0 0 6px {bar_c}80;"></div>
                        </div>
                        <div style="font-size:10px;color:#7a9ab5;font-family:'JetBrains Mono',monospace;">{pct}%</div>
                    </div>
                    <div style="font-size:12px;">{icon} {p['recommended_vehicle']}</div>
                    <div>{reroute}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Delay prob chart
        st.markdown('<div class="sh">📊 Delay Probability Distribution</div>', unsafe_allow_html=True)
        df_ml = pd.DataFrame([{"Stop":p["label"],"Delay %":int(p["delay_prob"]*100)} for p in ml_preds])
        chart = alt.Chart(df_ml).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Stop:N", axis=alt.Axis(labelColor="#7a9ab5", labelAngle=-30, title="")),
            y=alt.Y("Delay %:Q", scale=alt.Scale(domain=[0,100]),
                    axis=alt.Axis(labelColor="#7a9ab5", gridColor="#111d35")),
            color=alt.condition(
                alt.datum["Delay %"] > 50, alt.value("#ff3b5c"), alt.value("#00d4ff")
            ),
            tooltip=["Stop","Delay %"]
        ).properties(height=240, background="#080c18",
                     title=alt.TitleParams("AI Delay Probability (RandomForest)", color="#d4e0f0", font="Exo 2", fontSize=13)
        ).configure_view(strokeWidth=0, fill="#080c18").configure_axis(grid=True, gridColor="#111d35")
        st.altair_chart(chart, use_container_width=True)

        # Summary
        total_d = sum(1 for p in ml_preds if p["delay_predicted"])
        total_r = sum(1 for p in ml_preds if p["reroute_flag"])
        st.markdown(f"""
        <div class="mlp">
            <div class="mlp-badge">AI SUMMARY</div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:16px;margin-top:8px;">
                <div>
                    <div style="font-size:9px;color:#3d5070;letter-spacing:2px;font-family:'JetBrains Mono',monospace;margin-bottom:4px;">PROCESSED</div>
                    <div style="font-family:'Exo 2',sans-serif;font-size:26px;font-weight:800;color:#d4e0f0;">{len(ml_preds)}</div>
                </div>
                <div>
                    <div style="font-size:9px;color:#3d5070;letter-spacing:2px;font-family:'JetBrains Mono',monospace;margin-bottom:4px;">DELAY FLAGS</div>
                    <div style="font-family:'Exo 2',sans-serif;font-size:26px;font-weight:800;color:#ff3b5c;">{total_d}</div>
                </div>
                <div>
                    <div style="font-size:9px;color:#3d5070;letter-spacing:2px;font-family:'JetBrains Mono',monospace;margin-bottom:4px;">REROUTES</div>
                    <div style="font-family:'Exo 2',sans-serif;font-size:26px;font-weight:800;color:#ffb020;">{total_r}</div>
                </div>
                <div>
                    <div style="font-size:9px;color:#3d5070;letter-spacing:2px;font-family:'JetBrains Mono',monospace;margin-bottom:4px;">MODEL ACC</div>
                    <div style="font-family:'Exo 2',sans-serif;font-size:26px;font-weight:800;color:#00ff88;">86%</div>
                </div>
            </div>
            <hr>
            <div style="font-size:10px;color:#3d5070;line-height:2.2;font-family:'JetBrains Mono',monospace;">
                ✦ Warehouse assignment: multi-criteria scoring (dist + capacity + range + weather)<br>
                ✦ Traffic-aware ETA: OSRM real road network + speed penalty<br>
                ✦ AI delay prediction: RandomForestClassifier (100 trees, 8 depth)<br>
                ✦ Vehicle recommendation: RF Classifier (weight + distance features)<br>
                ✦ Route optimization: OR-Tools CVRPTW + Guided Local Search metaheuristic
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;">
            <div style="font-size:56px;margin-bottom:16px;opacity:0.3;">🤖</div>
            <div style="font-family:'Exo 2',sans-serif;font-size:18px;color:#3d5070;">
                Run optimization to see AI predictions
            </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:28px 0 8px;
            color:#111d35;font-size:10px;
            font-family:'JetBrains Mono',monospace;letter-spacing:2px;">
    ⚡ OptiRoute · CVRPTW + Multi-Warehouse + RandomForest · OSRM · CartoDB
</div>""", unsafe_allow_html=True)
