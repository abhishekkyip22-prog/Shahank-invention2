import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import requests
import os
import time
import math
import numpy as np
import pydeck as pdk
import altair as alt
from datetime import datetime, timedelta
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from folium.plugins import HeatMap, MarkerCluster
import json
import random

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be FIRST Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FleetIQ — Smart Routing",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛰️"
)

# ─────────────────────────────────────────────
#  PREMIUM DARK LOGISTICS UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background: #0a0d14;
    color: #e2e8f0;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: #0f1420 !important;
    border-right: 1px solid #1e2535;
}

/* headers */
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; font-weight: 700; }
h1 { font-size: 2rem !important; letter-spacing: -0.5px; }

/* cards */
.kpi-card {
    background: linear-gradient(135deg, #141927 0%, #1a2236 100%);
    border: 1px solid #1e2d4a;
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #3b82f6, #06b6d4);
    border-radius: 3px 0 0 3px;
}
.kpi-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px; }
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 600; color: #f1f5f9; }
.kpi-sub   { font-size: 12px; color: #94a3b8; margin-top: 4px; }

/* warehouse badge */
.wh-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: #1e2d4a; border: 1px solid #2d4a7a;
    border-radius: 8px; padding: 6px 12px;
    font-size: 12px; font-weight: 600; color: #60a5fa;
    margin: 3px;
}

/* constraint tag */
.ctag {
    display: inline-block;
    background: #1e293b; border: 1px solid #334155;
    border-radius: 6px; padding: 3px 9px;
    font-size: 11px; color: #94a3b8; margin: 2px;
}

/* vehicle card */
.v-card {
    background: #141927;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
}
.v-card:hover { border-color: #3b82f6; }

/* status pill */
.pill-green { background:#064e3b; color:#34d399; border:1px solid #065f46; border-radius:20px; padding:3px 12px; font-size:12px; font-weight:600; }
.pill-red   { background:#4c0519; color:#f87171; border:1px solid #7f1d1d; border-radius:20px; padding:3px 12px; font-size:12px; font-weight:600; }
.pill-amber { background:#451a03; color:#fbbf24; border:1px solid #78350f; border-radius:20px; padding:3px 12px; font-size:12px; font-weight:600; }

/* section heading */
.sec-head {
    font-size: 13px; font-weight: 600; color: #64748b;
    text-transform: uppercase; letter-spacing: 2px;
    border-bottom: 1px solid #1e2535; padding-bottom: 8px;
    margin: 20px 0 12px;
}

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(90deg,#3b82f6,#06b6d4) !important;
    color: white !important; font-weight: 600 !important;
    border: none !important; border-radius: 10px !important;
    padding: 10px 22px !important; width: 100%;
}
.stSelectbox > div, .stNumberInput > div > div { background: #141927 !important; border-color: #1e2535 !important; }
[data-testid="stMetric"] { background: #141927; border-radius: 10px; padding: 10px; }
div[data-testid="stExpander"] { background: #0f1420; border-color: #1e2535; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS & REAL-WORLD DATA STRUCTURES
# ─────────────────────────────────────────────

# Pre-seeded warehouses (easily editable)
DEFAULT_WAREHOUSES = [
    {"id": "WH-A", "name": "North Hub",   "lat": 28.6800, "lon": 77.2200, "capacity": 5000, "active": True},
    {"id": "WH-B", "name": "South Hub",   "lat": 28.5900, "lon": 77.0400, "capacity": 4000, "active": True},
    {"id": "WH-C", "name": "East Hub",    "lat": 28.6500, "lon": 77.3800, "capacity": 3500, "active": True},
    {"id": "WH-D", "name": "West Hub",    "lat": 28.6200, "lon": 76.9800, "capacity": 3000, "active": True},
    {"id": "WH-E", "name": "Central Hub", "lat": 28.6300, "lon": 77.2100, "capacity": 6000, "active": True},
]

VEHICLE_PROFILES = {
    "Bike":       {"speed_kmh": 20, "capacity_kg": 20,  "max_range_km": 30,  "cost_per_km": 2,   "co2_per_km": 0.02, "icon": "🛵"},
    "Car":        {"speed_kmh": 35, "capacity_kg": 150, "max_range_km": 120, "cost_per_km": 8,   "co2_per_km": 0.18, "icon": "🚗"},
    "Van":        {"speed_kmh": 30, "capacity_kg": 600, "max_range_km": 200, "cost_per_km": 12,  "co2_per_km": 0.28, "icon": "🚐"},
    "Truck":      {"speed_kmh": 25, "capacity_kg": 2000,"max_range_km": 400, "cost_per_km": 20,  "co2_per_km": 0.55, "icon": "🚚"},
    "Drone":      {"speed_kmh": 60, "capacity_kg": 3,   "max_range_km": 15,  "cost_per_km": 1,   "co2_per_km": 0.01, "icon": "🚁"},
    "E-Bike":     {"speed_kmh": 22, "capacity_kg": 30,  "max_range_km": 50,  "cost_per_km": 1.5, "co2_per_km": 0.01, "icon": "⚡"},
}

WEATHER_PROFILES = {
    "clear":  {"factor": 1.0,  "speed_pen": 0,    "risk": 5,  "icon": "☀️"},
    "cloudy": {"factor": 1.05, "speed_pen": 2,    "risk": 10, "icon": "⛅"},
    "rainy":  {"factor": 1.35, "speed_pen": 10,   "risk": 45, "icon": "🌧"},
    "foggy":  {"factor": 1.25, "speed_pen": 8,    "risk": 40, "icon": "🌫"},
    "hot":    {"factor": 1.10, "speed_pen": 3,    "risk": 20, "icon": "🌡"},
    "cold":   {"factor": 1.15, "speed_pen": 5,    "risk": 25, "icon": "❄️"},
    "stormy": {"factor": 1.60, "speed_pen": 20,   "risk": 80, "icon": "⛈"},
}

TIME_WINDOW_PRESETS = {
    "Standard (9am-6pm)":   (9, 18),
    "Morning (8am-12pm)":   (8, 12),
    "Afternoon (12pm-6pm)": (12, 18),
    "Evening (5pm-9pm)":    (17, 21),
    "Same-Day (Now+2hr)":   None,
    "Express (60 min)":     None,
}

PRIORITY_LEVELS = {
    "Standard":  {"sla_min": 120, "penalty": 1,   "color": "#64748b"},
    "Priority":  {"sla_min": 60,  "penalty": 3,   "color": "#f59e0b"},
    "Express":   {"sla_min": 30,  "penalty": 10,  "color": "#ef4444"},
    "Same-Day":  {"sla_min": 240, "penalty": 2,   "color": "#3b82f6"},
}

# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

ss("warehouses", [dict(w) for w in DEFAULT_WAREHOUSES])
ss("result", None)
ss("tracking_active", False)
ss("route_coords", [])
ss("selected_warehouse", None)
ss("vrp_solution", None)

# ─────────────────────────────────────────────
#  UTILITY FUNCTIONS
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
               f"{start[1]},{start[0]};{end[1]},{end[0]}"
               f"?overview=full&geometries=geojson")
        res = requests.get(url, timeout=5)
        return res.json()["routes"][0]
    except:
        return None

def score_warehouse(wh, delivery_points, vehicle, weather):
    """Multi-criteria warehouse scoring (lower = better)"""
    wh_coord = (wh["lat"], wh["lon"])
    vp = VEHICLE_PROFILES[vehicle]
    wp = WEATHER_PROFILES[weather]
    
    # Average distance to all delivery points
    if delivery_points:
        avg_dist = np.mean([haversine(wh_coord, dp["coord"]) for dp in delivery_points])
    else:
        avg_dist = 0
    
    # Max distance to farthest point
    if delivery_points:
        max_dist = max([haversine(wh_coord, dp["coord"]) for dp in delivery_points])
    else:
        max_dist = 0
    
    # Capacity score (penalize if near limit)
    total_demand = sum(dp.get("weight_kg", 1) for dp in delivery_points)
    cap_util = total_demand / max(wh["capacity"], 1)
    cap_penalty = cap_util * 20
    
    # Range feasibility penalty
    range_penalty = 50 if max_dist > vp["max_range_km"] else 0
    
    # Composite score
    score = (avg_dist * 2) + (max_dist * 0.5) + cap_penalty + range_penalty
    score *= wp["factor"]
    
    return round(score, 2)

def build_distance_matrix(points):
    n = len(points)
    matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = int(haversine(points[i], points[j]) * 1000)
    return matrix

# ─────────────────────────────────────────────
#  MULTI-VEHICLE VRP SOLVER  (OR-Tools CVRPTW)
# ─────────────────────────────────────────────
def solve_vrp(depot, deliveries, vehicles, weather, hard_constraints):
    """
    Full CVRPTW (Capacitated VRP with Time Windows) using OR-Tools.
    Returns route assignments per vehicle.
    """
    if not deliveries:
        return None

    n_vehicles = len(vehicles)
    points = [depot] + [d["coord"] for d in deliveries]
    n = len(points)

    # ---- Distance matrix (meters) ----
    dist_matrix = build_distance_matrix(points)

    # ---- Demand vector ----
    demands = [0] + [int(d.get("weight_kg", 1)) for d in deliveries]

    # ---- Time windows (seconds from now) ----
    wp = WEATHER_PROFILES[weather]
    base_speed_ms = (VEHICLE_PROFILES[vehicles[0]["type"]]["speed_kmh"] * 1000 / 3600)
    # Each node: (earliest_sec, latest_sec)
    time_windows = [(0, 86400)]  # depot always open
    for d in deliveries:
        tw_open  = d.get("tw_open_min", 0) * 60
        tw_close = d.get("tw_close_min", 1440) * 60
        time_windows.append((tw_open, tw_close))

    # ---- OR-Tools setup ----
    manager = pywrapcp.RoutingIndexManager(n, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Transit callback
    def dist_cb(from_idx, to_idx):
        return dist_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]
    transit_cb = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    # Capacity constraint
    if hard_constraints.get("capacity", True):
        def demand_cb(idx):
            return demands[manager.IndexToNode(idx)]
        demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
        capacities = [int(VEHICLE_PROFILES[v["type"]]["capacity_kg"]) for v in vehicles]
        routing.AddDimensionWithVehicleCapacity(demand_cb_idx, 0, capacities, True, "Capacity")

    # Time window constraint
    if hard_constraints.get("time_windows", True):
        def time_cb(from_idx, to_idx):
            node = manager.IndexToNode(from_idx)
            speed = VEHICLE_PROFILES[vehicles[min(0, n_vehicles-1)]["type"]]["speed_kmh"] * 1000 / 3600
            speed *= (1 - wp["speed_pen"] / 100)
            speed = max(speed, 1)
            travel_time = dist_matrix[node][manager.IndexToNode(to_idx)] / speed
            service_time = 300  # 5 min service per stop
            return int(travel_time + service_time)
        time_cb_idx = routing.RegisterTransitCallback(time_cb)
        routing.AddDimension(time_cb_idx, 3600, 86400, False, "Time")
        time_dim = routing.GetDimensionOrDie("Time")
        for node_idx in range(1, n):
            idx = manager.NodeToIndex(node_idx)
            tw_open, tw_close = time_windows[node_idx]
            time_dim.CumulVar(idx).SetRange(tw_open, tw_close)

    # Max range constraint
    if hard_constraints.get("range", True):
        def range_cb(from_idx, to_idx):
            return dist_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]
        range_cb_idx = routing.RegisterTransitCallback(range_cb)
        max_ranges = [int(VEHICLE_PROFILES[v["type"]]["max_range_km"] * 1000) for v in vehicles]
        routing.AddDimensionWithVehicleCapacity(range_cb_idx, 0, max_ranges, True, "Range")

    # Priority / penalty for unserved nodes
    for d_idx, d in enumerate(deliveries):
        node_idx = d_idx + 1
        penalty = PRIORITY_LEVELS[d.get("priority", "Standard")]["penalty"] * 10000
        routing.AddDisjunction([manager.NodeToIndex(node_idx)], penalty)

    # Search params
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 5  # 5-second solve limit

    solution = routing.SolveWithParameters(params)
    if not solution:
        # fallback: greedy nearest
        return fallback_greedy(depot, deliveries, vehicles)

    # Extract routes
    routes = []
    for v_idx in range(n_vehicles):
        route_nodes = []
        idx = routing.Start(v_idx)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            route_nodes.append(node)
            idx = solution.Value(routing.NextVar(idx))
        route_nodes.append(manager.IndexToNode(idx))
        if len(route_nodes) > 2:  # non-empty route
            routes.append({"vehicle": vehicles[v_idx], "nodes": route_nodes})

    return routes if routes else fallback_greedy(depot, deliveries, vehicles)

def fallback_greedy(depot, deliveries, vehicles):
    """Nearest-neighbor fallback when OR-Tools fails"""
    remaining = list(range(len(deliveries)))
    routes = []
    for v in vehicles:
        if not remaining:
            break
        route = [0]
        cap = VEHICLE_PROFILES[v["type"]]["capacity_kg"]
        used = 0
        current = depot
        while remaining:
            best_i, best_dist = None, float("inf")
            for i in remaining:
                d = haversine(current, deliveries[i]["coord"])
                w = deliveries[i].get("weight_kg", 1)
                if used + w <= cap and d < best_dist:
                    best_i, best_dist = i, d
            if best_i is None:
                break
            route.append(best_i + 1)
            used += deliveries[best_i].get("weight_kg", 1)
            current = deliveries[best_i]["coord"]
            remaining.remove(best_i)
        route.append(0)
        if len(route) > 2:
            routes.append({"vehicle": v, "nodes": route})
    return routes if routes else None

# ─────────────────────────────────────────────
#  SIDEBAR — WAREHOUSES
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sec-head">🏭 Warehouse Network</div>', unsafe_allow_html=True)

    for i, wh in enumerate(st.session_state.warehouses):
        with st.expander(f"{wh['id']} · {wh['name']}", expanded=False):
            col1, col2 = st.columns(2)
            wh["lat"] = col1.number_input("Lat", value=wh["lat"], key=f"wlat{i}", format="%.4f")
            wh["lon"] = col2.number_input("Lon", value=wh["lon"], key=f"wlon{i}", format="%.4f")
            wh["capacity"] = st.number_input("Capacity (kg)", value=wh["capacity"], step=500, key=f"wcap{i}")
            wh["active"] = st.checkbox("Active", value=wh["active"], key=f"wact{i}")

    if st.button("+ Add Warehouse"):
        n = len(st.session_state.warehouses) + 1
        st.session_state.warehouses.append({
            "id": f"WH-{chr(64+n)}", "name": f"Hub {n}",
            "lat": 28.63, "lon": 77.20, "capacity": 3000, "active": True
        })
        st.rerun()

    st.markdown('<div class="sec-head">🚗 Fleet Setup</div>', unsafe_allow_html=True)
    num_vehicles = st.slider("Number of Vehicles", 1, 6, 2)
    vehicle_types = []
    for i in range(num_vehicles):
        vtype = st.selectbox(
            f"Vehicle {i+1}", list(VEHICLE_PROFILES.keys()),
            index=i % len(VEHICLE_PROFILES), key=f"vtype{i}"
        )
        vehicle_types.append({"id": f"V{i+1}", "type": vtype})

    st.markdown('<div class="sec-head">🌤 Conditions</div>', unsafe_allow_html=True)
    weather = st.selectbox("Weather", list(WEATHER_PROFILES.keys()),
                           format_func=lambda w: f"{WEATHER_PROFILES[w]['icon']} {w.capitalize()}")
    hour_now = datetime.now().hour
    is_peak = (8 <= hour_now <= 11) or (17 <= hour_now <= 21)
    traffic_level = st.select_slider("Traffic Level", ["Low", "Moderate", "High", "Gridlock"],
                                     value="High" if is_peak else "Moderate")
    traffic_mult = {"Low": 1.0, "Moderate": 1.2, "High": 1.5, "Gridlock": 2.0}[traffic_level]

    st.markdown('<div class="sec-head">⚙️ Constraints</div>', unsafe_allow_html=True)
    hard_constraints = {
        "capacity":     st.checkbox("Vehicle Capacity Limit",   value=True),
        "time_windows": st.checkbox("Delivery Time Windows",    value=True),
        "range":        st.checkbox("Max Vehicle Range",        value=True),
        "priority":     st.checkbox("Order Priority / SLA",     value=True),
        "breaks":       st.checkbox("Driver Break Rules",       value=False),
        "road_type":    st.checkbox("Road Type Restrictions",   value=False),
    }

# ─────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:14px; margin-bottom:4px;">
    <span style="font-size:40px;">🛰️</span>
    <div>
        <h1 style="margin:0; color:#f1f5f9;">FleetIQ</h1>
        <div style="font-size:13px; color:#64748b; letter-spacing:1px;">MULTI-WAREHOUSE VEHICLE ROUTING SYSTEM</div>
    </div>
</div>
""", unsafe_allow_html=True)

active_wh = [w for w in st.session_state.warehouses if w["active"]]
wp = WEATHER_PROFILES[weather]

col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
col_stat1.markdown(f'<div class="kpi-card"><div class="kpi-label">Active Warehouses</div><div class="kpi-value">{len(active_wh)}</div><div class="kpi-sub">of {len(st.session_state.warehouses)} total</div></div>', unsafe_allow_html=True)
col_stat2.markdown(f'<div class="kpi-card"><div class="kpi-label">Fleet Size</div><div class="kpi-value">{num_vehicles}</div><div class="kpi-sub">{", ".join(set(v["type"] for v in vehicle_types))}</div></div>', unsafe_allow_html=True)
col_stat3.markdown(f'<div class="kpi-card"><div class="kpi-label">Weather</div><div class="kpi-value">{wp["icon"]} {weather.capitalize()}</div><div class="kpi-sub">Risk factor: {wp["risk"]}%</div></div>', unsafe_allow_html=True)
col_stat4.markdown(f'<div class="kpi-card"><div class="kpi-label">Traffic</div><div class="kpi-value">{traffic_level}</div><div class="kpi-sub">×{traffic_mult} speed penalty</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DELIVERY ORDERS INPUT
# ─────────────────────────────────────────────
tab_input, tab_map, tab_analysis, tab_live = st.tabs(
    ["📦 Orders", "🗺️ Route Map", "📊 Analysis", "📡 Live Tracking"]
)

with tab_input:
    st.markdown('<div class="sec-head">📦 Delivery Orders</div>', unsafe_allow_html=True)

    num_stops = st.slider("Number of Delivery Stops", 1, 10, 3)

    st.markdown("""
    <div style="display:grid; grid-template-columns: 2fr 1fr 1fr 1fr 1fr 1fr; gap:8px;
                font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:1px;
                padding:8px 4px; border-bottom:1px solid #1e2535;">
        <div>Stop</div><div>Latitude</div><div>Longitude</div>
        <div>Weight (kg)</div><div>Time Window</div><div>Priority</div>
    </div>
    """, unsafe_allow_html=True)

    deliveries = []
    for i in range(num_stops):
        c1, c2, c3, c4, c5, c6 = st.columns([2, 1, 1, 1, 1, 1])
        label = c1.text_input("Label", value=f"Customer {i+1}", key=f"lbl{i}", label_visibility="collapsed")
        lat   = c2.number_input("Lat",    value=28.60 + i*0.015, key=f"dlat{i}", format="%.4f", label_visibility="collapsed")
        lon   = c3.number_input("Lon",    value=77.15 + i*0.020, key=f"dlon{i}", format="%.4f", label_visibility="collapsed")
        wgt   = c4.number_input("kg",     value=float(5 + i*3),  key=f"dwgt{i}", min_value=0.1, label_visibility="collapsed")
        tw    = c5.selectbox("TW", list(TIME_WINDOW_PRESETS.keys()), key=f"dtw{i}", label_visibility="collapsed")
        prio  = c6.selectbox("Priority", list(PRIORITY_LEVELS.keys()), key=f"dprio{i}", label_visibility="collapsed")

        # Resolve time window
        tw_preset = TIME_WINDOW_PRESETS[tw]
        if tw_preset:
            tw_open_min  = tw_preset[0] * 60
            tw_close_min = tw_preset[1] * 60
        else:
            tw_open_min  = hour_now * 60
            tw_close_min = (hour_now + 4) * 60

        if lat != 0.0 or lon != 0.0:
            deliveries.append({
                "label": label, "coord": (lat, lon),
                "weight_kg": wgt,
                "tw_open_min": tw_open_min, "tw_close_min": tw_close_min,
                "priority": prio,
            })

    st.markdown("<br>", unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns(2)
    run_vrp = col_btn1.button("🚀 Optimize Routes (CVRPTW)", use_container_width=True)
    clear_btn = col_btn2.button("🔄 Reset", use_container_width=True)

    if clear_btn:
        st.session_state.result = None
        st.session_state.vrp_solution = None
        st.rerun()

# ─────────────────────────────────────────────
#  CORE VRP COMPUTATION
# ─────────────────────────────────────────────
if run_vrp and deliveries and active_wh:
    with st.spinner("⚙️ Running CVRPTW optimization + warehouse selection…"):

        # 1) Score all warehouses
        wh_scores = []
        for wh in active_wh:
            score = score_warehouse(wh, deliveries, vehicle_types[0]["type"], weather)
            wh_scores.append((score, wh))
        wh_scores.sort(key=lambda x: x[0])
        best_wh = wh_scores[0][1]
        st.session_state.selected_warehouse = best_wh

        depot_coord = (best_wh["lat"], best_wh["lon"])

        # 2) Solve VRP
        solution_routes = solve_vrp(
            depot=depot_coord,
            deliveries=deliveries,
            vehicles=vehicle_types,
            weather=weather,
            hard_constraints=hard_constraints,
        )

        # 3) Compute metrics per route
        all_route_coords = []
        total_dist = 0
        route_details = []

        if solution_routes:
            for route in solution_routes:
                nodes = route["nodes"]
                vtype = route["vehicle"]["type"]
                vp = VEHICLE_PROFILES[vtype]

                pts = [depot_coord] + [deliveries[n-1]["coord"] for n in nodes[1:-1] if 0 < n <= len(deliveries)]
                pts.append(depot_coord)

                route_dist = 0
                route_coords = []
                for j in range(len(pts)-1):
                    r = fetch_route(pts[j], pts[j+1])
                    if r:
                        seg_coords = [[c[1], c[0]] for c in r["geometry"]["coordinates"]]
                        route_coords.extend(seg_coords)
                        route_dist += r["distance"] / 1000
                    else:
                        route_coords.extend([pts[j], pts[j+1]])
                        route_dist += haversine(pts[j], pts[j+1])

                eff_speed = vp["speed_kmh"] * (1 - wp["speed_pen"]/100) / traffic_mult
                eff_speed = max(eff_speed, 5)
                eta_min = (route_dist / eff_speed) * 60
                fuel_cost = route_dist * vp["cost_per_km"]
                co2 = route_dist * vp["co2_per_km"]
                load = sum(deliveries[n-1].get("weight_kg", 1) for n in nodes[1:-1] if 0 < n <= len(deliveries))
                cap_util = (load / vp["capacity_kg"]) * 100

                all_route_coords.extend(route_coords)
                total_dist += route_dist

                route_details.append({
                    "vehicle": route["vehicle"],
                    "vtype": vtype,
                    "nodes": nodes,
                    "coords": route_coords,
                    "dist_km": round(route_dist, 2),
                    "eta_min": int(eta_min),
                    "fuel_cost": round(fuel_cost, 2),
                    "co2_kg": round(co2, 3),
                    "load_kg": round(load, 2),
                    "cap_util": round(cap_util, 1),
                    "stops": len(nodes) - 2,
                })

        st.session_state.result = {
            "best_wh": best_wh,
            "wh_scores": wh_scores,
            "routes": route_details,
            "deliveries": deliveries,
            "total_dist": round(total_dist, 2),
            "depot_coord": depot_coord,
            "all_route_coords": all_route_coords,
        }
        st.session_state.route_coords = all_route_coords

# ─────────────────────────────────────────────
#  RESULTS DISPLAY
# ─────────────────────────────────────────────
result = st.session_state.result

with tab_input:
    if result:
        st.markdown('<div class="sec-head">🏆 Warehouse Selection Result</div>', unsafe_allow_html=True)

        wh_cols = st.columns(min(len(result["wh_scores"]), 5))
        for idx, (score, wh) in enumerate(result["wh_scores"][:5]):
            dist_to_centroid = np.mean([haversine((wh["lat"], wh["lon"]), d["coord"]) for d in deliveries])
            badge_color = "#22c55e" if idx == 0 else "#64748b"
            is_best = "✅ SELECTED" if idx == 0 else f"#{idx+1}"
            wh_cols[idx].markdown(f"""
            <div class="kpi-card" style="border-color:{'#22c55e' if idx==0 else '#1e2d4a'};">
                <div style="font-size:10px; color:{badge_color}; font-weight:700; margin-bottom:6px;">{is_best}</div>
                <div style="font-weight:700; color:#f1f5f9;">{wh['id']} {wh['name']}</div>
                <div class="kpi-sub">Score: {score}</div>
                <div class="kpi-sub">Avg dist: {round(dist_to_centroid,1)} km</div>
                <div class="kpi-sub">Cap: {wh['capacity']} kg</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sec-head">🚛 Route Assignments</div>', unsafe_allow_html=True)
        ROUTE_COLORS = ["#3b82f6","#22c55e","#f59e0b","#ef4444","#a78bfa","#06b6d4"]

        for ri, rd in enumerate(result["routes"]):
            vp = VEHICLE_PROFILES[rd["vtype"]]
            color = ROUTE_COLORS[ri % len(ROUTE_COLORS)]
            st.markdown(f"""
            <div class="kpi-card" style="margin-bottom:10px; border-color:{color}40;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:20px;">{vp['icon']}</span>
                        <strong style="color:{color}; margin-left:8px;">{rd['vehicle']['id']} — {rd['vtype']}</strong>
                    </div>
                    <div style="display:flex; gap:8px;">
                        <span class="ctag">📏 {rd['dist_km']} km</span>
                        <span class="ctag">⏱ {rd['eta_min']} min</span>
                        <span class="ctag">💰 ₹{rd['fuel_cost']}</span>
                        <span class="ctag">🌿 {rd['co2_kg']} kg CO₂</span>
                        <span class="ctag">📦 {rd['load_kg']} kg ({rd['cap_util']}%)</span>
                        <span class="ctag">🛑 {rd['stops']} stops</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Summary KPIs
        total_eta  = max((rd["eta_min"] for rd in result["routes"]), default=0)
        total_cost = sum(rd["fuel_cost"] for rd in result["routes"])
        total_co2  = sum(rd["co2_kg"] for rd in result["routes"])
        st.markdown("<br>", unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Distance</div><div class="kpi-value">{result["total_dist"]} km</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi-card"><div class="kpi-label">Fleet ETA (max)</div><div class="kpi-value">{total_eta} min</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Cost</div><div class="kpi-value">₹{round(total_cost,2)}</div></div>', unsafe_allow_html=True)
        k4.markdown(f'<div class="kpi-card"><div class="kpi-label">CO₂ Footprint</div><div class="kpi-value">{round(total_co2,2)} kg</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAP TAB
# ─────────────────────────────────────────────
with tab_map:
    if result:
        ROUTE_COLORS_HEX = ["#3b82f6","#22c55e","#f59e0b","#ef4444","#a78bfa","#06b6d4"]
        depot = result["depot_coord"]
        fmap = folium.Map(location=[depot[0], depot[1]], zoom_start=12,
                          tiles="CartoDB dark_matter")

        # All warehouses
        for wh in active_wh:
            is_best = wh["id"] == result["best_wh"]["id"]
            folium.Marker(
                [wh["lat"], wh["lon"]],
                popup=folium.Popup(f"<b>{wh['id']}</b><br>{wh['name']}<br>Cap: {wh['capacity']} kg", max_width=200),
                icon=folium.Icon(color="green" if is_best else "gray",
                                 icon="home", prefix="fa")
            ).add_to(fmap)

        # Routes per vehicle
        for ri, rd in enumerate(result["routes"]):
            color = ROUTE_COLORS_HEX[ri % len(ROUTE_COLORS_HEX)]
            if rd["coords"]:
                folium.PolyLine(
                    rd["coords"], color=color, weight=4, opacity=0.85,
                    tooltip=f"{rd['vehicle']['id']} | {rd['dist_km']} km | {rd['eta_min']} min"
                ).add_to(fmap)

        # Delivery markers
        for i, d in enumerate(result["deliveries"]):
            pcolor = {"Express":"red","Priority":"orange","Same-Day":"blue","Standard":"lightgray"}[d["priority"]]
            folium.CircleMarker(
                d["coord"], radius=9, color=pcolor, fill=True, fill_opacity=0.9,
                popup=folium.Popup(
                    f"<b>{d['label']}</b><br>Weight: {d['weight_kg']} kg<br>Priority: {d['priority']}",
                    max_width=200
                )
            ).add_to(fmap)
            folium.Marker(
                d["coord"],
                icon=folium.DivIcon(html=f'<div style="font-size:10px;font-weight:700;color:white;background:{pcolor};border-radius:50%;width:20px;height:20px;display:flex;align-items:center;justify-content:center;">{i+1}</div>')
            ).add_to(fmap)

        # Heatmap of delivery density
        heat_data = [list(d["coord"]) for d in result["deliveries"]]
        if heat_data:
            HeatMap(heat_data, radius=25, blur=15, min_opacity=0.3).add_to(fmap)

        st_folium(fmap, width="100%", height=600)

        # Warehouse comparison table
        st.markdown('<div class="sec-head">📊 Warehouse Score Comparison</div>', unsafe_allow_html=True)
        df_wh = pd.DataFrame([
            {
                "ID": wh["id"], "Name": wh["name"],
                "Score (lower=better)": score,
                "Avg Dist (km)": round(np.mean([haversine((wh["lat"], wh["lon"]), d["coord"]) for d in result["deliveries"]]), 2),
                "Capacity (kg)": wh["capacity"],
                "Selected": "✅" if wh["id"] == result["best_wh"]["id"] else ""
            }
            for score, wh in result["wh_scores"]
        ])
        st.dataframe(df_wh, use_container_width=True, hide_index=True)

    else:
        st.info("Run optimization first to see route map.")

# ─────────────────────────────────────────────
#  ANALYSIS TAB
# ─────────────────────────────────────────────
with tab_analysis:
    if result and result["routes"]:
        routes = result["routes"]

        # ── Vehicle utilization
        st.markdown('<div class="sec-head">🚛 Vehicle Utilization</div>', unsafe_allow_html=True)
        df_util = pd.DataFrame([{
            "Vehicle": f"{rd['vehicle']['id']} ({rd['vtype']})",
            "Load %": rd["cap_util"],
            "Distance (km)": rd["dist_km"],
            "ETA (min)": rd["eta_min"],
            "Cost (₹)": rd["fuel_cost"],
            "CO₂ (kg)": rd["co2_kg"],
            "Stops": rd["stops"],
        } for rd in routes])

        bar_util = alt.Chart(df_util).mark_bar().encode(
            x=alt.X("Vehicle:N"),
            y=alt.Y("Load %:Q", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Vehicle:N", legend=None),
            tooltip=["Vehicle", "Load %", "Distance (km)", "ETA (min)", "Cost (₹)"]
        ).properties(height=250, title="Capacity Utilization %").interactive()
        st.altair_chart(bar_util, use_container_width=True)

        # ── Cost vs Distance scatter
        scatter = alt.Chart(df_util).mark_circle(size=120).encode(
            x=alt.X("Distance (km):Q"),
            y=alt.Y("Cost (₹):Q"),
            size="Stops:Q",
            color="Vehicle:N",
            tooltip=["Vehicle", "Distance (km)", "Cost (₹)", "Stops", "CO₂ (kg)"]
        ).properties(height=250, title="Cost vs Distance (bubble = stops)").interactive()
        st.altair_chart(scatter, use_container_width=True)

        # ── Priority breakdown
        st.markdown('<div class="sec-head">📦 Order Priority Breakdown</div>', unsafe_allow_html=True)
        prio_counts = {}
        for d in result["deliveries"]:
            prio_counts[d["priority"]] = prio_counts.get(d["priority"], 0) + 1

        df_prio = pd.DataFrame([{"Priority": k, "Count": v} for k, v in prio_counts.items()])
        pie = alt.Chart(df_prio).mark_arc().encode(
            theta="Count:Q",
            color="Priority:N",
            tooltip=["Priority", "Count"]
        ).properties(height=220, title="Order Priority Distribution")
        st.altair_chart(pie, use_container_width=True)

        # ── Delay risk gauge
        st.markdown('<div class="sec-head">⚠️ Risk Dashboard</div>', unsafe_allow_html=True)
        delay_risk = min(int(wp["risk"] + (traffic_mult - 1) * 30 + len(result["deliveries"]) * 2), 100)
        express_orders = sum(1 for d in result["deliveries"] if d["priority"] == "Express")

        rk1, rk2, rk3 = st.columns(3)
        rk1.markdown(f'<div class="kpi-card"><div class="kpi-label">Delay Risk Score</div><div class="kpi-value" style="color:{"#ef4444" if delay_risk>60 else "#f59e0b" if delay_risk>30 else "#22c55e"};">{delay_risk}%</div></div>', unsafe_allow_html=True)
        rk2.markdown(f'<div class="kpi-card"><div class="kpi-label">Express SLA at Risk</div><div class="kpi-value">{express_orders}</div><div class="kpi-sub">orders requiring ≤30 min</div></div>', unsafe_allow_html=True)
        rk3.markdown(f'<div class="kpi-card"><div class="kpi-label">Weather Severity</div><div class="kpi-value">{wp["icon"]} {wp["risk"]}%</div></div>', unsafe_allow_html=True)

        # AI insights
        st.markdown('<div class="sec-head">🧠 AI Recommendations</div>', unsafe_allow_html=True)
        if delay_risk > 60:
            st.error("🔴 HIGH DELAY RISK — Consider rescheduling or adding more vehicles.")
        elif delay_risk > 35:
            st.warning("🟡 MODERATE RISK — Monitor live conditions, re-optimize if traffic worsens.")
        else:
            st.success("🟢 LOW RISK — Routes are optimally planned for current conditions.")

        for rd in routes:
            vp = VEHICLE_PROFILES[rd["vtype"]]
            if rd["cap_util"] < 30:
                st.info(f"💡 {rd['vehicle']['id']} ({rd['vtype']}) is under-utilized ({rd['cap_util']}%). Consider consolidating loads.")
            if rd["dist_km"] > vp["max_range_km"] * 0.9:
                st.warning(f"⚠️ {rd['vehicle']['id']} is near its max range ({rd['dist_km']} km vs {vp['max_range_km']} km limit).")

        if weather in ["rainy", "stormy"]:
            st.info(f"🌧 {weather.capitalize()} weather increases accident risk. Reduce speed by {wp['speed_pen']}%.")

        total_fleet_cost = sum(rd["fuel_cost"] for rd in routes)
        cheapest = min(routes, key=lambda r: r["fuel_cost"])
        if len(routes) > 1 and cheapest["cap_util"] < 70:
            st.info(f"💰 Consolidating to fewer vehicles could reduce cost. Cheapest route: ₹{cheapest['fuel_cost']}.")

    else:
        st.info("Run optimization to see analysis.")

# ─────────────────────────────────────────────
#  LIVE TRACKING TAB
# ─────────────────────────────────────────────
with tab_live:
    if result and result["all_route_coords"]:
        st.markdown('<div class="sec-head">📡 Live Vehicle Tracking</div>', unsafe_allow_html=True)

        map_ph  = st.empty()
        stat_ph = st.empty()
        prog_ph = st.progress(0)

        if st.button("▶️ Start Simulation", use_container_width=True):
            route_coords = result["all_route_coords"]
            path = [[c[1], c[0]] for c in route_coords[::4]]

            ROUTE_COLORS_RGB = [
                [59,130,246],[34,197,94],[245,158,11],[239,68,68],[167,139,250],[6,182,212]
            ]

            for i in range(len(path)):
                p = i / max(len(path)-1, 1)
                prog_ph.progress(min(p, 1.0))
                current = path[i]
                done_path = path[:i+1]

                stat_ph.markdown(f"""
                <div class="kpi-card" style="margin:6px 0;">
                    🚚 Fleet Progress: <strong>{int(p*100)}%</strong> |
                    ETA remaining: <strong>{int((1-p) * max(rd['eta_min'] for rd in result['routes']))} min</strong>
                </div>
                """, unsafe_allow_html=True)

                layers = [
                    pdk.Layer("PathLayer",
                              data=[{"path": done_path}],
                              get_path="path", width_scale=8,
                              width_min_pixels=3,
                              get_color=ROUTE_COLORS_RGB[0]),
                    pdk.Layer("ScatterplotLayer",
                              data=[{"position": current}],
                              get_position="position",
                              get_color=[0, 255, 128],
                              get_radius=120),
                ]

                # Add warehouse markers
                for wh in active_wh:
                    layers.append(pdk.Layer(
                        "ScatterplotLayer",
                        data=[{"position": [wh["lon"], wh["lat"]]}],
                        get_position="position",
                        get_color=[255,165,0],
                        get_radius=200,
                    ))

                view = pdk.ViewState(latitude=current[1], longitude=current[0], zoom=13, pitch=30)
                deck = pdk.Deck(layers=layers, initial_view_state=view,
                                map_style="mapbox://styles/mapbox/dark-v10")

                with map_ph:
                    st.pydeck_chart(deck)

                time.sleep(0.04)

            prog_ph.progress(1.0)
            stat_ph.markdown('<div class="kpi-card">✅ <strong>All deliveries completed!</strong></div>', unsafe_allow_html=True)

    else:
        st.info("Run optimization first, then start live simulation.")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:30px 0 10px; color:#1e2535; font-size:12px;">
    FleetIQ · CVRPTW + Warehouse Selection · OR-Tools · Real-time Routing
</div>
""", unsafe_allow_html=True)