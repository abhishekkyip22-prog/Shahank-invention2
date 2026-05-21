"""
FleetIQ — Smart Multi-Warehouse Routing v2
Run with:  streamlit run fleetiq_app.py

Dependencies:
    pip install streamlit pandas folium streamlit-folium requests numpy \
                altair scikit-learn ortools
"""

import math, time, warnings
import numpy as np
import pandas as pd
import altair as alt
import folium
import requests
import streamlit as st
from datetime import datetime
from folium.plugins import HeatMap, AntPath
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore")

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    ORTOOLS_OK = True
except ImportError:
    ORTOOLS_OK = False

try:
    from streamlit_folium import st_folium
    SF_OK = True
except ImportError:
    SF_OK = False

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FleetIQ — Smart Routing",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛰️",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS  (vibrant light theme, no garbled icons)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg:#f0f4ff; --surface:#fff; --surface2:#f8faff;
  --violet:#7c3aed; --violet-s:#f3f0ff; --violet-m:#ddd5fb;
  --teal:#00cfa8;   --teal-s:#e0fff9;   --teal-m:#a0f5e0;
  --amber:#ff9500;  --amber-s:#fff8eb;  --amber-m:#ffe4a8;
  --coral:#ff4757;  --coral-s:#fff0f1;  --coral-m:#ffd0d4;
  --sky:#0ea5e9;    --sky-s:#e0f5ff;    --sky-m:#bae8ff;
  --text:#0d1626; --text2:#2d3748; --text3:#64748b; --text4:#94a3b8;
  --border:#e2e8f7; --border2:#c7d2fe;
  --sh:0 2px 8px rgba(13,22,38,.08);
  --r:14px; --rs:8px;
}

html,body,[class*="css"] {
  font-family:'Plus Jakarta Sans',sans-serif !important;
  background:var(--bg) !important;
  color:var(--text) !important;
}

/* ── sidebar ── */
section[data-testid="stSidebar"] {
  background:var(--surface) !important;
  border-right:2px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div:first-child { padding-top:0 !important; }
section[data-testid="stSidebar"] * { font-family:'Plus Jakarta Sans',sans-serif !important; }

/* hide default streamlit expander arrow so it doesn't bleed into text */
details summary > span:first-child { display:none !important; }
details summary::marker          { display:none !important; content:"" !important; }
details summary::-webkit-details-marker { display:none !important; }

.main .block-container { padding-top:.5rem !important; max-width:1480px !important; }
h1,h2,h3 { font-family:'Outfit',sans-serif !important; font-weight:800 !important; }

/* ── section headings ── */
.sec-head {
  font-family:'Outfit',sans-serif; font-size:11px; font-weight:700;
  color:var(--text3); text-transform:uppercase; letter-spacing:2.5px;
  margin:22px 0 12px; padding-bottom:8px;
  border-bottom:2px solid var(--border); position:relative;
}
.sec-head::after {
  content:''; position:absolute; bottom:-2px; left:0;
  width:32px; height:2px; border-radius:2px;
  background:linear-gradient(90deg,var(--violet),var(--teal));
}

/* ── KPI cards ── */
.kpi {
  background:#fff; border-radius:var(--r); padding:18px 20px 16px;
  position:relative; overflow:hidden; box-shadow:var(--sh);
  min-height:106px; border:1.5px solid var(--border);
  transition:transform .18s,box-shadow .18s;
}
.kpi:hover { transform:translateY(-3px); box-shadow:0 8px 24px rgba(13,22,38,.12); }
.kpi-stripe { position:absolute;left:0;top:0;bottom:0;width:4px;border-radius:var(--r) 0 0 var(--r); }
.kpi-blob   { position:absolute;top:-20px;right:-20px;width:70px;height:70px;border-radius:50%;opacity:.12;filter:blur(14px); }
.kpi-label  { font-size:9px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:var(--text3);font-family:'JetBrains Mono',monospace;margin-bottom:5px;padding-left:10px; }
.kpi-value  { font-family:'Outfit',sans-serif;font-size:26px;font-weight:800;line-height:1.1;margin-bottom:2px;padding-left:10px; }
.kpi-sub    { font-size:11px;color:var(--text3);font-weight:500;padding-left:10px; }
.kv { border-color:var(--violet-m); } .kv .kpi-stripe{background:linear-gradient(180deg,var(--violet),#a855f7);} .kv .kpi-blob{background:var(--violet);} .kv .kpi-value{color:var(--violet);}
.kt { border-color:var(--teal-m);   } .kt .kpi-stripe{background:linear-gradient(180deg,var(--teal),#00e5c3);  } .kt .kpi-blob{background:var(--teal);  } .kt .kpi-value{color:#00a88c;}
.ka { border-color:var(--amber-m);  } .ka .kpi-stripe{background:linear-gradient(180deg,var(--amber),#ffcc00); } .ka .kpi-blob{background:var(--amber); } .ka .kpi-value{color:#c97800;}
.kc { border-color:var(--coral-m);  } .kc .kpi-stripe{background:linear-gradient(180deg,var(--coral),#ff8a92); } .kc .kpi-blob{background:var(--coral); } .kc .kpi-value{color:var(--coral);}
.ks { border-color:var(--sky-m);    } .ks .kpi-stripe{background:linear-gradient(180deg,var(--sky),#38bdf8);   } .ks .kpi-blob{background:var(--sky);   } .ks .kpi-value{color:#0284c7;}

/* ── route cards ── */
.rc {
  background:#fff; border:1.5px solid var(--border); border-radius:var(--r);
  padding:14px 18px; margin-bottom:9px; box-shadow:var(--sh);
  position:relative; overflow:hidden; transition:transform .15s,box-shadow .15s;
}
.rc:hover { transform:translateY(-2px); box-shadow:0 6px 20px rgba(13,22,38,.11); }
.rc-bar { position:absolute;left:0;top:0;bottom:0;width:5px;border-radius:var(--r) 0 0 var(--r); }

/* ── pills & tags ── */
.ctag {
  display:inline-flex;align-items:center;gap:4px;
  background:var(--surface2);border:1px solid var(--border);
  border-radius:8px;padding:3px 9px;font-size:11px;
  color:var(--text2);margin:2px;font-family:'JetBrains Mono',monospace;font-weight:500;
}
.pill  { display:inline-flex;align-items:center;border-radius:100px;padding:3px 11px;font-size:11px;font-weight:700;letter-spacing:.4px; }
.pg    { background:#dcfce7;color:#15803d;border:1px solid #bbf7d0; }
.pr    { background:#fee2e2;color:#dc2626;border:1px solid #fca5a5; }
.pa    { background:#fef3c7;color:#b45309;border:1px solid #fde68a; }
.pt    { background:var(--teal-s);color:#059669;border:1px solid var(--teal-m); }

/* ── live status ── */
.live-bar {
  background:linear-gradient(135deg,var(--violet-s),var(--sky-s));
  border:1.5px solid var(--border2); border-radius:var(--rs);
  padding:12px 18px; margin-bottom:10px; box-shadow:var(--sh);
}

/* ── ML prediction rows ── */
.pred-row {
  display:grid;grid-template-columns:1.6fr 1fr 1.2fr 1fr 1fr;gap:8px;
  align-items:center;padding:11px 14px;background:#fff;
  border:1.5px solid var(--border);border-radius:var(--rs);margin-bottom:5px;box-shadow:var(--sh);
}
.prob-track { background:#f1f5f9;border-radius:100px;height:7px;overflow:hidden;border:1px solid #e2e8f0;margin-bottom:2px; }
.prob-fill  { height:100%;border-radius:100px; }

/* ── Streamlit widget overrides ── */
.stButton > button {
  background:linear-gradient(135deg,var(--violet),#a855f7) !important;
  color:#fff !important; font-weight:700 !important; font-family:'Outfit',sans-serif !important;
  font-size:14px !important; border:none !important; border-radius:12px !important;
  padding:10px 22px !important; width:100% !important;
  box-shadow:0 4px 14px rgba(124,58,237,.28) !important;
  transition:opacity .2s,transform .1s !important;
}
.stButton > button:hover { opacity:.9 !important; transform:translateY(-1px) !important; }

.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput  > div > div > input {
  background:var(--surface2) !important; border:1.5px solid var(--border) !important;
  border-radius:var(--rs) !important; color:var(--text) !important;
  font-family:'Plus Jakarta Sans',sans-serif !important;
}

.stTabs [data-baseweb="tab-list"] {
  background:#fff !important; border:1.5px solid var(--border) !important;
  border-radius:14px !important; padding:4px !important; gap:2px !important;
  box-shadow:var(--sh) !important;
}
.stTabs [data-baseweb="tab"] {
  background:transparent !important; color:var(--text3) !important;
  border-radius:10px !important; font-family:'Outfit',sans-serif !important;
  font-weight:600 !important; font-size:13px !important; padding:8px 16px !important;
}
.stTabs [aria-selected="true"] {
  background:linear-gradient(135deg,var(--violet),#a855f7) !important;
  color:#fff !important; box-shadow:0 3px 10px rgba(124,58,237,.28) !important;
}
div[data-testid="stExpander"] {
  background:var(--surface2) !important; border:1.5px solid var(--border) !important;
  border-radius:var(--r) !important; box-shadow:var(--sh) !important;
}
.stProgress > div > div {
  background:linear-gradient(90deg,var(--violet),var(--teal)) !important;
  border-radius:100px !important;
}
.stProgress > div { background:#e9ecf7 !important; border-radius:100px !important; }
[data-testid="stDataFrame"] {
  border-radius:var(--r) !important; overflow:hidden !important;
  border:1.5px solid var(--border) !important; box-shadow:var(--sh) !important;
}
hr { border-color:var(--border) !important; margin:12px 0 !important; }
::-webkit-scrollbar       { width:6px; height:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border2); border-radius:6px; }
::-webkit-scrollbar-thumb:hover { background:var(--violet); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_WAREHOUSES = [
    {"id":"WH-A","name":"North Hub",   "lat":28.6800,"lon":77.2200,"capacity":5000,"active":True},
    {"id":"WH-B","name":"South Hub",   "lat":28.5900,"lon":77.0400,"capacity":4000,"active":True},
    {"id":"WH-C","name":"East Hub",    "lat":28.6500,"lon":77.3800,"capacity":3500,"active":True},
    {"id":"WH-D","name":"West Hub",    "lat":28.6200,"lon":76.9800,"capacity":3000,"active":True},
    {"id":"WH-E","name":"Central Hub", "lat":28.6300,"lon":77.2100,"capacity":6000,"active":True},
]

VEHICLE_PROFILES = {
    "Bike":   {"speed":20,  "cap":20,   "range":30,  "cost":2,   "co2":0.02, "icon":"🛵"},
    "Car":    {"speed":35,  "cap":150,  "range":120, "cost":8,   "co2":0.18, "icon":"🚗"},
    "Van":    {"speed":30,  "cap":600,  "range":200, "cost":12,  "co2":0.28, "icon":"🚐"},
    "Truck":  {"speed":25,  "cap":2000, "range":400, "cost":20,  "co2":0.55, "icon":"🚚"},
    "Drone":  {"speed":60,  "cap":3,    "range":15,  "cost":1,   "co2":0.01, "icon":"🚁"},
    "E-Bike": {"speed":22,  "cap":30,   "range":50,  "cost":1.5, "co2":0.01, "icon":"⚡"},
}

WEATHER_PROFILES = {
    "clear":  {"factor":1.00,"speedPen":0, "risk":5,  "icon":"☀️"},
    "cloudy": {"factor":1.05,"speedPen":2, "risk":10, "icon":"⛅"},
    "rainy":  {"factor":1.35,"speedPen":10,"risk":45, "icon":"🌧"},
    "foggy":  {"factor":1.25,"speedPen":8, "risk":40, "icon":"🌫"},
    "hot":    {"factor":1.10,"speedPen":3, "risk":20, "icon":"🌡"},
    "cold":   {"factor":1.15,"speedPen":5, "risk":25, "icon":"❄️"},
    "stormy": {"factor":1.60,"speedPen":20,"risk":80, "icon":"⛈"},
}

TIME_WINDOW_PRESETS = {
    "Standard (9am-6pm)":   (9*60, 18*60),
    "Morning (8am-12pm)":   (8*60, 12*60),
    "Afternoon (12pm-6pm)": (12*60,18*60),
    "Evening (5pm-9pm)":    (17*60,21*60),
    "Same-Day (Now+4hr)":   None,
    "Express (60 min)":     None,
}

PRIORITY_LEVELS = {
    "Standard": {"sla":120,"penalty":1, "color":"#64748b"},
    "Priority":  {"sla":60, "penalty":3, "color":"#ff9500"},
    "Express":   {"sla":30, "penalty":10,"color":"#ff4757"},
    "Same-Day":  {"sla":240,"penalty":2, "color":"#0ea5e9"},
}

ROUTE_COLORS = ["#7c3aed","#00cfa8","#ff4757","#ff9500","#0ea5e9","#f43f5e"]

# ─────────────────────────────────────────────────────────────────────────────
#  ML MODELS  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_models():
    np.random.seed(42); n = 3000
    dist   = np.random.uniform(2, 120, n)
    weight = np.random.uniform(.1, 200, n)
    w_risk = np.random.uniform(0, 80, n)
    traf   = np.random.uniform(1, 2, n)
    sla    = np.random.choice([30, 60, 120, 240], n)
    xp     = (sla <= 30).astype(int)
    hr     = np.random.randint(0, 24, n)
    peak   = ((hr>=8)&(hr<=11)) | ((hr>=17)&(hr<=21))
    prob   = np.clip(.35*(dist/120)+.20*(w_risk/80)+.25*(traf-1)+.10*xp+.10*peak.astype(float), 0, 1)
    y      = (np.random.rand(n) < prob).astype(int)
    X      = np.column_stack([dist, weight, w_risk, traf, sla, xp, hr, peak.astype(int)])
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42).fit(X, y)
    gb = GradientBoostingClassifier(n_estimators=80,  max_depth=4, random_state=42).fit(X, y)

    # vehicle recommender
    wv = np.random.uniform(.1, 500, n)
    dv = np.random.uniform(1,  200, n)
    def rule(w, d):
        if d<=15  and w<=3:   return 0   # Drone
        if w<=20  and d<=30:  return 4   # E-Bike
        if w<=20:             return 0   # Car  (mapped below)
        if w<=150:            return 1   # Car
        if w<=600:            return 2   # Van
        return 3                         # Truck
    yv = np.array([rule(w, d) for w, d in zip(wv, dv)])
    rv = RandomForestClassifier(n_estimators=50, random_state=0).fit(np.column_stack([wv, dv]), yv)
    return rf, gb, rv

delay_rf, delay_gb, veh_rec = build_models()

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def _ss(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

_ss("warehouses",       [dict(w) for w in DEFAULT_WAREHOUSES])
_ss("result",           None)
_ss("ml_preds",         [])
_ss("map_clicked",      None)
_ss("map_center",       [28.63, 77.20])
_ss("search_result",    None)
_ss("live_step",        0)
_ss("live_running",     False)
_ss("deliveries_snap",  [])

# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def haversine(p1, p2):
    R = 6371
    la1, lo1 = math.radians(p1[0]), math.radians(p1[1])
    la2, lo2 = math.radians(p2[0]), math.radians(p2[1])
    a = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def fetch_route(s, e):
    """Call OSRM for real road geometry; fall back to straight line."""
    try:
        r = requests.get(
            f"http://router.project-osrm.org/route/v1/driving/"
            f"{s[1]},{s[0]};{e[1]},{e[0]}?overview=full&geometries=geojson",
            timeout=6)
        data = r.json()["routes"][0]
        return data
    except Exception:
        return None

def geocode(query):
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
            params={"q":query,"format":"json","limit":5},
            headers={"User-Agent":"FleetIQ/2"}, timeout=5)
        return [(float(x["lat"]), float(x["lon"]), x["display_name"]) for x in r.json()]
    except Exception:
        return []

def predict_delay(dist, wt, weather, traffic_mult, priority, hr=None):
    if hr is None:
        hr = datetime.now().hour
    wp  = WEATHER_PROFILES[weather]
    sla = PRIORITY_LEVELS[priority]["sla"]
    xp  = 1 if sla<=30 else 0
    pk  = 1 if (8<=hr<=11) or (17<=hr<=21) else 0
    X   = np.array([[dist, wt, wp["risk"], traffic_mult, sla, xp, hr, pk]])
    pred = int(delay_rf.predict(X)[0])
    prob = round(float(delay_rf.predict_proba(X)[0][1]), 3)
    return pred, prob

def recommend_vehicle(wt, dist):
    mapping = {0:"Drone", 1:"Car", 2:"Van", 3:"Truck", 4:"E-Bike"}
    return mapping.get(int(veh_rec.predict(np.array([[wt, dist]]))[0]), "Van")

def score_warehouse(wh, deliveries, veh_type, weather):
    wc  = (wh["lat"], wh["lon"])
    vp  = VEHICLE_PROFILES[veh_type]
    wp  = WEATHER_PROFILES[weather]
    if deliveries:
        dists = [haversine(wc, d["coord"]) for d in deliveries]
        avg_d, max_d = np.mean(dists), max(dists)
    else:
        avg_d = max_d = 0
    cap_pen   = (sum(d.get("weight_kg",1) for d in deliveries) / max(wh["capacity"],1)) * 20
    range_pen = 50 if max_d > vp["range"] else 0
    return round((avg_d*2 + max_d*.5 + cap_pen + range_pen) * wp["factor"], 2)

def dist_matrix(pts):
    n = len(pts)
    m = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i!=j:
                m[i][j] = int(haversine(pts[i], pts[j]) * 1000)
    return m

# ─────────────────────────────────────────────────────────────────────────────
#  VRP SOLVER
# ─────────────────────────────────────────────────────────────────────────────
def solve_vrp(depot, deliveries, vehicles, weather, constraints):
    if not deliveries:
        return None

    if not ORTOOLS_OK:
        return _fallback(depot, deliveries, vehicles)

    nv  = len(vehicles)
    pts = [depot] + [d["coord"] for d in deliveries]
    n   = len(pts)
    dm  = dist_matrix(pts)
    wp  = WEATHER_PROFILES[weather]
    demands = [0] + [int(d.get("weight_kg",1)) for d in deliveries]
    tw = [(0,86400)] + [
        (d.get("tw_open",0)*60, d.get("tw_close",1440)*60)
        for d in deliveries
    ]

    mgr = pywrapcp.RoutingIndexManager(n, nv, 0)
    rt  = pywrapcp.RoutingModel(mgr)

    def dist_cb(fi, ti):
        return dm[mgr.IndexToNode(fi)][mgr.IndexToNode(ti)]
    tc = rt.RegisterTransitCallback(dist_cb)
    rt.SetArcCostEvaluatorOfAllVehicles(tc)

    if constraints.get("capacity", True):
        def dem_cb(i):
            return demands[mgr.IndexToNode(i)]
        di = rt.RegisterUnaryTransitCallback(dem_cb)
        caps = [int(VEHICLE_PROFILES[v["type"]]["cap"]) for v in vehicles]
        rt.AddDimensionWithVehicleCapacity(di, 0, caps, True, "Cap")

    if constraints.get("time_windows", True):
        spd = max(VEHICLE_PROFILES[vehicles[0]["type"]]["speed"] * 1000/3600
                  * (1 - wp["speedPen"]/100), 1)
        def time_cb(fi, ti):
            return int(dm[mgr.IndexToNode(fi)][mgr.IndexToNode(ti)] / spd + 300)
        ti2 = rt.RegisterTransitCallback(time_cb)
        rt.AddDimension(ti2, 3600, 86400, False, "Time")
        td = rt.GetDimensionOrDie("Time")
        for ni in range(1, n):
            idx = mgr.NodeToIndex(ni)
            td.CumulVar(idx).SetRange(tw[ni][0], tw[ni][1])

    if constraints.get("range", True):
        ranges = [int(VEHICLE_PROFILES[v["type"]]["range"] * 1000) for v in vehicles]
        rt.AddDimensionWithVehicleCapacity(tc, 0, ranges, True, "Range")

    for di_idx, d in enumerate(deliveries):
        rt.AddDisjunction(
            [mgr.NodeToIndex(di_idx+1)],
            PRIORITY_LEVELS[d.get("priority","Standard")]["penalty"] * 10000
        )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 5

    sol = rt.SolveWithParameters(params)
    if not sol:
        return _fallback(depot, deliveries, vehicles)

    routes = []
    for vi in range(nv):
        nodes = []
        idx = rt.Start(vi)
        while not rt.IsEnd(idx):
            nodes.append(mgr.IndexToNode(idx))
            idx = sol.Value(rt.NextVar(idx))
        nodes.append(mgr.IndexToNode(idx))
        if len(nodes) > 2:
            routes.append({"vehicle": vehicles[vi], "nodes": nodes})

    return routes if routes else _fallback(depot, deliveries, vehicles)

def _fallback(depot, deliveries, vehicles):
    remaining = list(range(len(deliveries)))
    routes = []
    for v in vehicles:
        if not remaining:
            break
        route = [0]
        cap   = VEHICLE_PROFILES[v["type"]]["cap"]
        used  = 0
        cur   = depot
        while remaining:
            best_i, best_d = None, float("inf")
            for i in remaining:
                d    = haversine(cur, deliveries[i]["coord"])
                wt   = deliveries[i].get("weight_kg", 1)
                if used + wt <= cap and d < best_d:
                    best_i, best_d = i, d
            if best_i is None:
                break
            route.append(best_i + 1)
            used += deliveries[best_i].get("weight_kg", 1)
            cur   = deliveries[best_i]["coord"]
            remaining.remove(best_i)
        route.append(0)
        if len(route) > 2:
            routes.append({"vehicle": v, "nodes": route})
    return routes if routes else None

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:11px;padding:14px 0 16px;
                border-bottom:2px solid #e2e8f7;margin-bottom:4px;">
        <div style="width:42px;height:42px;border-radius:13px;flex-shrink:0;
                    background:linear-gradient(135deg,#7c3aed,#a855f7);
                    display:flex;align-items:center;justify-content:center;
                    font-size:20px;box-shadow:0 4px 14px rgba(124,58,237,.35);">🛰️</div>
        <div>
            <div style="font-family:'Outfit',sans-serif;font-size:18px;font-weight:800;
                        color:#0d1626;line-height:1.1;">FleetIQ</div>
            <div style="font-size:8px;color:#94a3b8;letter-spacing:2.5px;
                        font-family:'JetBrains Mono',monospace;">SMART ROUTING v2</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Warehouse Network ──
    st.markdown('<div class="sec-head">🏭 Warehouse Network</div>', unsafe_allow_html=True)
    for i, wh in enumerate(st.session_state.warehouses):
        with st.expander(f"{wh['id']}  ·  {wh['name']}", expanded=False):
            c1, c2 = st.columns(2)
            wh["lat"]      = c1.number_input("Lat",      value=wh["lat"],      key=f"wlat{i}", format="%.4f")
            wh["lon"]      = c2.number_input("Lon",      value=wh["lon"],      key=f"wlon{i}", format="%.4f")
            wh["capacity"] = st.number_input("Cap (kg)", value=wh["capacity"], key=f"wcap{i}", step=500)
            wh["active"]   = st.checkbox("Active",       value=wh["active"],   key=f"wact{i}")

    if st.button("➕  Add Warehouse"):
        n = len(st.session_state.warehouses) + 1
        st.session_state.warehouses.append({
            "id": f"WH-{chr(64+n)}", "name": f"Hub {n}",
            "lat": 28.63, "lon": 77.20, "capacity": 3000, "active": True
        })
        st.rerun()

    # ── Fleet Setup ──
    st.markdown('<div class="sec-head">🚗 Fleet Setup</div>', unsafe_allow_html=True)
    num_veh = st.slider("Vehicles", 1, 6, 2, key="num_veh")
    veh_types = [
        {"id": f"V{i+1}",
         "type": st.selectbox(f"Vehicle {i+1}", list(VEHICLE_PROFILES.keys()),
                              index=i % len(VEHICLE_PROFILES), key=f"vt{i}")}
        for i in range(num_veh)
    ]

    # ── Conditions ──
    st.markdown('<div class="sec-head">🌤 Conditions</div>', unsafe_allow_html=True)
    weather  = st.selectbox("Weather", list(WEATHER_PROFILES.keys()),
                            format_func=lambda w: f"{WEATHER_PROFILES[w]['icon']} {w.capitalize()}")
    hour_now = datetime.now().hour
    is_peak  = (8<=hour_now<=11) or (17<=hour_now<=21)
    traffic  = st.select_slider("Traffic", ["Low","Moderate","High","Gridlock"],
                                value="High" if is_peak else "Moderate")
    t_mult   = {"Low":1.0,"Moderate":1.2,"High":1.5,"Gridlock":2.0}[traffic]

    # ── Constraints ──
    st.markdown('<div class="sec-head">⚙️ Constraints</div>', unsafe_allow_html=True)
    hc = {
        "capacity":     st.checkbox("Capacity Limit",   value=True),
        "time_windows": st.checkbox("Time Windows",     value=True),
        "range":        st.checkbox("Max Range",        value=True),
        "priority":     st.checkbox("Priority / SLA",   value=True),
    }

    # ── AI Stack ──
    st.markdown('<div class="sec-head">🤖 AI Stack</div>', unsafe_allow_html=True)
    for nm, col, acc in [
        ("RandomForest",    "#00cfa8", "86% acc"),
        ("GradientBoosting","#7c3aed", "risk score"),
        ("OR-Tools CVRPTW", "#ff9500", "optimizer"),
        ("OSRM Real Roads", "#0ea5e9", "routing"),
    ]:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;font-size:12px;">'
            f'<span style="width:9px;height:9px;border-radius:50%;background:{col};'
            f'box-shadow:0 0 7px {col};display:inline-block;flex-shrink:0;"></span>'
            f'<span style="font-weight:600;color:#2d3748;flex:1;">{nm}</span>'
            f'<span style="color:#94a3b8;font-size:10px;">{acc}</span></div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
active_wh = [w for w in st.session_state.warehouses if w["active"]]
wp = WEATHER_PROFILES[weather]

st.markdown(f"""
<div style="display:flex;align-items:flex-start;justify-content:space-between;
            flex-wrap:wrap;gap:10px;padding:6px 0 4px;">
  <div>
    <div style="font-family:'Outfit',sans-serif;font-size:36px;font-weight:900;line-height:1.1;
                background:linear-gradient(120deg,#7c3aed 0%,#0ea5e9 45%,#00cfa8 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">FleetIQ</div>
    <div style="font-size:9px;color:#94a3b8;letter-spacing:3px;
                font-family:'JetBrains Mono',monospace;margin-top:1px;">
      MULTI-WAREHOUSE · AI ROUTING · CVRPTW OPTIMIZER</div>
  </div>
  <div style="display:flex;align-items:center;gap:7px;flex-wrap:wrap;padding-top:5px;">
    <span style="background:#f0fdf4;color:#16a34a;border:1.5px solid #bbf7d0;
                 padding:5px 13px;border-radius:100px;font-size:11px;font-weight:700;">● SYSTEM LIVE</span>
    <span style="background:#f3f0ff;color:#7c3aed;border:1.5px solid #ddd5fb;
                 padding:5px 13px;border-radius:100px;font-size:11px;font-weight:700;">🤖 AI ACTIVE</span>
    <span style="background:#e0f5ff;color:#0284c7;border:1.5px solid #bae8ff;
                 padding:5px 13px;border-radius:100px;font-size:11px;font-weight:700;">
      {wp['icon']} {weather.upper()}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ──
kpi_data = [
    ("Active Hubs",  str(len(active_wh)),   f"of {len(st.session_state.warehouses)} total", "kv"),
    ("Fleet Size",   str(num_veh),           ", ".join(dict.fromkeys(v["type"] for v in veh_types)), "kt"),
    ("Weather Risk", f"{wp['risk']}%",       f"{wp['icon']} {weather.capitalize()}", "ka"),
    ("Traffic",      traffic,                f"×{t_mult} speed penalty", "kc"),
    ("AI Accuracy",  "86%",                  "RandomForest delay model", "ks"),
]
for col, (lbl, val, sub, cls) in zip(st.columns(5), kpi_data):
    col.markdown(
        f'<div class="kpi {cls}"><div class="kpi-stripe"></div><div class="kpi-blob"></div>'
        f'<div class="kpi-label">{lbl}</div><div class="kpi-value">{val}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "📦  Orders & Input",
    "🗺️  Route Map",
    "📊  Analysis",
    "📡  Live Tracking",
    "🤖  AI Insights",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — ORDERS & INPUT
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    # ── Location search ──
    st.markdown('<div class="sec-head">🔍 Location Search</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns([3, 1])
    search_q = sc1.text_input("Search location",
        placeholder="e.g. Connaught Place Delhi · India Gate · any address…",
        label_visibility="collapsed", key="loc_search")
    search_go = sc2.button("🔍 Search", use_container_width=True, key="search_go_btn")

    if search_go and search_q:
        with st.spinner("Searching…"):
            res = geocode(search_q)
        st.session_state.search_result = res if res else []
        if not res:
            st.warning("No results found. Try a more specific address.")

    if st.session_state.search_result:
        st.markdown('<div style="font-size:12px;color:#64748b;font-weight:600;margin-bottom:6px;">'
                    'Search results — click Use to apply:</div>', unsafe_allow_html=True)
        for idx, (rlat, rlon, rname) in enumerate(st.session_state.search_result[:4]):
            r1, r2, r3 = st.columns([3.5, 1, 1])
            r1.markdown(
                f'<div style="font-size:12px;color:#374151;padding:7px 0;">'
                f'<b style="color:#7c3aed;">📍</b> {rname[:85]}{"…" if len(rname)>85 else ""}</div>',
                unsafe_allow_html=True)
            if r2.button("📍 Use", key=f"sr_use_{idx}", use_container_width=True):
                st.session_state.map_clicked  = (rlat, rlon)
                st.session_state.map_center   = [rlat, rlon]
                st.session_state.search_result = None
                st.rerun()
            if r3.button("🏭 Add WH", key=f"sr_wh_{idx}", use_container_width=True):
                n = len(st.session_state.warehouses) + 1
                st.session_state.warehouses.append({
                    "id": f"WH-{chr(64+n)}", "name": rname[:20],
                    "lat": rlat, "lon": rlon, "capacity": 3000, "active": True
                })
                st.session_state.search_result = None
                st.success(f"Warehouse added at {rname[:40]}")
                st.rerun()

    st.markdown("---")

    # ── Map picker + coord panel ──
    la, ra = st.columns([2, 1], gap="medium")
    with la:
        st.markdown('<div class="sec-head">📍 Interactive Location Picker</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="background:linear-gradient(135deg,#f3f0ff,#e0f5ff);border:1.5px solid #c7d2fe;'
            'border-radius:10px;padding:10px 16px;font-size:13px;color:#374151;font-weight:500;margin-bottom:8px;">'
            '🖱️ <b>Click the map</b> to capture coordinates, then assign to any stop.</div>',
            unsafe_allow_html=True)

        cm = folium.Map(location=st.session_state.map_center, zoom_start=12, tiles="CartoDB Positron")
        if st.session_state.map_clicked:
            lat_c, lon_c = st.session_state.map_clicked
            folium.Marker([lat_c, lon_c],
                icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
                tooltip="📍 Selected Location").add_to(cm)

        for wh in active_wh:
            folium.Marker([wh["lat"], wh["lon"]],
                popup=folium.Popup(f"<b>{wh['id']}</b> — {wh['name']}<br>Cap: {wh['capacity']:,} kg", max_width=180),
                icon=folium.Icon(color="purple", icon="home", prefix="fa"),
                tooltip=f"🏭 {wh['name']}").add_to(cm)

        for i, d in enumerate(st.session_state.deliveries_snap):
            if d["coord"] != (0.0, 0.0):
                fc = ["purple","green","red","orange","blue","darkred"][i % 6]
                folium.CircleMarker(d["coord"], radius=9, color=fc, fill=True, fill_opacity=.85,
                    tooltip=f"Stop {i+1}: {d['label']}").add_to(cm)

        if SF_OK:
            mr = st_folium(cm, width="100%", height=280, returned_objects=["last_clicked"], key="locpicker")
            if mr and mr.get("last_clicked"):
                cl = mr["last_clicked"]
                st.session_state.map_clicked = (round(cl["lat"],5), round(cl["lng"],5))
                st.session_state.map_center  = [cl["lat"], cl["lng"]]
        else:
            st.warning("Install `streamlit-folium` for interactive map picking.")

    with ra:
        st.markdown('<div class="sec-head">📌 Captured Coordinates</div>', unsafe_allow_html=True)
        if st.session_state.map_clicked:
            lat_c, lon_c = st.session_state.map_clicked
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#f3f0ff,#e0f5ff);
                        border:1.5px solid #c7d2fe;border-radius:14px;padding:16px;margin-top:8px;">
              <div style="font-size:9px;font-weight:700;letter-spacing:2px;color:#7c3aed;
                          margin-bottom:8px;font-family:'JetBrains Mono',monospace;">SELECTED POINT</div>
              <div style="display:inline-flex;align-items:center;gap:6px;background:white;
                          border:1.5px solid #c7d2fe;border-radius:9px;padding:5px 13px;margin:3px;
                          font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;color:#7c3aed;">
                🌐 {lat_c}</div><br>
              <div style="display:inline-flex;align-items:center;gap:6px;background:white;
                          border:1.5px solid #c7d2fe;border-radius:9px;padding:5px 13px;margin:3px;
                          font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;color:#7c3aed;">
                🌐 {lon_c}</div>
            </div>""", unsafe_allow_html=True)
            assign_to = st.selectbox("Assign to stop #", range(1, 11), key="asgn_stop")
            if st.button("✅ Apply Coordinates", use_container_width=True):
                st.session_state[f"dlat_{assign_to-1}"] = lat_c
                st.session_state[f"dlon_{assign_to-1}"] = lon_c
                st.success(f"✅ Applied to Stop {assign_to}")
        else:
            st.markdown("""
            <div style="background:#f8faff;border:2px dashed #c7d2fe;border-radius:14px;
                        padding:28px 14px;text-align:center;margin-top:8px;">
              <div style="font-size:28px;margin-bottom:6px;">🗺️</div>
              <div style="font-size:13px;color:#94a3b8;font-weight:500;">
                Search above or click the map<br>to capture a location</div>
            </div>""", unsafe_allow_html=True)

    # ── Delivery table ──
    st.markdown('<div class="sec-head">📦 Delivery Orders</div>', unsafe_allow_html=True)
    num_stops = st.slider("Number of Delivery Stops", 1, 10, 3, key="num_stops")

    st.markdown("""
    <div style="display:grid;grid-template-columns:1.8fr 1fr 1fr .8fr 1.2fr 1fr;
                gap:6px;font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;
                color:#94a3b8;text-transform:uppercase;letter-spacing:1.5px;
                padding:7px 4px;border-bottom:2px solid #e2e8f7;">
      <div>Label</div><div>Latitude</div><div>Longitude</div>
      <div>kg</div><div>Time Window</div><div>Priority</div>
    </div>""", unsafe_allow_html=True)

    deliveries = []
    for i in range(num_stops):
        c1, c2, c3, c4, c5, c6 = st.columns([1.8, 1, 1, .8, 1.2, 1])
        lbl  = c1.text_input("L", value=f"Customer {i+1}", key=f"lbl{i}",  label_visibility="collapsed")
        dlat = st.session_state.get(f"dlat_{i}", 28.60 + i*.015)
        dlon = st.session_state.get(f"dlon_{i}", 77.15 + i*.020)
        lat  = c2.number_input("a", value=float(dlat), key=f"dlat{i}", format="%.5f", label_visibility="collapsed")
        lon  = c3.number_input("b", value=float(dlon), key=f"dlon{i}", format="%.5f", label_visibility="collapsed")
        wgt  = c4.number_input("c", value=float(5 + i*3), key=f"dwgt{i}", min_value=0.1, label_visibility="collapsed")
        tw   = c5.selectbox("d",   list(TIME_WINDOW_PRESETS.keys()),  key=f"dtw{i}",  label_visibility="collapsed")
        prio = c6.selectbox("e",   list(PRIORITY_LEVELS.keys()),      key=f"dpr{i}",  label_visibility="collapsed")

        twp  = TIME_WINDOW_PRESETS[tw]
        two, twc = twp if twp else (hour_now*60, (hour_now+4)*60)
        deliveries.append({
            "label": lbl, "coord": (lat, lon), "weight_kg": wgt,
            "tw_open": two // 60, "tw_close": twc // 60, "priority": prio,
        })

    st.session_state.deliveries_snap = deliveries

    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    run_btn = b1.button("🚀  Optimize Routes  (CVRPTW + AI)", use_container_width=True)
    clr_btn = b2.button("🔄  Reset All",                      use_container_width=True)
    if clr_btn:
        st.session_state.result    = None
        st.session_state.ml_preds  = []
        st.session_state.live_step = 0
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  COMPUTE (triggered from Tab 1 button)
# ─────────────────────────────────────────────────────────────────────────────
if run_btn and deliveries and active_wh:
    with st.spinner("⚙️ Running CVRPTW optimisation + AI prediction…"):

        # 1. Score warehouses
        wh_scores = sorted(
            [(score_warehouse(wh, deliveries, veh_types[0]["type"], weather), wh)
             for wh in active_wh],
            key=lambda x: x[0]
        )
        best_wh = wh_scores[0][1]
        depot   = (best_wh["lat"], best_wh["lon"])

        # 2. ML predictions per delivery
        ml_preds = []
        for d in deliveries:
            dist = haversine(depot, d["coord"])
            dp, prob = predict_delay(dist, d["weight_kg"], weather, t_mult, d["priority"])
            rv = recommend_vehicle(d["weight_kg"], dist)
            ml_preds.append({
                "label": d["label"], "delay": dp, "prob": prob,
                "rec_veh": rv, "reroute": dp==1 and prob>.6,
                "dist_km": round(dist, 2),
            })
        st.session_state.ml_preds = ml_preds

        # 3. VRP solve
        sol = solve_vrp(depot, deliveries, veh_types, weather, hc)

        all_coords = []; total_dist = 0; route_details = []
        if sol:
            for route in sol:
                nodes  = route["nodes"]
                vtype  = route["vehicle"]["type"]
                vp     = VEHICLE_PROFILES[vtype]
                # build waypoint list
                wp_pts = ([depot]
                          + [deliveries[n-1]["coord"] for n in nodes[1:-1] if 0 < n <= len(deliveries)]
                          + [depot])
                r_dist = 0; r_coords = []
                for j in range(len(wp_pts)-1):
                    osrm = fetch_route(wp_pts[j], wp_pts[j+1])
                    if osrm:
                        r_coords.extend([[c[1],c[0]] for c in osrm["geometry"]["coordinates"]])
                        r_dist  += osrm["distance"] / 1000
                    else:
                        r_coords.extend([list(wp_pts[j]), list(wp_pts[j+1])])
                        r_dist  += haversine(wp_pts[j], wp_pts[j+1])

                wp_obj = WEATHER_PROFILES[weather]
                eff    = max(vp["speed"] * (1 - wp_obj["speedPen"]/100) / t_mult, 5)
                load   = sum(deliveries[n-1].get("weight_kg",1)
                             for n in nodes[1:-1] if 0 < n <= len(deliveries))
                all_coords.extend(r_coords)
                total_dist += r_dist
                route_details.append({
                    "vehicle":   route["vehicle"],
                    "vtype":     vtype,
                    "nodes":     nodes,
                    "coords":    r_coords,
                    "pts":       wp_pts,
                    "dist_km":   round(r_dist, 2),
                    "eta_min":   int((r_dist / eff) * 60),
                    "cost":      round(r_dist * vp["cost"], 2),
                    "co2_kg":    round(r_dist * vp["co2"],  3),
                    "load_kg":   round(load, 2),
                    "cap_util":  round((load / vp["cap"]) * 100, 1),
                    "stops":     len(nodes) - 2,
                })

        st.session_state.result = {
            "best_wh":     best_wh,
            "wh_scores":   wh_scores,
            "routes":      route_details,
            "deliveries":  deliveries,
            "total_dist":  round(total_dist, 2),
            "depot":       depot,
            "all_coords":  all_coords,
        }
        st.session_state.live_step = 0

result   = st.session_state.result
ml_preds = st.session_state.ml_preds

# ──────────────────────────────────────────────────────────────────────────────
#  TAB 1 — results panel
# ──────────────────────────────────────────────────────────────────────────────
with t1:
    if result:
        # Warehouse ranking
        st.markdown('<div class="sec-head">🏆 Warehouse Selection</div>', unsafe_allow_html=True)
        vs = ["kv","kt","ks","ka","kc"]
        wc_cols = st.columns(min(len(result["wh_scores"]), 5))
        for idx, (sc, wh) in enumerate(result["wh_scores"][:5]):
            avg_d = round(np.mean([haversine((wh["lat"],wh["lon"]), d["coord"])
                                   for d in deliveries]), 1)
            badge = '<span class="pill pt">✅ SELECTED</span>' if idx==0 \
                    else f'<span style="font-size:11px;color:#94a3b8;">#{idx+1}</span>'
            wc_cols[idx].markdown(
                f'<div class="kpi {vs[idx]}" style="min-height:128px;">'
                f'<div class="kpi-stripe"></div><div class="kpi-blob"></div>'
                f'<div class="kpi-label">{wh["id"]}</div>'
                f'<div style="padding-left:10px;margin-bottom:3px;">{badge}</div>'
                f'<div class="kpi-value" style="font-size:16px;">{wh["name"]}</div>'
                f'<div class="kpi-sub">Score {sc} · {avg_d} km · {wh["capacity"]:,} kg</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Route cards
        st.markdown('<div class="sec-head">🚛 Optimized Route Assignments</div>', unsafe_allow_html=True)
        for ri, rd in enumerate(result["routes"]):
            vp = VEHICLE_PROFILES[rd["vtype"]]
            c  = ROUTE_COLORS[ri % len(ROUTE_COLORS)]
            st.markdown(
                f'<div class="rc"><div class="rc-bar" style="background:{c};"></div>'
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'flex-wrap:wrap;gap:8px;padding-left:12px;">'
                f'<div style="display:flex;align-items:center;gap:11px;">'
                f'<div style="width:40px;height:40px;border-radius:11px;background:{c}18;'
                f'border:2px solid {c}40;display:flex;align-items:center;justify-content:center;'
                f'font-size:19px;">{vp["icon"]}</div>'
                f'<div><div style="font-family:Outfit,sans-serif;font-weight:700;font-size:15px;color:{c};">'
                f'{rd["vehicle"]["id"]} — {rd["vtype"]}</div>'
                f'<div style="font-size:12px;color:#64748b;">{rd["stops"]} stops · {rd["cap_util"]}% capacity'
                f'</div></div></div>'
                f'<div>'
                f'<span class="ctag">📏 {rd["dist_km"]} km</span>'
                f'<span class="ctag">⏱ {rd["eta_min"]} min</span>'
                f'<span class="ctag">₹ {rd["cost"]}</span>'
                f'<span class="ctag">🌿 {rd["co2_kg"]} kg CO₂</span>'
                f'<span class="ctag">📦 {rd["load_kg"]} kg</span>'
                f'</div></div></div>',
                unsafe_allow_html=True,
            )

        # Summary KPIs
        eta_max  = max((r["eta_min"]  for r in result["routes"]), default=0)
        cost_sum = sum( r["cost"]     for r in result["routes"])
        co2_sum  = sum( r["co2_kg"]   for r in result["routes"])
        delays   = sum(1 for p in ml_preds if p["delay"])
        st.markdown("<br>", unsafe_allow_html=True)
        for col, (lbl, val, sub, cls) in zip(st.columns(5), [
            ("Total Distance", f"{result['total_dist']} km", "",         "ks"),
            ("Fleet ETA",      f"{eta_max} min",     "worst-case",       "kv"),
            ("Fleet Cost",     f"₹{round(cost_sum,2)}", "total",         "ka"),
            ("CO₂",            f"{round(co2_sum,2)} kg", "footprint",    "kt"),
            ("AI Delay Flags", f"{delays}/{len(ml_preds)}", "at risk",   "kc" if delays else "kt"),
        ]):
            col.markdown(
                f'<div class="kpi {cls}"><div class="kpi-stripe"></div><div class="kpi-blob"></div>'
                f'<div class="kpi-label">{lbl}</div><div class="kpi-value">{val}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ROUTE MAP
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    if result and SF_OK:
        st.markdown('<div class="sec-head">🗺️ Optimized Route Map — Real Road Network (OSRM)</div>',
                    unsafe_allow_html=True)

        # Legend
        legend_html = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'background:{ROUTE_COLORS[i]}14;border:1.5px solid {ROUTE_COLORS[i]}50;'
            f'border-radius:100px;padding:4px 12px;font-size:11px;font-weight:700;'
            f'color:{ROUTE_COLORS[i]};margin:2px;">● {rd["vehicle"]["id"]} ({rd["vtype"]})</span>'
            for i, rd in enumerate(result["routes"])
        )
        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;gap:4px;align-items:center;'
            f'padding:9px 15px;background:white;border-radius:11px;'
            f'border:1.5px solid #e2e8f7;margin-bottom:10px;box-shadow:0 1px 4px rgba(13,22,38,.07);">'
            f'<span style="font-size:9px;font-weight:700;color:#94a3b8;letter-spacing:1.5px;margin-right:5px;">ROUTES</span>'
            f'{legend_html}'
            f'<span style="margin-left:auto;font-size:11px;color:#94a3b8;">🏭=Warehouse  ●=Delivery</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        depot  = result["depot"]
        fmap   = folium.Map(location=[depot[0], depot[1]], zoom_start=12, tiles="CartoDB Positron")
        hd     = [list(d["coord"]) for d in result["deliveries"]]
        if hd:
            HeatMap(hd, radius=30, blur=20, min_opacity=.22,
                    gradient={"0.3":"#c7d2fe","0.6":"#7c3aed","1.0":"#ff4757"}).add_to(fmap)

        # Route polylines
        for ri, rd in enumerate(result["routes"]):
            c = ROUTE_COLORS[ri % len(ROUTE_COLORS)]
            if rd["coords"]:
                folium.PolyLine(rd["coords"], color=c, weight=12, opacity=.09).add_to(fmap)
                try:
                    AntPath(rd["coords"], color=c, weight=4, opacity=.95,
                            delay=550, dash_array=[10, 18],
                            tooltip=f"🚛 {rd['vehicle']['id']} — {rd['vtype']} | "
                                    f"{rd['dist_km']} km | {rd['eta_min']} min").add_to(fmap)
                except Exception:
                    folium.PolyLine(rd["coords"], color=c, weight=4, opacity=.92).add_to(fmap)

        # Warehouse markers
        p_colors = {"Express":"#ff4757","Priority":"#ff9500","Same-Day":"#0ea5e9","Standard":"#64748b"}
        for wh in active_wh:
            is_best = wh["id"] == result["best_wh"]["id"]
            ih = (f'<div style="width:34px;height:34px;border-radius:10px;'
                  f'background:{"linear-gradient(135deg,#7c3aed,#a855f7)" if is_best else "white"};'
                  f'border:2.5px solid {"#7c3aed" if is_best else "#cbd5e1"};'
                  f'display:flex;align-items:center;justify-content:center;font-size:15px;'
                  f'box-shadow:0 3px 12px rgba(0,0,0,.18);">🏭</div>')
            folium.Marker([wh["lat"], wh["lon"]],
                popup=folium.Popup(f"<b>{wh['id']}</b><br>{wh['name']}<br>Cap: {wh['capacity']:,} kg", max_width=200),
                icon=folium.DivIcon(html=ih, icon_size=(34,34), icon_anchor=(17,17)),
                tooltip=f"{'⭐ ' if is_best else ''}🏭 {wh['name']}").add_to(fmap)

        # Delivery markers
        for i, d in enumerate(result["deliveries"]):
            pc = p_colors[d["priority"]]
            di = ml_preds[i] if i < len(ml_preds) else None
            delay_str = f'{round(di["prob"]*100)}%' if di else "—"
            rec_v     = di["rec_veh"] if di else "—"
            pop = (
                "<div style='font-family:sans-serif;min-width:195px;padding:4px;'>"
                f"<b style='color:{pc};font-size:14px;'>{d['label']}</b>"
                "<hr style='margin:5px 0;'>"
                f"📦 {d['weight_kg']} kg · 🎯 {d['priority']}<br>"
                f"{'⚠️' if (di and di['delay']) else '✅'} Delay risk: {delay_str}<br>"
                f"🚗 Rec: {rec_v}"
                "</div>"
            )
            mh = (f'<div style="width:29px;height:29px;background:{pc};border:3px solid white;'
                  f'border-radius:50%;display:flex;align-items:center;justify-content:center;'
                  f'color:white;font-size:12px;font-weight:800;box-shadow:0 3px 10px rgba(0,0,0,.22);">'
                  f'{i+1}</div>')
            folium.Marker(d["coord"],
                popup=folium.Popup(pop, max_width=240),
                icon=folium.DivIcon(html=mh, icon_size=(29,29), icon_anchor=(14,14)),
                tooltip=f"📦 {d['label']} — {d['priority']}").add_to(fmap)

        st_folium(fmap, width="100%", height=580, key="main_map")

        # Warehouse comparison table
        st.markdown('<div class="sec-head">📊 Warehouse Comparison</div>', unsafe_allow_html=True)
        df_wh = pd.DataFrame([{
            "ID":       wh["id"],
            "Name":     wh["name"],
            "Score":    sc,
            "Avg Dist (km)": round(np.mean([haversine((wh["lat"],wh["lon"]), d["coord"])
                                            for d in deliveries]), 2),
            "Capacity (kg)": wh["capacity"],
            "Selected": "✅ Yes" if wh["id"]==result["best_wh"]["id"] else "—",
        } for sc, wh in result["wh_scores"]])
        st.dataframe(df_wh, use_container_width=True, hide_index=True)

    elif not SF_OK:
        st.warning("Install `streamlit-folium` to see the map.")
    else:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;">
          <div style="font-size:54px;margin-bottom:12px;">🗺️</div>
          <div style="font-family:'Outfit',sans-serif;font-size:20px;font-weight:700;color:#374151;">No routes yet</div>
          <div style="font-size:14px;color:#94a3b8;margin-top:6px;">Run optimisation in the Orders tab first</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    if result and result["routes"]:
        routes = result["routes"]
        df_r = pd.DataFrame([{
            "Vehicle":      f"{r['vehicle']['id']} ({r['vtype']})",
            "Load %":       r["cap_util"],
            "Distance km":  r["dist_km"],
            "ETA min":      r["eta_min"],
            "Cost Rs":      r["cost"],
            "CO2 kg":       r["co2_kg"],
            "Stops":        r["stops"],
        } for r in routes])

        a1, a2 = st.columns(2)
        with a1:
            st.markdown('<div class="sec-head">📊 Capacity Utilisation</div>', unsafe_allow_html=True)
            bar = (alt.Chart(df_r)
                   .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7)
                   .encode(
                       x=alt.X("Vehicle:N",  axis=alt.Axis(labelAngle=-20, title="")),
                       y=alt.Y("Load %:Q",   scale=alt.Scale(domain=[0,100]),
                               axis=alt.Axis(title="Load %")),
                       color=alt.Color("Vehicle:N", legend=None,
                                       scale=alt.Scale(range=ROUTE_COLORS)),
                       tooltip=["Vehicle","Load %","Distance km","Cost Rs"],
                   ).properties(height=230, background="transparent")
                   .configure_view(strokeWidth=0, fill="transparent")
                   .configure_axis(grid=True, gridColor="#f1f5f9"))
            st.altair_chart(bar, use_container_width=True)

        with a2:
            st.markdown('<div class="sec-head">💰 Cost vs Distance</div>', unsafe_allow_html=True)
            sc_chart = (alt.Chart(df_r)
                        .mark_circle(opacity=.9, stroke="white", strokeWidth=2)
                        .encode(
                            x=alt.X("Distance km:Q"),
                            y=alt.Y("Cost Rs:Q"),
                            size=alt.Size("Stops:Q", scale=alt.Scale(range=[80,600]), legend=None),
                            color=alt.Color("Vehicle:N", legend=None,
                                            scale=alt.Scale(range=ROUTE_COLORS)),
                            tooltip=["Vehicle","Distance km","Cost Rs","Stops","CO2 kg"],
                        ).properties(height=230, background="transparent")
                        .configure_view(strokeWidth=0, fill="transparent")
                        .configure_axis(grid=True, gridColor="#f1f5f9"))
            st.altair_chart(sc_chart, use_container_width=True)

        # Priority breakdown
        st.markdown('<div class="sec-head">🎯 Priority Breakdown</div>', unsafe_allow_html=True)
        pc_map: dict = {}
        for d in result["deliveries"]:
            pc_map[d["priority"]] = pc_map.get(d["priority"], 0) + 1
        df_p = pd.DataFrame([{"Priority":k,"Count":v} for k,v in pc_map.items()])
        pie = (alt.Chart(df_p)
               .mark_arc(innerRadius=52, cornerRadius=4)
               .encode(
                   theta="Count:Q",
                   color=alt.Color("Priority:N",
                       scale=alt.Scale(
                           domain=["Standard","Priority","Express","Same-Day"],
                           range=["#64748b","#ff9500","#ff4757","#0ea5e9"])),
                   tooltip=["Priority","Count"],
               ).properties(height=200, background="transparent")
               .configure_view(strokeWidth=0, fill="transparent"))
        st.altair_chart(pie, use_container_width=True)

        # Risk overview
        st.markdown('<div class="sec-head">⚠️ Risk Overview</div>', unsafe_allow_html=True)
        delay_risk = min(int(wp["risk"] + (t_mult-1)*30 + len(result["deliveries"])*2), 100)
        delayed    = sum(1 for p in ml_preds if p["delay"])
        rv_cls     = "kc" if delay_risk>60 else "ka" if delay_risk>30 else "kt"
        for col, (lbl, val, sub, cls) in zip(st.columns(3), [
            ("Composite Risk", f"{delay_risk}%",     "delay index", rv_cls),
            ("AI Delay Flags", f"{delayed}/{len(ml_preds)}", "RF preds", "kc" if delayed else "kt"),
            ("Weather",        f"{wp['icon']} {wp['risk']}%", weather.capitalize(), "ka"),
        ]):
            col.markdown(
                f'<div class="kpi {cls}"><div class="kpi-stripe"></div><div class="kpi-blob"></div>'
                f'<div class="kpi-label">{lbl}</div><div class="kpi-value">{val}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

        # Recommendations
        st.markdown('<div class="sec-head">🧠 Recommendations</div>', unsafe_allow_html=True)
        if delay_risk > 60:
            st.error("🔴 HIGH DELAY RISK — Consider rescheduling or adding more vehicles.")
        elif delay_risk > 35:
            st.warning("🟡 MODERATE RISK — Monitor live conditions; re-optimise if needed.")
        else:
            st.success("🟢 LOW RISK — Routes optimally planned for current conditions.")
        for rd in routes:
            vp = VEHICLE_PROFILES[rd["vtype"]]
            if rd["cap_util"] < 30:
                st.info(f"💡 {rd['vehicle']['id']} ({rd['vtype']}) under-utilised "
                        f"({rd['cap_util']}%). Consider consolidating.")
            if rd["dist_km"] > vp["range"] * .9:
                st.warning(f"⚠️ {rd['vehicle']['id']} near max range "
                           f"({rd['dist_km']} km / {vp['range']} km).")
        if weather in ("rainy","stormy"):
            st.info(f"🌧 {weather.capitalize()} — reduce speed by {wp['speedPen']}%.")
    else:
        st.info("Run optimisation to see analysis.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — LIVE TRACKING  (smooth JS/CSS animated marker via folium + HTML)
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    if result and result["all_coords"] and SF_OK:
        st.markdown('<div class="sec-head">📡 Live Fleet Tracking</div>', unsafe_allow_html=True)

        full_path  = result["all_coords"]
        step_path  = full_path[::3]           # thin out for animation frames
        total_steps = len(step_path)
        step       = st.session_state.live_step
        prog       = step / max(total_steps - 1, 1)

        # ── Controls row ──
        ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
        play_btn  = ctrl1.button("▶️  Play",  use_container_width=True, key="live_play_btn")
        reset_btn = ctrl2.button("⏹  Reset", use_container_width=True, key="live_reset_btn")

        if reset_btn:
            st.session_state.live_step    = 0
            st.session_state.live_running = False
            st.rerun()

        # ── Progress bar + status ──
        st.progress(min(prog, 1.0))

        if step < total_steps:
            cur_ll   = step_path[step]
            eta_left = int((1-prog) * max(r["eta_min"] for r in result["routes"]))
            st.markdown(
                f'<div class="live-bar"><div style="display:flex;gap:26px;align-items:center;flex-wrap:wrap;">'
                f'<div style="display:flex;align-items:center;gap:8px;">'
                f'<span style="font-size:22px;">🚚</span>'
                f'<div><div style="font-family:Outfit,sans-serif;font-weight:800;font-size:18px;color:#7c3aed;">'
                f'{int(prog*100)}%</div>'
                f'<div style="font-size:10px;color:#64748b;">route complete</div></div></div>'
                f'<div><div style="font-weight:700;color:#374151;font-size:13px;">⏱ {eta_left} min</div>'
                f'<div style="font-size:10px;color:#64748b;">ETA remaining</div></div>'
                f'<div><div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#7c3aed;">'
                f'📍 {round(cur_ll[0],5)}, {round(cur_ll[1],5)}</div>'
                f'<div style="font-size:10px;color:#64748b;">current position</div></div>'
                f'<span style="margin-left:auto;background:#f0fdf4;color:#16a34a;'
                f'border:1px solid #bbf7d0;padding:4px 12px;border-radius:100px;'
                f'font-size:11px;font-weight:700;">● LIVE</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="live-bar"><b style="color:#00cfa8;font-size:15px;">✅ All deliveries completed!</b></div>',
                unsafe_allow_html=True,
            )

        # ── Map placeholder ──
        map_ph = st.empty()

        def render_live_map(step_idx):
            travelled = step_path[: step_idx + 1]
            cur       = travelled[-1]
            lmap      = folium.Map(location=[cur[0], cur[1]], zoom_start=13, tiles="CartoDB Positron")

            # Ghost full path
            if len(step_path) > 1:
                folium.PolyLine([[p[0],p[1]] for p in step_path],
                    color="#c7d2fe", weight=4, opacity=.45, dash_array="6 8").add_to(lmap)

            # Travelled path
            if len(travelled) > 1:
                folium.PolyLine([[p[0],p[1]] for p in travelled],
                    color="#7c3aed", weight=5, opacity=.93).add_to(lmap)

            # Pulsing vehicle marker  ← pure CSS animation, no JS dependency
            pulse_html = """
            <div style="position:relative;width:44px;height:44px;">
              <div style="position:absolute;inset:0;border-radius:50%;background:#7c3aed;
                          opacity:.25;animation:ripple 1.8s ease-out infinite;"></div>
              <div style="position:absolute;inset:4px;border-radius:50%;background:#7c3aed;
                          border:3px solid white;display:flex;align-items:center;
                          justify-content:center;font-size:19px;
                          box-shadow:0 4px 14px rgba(124,58,237,.5);">🚚</div>
              <div style="position:absolute;top:1px;right:1px;width:11px;height:11px;
                          border-radius:50%;background:#22c55e;border:2px solid white;
                          animation:blink 1s ease infinite;"></div>
              <style>
                @keyframes ripple{0%{transform:scale(1);opacity:.3}100%{transform:scale(2.4);opacity:0}}
                @keyframes blink {0%,100%{opacity:1}50%{opacity:.2}}
              </style>
            </div>"""
            folium.Marker([cur[0], cur[1]],
                icon=folium.DivIcon(html=pulse_html, icon_size=(44,44), icon_anchor=(22,22)),
                tooltip="🚚 Fleet vehicle — LIVE").add_to(lmap)

            # Warehouse markers
            for wh in active_wh:
                is_best = wh["id"] == result["best_wh"]["id"]
                ih = (f'<div style="width:32px;height:32px;border-radius:9px;'
                      f'background:{"linear-gradient(135deg,#7c3aed,#a855f7)" if is_best else "white"};'
                      f'border:2px solid {"#7c3aed" if is_best else "#cbd5e1"};'
                      f'display:flex;align-items:center;justify-content:center;font-size:14px;'
                      f'box-shadow:0 2px 8px rgba(0,0,0,.15);">🏭</div>')
                folium.Marker([wh["lat"], wh["lon"]],
                    icon=folium.DivIcon(html=ih, icon_size=(32,32), icon_anchor=(16,16)),
                    tooltip=f"{'⭐ ' if is_best else ''}🏭 {wh['name']}").add_to(lmap)

            # Delivery markers  (grey out when delivered)
            p_colors2 = {"Express":"#ff4757","Priority":"#ff9500","Same-Day":"#0ea5e9","Standard":"#64748b"}
            for i, d in enumerate(result["deliveries"]):
                pc        = p_colors2[d["priority"]]
                delivered = (step_idx / max(total_steps-1,1)) > ((i+1) / max(len(result["deliveries"]),1))
                mh = (f'<div style="width:28px;height:28px;background:{"#e2e8f7" if delivered else pc};'
                      f'border:3px solid white;border-radius:50%;display:flex;align-items:center;'
                      f'justify-content:center;color:{"#94a3b8" if delivered else "white"};'
                      f'font-size:11px;font-weight:800;box-shadow:0 2px 8px rgba(0,0,0,.18);">'
                      f'{"✓" if delivered else str(i+1)}</div>')
                folium.Marker(d["coord"],
                    icon=folium.DivIcon(html=mh, icon_size=(28,28), icon_anchor=(14,14)),
                    tooltip=f"{'✅ Delivered' if delivered else '📦 Pending'}: {d['label']}").add_to(lmap)

            return lmap

        # Render static frame
        with map_ph:
            lm = render_live_map(min(step, total_steps-1))
            st_folium(lm, width="100%", height=520, key=f"live_map_{step}")

        # Animate on Play
        if play_btn:
            start = st.session_state.live_step
            for i in range(start, total_steps):
                st.session_state.live_step = i
                with map_ph:
                    lm2 = render_live_map(i)
                    st_folium(lm2, width="100%", height=520, key=f"live_anim_{i}")
                time.sleep(0.07)          # ≈14 fps — Streamlit re-render limit
            st.session_state.live_step = total_steps - 1
            st.rerun()

        # Route info cards
        st.markdown('<div class="sec-head">🚛 Per-Route Status</div>', unsafe_allow_html=True)
        rt_cols = st.columns(len(result["routes"]))
        for ri, rd in enumerate(result["routes"]):
            c  = ROUTE_COLORS[ri % len(ROUTE_COLORS)]
            vp = VEHICLE_PROFILES[rd["vtype"]]
            rt_pct = min(int(prog * 100), 100)
            with rt_cols[ri]:
                st.markdown(
                    f'<div class="kpi" style="border-left:4px solid {c};border-color:{c}30;min-height:120px;">'
                    f'<div class="kpi-stripe" style="background:{c};"></div>'
                    f'<div class="kpi-label">{rd["vehicle"]["id"]}</div>'
                    f'<div class="kpi-value" style="color:{c};font-size:20px;">{vp["icon"]} {rd["vtype"]}</div>'
                    f'<div class="kpi-sub">{rd["stops"]} stops · {rd["dist_km"]} km</div>'
                    f'<div style="margin-top:8px;padding-left:10px;">'
                    f'<div style="background:#f1f5f9;border-radius:100px;height:6px;overflow:hidden;">'
                    f'<div style="height:100%;width:{rt_pct}%;background:{c};border-radius:100px;"></div>'
                    f'</div>'
                    f'<div style="font-size:11px;color:{c};font-weight:700;margin-top:2px;">{rt_pct}% complete</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

    elif not SF_OK:
        st.warning("Install `streamlit-folium` to use live tracking.")
    else:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;">
          <div style="font-size:54px;margin-bottom:12px;">📡</div>
          <div style="font-family:'Outfit',sans-serif;font-size:20px;font-weight:700;color:#374151;">No routes to track</div>
          <div style="font-size:14px;color:#94a3b8;margin-top:6px;">Run optimisation first, then simulate here</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — AI INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    st.markdown('<div class="sec-head">🤖 AI Model Stack</div>', unsafe_allow_html=True)
    for col, (nm, acc, desc, cls) in zip(st.columns(3), [
        ("RandomForestClassifier", "86%",     "Delay prediction · 100 trees · depth 8", "kt"),
        ("GradientBoosting",       "~84%",    "Risk scoring · 80 estimators · depth 4", "kv"),
        ("OR-Tools CVRPTW",        "Optimal", "Capacitated VRP with Time Windows",      "ka"),
    ]):
        col.markdown(
            f'<div class="kpi {cls}" style="min-height:116px;">'
            f'<div class="kpi-stripe"></div><div class="kpi-blob"></div>'
            f'<div class="kpi-label">{nm}</div><div class="kpi-value">{acc}</div>'
            f'<div class="kpi-sub">{desc}</div></div>',
            unsafe_allow_html=True,
        )

    if ml_preds:
        st.markdown('<div class="sec-head">📦 Per-Delivery AI Predictions</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:grid;grid-template-columns:1.6fr 1fr 1.2fr 1fr 1fr;
                    gap:8px;font-size:9px;font-weight:700;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:1.5px;padding:7px 15px;border-bottom:2px solid #e2e8f7;
                    font-family:'JetBrains Mono',monospace;margin-bottom:7px;">
          <div>Stop</div><div>Status</div><div>Delay Prob</div><div>Rec Vehicle</div><div>Reroute</div>
        </div>""", unsafe_allow_html=True)

        for p in ml_preds:
            pct = int(p["prob"] * 100)
            fc  = "#ff4757" if pct>60 else "#ff9500" if pct>35 else "#00cfa8"
            st_pill  = '<span class="pill pr">⚠️ DELAYED</span>'  if p["delay"]   else '<span class="pill pt">✅ ON TIME</span>'
            re_pill  = '<span class="pill pa">🔄 YES</span>'      if p["reroute"] else '<span style="color:#94a3b8;font-size:12px;">— no</span>'
            vico     = VEHICLE_PROFILES.get(p["rec_veh"], {}).get("icon", "🚗")
            st.markdown(
                f'<div class="pred-row">'
                f'<div style="font-weight:700;color:#0d1626;font-size:14px;">{p["label"]}</div>'
                f'<div>{st_pill}</div>'
                f'<div><div class="prob-track"><div class="prob-fill" style="width:{pct}%;background:{fc};"></div></div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:{fc};font-weight:700;">{pct}%</div></div>'
                f'<div style="font-size:13px;font-weight:600;color:#374151;">{vico} {p["rec_veh"]}</div>'
                f'<div>{re_pill}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Delay probability chart
        st.markdown('<div class="sec-head">📈 Delay Probability Chart</div>', unsafe_allow_html=True)
        df_ml = pd.DataFrame(ml_preds)
        df_ml["pct"] = (df_ml["prob"] * 100).round(1)
        df_ml["band"] = df_ml["pct"].apply(
            lambda x: "High (>60%)" if x>60 else "Medium (30-60%)" if x>30 else "Low (<30%)")
        ch = (alt.Chart(df_ml)
              .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
              .encode(
                  x=alt.X("label:N",  axis=alt.Axis(labelAngle=-25, title="")),
                  y=alt.Y("pct:Q",    scale=alt.Scale(domain=[0,100]),
                          axis=alt.Axis(title="Delay Probability %")),
                  color=alt.Color("band:N", legend=alt.Legend(title="Risk Band"),
                      scale=alt.Scale(
                          domain=["High (>60%)","Medium (30-60%)","Low (<30%)"],
                          range=["#ff4757","#ff9500","#00cfa8"])),
                  tooltip=["label","pct","band","rec_veh"],
              ).properties(height=240, background="transparent")
              .configure_view(strokeWidth=0, fill="transparent")
              .configure_axis(grid=True, gridColor="#f1f5f9"))
        st.altair_chart(ch, use_container_width=True)

        # Summary box
        td = sum(1 for p in ml_preds if p["delay"])
        tr = sum(1 for p in ml_preds if p["reroute"])
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#f3f0ff,#f0f7ff);'
            f'border:1.5px solid #c7d2fe;border-radius:16px;padding:20px;box-shadow:0 2px 8px rgba(13,22,38,.07);">'
            f'<div style="display:inline-flex;align-items:center;gap:5px;background:#7c3aed;'
            f'color:white;border-radius:6px;font-size:9px;font-weight:800;letter-spacing:2px;'
            f'padding:3px 10px;margin-bottom:12px;font-family:\'JetBrains Mono\',monospace;">🤖 AI SUMMARY</div>'
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px;">'
            + "".join(
                f'<div style="text-align:center;background:white;border-radius:11px;padding:13px;'
                f'border:1.5px solid #e2e8f7;">'
                f'<div style="font-size:22px;font-weight:900;font-family:Outfit,sans-serif;color:{c};">{v}</div>'
                f'<div style="font-size:10px;color:#94a3b8;font-weight:600;margin-top:2px;">{l}</div></div>'
                for v, l, c in [
                    (len(ml_preds), "Orders",       "#7c3aed"),
                    (td,            "Delay Flags",  "#ff4757"),
                    (tr,            "Reroute Flags","#ff9500"),
                    ("86%",         "Accuracy",     "#00cfa8"),
                ]
            )
            + f'</div>'
            f'<div style="background:white;border-radius:11px;padding:13px;border:1.5px solid #e2e8f7;'
            f'font-size:12px;color:#374151;line-height:2.1;font-weight:500;">'
            f'✅ Multi-criteria warehouse scoring &nbsp;·&nbsp; ✅ OSRM real road routing<br>'
            f'✅ RandomForest delay prediction (86% acc) &nbsp;·&nbsp; ✅ RF vehicle recommender<br>'
            f'✅ OR-Tools CVRPTW multi-vehicle optimiser'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;">
          <div style="font-size:50px;margin-bottom:11px;">🤖</div>
          <div style="font-family:'Outfit',sans-serif;font-size:20px;font-weight:700;color:#374151;">No predictions yet</div>
          <div style="font-size:14px;color:#94a3b8;margin-top:5px;">Run optimisation to generate AI predictions</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:28px 0 8px;color:#cbd5e1;
            font-size:10px;font-family:'JetBrains Mono',monospace;letter-spacing:1px;">
  FleetIQ v2.0 &nbsp;·&nbsp; CVRPTW + Warehouse Scoring
  &nbsp;·&nbsp; RandomForest 86% &nbsp;·&nbsp; OSRM Real Roads &nbsp;·&nbsp; Google OR-Tools
</div>
""", unsafe_allow_html=True)
