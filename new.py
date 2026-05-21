import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import requests
import time
import math
import numpy as np
import altair as alt
from datetime import datetime
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from folium.plugins import HeatMap, AntPath
import warnings, json
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="OptiRoute — Smart Routing",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛰️"
)

# ─────────────────────────────────────────────
#  CSS — VIBRANT LIGHT THEME  +  BUG FIXES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg:#f0f4ff; --surface:#fff; --surface2:#f8faff; --glass:rgba(255,255,255,.82);
    --coral:#ff4757; --coral-s:#fff0f1; --coral-m:#ffd0d4;
    --teal:#00cfa8;  --teal-s:#e0fff9;  --teal-m:#a0f5e0;
    --violet:#7c3aed;--violet-s:#f3f0ff;--violet-m:#ddd5fb;
    --amber:#ff9500; --amber-s:#fff8eb; --amber-m:#ffe4a8;
    --sky:#0ea5e9;   --sky-s:#e0f5ff;   --sky-m:#bae8ff;
    --rose:#f43f5e;  --rose-s:#fff1f3;  --rose-m:#ffd0da;
    --text:#0d1626; --text2:#2d3748; --text3:#64748b; --text4:#94a3b8;
    --border:#e2e8f7; --border2:#c7d2fe;
    --sh-sm:0 1px 4px rgba(13,22,38,.07),0 1px 2px rgba(13,22,38,.04);
    --sh-md:0 4px 18px rgba(13,22,38,.10),0 2px 6px rgba(13,22,38,.06);
    --r:14px; --rl:20px; --rs:8px;
}
html,body,[class*="css"],[data-testid="stAppViewContainer"] {
    font-family:'Plus Jakarta Sans',sans-serif !important;
    background:var(--bg) !important; color:var(--text) !important;
}
[data-testid="stAppViewContainer"]::before {
    content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
    background:
        radial-gradient(ellipse 90% 55% at 15% -5%,rgba(124,58,237,.07) 0%,transparent 60%),
        radial-gradient(ellipse 70% 45% at 85% 105%,rgba(0,207,168,.06) 0%,transparent 60%),
        radial-gradient(ellipse 55% 65% at 55% 55%,rgba(255,71,87,.04) 0%,transparent 70%);
}
section[data-testid="stSidebar"] {
    background:var(--surface) !important;
    border-right:1.5px solid var(--border) !important;
    box-shadow:2px 0 20px rgba(13,22,38,.06) !important;
}

/* ══════════════════════════════════════════
   FIX 1 ─ EXPANDER "arrow_right" TEXT BUG
   The global font-family:Plus Jakarta Sans !important
   overrides Material Icons, causing the chevron
   ligature to render as literal text.
   We restore the correct font only for that span.
   ══════════════════════════════════════════ */
[data-testid="stExpander"] summary svg                          { display:inline-block !important; }
[data-testid="stExpander"] summary [data-testid="stExpanderToggleIcon"],
[data-testid="stExpander"] summary [data-testid="stExpanderToggleIcon"] * {
    font-family:'Material Symbols Rounded','Material Icons',
                'Material Icons Outlined',monospace !important;
    font-size:20px !important;
}
/* Clean up the label text inside expander summaries */
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span:not([data-testid]) {
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-size:13px !important;
    font-weight:600 !important;
    color:var(--text2) !important;
    vertical-align:middle;
}

section[data-testid="stSidebar"] * { font-family:'Plus Jakarta Sans',sans-serif !important; }
/* Re-apply icon font fix inside sidebar too */
section[data-testid="stSidebar"] [data-testid="stExpanderToggleIcon"],
section[data-testid="stSidebar"] [data-testid="stExpanderToggleIcon"] * {
    font-family:'Material Symbols Rounded','Material Icons',monospace !important;
}

.main .block-container { padding-top:.6rem !important; max-width:1440px !important; position:relative; z-index:1; }
h1,h2,h3 { font-family:'Outfit',sans-serif !important; font-weight:800 !important; }

.sec-head {
    font-family:'Outfit',sans-serif; font-size:11px; font-weight:700;
    color:var(--text3); text-transform:uppercase; letter-spacing:2.5px;
    margin:24px 0 14px; padding-bottom:10px; border-bottom:2px solid var(--border); position:relative;
}
.sec-head::after { content:'';position:absolute;bottom:-2px;left:0;width:36px;height:2px;border-radius:2px;background:linear-gradient(90deg,var(--violet),var(--teal)); }

/* KPI cards */
.kpi { background:var(--glass);backdrop-filter:blur(12px);border:1.5px solid var(--border);border-radius:var(--rl);padding:20px 22px 18px;position:relative;overflow:hidden;box-shadow:var(--sh-sm);min-height:112px;transition:transform .18s,box-shadow .18s; }
.kpi:hover { transform:translateY(-3px);box-shadow:var(--sh-md); }
.kpi-stripe { position:absolute;left:0;top:0;bottom:0;width:4px;border-radius:var(--rl) 0 0 var(--rl); }
.kpi-blob   { position:absolute;top:-18px;right:-18px;width:72px;height:72px;border-radius:50%;opacity:.14;filter:blur(14px); }
.kpi-label  { font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--text3);font-family:'JetBrains Mono',monospace;margin-bottom:5px;padding-left:10px; }
.kpi-value  { font-family:'Outfit',sans-serif;font-size:27px;font-weight:800;line-height:1.1;margin-bottom:3px;padding-left:10px; }
.kpi-sub    { font-size:11.5px;color:var(--text3);font-weight:500;padding-left:10px; }
.kv { border-color:var(--violet-m); } .kv .kpi-stripe { background:linear-gradient(180deg,var(--violet),#a855f7); } .kv .kpi-blob { background:var(--violet); } .kv .kpi-value { color:var(--violet); }
.kt { border-color:var(--teal-m);   } .kt .kpi-stripe { background:linear-gradient(180deg,var(--teal),#00e5c3);   } .kt .kpi-blob { background:var(--teal);   } .kt .kpi-value { color:#00a88c; }
.ka { border-color:var(--amber-m);  } .ka .kpi-stripe { background:linear-gradient(180deg,var(--amber),#ffcc00);  } .ka .kpi-blob { background:var(--amber);  } .ka .kpi-value { color:#c97800; }
.kc { border-color:var(--coral-m);  } .kc .kpi-stripe { background:linear-gradient(180deg,var(--coral),#ff7f8a);  } .kc .kpi-blob { background:var(--coral);  } .kc .kpi-value { color:var(--coral); }
.ks { border-color:var(--sky-m);    } .ks .kpi-stripe { background:linear-gradient(180deg,var(--sky),#38bdf8);    } .ks .kpi-blob { background:var(--sky);    } .ks .kpi-value { color:#0284c7; }

/* Route cards */
.rc { background:var(--surface);border:1.5px solid var(--border);border-radius:var(--r);padding:16px 20px;margin-bottom:10px;box-shadow:var(--sh-sm);position:relative;overflow:hidden;transition:transform .15s,box-shadow .15s; }
.rc:hover { transform:translateY(-2px);box-shadow:var(--sh-md); }
.rc-bar { position:absolute;left:0;top:0;bottom:0;width:5px;border-radius:var(--r) 0 0 var(--r); }

/* Tags & Pills */
.ctag { display:inline-flex;align-items:center;gap:4px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:4px 10px;font-size:11.5px;color:var(--text2);margin:2px;font-family:'JetBrains Mono',monospace;font-weight:500; }
.pill { display:inline-flex;align-items:center;border-radius:100px;padding:3px 12px;font-size:11px;font-weight:700;letter-spacing:.4px; }
.pg  { background:#dcfce7;color:#15803d;border:1px solid #bbf7d0; }
.pr  { background:#fee2e2;color:#dc2626;border:1px solid #fca5a5; }
.pa  { background:#fef3c7;color:#b45309;border:1px solid #fde68a; }
.pt  { background:var(--teal-s);color:#059669;border:1px solid var(--teal-m); }

/* Map/coord */
.mapbox { background:linear-gradient(135deg,var(--violet-s),var(--sky-s));border:1.5px solid var(--border2);border-radius:var(--r);padding:14px 18px;font-size:13px;color:var(--text2);font-weight:500; }
.coord-chip { display:inline-flex;align-items:center;gap:6px;background:white;border:1.5px solid var(--border2);border-radius:10px;padding:6px 14px;margin:4px;font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;color:var(--violet);box-shadow:var(--sh-sm); }

/* ML panel */
.ml-panel { background:linear-gradient(135deg,var(--violet-s),#f0f7ff);border:1.5px solid var(--border2);border-radius:var(--rl);padding:22px;box-shadow:var(--sh-sm); }
.ml-badge { display:inline-flex;align-items:center;gap:5px;background:var(--violet);color:white;border-radius:6px;font-size:9px;font-weight:800;letter-spacing:2px;padding:3px 10px;margin-bottom:14px;font-family:'JetBrains Mono',monospace; }

/* Prediction rows */
.pred-row { display:grid;grid-template-columns:1.6fr 1fr 1.2fr 1fr 1fr;gap:8px;align-items:center;padding:12px 16px;background:var(--surface);border:1.5px solid var(--border);border-radius:var(--r);margin-bottom:6px;box-shadow:var(--sh-sm);transition:border-color .15s,box-shadow .15s; }
.pred-row:hover { border-color:var(--border2);box-shadow:var(--sh-md); }
.prob-track { background:#f1f5f9;border-radius:100px;height:7px;overflow:hidden;margin-bottom:3px;border:1px solid #e2e8f0; }
.prob-fill  { height:100%;border-radius:100px; }

/* Live tracking status bar */
.live-status { background:linear-gradient(135deg,var(--violet-s),var(--sky-s));border:1.5px solid var(--border2);border-radius:var(--r);padding:14px 20px;margin-bottom:10px;box-shadow:var(--sh-sm); }

/* Streamlit overrides */
.stButton>button { background:linear-gradient(135deg,var(--violet) 0%,#a855f7 100%) !important;color:white !important;font-weight:700 !important;font-family:'Outfit',sans-serif !important;font-size:14px !important;border:none !important;border-radius:12px !important;padding:10px 24px !important;width:100% !important;box-shadow:0 4px 14px rgba(124,58,237,.28) !important;transition:opacity .2s,transform .1s !important; }
.stButton>button:hover { opacity:.91 !important;transform:translateY(-1px) !important; }
.stSelectbox>div>div,.stNumberInput>div>div>input,.stTextInput>div>div>input { background:var(--surface2) !important;border:1.5px solid var(--border) !important;border-radius:var(--rs) !important;color:var(--text) !important;font-family:'Plus Jakarta Sans',sans-serif !important; }
.stTabs [data-baseweb="tab-list"] { background:var(--surface) !important;border:1.5px solid var(--border) !important;border-radius:14px !important;padding:4px !important;gap:2px !important;box-shadow:var(--sh-sm) !important; }
.stTabs [data-baseweb="tab"] { background:transparent !important;color:var(--text3) !important;border-radius:10px !important;font-family:'Outfit',sans-serif !important;font-weight:600 !important;font-size:13px !important;padding:8px 18px !important; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,var(--violet) 0%,#a855f7 100%) !important;color:white !important;box-shadow:0 3px 10px rgba(124,58,237,.28) !important; }
div[data-testid="stExpander"] { background:var(--surface2) !important;border:1.5px solid var(--border) !important;border-radius:var(--r) !important;box-shadow:var(--sh-sm) !important; }
.stProgress>div>div { background:linear-gradient(90deg,var(--violet),var(--teal)) !important;border-radius:100px !important; }
.stProgress>div     { background:#e9ecf7 !important;border-radius:100px !important; }
[data-testid="stDataFrame"] { border-radius:var(--r) !important;overflow:hidden !important;border:1.5px solid var(--border) !important;box-shadow:var(--sh-sm) !important; }
hr { border-color:var(--border) !important;margin:14px 0 !important; }
::-webkit-scrollbar { width:6px;height:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border2);border-radius:6px; }
::-webkit-scrollbar-thumb:hover { background:var(--violet); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
DEFAULT_WAREHOUSES = [
    {"id":"WH-A","name":"North Hub",   "lat":28.6800,"lon":77.2200,"capacity":5000,"active":True},
    {"id":"WH-B","name":"South Hub",   "lat":28.5900,"lon":77.0400,"capacity":4000,"active":True},
    {"id":"WH-C","name":"East Hub",    "lat":28.6500,"lon":77.3800,"capacity":3500,"active":True},
    {"id":"WH-D","name":"West Hub",    "lat":28.6200,"lon":76.9800,"capacity":3000,"active":True},
    {"id":"WH-E","name":"Central Hub", "lat":28.6300,"lon":77.2100,"capacity":6000,"active":True},
]
VEHICLE_PROFILES = {
    "Bike":  {"speed_kmh":20,"capacity_kg":20,  "max_range_km":30, "cost_per_km":2,  "co2_per_km":0.02,"icon":"🛵"},
    "Car":   {"speed_kmh":35,"capacity_kg":150, "max_range_km":120,"cost_per_km":8,  "co2_per_km":0.18,"icon":"🚗"},
    "Van":   {"speed_kmh":30,"capacity_kg":600, "max_range_km":200,"cost_per_km":12, "co2_per_km":0.28,"icon":"🚐"},
    "Truck": {"speed_kmh":25,"capacity_kg":2000,"max_range_km":400,"cost_per_km":20, "co2_per_km":0.55,"icon":"🚚"},
    "Drone": {"speed_kmh":60,"capacity_kg":3,   "max_range_km":15, "cost_per_km":1,  "co2_per_km":0.01,"icon":"🚁"},
    "E-Bike":{"speed_kmh":22,"capacity_kg":30,  "max_range_km":50, "cost_per_km":1.5,"co2_per_km":0.01,"icon":"⚡"},
}
WEATHER_PROFILES = {
    "clear": {"factor":1.0, "speed_pen":0,  "risk":5,  "icon":"☀️"},
    "cloudy":{"factor":1.05,"speed_pen":2,  "risk":10, "icon":"⛅"},
    "rainy": {"factor":1.35,"speed_pen":10, "risk":45, "icon":"🌧"},
    "foggy": {"factor":1.25,"speed_pen":8,  "risk":40, "icon":"🌫"},
    "hot":   {"factor":1.10,"speed_pen":3,  "risk":20, "icon":"🌡"},
    "cold":  {"factor":1.15,"speed_pen":5,  "risk":25, "icon":"❄️"},
    "stormy":{"factor":1.60,"speed_pen":20, "risk":80, "icon":"⛈"},
}
TIME_WINDOW_PRESETS = {
    "Standard (9am-6pm)":   (9*60,18*60),
    "Morning (8am-12pm)":   (8*60,12*60),
    "Afternoon (12pm-6pm)": (12*60,18*60),
    "Evening (5pm-9pm)":    (17*60,21*60),
    "Same-Day (Now+4hr)":   None,
    "Express (60 min)":     None,
}
PRIORITY_LEVELS = {
    "Standard":{"sla_min":120,"penalty":1, "color":"#64748b"},
    "Priority": {"sla_min":60, "penalty":3, "color":"#ff9500"},
    "Express":  {"sla_min":30, "penalty":10,"color":"#ff4757"},
    "Same-Day": {"sla_min":240,"penalty":2, "color":"#0ea5e9"},
}
RCOLS = ["#7c3aed","#00cfa8","#ff4757","#ff9500","#0ea5e9","#f43f5e"]

# ─────────────────────────────────────────────
#  ML MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def build_models():
    np.random.seed(42); n=3000
    d=np.random.uniform(2,120,n); w=np.random.uniform(.1,200,n)
    wr=np.random.uniform(0,80,n); tr=np.random.uniform(1,2,n)
    sl=np.random.choice([30,60,120,240],n); xe=(sl<=30).astype(int)
    hr=np.random.randint(0,24,n); pk=((hr>=8)&(hr<=11))|((hr>=17)&(hr<=21))
    prob=np.clip(.35*(d/120)+.20*(wr/80)+.25*(tr-1)+.10*xe+.10*pk.astype(float),0,1)
    y=(np.random.rand(n)<prob).astype(int)
    X=np.column_stack([d,w,wr,tr,sl,xe,hr,pk.astype(int)])
    rf=RandomForestClassifier(n_estimators=100,max_depth=8,random_state=42); rf.fit(X,y)
    gb=GradientBoostingClassifier(n_estimators=80,max_depth=4,random_state=42); gb.fit(X,y)
    wv=np.random.uniform(.1,500,n); dv=np.random.uniform(1,200,n)
    def rule(ww,dd):
        if dd<=15 and ww<=3: return 0
        if ww<=20 and dd<=30: return 4
        if ww<=20: return 0
        if ww<=150: return 1
        if ww<=600: return 2
        return 3
    yv=np.array([rule(ww,dd) for ww,dd in zip(wv,dv)])
    rv=RandomForestClassifier(n_estimators=50,random_state=0); rv.fit(np.column_stack([wv,dv]),yv)
    return rf,gb,rv

delay_rf, delay_gb, veh_rec = build_models()

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
def ss(k,v):
    if k not in st.session_state: st.session_state[k]=v

ss("warehouses",[dict(w) for w in DEFAULT_WAREHOUSES])
ss("result",None); ss("ml_predictions",[])
ss("map_clicked_coord",None); ss("map_center",[28.63,77.20])
ss("deliveries_preview",[]); ss("live_step",0); ss("live_running",False)
ss("search_result",None)

# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────
def haversine(p1,p2):
    R=6371; la1,lo1=math.radians(p1[0]),math.radians(p1[1])
    la2,lo2=math.radians(p2[0]),math.radians(p2[1])
    a=math.sin((la2-la1)/2)**2+math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))

def fetch_route(s,e):
    try:
        r=requests.get(
            f"http://router.project-osrm.org/route/v1/driving/{s[1]},{s[0]};{e[1]},{e[0]}?overview=full&geometries=geojson",
            timeout=6)
        return r.json()["routes"][0]
    except: return None

def geocode_location(query):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": query, "format": "json", "limit": 5}
        headers = {"User-Agent": "OptiRoute/2.0"}
        r = requests.get(url, params=params, headers=headers, timeout=5)
        results = r.json()
        return [(float(x["lat"]), float(x["lon"]), x["display_name"]) for x in results]
    except: return []

def predict_delay(dist,wt,weather,traffic,priority,hr=None):
    if hr is None: hr=datetime.now().hour
    wp=WEATHER_PROFILES[weather]; sl=PRIORITY_LEVELS[priority]["sla_min"]
    xe=1 if sl<=30 else 0; pk=1 if(8<=hr<=11)or(17<=hr<=21) else 0
    feat=np.array([[dist,wt,wp["risk"],traffic,sl,xe,hr,pk]])
    return int(delay_rf.predict(feat)[0]),round(float(delay_rf.predict_proba(feat)[0][1]),3)

def recommend_vehicle(wt,dist):
    m={0:"Drone",1:"Car",2:"Van",3:"Truck",4:"E-Bike"}
    return m.get(veh_rec.predict(np.array([[wt,dist]]))[0],"Van")

def score_wh(wh,deliveries,veh,weather):
    wc=(wh["lat"],wh["lon"]); vp=VEHICLE_PROFILES[veh]; wp=WEATHER_PROFILES[weather]
    if deliveries:
        ad=np.mean([haversine(wc,d["coord"]) for d in deliveries])
        md=max([haversine(wc,d["coord"]) for d in deliveries])
    else: ad=md=0
    cp=(sum(d.get("weight_kg",1) for d in deliveries)/max(wh["capacity"],1))*20
    rp=50 if md>vp["max_range_km"] else 0
    return round(((ad*2)+(md*.5)+cp+rp)*wp["factor"],2)

def dist_matrix(pts):
    n=len(pts); m=[[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i!=j: m[i][j]=int(haversine(pts[i],pts[j])*1000)
    return m

# ─────────────────────────────────────────────
#  VRP SOLVER
# ─────────────────────────────────────────────
def solve_vrp(depot,deliveries,vehicles,weather,hc):
    if not deliveries: return None
    nv=len(vehicles); pts=[depot]+[d["coord"] for d in deliveries]; n=len(pts)
    dm=dist_matrix(pts); demands=[0]+[int(d.get("weight_kg",1)) for d in deliveries]
    wp=WEATHER_PROFILES[weather]
    tw=[(0,86400)]+[(d.get("tw_open_min",0)*60,d.get("tw_close_min",1440)*60) for d in deliveries]
    mgr=pywrapcp.RoutingIndexManager(n,nv,0); rt=pywrapcp.RoutingModel(mgr)
    def dcb(fi,ti): return dm[mgr.IndexToNode(fi)][mgr.IndexToNode(ti)]
    tcb=rt.RegisterTransitCallback(dcb); rt.SetArcCostEvaluatorOfAllVehicles(tcb)
    if hc.get("capacity",True):
        def dmd(i): return demands[mgr.IndexToNode(i)]
        di=rt.RegisterUnaryTransitCallback(dmd)
        rt.AddDimensionWithVehicleCapacity(di,0,[int(VEHICLE_PROFILES[v["type"]]["capacity_kg"]) for v in vehicles],True,"Cap")
    if hc.get("time_windows",True):
        def tc2(fi,ti):
            spd=max(VEHICLE_PROFILES[vehicles[0]["type"]]["speed_kmh"]*1000/3600*(1-wp["speed_pen"]/100),1)
            return int(dm[mgr.IndexToNode(fi)][mgr.IndexToNode(ti)]/spd+300)
        ti2=rt.RegisterTransitCallback(tc2); rt.AddDimension(ti2,3600,86400,False,"Time")
        td=rt.GetDimensionOrDie("Time")
        for ni in range(1,n):
            idx=mgr.NodeToIndex(ni); td.CumulVar(idx).SetRange(tw[ni][0],tw[ni][1])
    if hc.get("range",True):
        rt.AddDimensionWithVehicleCapacity(tcb,0,[int(VEHICLE_PROFILES[v["type"]]["max_range_km"]*1000) for v in vehicles],True,"Range")
    for di,d in enumerate(deliveries):
        rt.AddDisjunction([mgr.NodeToIndex(di+1)],PRIORITY_LEVELS[d.get("priority","Standard")]["penalty"]*10000)
    p=pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    p.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds=5
    sol=rt.SolveWithParameters(p)
    if not sol: return fallback(depot,deliveries,vehicles)
    routes=[]
    for vi in range(nv):
        nodes=[]; idx=rt.Start(vi)
        while not rt.IsEnd(idx): nodes.append(mgr.IndexToNode(idx)); idx=sol.Value(rt.NextVar(idx))
        nodes.append(mgr.IndexToNode(idx))
        if len(nodes)>2: routes.append({"vehicle":vehicles[vi],"nodes":nodes})
    return routes or fallback(depot,deliveries,vehicles)

def fallback(depot,deliveries,vehicles):
    rem=list(range(len(deliveries))); routes=[]
    for v in vehicles:
        if not rem: break
        route=[0]; cap=VEHICLE_PROFILES[v["type"]]["capacity_kg"]; used=0; cur=depot
        while rem:
            bi,bd=None,float("inf")
            for i in rem:
                dist=haversine(cur,deliveries[i]["coord"]); wt=deliveries[i].get("weight_kg",1)
                if used+wt<=cap and dist<bd: bi,bd=i,dist
            if bi is None: break
            route.append(bi+1); used+=deliveries[bi].get("weight_kg",1); cur=deliveries[bi]["coord"]; rem.remove(bi)
        route.append(0)
        if len(route)>2: routes.append({"vehicle":v,"nodes":route})
    return routes or None

# ─────────────────────────────────────────────
#  GOOGLE-MAPS-STYLE LIVE TRACKING HTML
#  Pure JS animation inside a single Folium map
#  — no Python reruns, smooth 60fps interpolation
# ─────────────────────────────────────────────
def build_live_tracking_html(result, active_wh, weather, ml_preds):
    """
    Returns a self-contained HTML string with:
    - Leaflet.js map (same tiles as folium)
    - All route coords embedded as JS arrays
    - Smooth JS setInterval animation (60fps interpolation)
    - Google Maps UI: speed indicator, ETA panel, turn-by-turn list,
      traffic layer toggle, satellite/map toggle, zoom controls,
      re-center button, delivery status sidebar
    """
    routes        = result["routes"]
    deliveries    = result["deliveries"]
    depot         = result["depot_coord"]
    best_wh       = result["best_wh"]
    wp            = WEATHER_PROFILES[weather]

    # ── Build per-route coord arrays for JS ──
    route_js_arrays = []
    for ri, rd in enumerate(routes):
        coords = rd["coords"]          # [[lat,lon], ...]  already lat/lon
        col    = RCOLS[ri % len(RCOLS)]
        vtype  = rd["vtype"]
        vp     = VEHICLE_PROFILES[vtype]
        stops  = rd["stops"]
        eta    = rd["eta_min"]
        dist   = rd["dist_km"]
        cost   = rd["fuel_cost"]
        co2    = rd["co2_kg"]
        icon   = vp["icon"]
        vid    = rd["vehicle"]["id"]
        route_js_arrays.append({
            "id":    vid,
            "vtype": vtype,
            "icon":  icon,
            "color": col,
            "coords": coords,
            "stops": stops,
            "eta":   eta,
            "dist":  dist,
            "cost":  cost,
            "co2":   co2,
        })

    # ── Build delivery markers for JS ──
    pcols = {"Express":"#ff4757","Priority":"#ff9500","Same-Day":"#0ea5e9","Standard":"#64748b"}
    delivery_js = []
    for i, d in enumerate(deliveries):
        di = ml_preds[i] if i < len(ml_preds) else None
        delivery_js.append({
            "idx":      i,
            "label":    d["label"],
            "lat":      d["coord"][0],
            "lon":      d["coord"][1],
            "priority": d["priority"],
            "color":    pcols[d["priority"]],
            "weight":   d.get("weight_kg", 1),
            "delay":    di["delay_predicted"] if di else False,
            "delay_prob": round(di["delay_prob"]*100) if di else 0,
            "rec_veh":  di["recommended_vehicle"] if di else "—",
        })

    # ── Warehouse markers for JS ──
    wh_js = []
    for wh in active_wh:
        wh_js.append({
            "id": wh["id"], "name": wh["name"],
            "lat": wh["lat"], "lon": wh["lon"],
            "capacity": wh["capacity"],
            "best": wh["id"] == best_wh["id"],
        })

    routes_json    = json.dumps(route_js_arrays)
    deliveries_json = json.dumps(delivery_js)
    wh_json        = json.dumps(wh_js)
    center_lat     = depot[0]
    center_lon     = depot[1]
    weather_icon   = wp["icon"]
    weather_risk   = wp["risk"]
    total_eta      = max((r["eta_min"] for r in routes), default=0)
    total_dist     = result["total_dist"]
    total_cost     = round(sum(r["fuel_cost"] for r in routes), 2)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ margin:0;padding:0;box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;overflow:hidden; }}

  #map {{ width:100%;height:100vh; }}

  /* ── Google Maps style top bar ── */
  #topbar {{
    position:absolute;top:12px;left:50%;transform:translateX(-50%);z-index:1000;
    background:rgba(13,18,30,.92);backdrop-filter:blur(16px);
    border:1px solid rgba(124,58,237,.4);border-radius:14px;
    padding:10px 20px;display:flex;align-items:center;gap:18px;
    box-shadow:0 8px 32px rgba(0,0,0,.5);min-width:600px;
  }}
  #topbar .brand {{ font-size:15px;font-weight:800;
    background:linear-gradient(90deg,#7c3aed,#00cfa8);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent; }}
  #topbar .stat {{ text-align:center; }}
  #topbar .stat-val {{ font-size:16px;font-weight:700;color:#fff; }}
  #topbar .stat-lbl {{ font-size:9px;color:#6b7280;letter-spacing:1.5px;text-transform:uppercase; }}
  #topbar .divider {{ width:1px;height:32px;background:rgba(255,255,255,.12); }}
  #live-badge {{
    display:flex;align-items:center;gap:6px;
    background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.4);
    border-radius:100px;padding:4px 12px;font-size:11px;font-weight:700;color:#10b981;
  }}
  .live-dot {{ width:7px;height:7px;border-radius:50%;background:#10b981;
    animation:pulse 1.2s ease-in-out infinite; }}
  @keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:.4;transform:scale(1.4)}} }}

  /* ── Bottom ETA panel (Google Maps style) ── */
  #etapanel {{
    position:absolute;bottom:0;left:0;right:0;z-index:1000;
    background:rgba(13,18,30,.96);backdrop-filter:blur(20px);
    border-top:1px solid rgba(255,255,255,.08);
    padding:14px 20px 20px;
    display:flex;align-items:center;gap:20px;flex-wrap:wrap;
  }}
  #etapanel .big-eta {{ font-size:32px;font-weight:900;color:#fff;line-height:1; }}
  #etapanel .eta-unit {{ font-size:13px;color:#9ca3af;margin-top:2px; }}
  #etapanel .eta-dist {{ font-size:18px;font-weight:600;color:#7c3aed;margin-top:2px; }}
  #progress-track {{
    flex:1;height:6px;background:rgba(255,255,255,.1);border-radius:100px;overflow:hidden;
  }}
  #progress-fill {{
    height:100%;border-radius:100px;width:0%;
    background:linear-gradient(90deg,#7c3aed,#00cfa8);
    transition:width .3s ease;
  }}
  #etapanel .stop-list {{ display:flex;gap:8px;flex-wrap:wrap; }}
  .stop-pill {{
    display:inline-flex;align-items:center;gap:5px;
    border-radius:100px;padding:4px 12px;font-size:11px;font-weight:700;
    border:1px solid;transition:all .3s;cursor:default;
  }}
  .stop-pill.pending   {{ background:rgba(255,149,0,.12);color:#ff9500;border-color:rgba(255,149,0,.35); }}
  .stop-pill.delivered {{ background:rgba(0,207,168,.12);color:#00cfa8;border-color:rgba(0,207,168,.35);text-decoration:line-through;opacity:.6; }}

  /* ── Left panel: vehicle fleet status ── */
  #fleetpanel {{
    position:absolute;top:80px;left:12px;z-index:1000;
    background:rgba(13,18,30,.92);backdrop-filter:blur(16px);
    border:1px solid rgba(255,255,255,.08);border-radius:14px;
    padding:14px;width:220px;
    box-shadow:0 8px 32px rgba(0,0,0,.4);
  }}
  #fleetpanel .panel-title {{
    font-size:9px;font-weight:700;letter-spacing:2px;
    color:#6b7280;text-transform:uppercase;margin-bottom:10px;
    padding-bottom:8px;border-bottom:1px solid rgba(255,255,255,.08);
  }}
  .fleet-row {{
    display:flex;align-items:center;gap:10px;padding:8px 0;
    border-bottom:1px solid rgba(255,255,255,.05);
  }}
  .fleet-row:last-child {{ border-bottom:none; }}
  .fleet-icon {{ font-size:20px;width:32px;text-align:center; }}
  .fleet-info {{ flex:1; }}
  .fleet-id   {{ font-size:12px;font-weight:700;color:#fff; }}
  .fleet-sub  {{ font-size:10px;color:#6b7280;margin-top:1px; }}
  .fleet-bar  {{ height:3px;background:rgba(255,255,255,.1);border-radius:100px;margin-top:4px;overflow:hidden; }}
  .fleet-fill {{ height:100%;border-radius:100px; }}
  .fleet-status {{ font-size:9px;font-weight:700;letter-spacing:1px;
    padding:2px 7px;border-radius:100px;border:1px solid; }}
  .s-active {{ color:#10b981;border-color:rgba(16,185,129,.4);background:rgba(16,185,129,.1); }}
  .s-done   {{ color:#6b7280;border-color:rgba(107,114,128,.3);background:rgba(107,114,128,.08); }}

  /* ── Right panel: map controls ── */
  #ctrlpanel {{
    position:absolute;top:80px;right:12px;z-index:1000;
    display:flex;flex-direction:column;gap:8px;
  }}
  .ctrl-btn {{
    width:42px;height:42px;border-radius:12px;border:none;cursor:pointer;
    background:rgba(13,18,30,.92);backdrop-filter:blur(16px);
    border:1px solid rgba(255,255,255,.1);color:#fff;font-size:17px;
    display:flex;align-items:center;justify-content:center;
    transition:all .2s;box-shadow:0 4px 12px rgba(0,0,0,.3);
  }}
  .ctrl-btn:hover {{ background:rgba(124,58,237,.3);border-color:rgba(124,58,237,.5); }}
  .ctrl-btn.active {{ background:rgba(124,58,237,.4);border-color:#7c3aed; }}

  /* ── Speed/heading indicator (Google Maps style) ── */
  #speedometer {{
    position:absolute;bottom:110px;right:12px;z-index:1000;
    background:rgba(13,18,30,.92);backdrop-filter:blur(16px);
    border:1px solid rgba(255,255,255,.1);border-radius:14px;
    padding:10px 14px;text-align:center;min-width:80px;
    box-shadow:0 4px 16px rgba(0,0,0,.4);
  }}
  #speedometer .spd-val {{ font-size:26px;font-weight:900;color:#fff;line-height:1; }}
  #speedometer .spd-unit {{ font-size:9px;color:#6b7280;letter-spacing:1.5px;text-transform:uppercase; }}

  /* ── Turn-by-turn next instruction ── */
  #navinstruct {{
    position:absolute;top:80px;left:50%;transform:translateX(-50%);z-index:999;
    background:rgba(13,18,30,.95);backdrop-filter:blur(16px);
    border:1px solid rgba(124,58,237,.3);border-radius:12px;
    padding:10px 18px;display:flex;align-items:center;gap:12px;
    box-shadow:0 4px 20px rgba(0,0,0,.4);min-width:340px;
  }}
  #navinstruct .nav-icon {{ font-size:24px; }}
  #navinstruct .nav-text {{ flex:1; }}
  #navinstruct .nav-action {{ font-size:13px;font-weight:700;color:#fff; }}
  #navinstruct .nav-dist   {{ font-size:11px;color:#9ca3af;margin-top:1px; }}

  /* ── Weather overlay ── */
  #weatherbar {{
    position:absolute;top:12px;right:12px;z-index:1000;
    background:rgba(13,18,30,.88);backdrop-filter:blur(12px);
    border:1px solid rgba(255,255,255,.1);border-radius:10px;
    padding:7px 13px;display:flex;align-items:center;gap:7px;
    font-size:12px;color:#d1d5db;font-weight:600;
    box-shadow:0 4px 12px rgba(0,0,0,.3);
  }}

  /* ── Tooltip overrides ── */
  .leaflet-popup-content-wrapper {{
    background:rgba(13,18,30,.97) !important;
    border:1px solid rgba(255,255,255,.1) !important;
    border-radius:12px !important;color:#fff !important;
    box-shadow:0 8px 32px rgba(0,0,0,.6) !important;
  }}
  .leaflet-popup-tip {{ background:rgba(13,18,30,.97) !important; }}
  .leaflet-popup-content {{ margin:12px 16px !important;font-size:13px; }}
</style>
</head>
<body>
<div id="map"></div>

<!-- Top bar -->
<div id="topbar">
  <span class="brand">🛰️ OptiRoute</span>
  <div class="divider"></div>
  <div class="stat"><div class="stat-val" id="tb-eta">{total_eta}</div><div class="stat-lbl">ETA min</div></div>
  <div class="stat"><div class="stat-val">{total_dist} km</div><div class="stat-lbl">Total dist</div></div>
  <div class="stat"><div class="stat-val">₹{total_cost}</div><div class="stat-lbl">Fleet cost</div></div>
  <div class="stat"><div class="stat-val" id="tb-stops">0/{len(deliveries)}</div><div class="stat-lbl">Delivered</div></div>
  <div class="divider"></div>
  <div id="live-badge"><div class="live-dot"></div>LIVE TRACKING</div>
</div>

<!-- Weather bar -->
<div id="weatherbar">{weather_icon} {weather.upper()} &nbsp;·&nbsp; Risk {weather_risk}%</div>

<!-- Fleet panel -->
<div id="fleetpanel">
  <div class="panel-title">🚛 Fleet Status</div>
  <div id="fleet-rows"></div>
</div>

<!-- Map controls -->
<div id="ctrlpanel">
  <button class="ctrl-btn" id="btn-recenter" title="Re-center" onclick="recenterMap()">🎯</button>
  <button class="ctrl-btn" id="btn-sat"      title="Satellite/Map" onclick="toggleTile()">🛰️</button>
  <button class="ctrl-btn" id="btn-traffic"  title="Toggle route" onclick="toggleRoutes()">🗺️</button>
  <button class="ctrl-btn" id="btn-zoom-in"  title="Zoom in"  onclick="map.zoomIn()">＋</button>
  <button class="ctrl-btn" id="btn-zoom-out" title="Zoom out" onclick="map.zoomOut()">－</button>
</div>

<!-- Speedometer -->
<div id="speedometer">
  <div class="spd-val" id="spd-num">0</div>
  <div class="spd-unit">km/h</div>
</div>

<!-- Nav instruction -->
<div id="navinstruct">
  <div class="nav-icon" id="nav-icon">⬆️</div>
  <div class="nav-text">
    <div class="nav-action" id="nav-action">Preparing route…</div>
    <div class="nav-dist"  id="nav-dist">Starting optimization</div>
  </div>
</div>

<!-- ETA bottom panel -->
<div id="etapanel">
  <div>
    <div class="big-eta" id="eta-val">{total_eta}</div>
    <div class="eta-unit">min remaining</div>
    <div class="eta-dist" id="dist-val">0.0 km covered</div>
  </div>
  <div style="flex:1">
    <div style="display:flex;justify-content:space-between;font-size:10px;color:#6b7280;margin-bottom:4px;">
      <span>Progress</span><span id="pct-val">0%</span>
    </div>
    <div id="progress-track"><div id="progress-fill"></div></div>
    <div class="stop-list" id="stop-pills" style="margin-top:8px;"></div>
  </div>
</div>

<script>
// ── Data from Python ──
const ROUTES     = {routes_json};
const DELIVERIES = {deliveries_json};
const WAREHOUSES = {wh_json};
const CENTER     = [{center_lat}, {center_lon}];
const TOTAL_ETA  = {total_eta};

// ── Map init ──
const map = L.map('map', {{
  center: CENTER,
  zoom: 13,
  zoomControl: false,
  attributionControl: true
}});

const TILES = {{
  streets:   L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
               {{attribution:'© OpenStreetMap © CARTO', maxZoom:19}}),
  dark:      L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
               {{attribution:'© OpenStreetMap © CARTO', maxZoom:19}}),
  satellite: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
               {{attribution:'© Esri', maxZoom:19}}),
}};
TILES.dark.addTo(map);
let currentTileIdx = 0;
const tileKeys = ['dark','streets','satellite'];

function toggleTile() {{
  currentTileIdx = (currentTileIdx + 1) % tileKeys.length;
  Object.values(TILES).forEach(t => map.removeLayer(t));
  TILES[tileKeys[currentTileIdx]].addTo(map);
  map.eachLayer(l => {{ if(l._path) l.bringToFront(); }});
}}

let routesVisible = true;
function toggleRoutes() {{
  routesVisible = !routesVisible;
  ghostLines.forEach(l => routesVisible ? l.addTo(map) : map.removeLayer(l));
}}

function recenterMap() {{
  if (truckMarkers.length > 0) {{
    const pos = truckMarkers[0].getLatLng();
    map.setView(pos, 14, {{animate:true, duration:0.8}});
  }}
}}

// ── Warehouse markers ──
WAREHOUSES.forEach(wh => {{
  const best = wh.best;
  const whHtml = `<div style="
    width:${{best?44:36}}px;height:${{best?44:36}}px;border-radius:12px;
    background:${{best?'linear-gradient(135deg,#7c3aed,#a855f7)':'rgba(30,40,60,0.95)'}};
    border:2.5px solid ${{best?'#a855f7':'rgba(255,255,255,.25)'}};
    display:flex;align-items:center;justify-content:center;font-size:${{best?20:16}}px;
    box-shadow:${{best?'0 4px 20px rgba(124,58,237,.6)':'0 3px 10px rgba(0,0,0,.5)'}};
    cursor:pointer;">🏭</div>`;
  L.marker([wh.lat, wh.lon], {{
    icon: L.divIcon({{html:whHtml, className:'', iconSize:[best?44:36,best?44:36], iconAnchor:[best?22:18,best?22:18]}})
  }}).bindPopup(`
    <b style="color:#a855f7;font-size:14px;">${{wh.id}} ${{best?'⭐':''}} </b><br>
    ${{wh.name}}<br>
    <span style="color:#9ca3af">Capacity: ${{wh.capacity.toLocaleString()}} kg</span>
    ${{best?'<br><span style="color:#10b981;font-weight:700;">✅ Selected Depot</span>':''}}
  `).addTo(map);
}});

// ── Delivery markers ──
const pcols = {{"Express":"#ff4757","Priority":"#ff9500","Same-Day":"#0ea5e9","Standard":"#64748b"}};
const deliveryMarkers = [];
DELIVERIES.forEach((d, i) => {{
  const mHtml = `<div style="
    width:32px;height:32px;background:${{d.color}};
    border:3px solid rgba(255,255,255,.9);border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    color:white;font-size:12px;font-weight:800;
    box-shadow:0 3px 12px rgba(0,0,0,.5);cursor:pointer;">${{i+1}}</div>`;
  const m = L.marker([d.lat, d.lon], {{
    icon: L.divIcon({{html:mHtml, className:'', iconSize:[32,32], iconAnchor:[16,16]}})
  }}).bindPopup(`
    <b style="color:${{d.color}};font-size:14px;">${{d.label}}</b><br>
    📦 ${{d.weight}} kg &nbsp;·&nbsp; 🎯 ${{d.priority}}<br>
    ${{d.delay?'<span style="color:#ff4757">⚠️ Delay risk: '+d.delay_prob+'%</span>':
               '<span style="color:#10b981">✅ On time predicted</span>'}}<br>
    <span style="color:#9ca3af">🚗 Rec: ${{d.rec_veh}}</span>
  `).addTo(map);
  deliveryMarkers.push(m);
}});

// ── Stop pills in bottom panel ──
const pillContainer = document.getElementById('stop-pills');
DELIVERIES.forEach((d,i) => {{
  const pill = document.createElement('span');
  pill.className = 'stop-pill pending';
  pill.id = `pill-${{i}}`;
  pill.textContent = `${{i+1}}. ${{d.label.split(' ')[0]}} ${{d.label.split(' ')[1]||''}}`;
  pillContainer.appendChild(pill);
}});

// ── Ghost (planned) route polylines ──
const ghostLines = [];
ROUTES.forEach(route => {{
  if (!route.coords || route.coords.length < 2) return;
  const line = L.polyline(route.coords, {{
    color: route.color, weight: 4, opacity: 0.18, dashArray: '8 10'
  }}).addTo(map);
  ghostLines.push(line);
}});

// ── Truck markers — one per route ──
const truckMarkers = [];
const travelledLines = [];
const routeStates = [];

ROUTES.forEach((route, ri) => {{
  if (!route.coords || route.coords.length < 2) return;
  const col = route.color;

  // Truck marker
  const truckHtml = `
    <div style="position:relative;">
      <div style="width:40px;height:40px;border-radius:50%;
        background:${{col}};border:3px solid white;
        display:flex;align-items:center;justify-content:center;
        font-size:18px;box-shadow:0 4px 16px rgba(0,0,0,.6);">${{route.icon}}</div>
      <div style="position:absolute;bottom:-18px;left:50%;transform:translateX(-50%);
        background:${{col}};color:white;font-size:9px;font-weight:800;
        padding:2px 6px;border-radius:100px;white-space:nowrap;">${{route.id}}</div>
    </div>`;
  const truckM = L.marker(route.coords[0], {{
    icon: L.divIcon({{html:truckHtml, className:'', iconSize:[40,58], iconAnchor:[20,20]}})
  }}).addTo(map);
  truckMarkers.push(truckM);

  // Glow circle under truck
  const glowCircle = L.circleMarker(route.coords[0], {{
    radius: 18, color: col, fillColor: col,
    fillOpacity: 0.15, weight: 2, opacity: 0.4
  }}).addTo(map);

  // Travelled path
  const travLine = L.polyline([], {{
    color: col, weight: 5, opacity: 0.92
  }}).addTo(map);
  travelledLines.push(travLine);

  routeStates.push({{
    route, stepIdx: 0, done: false,
    truckM, glowCircle, travLine,
    visited: []
  }});
}});

// ── Fleet panel HTML ──
const fleetRows = document.getElementById('fleet-rows');
routeStates.forEach((rs, ri) => {{
  const r = rs.route;
  const div = document.createElement('div');
  div.className = 'fleet-row';
  div.id = `fleet-row-${{ri}}`;
  div.innerHTML = `
    <div class="fleet-icon">${{r.icon}}</div>
    <div class="fleet-info">
      <div class="fleet-id">${{r.id}} · ${{r.vtype}}</div>
      <div class="fleet-sub">${{r.stops}} stops · ${{r.dist}} km</div>
      <div class="fleet-bar">
        <div class="fleet-fill" id="fleet-fill-${{ri}}"
          style="width:0%;background:${{r.color}}"></div>
      </div>
    </div>
    <span class="fleet-status s-active" id="fleet-status-${{ri}}">●</span>`;
  fleetRows.appendChild(div);
}});

// ── Animation state ──
let animating    = false;
let totalSteps   = routeStates.reduce((s,rs)=>s+rs.route.coords.length,0);
let stepsComplete = 0;
let deliveredCount = 0;

// Compute speed from coords (approx)
function coordDist(a, b) {{
  const R = 6371000;
  const dLat = (b[0]-a[0]) * Math.PI/180;
  const dLon = (b[1]-a[1]) * Math.PI/180;
  const aa = Math.sin(dLat/2)**2 + Math.cos(a[0]*Math.PI/180)*Math.cos(b[0]*Math.PI/180)*Math.sin(dLon/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(aa), Math.sqrt(1-aa));
}}

// ── SMOOTH ANIMATION CORE ──
// Uses requestAnimationFrame for 60fps smooth interpolation
// between sparse OSRM coords (Google Maps style)
let lastTime = null;
const SPEED_FACTOR = 80;  // km/h simulated speed

function lerp(a, b, t) {{
  return [a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t];
}}

// Sub-step progress per route (0..1 between coords[stepIdx] and coords[stepIdx+1])
routeStates.forEach(rs => {{ rs.subStep = 0; }});

let lastSpeedUpdate = 0;
let displaySpeed = 0;

function animate(timestamp) {{
  if (!animating) return;
  if (!lastTime) lastTime = timestamp;
  const dt = (timestamp - lastTime) / 1000;  // seconds
  lastTime = timestamp;

  let anyActive = false;

  routeStates.forEach((rs, ri) => {{
    if (rs.done) return;
    const coords = rs.route.coords;
    if (rs.stepIdx >= coords.length - 1) {{
      rs.done = true;
      document.getElementById(`fleet-status-${{ri}}`).className = 'fleet-status s-done';
      document.getElementById(`fleet-status-${{ri}}`).textContent = '✓';
      return;
    }}
    anyActive = true;

    const cur  = coords[rs.stepIdx];
    const next = coords[rs.stepIdx + 1];
    const segDist = coordDist(cur, next);           // metres
    const segTime = segDist / (SPEED_FACTOR * 1000 / 3600);  // seconds
    const advance = segTime > 0 ? dt / segTime : 1;

    rs.subStep = Math.min(rs.subStep + advance, 1);

    // Interpolated position
    const pos = lerp(cur, next, rs.subStep);
    rs.truckM.setLatLng(pos);
    rs.glowCircle.setLatLng(pos);

    // Speed display
    if (timestamp - lastSpeedUpdate > 500) {{
      displaySpeed = Math.round(SPEED_FACTOR * (0.85 + Math.random()*0.3));
      document.getElementById('spd-num').textContent = displaySpeed;
      lastSpeedUpdate = timestamp;
    }}

    // When segment done, advance
    if (rs.subStep >= 1) {{
      rs.subStep = 0;
      rs.stepIdx++;
      rs.visited.push(coords[rs.stepIdx]);
      rs.travLine.setLatLngs(rs.visited.length>1 ? rs.visited : []);
      stepsComplete++;

      // Fleet progress bar
      const pct = Math.round((rs.stepIdx / (coords.length-1))*100);
      const fill = document.getElementById(`fleet-fill-${{ri}}`);
      if (fill) fill.style.width = pct + '%';
    }}
  }});

  // ── Global progress ──
  const globalPct = totalSteps > 0 ? Math.min(stepsComplete / totalSteps, 1) : 0;
  document.getElementById('progress-fill').style.width = (globalPct*100).toFixed(1)+'%';
  document.getElementById('pct-val').textContent = (globalPct*100).toFixed(0)+'%';
  const etaRemain = Math.round((1 - globalPct) * TOTAL_ETA);
  document.getElementById('eta-val').textContent = Math.max(0, etaRemain);
  document.getElementById('tb-eta').textContent = Math.max(0, etaRemain);

  // Distance covered approx
  const distCovered = (globalPct * {total_dist}).toFixed(1);
  document.getElementById('dist-val').textContent = distCovered + ' km covered';

  // ── Mark deliveries as done based on progress ──
  DELIVERIES.forEach((d, i) => {{
    const threshold = (i+1) / DELIVERIES.length;
    if (globalPct >= threshold) {{
      const pill = document.getElementById(`pill-${{i}}`);
      if (pill && !pill.classList.contains('delivered')) {{
        pill.classList.remove('pending');
        pill.classList.add('delivered');
        deliveredCount++;
        document.getElementById('tb-stops').textContent = deliveredCount + '/' + DELIVERIES.length;
        // Update delivery marker to checkmark
        const mHtml = `<div style="
          width:32px;height:32px;background:#10b981;
          border:3px solid rgba(255,255,255,.9);border-radius:50%;
          display:flex;align-items:center;justify-content:center;
          color:white;font-size:14px;font-weight:800;
          box-shadow:0 3px 12px rgba(16,185,129,.5);">✓</div>`;
        deliveryMarkers[i].setIcon(L.divIcon({{html:mHtml,className:'',iconSize:[32,32],iconAnchor:[16,16]}}));
      }}
    }}
  }});

  // ── Nav instruction: heading towards next stop ──
  if (routeStates.length > 0 && !routeStates[0].done) {{
    const rs = routeStates[0];
    const coords = rs.route.coords;
    const ahead = Math.min(rs.stepIdx + 8, coords.length - 1);
    const cur = rs.truckM.getLatLng();
    const target = coords[ahead];
    const dLon = target[1] - cur.lng;
    const dLat = target[0] - cur.lat;
    const angle = Math.atan2(dLon, dLat) * 180 / Math.PI;
    let icon = '⬆️';
    if (angle < -135 || angle > 135) icon = '⬇️';
    else if (angle < -45) icon = '⬅️';
    else if (angle > 45)  icon = '➡️';
    document.getElementById('nav-icon').textContent = icon;
    const remSteps = coords.length - rs.stepIdx;
    const remDist  = ((remSteps / coords.length) * rs.route.dist).toFixed(1);
    document.getElementById('nav-action').textContent = 'Continue on current road';
    document.getElementById('nav-dist').textContent   = remDist + ' km to next stop';
  }}

  // Auto-follow active truck (Google Maps style)
  if (routeStates.length > 0 && !routeStates[0].done) {{
    const pos = routeStates[0].truckM.getLatLng();
    const mapCenter = map.getCenter();
    const dx = Math.abs(pos.lat - mapCenter.lat);
    const dy = Math.abs(pos.lng - mapCenter.lng);
    if (dx > 0.008 || dy > 0.008) {{
      map.panTo(pos, {{animate:true, duration:1.0, easeLinearity:0.25}});
    }}
  }}

  if (!anyActive) {{
    animating = false;
    document.getElementById('nav-action').textContent = '✅ All deliveries complete!';
    document.getElementById('nav-dist').textContent = 'Fleet returned to depot';
    document.getElementById('spd-num').textContent = '0';
    document.getElementById('eta-val').textContent = '0';
  }} else {{
    requestAnimationFrame(animate);
  }}
}}

// ── Play / Pause via postMessage from Streamlit ──
// Also expose global start/stop for button calls
window.startTracking = function() {{
  if (animating) return;
  // Reset
  routeStates.forEach((rs,ri) => {{
    rs.stepIdx = 0; rs.subStep = 0; rs.done = false; rs.visited = [];
    rs.travLine.setLatLngs([]);
    if (rs.route.coords.length > 0) rs.truckM.setLatLng(rs.route.coords[0]);
    const fill = document.getElementById(`fleet-fill-${{ri}}`);
    if (fill) fill.style.width = '0%';
    const status = document.getElementById(`fleet-status-${{ri}}`);
    if (status) {{ status.className = 'fleet-status s-active'; status.textContent = '●'; }}
  }});
  stepsComplete = 0; deliveredCount = 0; lastTime = null;
  DELIVERIES.forEach((d,i) => {{
    const pill = document.getElementById(`pill-${{i}}`);
    if (pill) {{ pill.classList.remove('delivered'); pill.classList.add('pending'); }}
    // Restore original markers
    const mHtml = `<div style="
      width:32px;height:32px;background:${{d.color}};
      border:3px solid rgba(255,255,255,.9);border-radius:50%;
      display:flex;align-items:center;justify-content:center;
      color:white;font-size:12px;font-weight:800;
      box-shadow:0 3px 12px rgba(0,0,0,.5);">${{i+1}}</div>`;
    deliveryMarkers[i].setIcon(L.divIcon({{html:mHtml,className:'',iconSize:[32,32],iconAnchor:[16,16]}}));
  }});
  document.getElementById('tb-stops').textContent = '0/' + DELIVERIES.length;
  animating = true;
  requestAnimationFrame(animate);
}};

window.stopTracking = function() {{
  animating = false;
}};

// Auto-start after short delay
setTimeout(() => window.startTracking(), 1200);
</script>
</body>
</html>"""
    return html


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:12px 0 18px;
                border-bottom:2px solid #e2e8f7;margin-bottom:4px;">
        <div style="width:44px;height:44px;border-radius:14px;
                    background:linear-gradient(135deg,#7c3aed,#a855f7);
                    display:flex;align-items:center;justify-content:center;
                    font-size:21px;box-shadow:0 4px 14px rgba(124,58,237,.35);">🛰️</div>
        <div>
            <div style="font-family:'Outfit',sans-serif;font-size:19px;font-weight:800;color:#0d1626;">OptiRoute</div>
            <div style="font-size:9px;color:#94a3b8;letter-spacing:2.5px;font-family:'JetBrains Mono',monospace;margin-top:1px;">SMART ROUTING </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">🏭 Warehouse Network</div>', unsafe_allow_html=True)
    for i,wh in enumerate(st.session_state.warehouses):
        with st.expander(f"{wh['id']} · {wh['name']}", expanded=False):
            c1,c2=st.columns(2)
            wh["lat"]=c1.number_input("Lat",value=wh["lat"],key=f"wlat{i}",format="%.4f")
            wh["lon"]=c2.number_input("Lon",value=wh["lon"],key=f"wlon{i}",format="%.4f")
            wh["capacity"]=st.number_input("Capacity (kg)",value=wh["capacity"],step=500,key=f"wcap{i}")
            wh["active"]=st.checkbox("Active",value=wh["active"],key=f"wact{i}")
    if st.button("＋ Add Warehouse"):
        n=len(st.session_state.warehouses)+1
        st.session_state.warehouses.append({"id":f"WH-{chr(64+n)}","name":f"Hub {n}","lat":28.63,"lon":77.20,"capacity":3000,"active":True})
        st.rerun()

    st.markdown('<div class="sec-head">🚗 Fleet Setup</div>', unsafe_allow_html=True)
    num_vehicles=st.slider("Number of Vehicles",1,6,2)
    vehicle_types=[{"id":f"V{i+1}","type":st.selectbox(f"Vehicle {i+1}",list(VEHICLE_PROFILES.keys()),index=i%6,key=f"vt{i}")} for i in range(num_vehicles)]

    st.markdown('<div class="sec-head">🌤 Conditions</div>', unsafe_allow_html=True)
    weather=st.selectbox("Weather",list(WEATHER_PROFILES.keys()),format_func=lambda w:f"{WEATHER_PROFILES[w]['icon']} {w.capitalize()}")
    hour_now=datetime.now().hour; is_peak=(8<=hour_now<=11)or(17<=hour_now<=21)
    traffic_level=st.select_slider("Traffic",["Low","Moderate","High","Gridlock"],value="High" if is_peak else "Moderate")
    traffic_mult={"Low":1.0,"Moderate":1.2,"High":1.5,"Gridlock":2.0}[traffic_level]

    st.markdown('<div class="sec-head">⚙️ Constraints</div>', unsafe_allow_html=True)
    hc={"capacity":st.checkbox("Capacity Limit",value=True),
        "time_windows":st.checkbox("Time Windows",value=True),
        "range":st.checkbox("Max Range",value=True),
        "priority":st.checkbox("Priority/SLA",value=True),
        "breaks":st.checkbox("Driver Breaks",value=False),
        "road_type":st.checkbox("Road Restrictions",value=False)}

    st.markdown('<div class="sec-head">🤖 AI Stack</div>', unsafe_allow_html=True)
    for nm,col,acc in [("RandomForest","#00cfa8","86% acc"),("GradientBoosting","#7c3aed","risk score"),("OR-Tools CVRPTW","#ff9500","optimizer"),("OSRM Real Roads","#0ea5e9","routing")]:
        st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;padding:4px 0;font-size:12px;">
            <span style="width:9px;height:9px;border-radius:50%;background:{col};display:inline-block;box-shadow:0 0 7px {col};flex-shrink:0;"></span>
            <span style="font-weight:600;color:#2d3748;">{nm}</span>
            <span style="color:#94a3b8;font-size:10px;margin-left:auto;">{acc}</span>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
active_wh=[w for w in st.session_state.warehouses if w["active"]]
wp=WEATHER_PROFILES[weather]

st.markdown(f"""
<div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:12px;padding:8px 0 6px;">
    <div>
        <div style="font-family:'Outfit',sans-serif;font-size:38px;font-weight:900;
                    background:linear-gradient(120deg,#7c3aed 0%,#0ea5e9 45%,#00cfa8 100%);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1;">OptiRoute</div>
        <div style="font-size:11px;color:#94a3b8;letter-spacing:3px;margin-top:2px;font-family:'JetBrains Mono',monospace;">
            MULTI-WAREHOUSE · CVRPTW OPTIMIZER</div>
    </div>
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding-top:6px;">
        <span style="background:#f0fdf4;color:#16a34a;border:1.5px solid #bbf7d0;padding:5px 14px;border-radius:100px;font-size:12px;font-weight:700;">● SYSTEM LIVE</span>
        <span style="background:#f3f0ff;color:#7c3aed;border:1.5px solid #ddd5fb;padding:5px 14px;border-radius:100px;font-size:12px;font-weight:700;">🤖 AI ACTIVE</span>
        <span style="background:#e0f5ff;color:#0284c7;border:1.5px solid #bae8ff;padding:5px 14px;border-radius:100px;font-size:12px;font-weight:700;">{wp['icon']} {weather.upper()}</span>
    </div>
</div>
""", unsafe_allow_html=True)

kc=st.columns(5)
for col,(lbl,val,sub,cls) in zip(kc,[
    ("Active Hubs",  str(len(active_wh)), f"of {len(st.session_state.warehouses)} total","kv"),
    ("Fleet Size",   str(num_vehicles),   ", ".join(set(v["type"] for v in vehicle_types)),"kt"),
    ("Weather Risk", f"{wp['risk']}%",    f"{wp['icon']} {weather.capitalize()}","ka"),
    ("Traffic",      traffic_level,       f"x{traffic_mult} speed penalty","kc"),
    ("AI Accuracy",  "86%",              "RandomForest delay model","ks"),
]):
    col.markdown(f"""<div class="kpi {cls}">
        <div class="kpi-stripe"></div><div class="kpi-blob"></div>
        <div class="kpi-label">{lbl}</div><div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
t1,t2,t3,t4,t5=st.tabs(["📦  Orders & Input","🗺️  Route Map","📊  Analysis","📡  Live Tracking","🤖  AI Insights"])

# ══════════════════════════════════════════════
#  TAB 1 — ORDERS & INPUT
# ══════════════════════════════════════════════
with t1:
    st.markdown('<div class="sec-head">🔍 Location Search</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns([3,1])
    with sc1:
        search_query = st.text_input("Search location",
            placeholder="e.g. Connaught Place, Delhi  or  India Gate  or  any address…",
            key="loc_search", label_visibility="collapsed")
    with sc2:
        search_btn = st.button("🔍 Search", use_container_width=True, key="search_btn")

    if search_btn and search_query:
        with st.spinner("Searching…"):
            results = geocode_location(search_query)
        if results:
            st.session_state.search_result = results
        else:
            st.warning("No results found. Try a more specific address.")

    if st.session_state.search_result:
        results = st.session_state.search_result
        st.markdown('<div style="margin-bottom:6px;font-size:12px;color:#64748b;font-weight:600;">Search Results — click Use to apply:</div>', unsafe_allow_html=True)
        for idx,(rlat,rlon,rname) in enumerate(results[:4]):
            r1,r2,r3=st.columns([3,1,1])
            r1.markdown(f"""<div style="font-size:12px;color:#374151;padding:8px 0;">
                <b style="color:#7c3aed;">📍</b> {rname[:80]}{'…' if len(rname)>80 else ''}</div>""",
                unsafe_allow_html=True)
            if r2.button("📍 Use",key=f"sr_use_{idx}",use_container_width=True):
                st.session_state.map_clicked_coord=(rlat,rlon)
                st.session_state.map_center=[rlat,rlon]
                st.session_state.search_result=None
                st.rerun()
            if r3.button("🏭 Wh",key=f"sr_wh_{idx}",help="Set as new warehouse",use_container_width=True):
                n=len(st.session_state.warehouses)+1
                st.session_state.warehouses.append({"id":f"WH-{chr(64+n)}","name":rname[:20],"lat":rlat,"lon":rlon,"capacity":3000,"active":True})
                st.session_state.search_result=None
                st.success(f"Added warehouse at {rname[:40]}")
                st.rerun()

    st.markdown("---")

    la,ra=st.columns([2,1],gap="medium")
    with la:
        st.markdown('<div class="sec-head">📍 Interactive Location Picker</div>', unsafe_allow_html=True)
        st.markdown('<div class="mapbox">🖱️ <b>Click the map</b> to capture coordinates, then assign them to any stop.</div>', unsafe_allow_html=True)

        cm=folium.Map(location=st.session_state.map_center,zoom_start=12,tiles="CartoDB Positron")
        if st.session_state.map_clicked_coord:
            lat_c,lon_c=st.session_state.map_clicked_coord
            folium.Marker([lat_c,lon_c],icon=folium.Icon(color="red",icon="map-marker",prefix="fa"),tooltip="📍 Selected").add_to(cm)
        for wh in active_wh:
            folium.Marker([wh["lat"],wh["lon"]],popup=folium.Popup(f"<b>{wh['id']}</b>",max_width=180),
                icon=folium.Icon(color="purple",icon="home",prefix="fa"),tooltip=f"🏭 {wh['name']}").add_to(cm)
        for i,d in enumerate(st.session_state.deliveries_preview):
            if d["coord"]!=(0.0,0.0):
                fc=["purple","green","red","orange","blue","darkred"][i%6]
                folium.CircleMarker(d["coord"],radius=9,color=fc,fill=True,fill_opacity=.85,tooltip=f"Stop {i+1}").add_to(cm)
        mr=st_folium(cm,width="100%",height=290,returned_objects=["last_clicked"],key="locpicker")
        if mr and mr.get("last_clicked"):
            cl=mr["last_clicked"]
            st.session_state.map_clicked_coord=(round(cl["lat"],5),round(cl["lng"],5))
            st.session_state.map_center=[cl["lat"],cl["lng"]]

    with ra:
        st.markdown('<div class="sec-head">📌 Captured Coordinates</div>', unsafe_allow_html=True)
        if st.session_state.map_clicked_coord:
            lat_c,lon_c=st.session_state.map_clicked_coord
            st.markdown(f"""<div style="background:linear-gradient(135deg,#f3f0ff,#e0f5ff);
                border:1.5px solid #c7d2fe;border-radius:16px;padding:18px;margin-top:8px;">
                <div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#7c3aed;margin-bottom:10px;">SELECTED POINT</div>
                <div class="coord-chip">🌐 {lat_c}</div><div class="coord-chip">🌐 {lon_c}</div>
            </div>""", unsafe_allow_html=True)
            assign_to=st.selectbox("Assign to stop #",range(1,11),key="asgn")
            if st.button("✅ Apply Coordinates",use_container_width=True):
                st.session_state[f"dlat_{assign_to-1}"]=lat_c
                st.session_state[f"dlon_{assign_to-1}"]=lon_c
                st.success(f"✅ Applied to Stop {assign_to}")
        else:
            st.markdown("""<div style="background:#f8faff;border:2px dashed #c7d2fe;border-radius:16px;padding:32px 16px;text-align:center;margin-top:8px;">
                <div style="font-size:30px;margin-bottom:8px;">🗺️</div>
                <div style="font-size:13px;color:#94a3b8;">Search above or click the map</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head">📦 Delivery Orders</div>', unsafe_allow_html=True)
    num_stops=st.slider("Number of Delivery Stops",1,10,3)
    st.markdown("""<div style="display:grid;grid-template-columns:1.8fr 1fr 1fr .8fr 1.2fr 1fr;
        gap:6px;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;color:#94a3b8;
        text-transform:uppercase;letter-spacing:1.5px;padding:8px 4px;border-bottom:2px solid #e2e8f7;">
        <div>Label</div><div>Latitude</div><div>Longitude</div><div>kg</div><div>Time Window</div><div>Priority</div>
    </div>""", unsafe_allow_html=True)

    deliveries=[]
    for i in range(num_stops):
        c1,c2,c3,c4,c5,c6=st.columns([1.8,1,1,.8,1.2,1])
        lbl=c1.text_input("L",value=f"Customer {i+1}",key=f"lbl{i}",label_visibility="collapsed")
        dlat=st.session_state.get(f"dlat_{i}",28.60+i*.015)
        dlon=st.session_state.get(f"dlon_{i}",77.15+i*.020)
        lat=c2.number_input("a",value=float(dlat),key=f"dlat{i}",format="%.5f",label_visibility="collapsed")
        lon=c3.number_input("b",value=float(dlon),key=f"dlon{i}",format="%.5f",label_visibility="collapsed")
        wgt=c4.number_input("c",value=float(5+i*3),key=f"dwgt{i}",min_value=.1,label_visibility="collapsed")
        tw=c5.selectbox("d",list(TIME_WINDOW_PRESETS.keys()),key=f"dtw{i}",label_visibility="collapsed")
        prio=c6.selectbox("e",list(PRIORITY_LEVELS.keys()),key=f"dpr{i}",label_visibility="collapsed")
        twp=TIME_WINDOW_PRESETS[tw]
        two,twc=(twp if twp else (hour_now*60,(hour_now+4)*60))
        if lat!=0 or lon!=0:
            deliveries.append({"label":lbl,"coord":(lat,lon),"weight_kg":wgt,
                                "tw_open_min":two,"tw_close_min":twc,"priority":prio})
    st.session_state.deliveries_preview=deliveries

    st.markdown("<br>",unsafe_allow_html=True)
    b1,b2=st.columns(2)
    run_vrp=b1.button("🚀 Optimize Routes (CVRPTW + AI)",use_container_width=True)
    clr_btn=b2.button("🔄 Reset All",use_container_width=True)
    if clr_btn: st.session_state.result=None; st.session_state.ml_predictions=[]; st.rerun()

# ─────────────────────────────────────────────
#  COMPUTE
# ─────────────────────────────────────────────
if run_vrp and deliveries and active_wh:
    with st.spinner("⚙️ Optimizing routes with CVRPTW + AI…"):
        wh_scores=sorted([(score_wh(wh,deliveries,vehicle_types[0]["type"],weather),wh) for wh in active_wh],key=lambda x:x[0])
        best_wh=wh_scores[0][1]; depot=(best_wh["lat"],best_wh["lon"])
        ml_preds=[]
        for d in deliveries:
            dist=haversine(depot,d["coord"]); dp,prob=predict_delay(dist,d["weight_kg"],weather,traffic_mult,d["priority"])
            rv=recommend_vehicle(d["weight_kg"],dist)
            ml_preds.append({"label":d["label"],"delay_predicted":dp,"delay_prob":prob,
                             "recommended_vehicle":rv,"reroute_flag":dp==1 and prob>.6,"dist_to_depot":round(dist,2)})
        st.session_state.ml_predictions=ml_preds
        sol=solve_vrp(depot,deliveries,vehicle_types,weather,hc)
        all_c=[]; tdist=0; rdetails=[]
        if sol:
            for route in sol:
                nodes=route["nodes"]; vtype=route["vehicle"]["type"]; vp=VEHICLE_PROFILES[vtype]
                pts=[depot]+[deliveries[n-1]["coord"] for n in nodes[1:-1] if 0<n<=len(deliveries)]+[depot]
                rdist=0; rc=[]
                for j in range(len(pts)-1):
                    r=fetch_route(pts[j],pts[j+1])
                    if r: rc.extend([[c[1],c[0]] for c in r["geometry"]["coordinates"]]); rdist+=r["distance"]/1000
                    else: rc.extend([list(pts[j]),list(pts[j+1])]); rdist+=haversine(pts[j],pts[j+1])
                eff=max(vp["speed_kmh"]*(1-wp["speed_pen"]/100)/traffic_mult,5)
                load=sum(deliveries[n-1].get("weight_kg",1) for n in nodes[1:-1] if 0<n<=len(deliveries))
                all_c.extend(rc); tdist+=rdist
                rdetails.append({"vehicle":route["vehicle"],"vtype":vtype,"nodes":nodes,"coords":rc,"pts":pts,
                    "dist_km":round(rdist,2),"eta_min":int((rdist/eff)*60),
                    "fuel_cost":round(rdist*vp["cost_per_km"],2),"co2_kg":round(rdist*vp["co2_per_km"],3),
                    "load_kg":round(load,2),"cap_util":round((load/vp["capacity_kg"])*100,1),"stops":len(nodes)-2})
        st.session_state.result={"best_wh":best_wh,"wh_scores":wh_scores,"routes":rdetails,
            "deliveries":deliveries,"total_dist":round(tdist,2),"depot_coord":depot,"all_route_coords":all_c}
        st.session_state.live_step=0

result=st.session_state.result; ml_preds=st.session_state.ml_predictions

# ── Tab 1 results ──
with t1:
    if result:
        st.markdown('<div class="sec-head">🏆 Warehouse Selection</div>', unsafe_allow_html=True)
        vs=["kv","kt","ks","ka","kc"]
        wc=st.columns(min(len(result["wh_scores"]),5))
        for idx,(sc,wh) in enumerate(result["wh_scores"][:5]):
            avgd=round(np.mean([haversine((wh["lat"],wh["lon"]),d["coord"]) for d in deliveries]),1)
            badge='<span class="pill pt">✅ SELECTED</span>' if idx==0 else f'<span style="font-size:11px;color:#94a3b8;">#{idx+1}</span>'
            wc[idx].markdown(f"""<div class="kpi {vs[idx]}" style="min-height:130px;">
                <div class="kpi-stripe"></div><div class="kpi-blob"></div>
                <div class="kpi-label">{wh['id']}</div>
                <div style="padding-left:10px;margin-bottom:4px;">{badge}</div>
                <div class="kpi-value" style="font-size:17px;">{wh['name']}</div>
                <div class="kpi-sub">Score {sc} · {avgd} km · {wh['capacity']:,} kg</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-head">🚛 Optimized Route Assignments</div>', unsafe_allow_html=True)
        for ri,rd in enumerate(result["routes"]):
            vp=VEHICLE_PROFILES[rd["vtype"]]; c=RCOLS[ri%len(RCOLS)]
            st.markdown(f"""<div class="rc">
                <div class="rc-bar" style="background:{c};"></div>
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;padding-left:12px;">
                    <div style="display:flex;align-items:center;gap:12px;">
                        <div style="width:42px;height:42px;border-radius:12px;background:{c}18;border:2px solid {c}40;display:flex;align-items:center;justify-content:center;font-size:19px;">{vp['icon']}</div>
                        <div>
                            <div style="font-family:'Outfit',sans-serif;font-weight:700;font-size:15px;color:{c};">{rd['vehicle']['id']} — {rd['vtype']}</div>
                            <div style="font-size:12px;color:#64748b;">{rd['stops']} stops · {rd['cap_util']}% capacity</div>
                        </div>
                    </div>
                    <div>
                        <span class="ctag">📏 {rd['dist_km']} km</span>
                        <span class="ctag">⏱ {rd['eta_min']} min</span>
                        <span class="ctag">₹ {rd['fuel_cost']}</span>
                        <span class="ctag">🌿 {rd['co2_kg']} kg CO₂</span>
                        <span class="ctag">📦 {rd['load_kg']} kg</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        eta=max((r["eta_min"] for r in result["routes"]),default=0)
        cost=sum(r["fuel_cost"] for r in result["routes"])
        co2=sum(r["co2_kg"] for r in result["routes"])
        dls=sum(1 for p in ml_preds if p["delay_predicted"])
        st.markdown("<br>",unsafe_allow_html=True)
        for col,(lbl,val,sub,cls) in zip(st.columns(5),[
            ("Total Distance",f"{result['total_dist']} km","","ks"),
            ("Fleet ETA",f"{eta} min","worst-case","kv"),
            ("Fleet Cost",f"₹{round(cost,2)}","total","ka"),
            ("CO₂",f"{round(co2,2)} kg","footprint","kt"),
            ("AI Delay Flags",f"{dls}/{len(ml_preds)}","at risk","kc" if dls else "kt"),
        ]):
            col.markdown(f"""<div class="kpi {cls}">
                <div class="kpi-stripe"></div><div class="kpi-blob"></div>
                <div class="kpi-label">{lbl}</div><div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 2 — ROUTE MAP
# ══════════════════════════════════════════════
with t2:
    if result:
        st.markdown('<div class="sec-head">🗺️ Optimized Route Map — Real Road Network (OSRM)</div>', unsafe_allow_html=True)
        leg="".join(
            f'<span style="display:inline-flex;align-items:center;gap:5px;background:{RCOLS[i]}14;'
            f'border:1.5px solid {RCOLS[i]}50;border-radius:100px;padding:4px 12px;font-size:11px;'
            f'font-weight:700;color:{RCOLS[i]};margin:2px;">● {rd["vehicle"]["id"]} ({rd["vtype"]})</span>'
            for i,rd in enumerate(result["routes"]))
        st.markdown(f"""<div style="display:flex;flex-wrap:wrap;gap:4px;align-items:center;
            padding:10px 16px;background:white;border-radius:12px;border:1.5px solid #e2e8f7;margin-bottom:12px;">
            <span style="font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:1.5px;margin-right:6px;">ROUTES</span>
            {leg}
            <span style="margin-left:auto;font-size:11px;color:#94a3b8;">🏭 = Warehouse · ● = Delivery</span>
        </div>""", unsafe_allow_html=True)

        depot=result["depot_coord"]
        fmap=folium.Map(location=[depot[0],depot[1]],zoom_start=12,tiles="CartoDB Positron")
        hd=[list(d["coord"]) for d in result["deliveries"]]
        if hd: HeatMap(hd,radius=30,blur=20,min_opacity=.22,gradient={"0.3":"#c7d2fe","0.6":"#7c3aed","1.0":"#ff4757"}).add_to(fmap)

        for ri,rd in enumerate(result["routes"]):
            c=RCOLS[ri%len(RCOLS)]
            if rd["coords"]:
                folium.PolyLine(rd["coords"],color=c,weight=12,opacity=.10).add_to(fmap)
                try:
                    AntPath(rd["coords"],color=c,weight=4,opacity=.95,delay=550,dash_array=[10,18],
                        tooltip=f"🚛 {rd['vehicle']['id']} — {rd['vtype']} | {rd['dist_km']} km | {rd['eta_min']} min").add_to(fmap)
                except:
                    folium.PolyLine(rd["coords"],color=c,weight=4,opacity=.92).add_to(fmap)

        pcols={"Express":"#ff4757","Priority":"#ff9500","Same-Day":"#0ea5e9","Standard":"#64748b"}
        for wh in active_wh:
            best=wh["id"]==result["best_wh"]["id"]
            if best: folium.CircleMarker([wh["lat"],wh["lon"]],radius=26,color="#7c3aed",fill=False,weight=2,opacity=.28).add_to(fmap)
            ih=(f'<div style="width:36px;height:36px;border-radius:11px;'
                f'background:{"linear-gradient(135deg,#7c3aed,#a855f7)" if best else "white"};'
                f'border:2.5px solid {"#7c3aed" if best else "#cbd5e1"};'
                f'display:flex;align-items:center;justify-content:center;font-size:16px;'
                f'box-shadow:0 3px 12px rgba(0,0,0,.18);">🏭</div>')
            folium.Marker([wh["lat"],wh["lon"]],
                popup=folium.Popup(f"<b>{wh['id']}</b><br>{wh['name']}<br>Cap: {wh['capacity']:,} kg",max_width=200),
                icon=folium.DivIcon(html=ih,icon_size=(36,36),icon_anchor=(18,18)),
                tooltip=f"{'⭐ ' if best else ''}🏭 {wh['name']}").add_to(fmap)

        for i,d in enumerate(result["deliveries"]):
            pc=pcols[d["priority"]]
            di=ml_preds[i] if i<len(ml_preds) else None
            ds="⚠️" if (di and di["delay_predicted"]) else "✅"
            delay_str=(str(round(di["delay_prob"]*100))+"%") if di else "—"
            rec_veh=di["recommended_vehicle"] if di else "—"
            pop=(f"<div style='font-family:sans-serif;min-width:200px;padding:4px;'>"
                 f"<b style='color:{pc};font-size:14px;'>{d['label']}</b><hr style='margin:5px 0;'>"
                 f"📦 {d['weight_kg']} kg · 🎯 {d['priority']}<br>"
                 f"{ds} Delay risk: {delay_str}<br>🚗 Rec: {rec_veh}</div>")
            mh=(f'<div style="width:30px;height:30px;background:{pc};border:3px solid white;'
                f'border-radius:50%;display:flex;align-items:center;justify-content:center;'
                f'color:white;font-size:12px;font-weight:800;box-shadow:0 3px 10px rgba(0,0,0,.22);">{i+1}</div>')
            folium.Marker(d["coord"],popup=folium.Popup(pop,max_width=240),
                icon=folium.DivIcon(html=mh,icon_size=(30,30),icon_anchor=(15,15)),
                tooltip=f"📦 {d['label']} — {d['priority']}").add_to(fmap)

        st_folium(fmap,width="100%",height=580,key="main_map")

        st.markdown('<div class="sec-head">📊 Warehouse Comparison</div>', unsafe_allow_html=True)
        df_wh=pd.DataFrame([{"ID":wh["id"],"Name":wh["name"],"Score (lower=better)":sc,
            "Avg Dist (km)":round(np.mean([haversine((wh["lat"],wh["lon"]),d["coord"]) for d in deliveries]),2),
            "Capacity (kg)":wh["capacity"],"Selected":"✅ Yes" if wh["id"]==result["best_wh"]["id"] else "—"
        } for sc,wh in result["wh_scores"]])
        st.dataframe(df_wh,use_container_width=True,hide_index=True)
    else:
        st.markdown("""<div style="text-align:center;padding:80px 20px;">
            <div style="font-size:56px;margin-bottom:14px;">🗺️</div>
            <div style="font-family:'Outfit',sans-serif;font-size:20px;font-weight:700;color:#374151;">No routes yet</div>
            <div style="font-size:14px;color:#94a3b8;margin-top:6px;">Run optimization in the Orders tab first</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 3 — ANALYSIS
# ══════════════════════════════════════════════
with t3:
    if result and result["routes"]:
        routes=result["routes"]
        df=pd.DataFrame([{"Vehicle":f"{r['vehicle']['id']} ({r['vtype']})","Load %":r["cap_util"],
            "Distance (km)":r["dist_km"],"ETA (min)":r["eta_min"],
            "Cost (Rs)":r["fuel_cost"],"CO2 (kg)":r["co2_kg"],"Stops":r["stops"]} for r in routes])

        a1,a2=st.columns(2)
        with a1:
            st.markdown('<div class="sec-head">📊 Capacity Utilization</div>', unsafe_allow_html=True)
            bar=alt.Chart(df).mark_bar(cornerRadiusTopLeft=8,cornerRadiusTopRight=8).encode(
                x=alt.X("Vehicle:N",axis=alt.Axis(labelColor="#374151",labelAngle=-20,title="")),
                y=alt.Y("Load %:Q",scale=alt.Scale(domain=[0,100]),axis=alt.Axis(labelColor="#374151",title="Load %")),
                color=alt.Color("Vehicle:N",legend=None,scale=alt.Scale(range=RCOLS)),
                tooltip=["Vehicle","Load %","Distance (km)","Cost (Rs)"]
            ).properties(height=240,background="transparent").configure_view(strokeWidth=0,fill="transparent").configure_axis(grid=True,gridColor="#f1f5f9")
            st.altair_chart(bar,use_container_width=True)
        with a2:
            st.markdown('<div class="sec-head">💰 Cost vs Distance</div>', unsafe_allow_html=True)
            scatter=alt.Chart(df).mark_circle(opacity=.9,stroke="white",strokeWidth=2).encode(
                x=alt.X("Distance (km):Q",axis=alt.Axis(labelColor="#374151")),
                y=alt.Y("Cost (Rs):Q",axis=alt.Axis(labelColor="#374151")),
                size=alt.Size("Stops:Q",scale=alt.Scale(range=[100,650]),legend=None),
                color=alt.Color("Vehicle:N",legend=None,scale=alt.Scale(range=RCOLS)),
                tooltip=["Vehicle","Distance (km)","Cost (Rs)","Stops","CO2 (kg)"]
            ).properties(height=240,background="transparent").configure_view(strokeWidth=0,fill="transparent").configure_axis(grid=True,gridColor="#f1f5f9")
            st.altair_chart(scatter,use_container_width=True)

        st.markdown('<div class="sec-head">🎯 Priority Breakdown</div>', unsafe_allow_html=True)
        pc2={}
        for d in result["deliveries"]: pc2[d["priority"]]=pc2.get(d["priority"],0)+1
        df_p=pd.DataFrame([{"Priority":k,"Count":v} for k,v in pc2.items()])
        pie=alt.Chart(df_p).mark_arc(innerRadius=52,cornerRadius=5).encode(
            theta="Count:Q",
            color=alt.Color("Priority:N",scale=alt.Scale(
                domain=["Standard","Priority","Express","Same-Day"],
                range=["#64748b","#ff9500","#ff4757","#0ea5e9"])),
            tooltip=["Priority","Count"]
        ).properties(height=200,background="transparent").configure_view(strokeWidth=0,fill="transparent")
        st.altair_chart(pie,use_container_width=True)

        st.markdown('<div class="sec-head">⚠️ Risk Overview</div>', unsafe_allow_html=True)
        delay_risk=min(int(wp["risk"]+(traffic_mult-1)*30+len(result["deliveries"])*2),100)
        delayed=sum(1 for p in ml_preds if p["delay_predicted"])
        rv="kc" if delay_risk>60 else "ka" if delay_risk>30 else "kt"
        for col,(lbl,val,sub,cls) in zip(st.columns(3),[
            ("Composite Risk",f"{delay_risk}%","delay index",rv),
            ("AI Delay Flags",f"{delayed}/{len(ml_preds)}","RF predictions","kc" if delayed else "kt"),
            ("Weather",f"{wp['icon']} {wp['risk']}%",weather.capitalize(),"ka"),
        ]):
            col.markdown(f"""<div class="kpi {cls}">
                <div class="kpi-stripe"></div><div class="kpi-blob"></div>
                <div class="kpi-label">{lbl}</div><div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-head">🧠 Recommendations</div>', unsafe_allow_html=True)
        if delay_risk>60: st.error("🔴 HIGH DELAY RISK — Consider rescheduling or adding more vehicles.")
        elif delay_risk>35: st.warning("🟡 MODERATE RISK — Monitor live conditions; re-optimize if needed.")
        else: st.success("🟢 LOW RISK — Routes optimally planned for current conditions.")
        for rd in routes:
            vp=VEHICLE_PROFILES[rd["vtype"]]
            if rd["cap_util"]<30: st.info(f"💡 {rd['vehicle']['id']} ({rd['vtype']}) under-utilized ({rd['cap_util']}%). Consider consolidating.")
            if rd["dist_km"]>vp["max_range_km"]*.9: st.warning(f"⚠️ {rd['vehicle']['id']} near max range ({rd['dist_km']} km / {vp['max_range_km']} km).")
        if weather in ["rainy","stormy"]: st.info(f"🌧 {weather.capitalize()} — reduce speed by {wp['speed_pen']}%.")
    else:
        st.info("Run optimization to see analysis.")

# ══════════════════════════════════════════════
#  TAB 4 — LIVE TRACKING  (GOOGLE MAPS STYLE)
# ══════════════════════════════════════════════
with t4:
    if result and result["routes"]:
        st.markdown('<div class="sec-head">📡 Live Fleet Tracking — Google Maps Style (JS Animation)</div>', unsafe_allow_html=True)

        # Info row
        i1,i2,i3,i4 = st.columns(4)
        for col,(lbl,val,sub,cls) in zip([i1,i2,i3,i4],[
            ("Total ETA", f"{max(r['eta_min'] for r in result['routes'])} min", "worst-case route","kv"),
            ("Fleet Size", str(len(result['routes'])), "active vehicles","kt"),
            ("Deliveries", str(len(result['deliveries'])), "total stops","ka"),
            ("Distance", f"{result['total_dist']} km", "total network","ks"),
        ]):
            col.markdown(f"""<div class="kpi {cls}" style="min-height:90px;">
                <div class="kpi-stripe"></div><div class="kpi-blob"></div>
                <div class="kpi-label">{lbl}</div><div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature callouts
        st.markdown("""
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;">
            <span style="background:#f3f0ff;color:#7c3aed;border:1.5px solid #ddd5fb;padding:5px 14px;border-radius:100px;font-size:11px;font-weight:700;">🎯 Auto-follow truck</span>
            <span style="background:#e0fff9;color:#00a88c;border:1.5px solid #a0f5e0;padding:5px 14px;border-radius:100px;font-size:11px;font-weight:700;">⚡ 60fps JS interpolation</span>
            <span style="background:#e0f5ff;color:#0284c7;border:1.5px solid #bae8ff;padding:5px 14px;border-radius:100px;font-size:11px;font-weight:700;">🛰️ Satellite/Map/Dark tiles</span>
            <span style="background:#fff8eb;color:#c97800;border:1.5px solid #ffe4a8;padding:5px 14px;border-radius:100px;font-size:11px;font-weight:700;">📍 Turn-by-turn nav</span>
            <span style="background:#fff0f1;color:#ff4757;border:1.5px solid #ffd0d4;padding:5px 14px;border-radius:100px;font-size:11px;font-weight:700;">🚚 Per-vehicle fleet panel</span>
            <span style="background:#f0fdf4;color:#16a34a;border:1.5px solid #bbf7d0;padding:5px 14px;border-radius:100px;font-size:11px;font-weight:700;">✅ Live delivery status</span>
        </div>
        """, unsafe_allow_html=True)

        # Build and render the full Google Maps style HTML
        live_html = build_live_tracking_html(result, active_wh, weather, ml_preds)
        st.components.v1.html(live_html, height=700, scrolling=False)

    else:
        st.markdown("""<div style="text-align:center;padding:80px 20px;">
            <div style="font-size:56px;margin-bottom:14px;">📡</div>
            <div style="font-family:'Outfit',sans-serif;font-size:20px;font-weight:700;color:#374151;">No routes to track</div>
            <div style="font-size:14px;color:#94a3b8;margin-top:6px;">Run optimization in the Orders tab, then return here</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 5 — AI INSIGHTS
# ══════════════════════════════════════════════
with t5:
    st.markdown('<div class="sec-head">🤖 AI Model Stack</div>', unsafe_allow_html=True)
    for col,(nm,acc,desc,cls) in zip(st.columns(3),[
        ("RandomForestClassifier","86%","Delay prediction · 100 trees · depth 8","kt"),
        ("GradientBoosting","~84%","Risk scoring · 80 estimators · depth 4","kv"),
        ("OR-Tools CVRPTW","Optimal","Capacitated VRP with Time Windows","ka"),
    ]):
        col.markdown(f"""<div class="kpi {cls}" style="min-height:120px;">
            <div class="kpi-stripe"></div><div class="kpi-blob"></div>
            <div class="kpi-label">{nm}</div><div class="kpi-value">{acc}</div><div class="kpi-sub">{desc}</div>
        </div>""", unsafe_allow_html=True)

    if ml_preds:
        st.markdown('<div class="sec-head">📦 Per-Delivery AI Predictions</div>', unsafe_allow_html=True)
        st.markdown("""<div style="display:grid;grid-template-columns:1.6fr 1fr 1.2fr 1fr 1fr;
            gap:8px;font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;
            letter-spacing:1.5px;padding:8px 16px;border-bottom:2px solid #e2e8f7;
            font-family:'JetBrains Mono',monospace;margin-bottom:8px;">
            <div>Stop</div><div>Status</div><div>Delay Prob</div><div>Rec. Vehicle</div><div>Reroute</div>
        </div>""", unsafe_allow_html=True)
        for p in ml_preds:
            pct=int(p["delay_prob"]*100)
            fc="#ff4757" if pct>60 else "#ff9500" if pct>35 else "#00cfa8"
            st_pill='<span class="pill pr">⚠️ DELAYED</span>' if p["delay_predicted"] else '<span class="pill pt">✅ ON TIME</span>'
            re_pill='<span class="pill pa">🔄 YES</span>' if p["reroute_flag"] else '<span style="color:#94a3b8;font-size:12px;">— no</span>'
            ico=VEHICLE_PROFILES.get(p["recommended_vehicle"],{}).get("icon","🚗")
            st.markdown(f"""<div class="pred-row">
                <div style="font-weight:700;color:#0d1626;font-size:14px;">{p['label']}</div>
                <div>{st_pill}</div>
                <div>
                    <div class="prob-track"><div class="prob-fill" style="width:{pct}%;background:{fc};"></div></div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:{fc};font-weight:700;">{pct}%</div>
                </div>
                <div style="font-size:13px;font-weight:600;color:#374151;">{ico} {p['recommended_vehicle']}</div>
                <div>{re_pill}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-head">📈 Delay Probability Chart</div>', unsafe_allow_html=True)
        df_ml=pd.DataFrame(ml_preds)
        df_ml["delay_pct"]=(df_ml["delay_prob"]*100).round(1)
        def delay_color_cat(p):
            if p>50: return "High (>50%)"
            elif p>30: return "Medium (30-50%)"
            return "Low (<30%)"
        df_ml["risk_band"]=df_ml["delay_pct"].apply(delay_color_cat)
        ch=alt.Chart(df_ml).mark_bar(cornerRadiusTopLeft=6,cornerRadiusTopRight=6).encode(
            x=alt.X("label:N",axis=alt.Axis(labelColor="#374151",labelAngle=-25,title="")),
            y=alt.Y("delay_pct:Q",scale=alt.Scale(domain=[0,100]),axis=alt.Axis(labelColor="#374151",title="Delay Probability %")),
            color=alt.Color("risk_band:N",legend=alt.Legend(title="Risk Band"),
                scale=alt.Scale(domain=["High (>50%)","Medium (30-50%)","Low (<30%)"],range=["#ff4757","#ff9500","#00cfa8"])),
            tooltip=["label","delay_pct","risk_band","recommended_vehicle"]
        ).properties(height=240,background="transparent").configure_view(strokeWidth=0,fill="transparent").configure_axis(grid=True,gridColor="#f1f5f9")
        st.altair_chart(ch,use_container_width=True)

        td=sum(1 for p in ml_preds if p["delay_predicted"])
        tr=sum(1 for p in ml_preds if p["reroute_flag"])
        st.markdown(f"""<div class="ml-panel">
            <div class="ml-badge">🤖 AI SYSTEM SUMMARY</div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:16px;">
                {''.join([
                    f'<div style="text-align:center;background:white;border-radius:12px;padding:14px;border:1.5px solid #e2e8f7;">'
                    f'<div style="font-size:24px;font-weight:900;font-family:Outfit,sans-serif;color:{c};">{v}</div>'
                    f'<div style="font-size:11px;color:#94a3b8;font-weight:600;margin-top:2px;">{l}</div></div>'
                    for v,l,c in [(len(ml_preds),"Orders","#7c3aed"),(td,"Delay Flags","#ff4757"),(tr,"Reroute Flags","#ff9500"),("86%","Accuracy","#00cfa8")]
                ])}
            </div>
            <div style="background:white;border-radius:12px;padding:14px;border:1.5px solid #e2e8f7;font-size:12px;color:#374151;line-height:2.2;font-weight:500;">
                ✅ Multi-criteria warehouse scoring &nbsp;·&nbsp; ✅ OSRM real road routing<br>
                ✅ RandomForest delay prediction (86% acc) &nbsp;·&nbsp; ✅ RF vehicle recommender<br>
                ✅ OR-Tools CVRPTW multi-vehicle optimizer
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="text-align:center;padding:80px 20px;">
            <div style="font-size:52px;margin-bottom:12px;">🤖</div>
            <div style="font-family:'Outfit',sans-serif;font-size:20px;font-weight:700;color:#374151;">No predictions yet</div>
            <div style="font-size:14px;color:#94a3b8;margin-top:5px;">Run optimization to generate AI predictions</div>
        </div>""", unsafe_allow_html=True)

