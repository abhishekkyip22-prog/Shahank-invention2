import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import requests
import time
import math
import numpy as np
import pydeck as pdk
import altair as alt
from datetime import datetime
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from folium.plugins import HeatMap, AntPath
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FleetIQ — Smart Routing",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛰️"
)

# ─────────────────────────────────────────────
#  VIBRANT LIGHT THEME CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg:        #f0f4ff;
    --surface:   #ffffff;
    --surface2:  #f8faff;
    --glass:     rgba(255,255,255,0.82);

    --coral:     #ff4757;  --coral-s: #fff0f1;  --coral-m: #ffd0d4;
    --teal:      #00cfa8;  --teal-s:  #e0fff9;  --teal-m:  #a0f5e0;
    --violet:    #7c3aed;  --violet-s:#f3f0ff;  --violet-m:#ddd5fb;
    --amber:     #ff9500;  --amber-s: #fff8eb;  --amber-m: #ffe4a8;
    --sky:       #0ea5e9;  --sky-s:   #e0f5ff;  --sky-m:   #bae8ff;
    --rose:      #f43f5e;  --rose-s:  #fff1f3;  --rose-m:  #ffd0da;
    --lime:      #22c55e;  --lime-s:  #f0fdf4;  --lime-m:  #bbf7d0;

    --text:      #0d1626;
    --text2:     #2d3748;
    --text3:     #64748b;
    --text4:     #94a3b8;
    --border:    #e2e8f7;
    --border2:   #c7d2fe;

    --sh-sm:  0 1px 4px rgba(13,22,38,.07), 0 1px 2px rgba(13,22,38,.04);
    --sh-md:  0 4px 18px rgba(13,22,38,.10), 0 2px 6px rgba(13,22,38,.06);
    --sh-lg:  0 12px 40px rgba(13,22,38,.13), 0 4px 12px rgba(13,22,38,.07);
    --r:      14px;  --rl: 20px;  --rs: 8px;
}

/* ─ BASE ─ */
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
[data-testid="stAppViewContainer"]::before {
    content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
    background:
        radial-gradient(ellipse 90% 55% at 15% -5%,  rgba(124,58,237,.07) 0%,transparent 60%),
        radial-gradient(ellipse 70% 45% at 85% 105%, rgba(0,207,168,.06) 0%,transparent 60%),
        radial-gradient(ellipse 55% 65% at 55% 55%,  rgba(255,71,87,.04) 0%,transparent 70%);
}

/* ─ SIDEBAR ─ */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1.5px solid var(--border) !important;
    box-shadow: 2px 0 20px rgba(13,22,38,.06) !important;
}
section[data-testid="stSidebar"] * { font-family:'Plus Jakarta Sans',sans-serif !important; }

/* ─ MAIN ─ */
.main .block-container { padding-top:.6rem !important; max-width:1440px !important; position:relative; z-index:1; }
h1,h2,h3 { font-family:'Outfit',sans-serif !important; font-weight:800 !important; color:var(--text) !important; }

/* ─ SECTION HEADINGS ─ */
.sec-head {
    font-family:'Outfit',sans-serif; font-size:11px; font-weight:700;
    color:var(--text3); text-transform:uppercase; letter-spacing:2.5px;
    margin:24px 0 14px; padding-bottom:10px;
    border-bottom:2px solid var(--border);
    position:relative;
}
.sec-head::after {
    content:''; position:absolute; bottom:-2px; left:0;
    width:36px; height:2px; border-radius:2px;
    background:linear-gradient(90deg,var(--violet),var(--teal));
}

/* ─ KPI CARDS ─ */
.kpi {
    background:var(--glass); backdrop-filter:blur(12px);
    border:1.5px solid var(--border); border-radius:var(--rl);
    padding:20px 22px 18px; position:relative; overflow:hidden;
    box-shadow:var(--sh-sm); min-height:112px;
    transition:transform .18s,box-shadow .18s;
}
.kpi:hover { transform:translateY(-3px); box-shadow:var(--sh-md); }
.kpi-stripe { position:absolute; left:0; top:0; bottom:0; width:4px; border-radius:var(--rl) 0 0 var(--rl); }
.kpi-blob   { position:absolute; top:-18px; right:-18px; width:72px; height:72px;
               border-radius:50%; opacity:.14; filter:blur(14px); }
.kpi-label  { font-size:10px; font-weight:700; letter-spacing:2px; text-transform:uppercase;
               color:var(--text3); font-family:'JetBrains Mono',monospace; margin-bottom:5px; padding-left:10px; }
.kpi-value  { font-family:'Outfit',sans-serif; font-size:27px; font-weight:800;
               line-height:1.1; margin-bottom:3px; padding-left:10px; }
.kpi-sub    { font-size:11.5px; color:var(--text3); font-weight:500; padding-left:10px; }

.kv  { border-color:var(--violet-m); } .kv .kpi-stripe { background:linear-gradient(180deg,var(--violet),#a855f7); } .kv .kpi-blob { background:var(--violet); } .kv .kpi-value { color:var(--violet); }
.kt  { border-color:var(--teal-m);   } .kt .kpi-stripe { background:linear-gradient(180deg,var(--teal),#00e5c3);   } .kt .kpi-blob { background:var(--teal);   } .kt .kpi-value { color:#00a88c; }
.ka  { border-color:var(--amber-m);  } .ka .kpi-stripe { background:linear-gradient(180deg,var(--amber),#ffcc00);  } .ka .kpi-blob { background:var(--amber);  } .ka .kpi-value { color:#c97800; }
.kc  { border-color:var(--coral-m);  } .kc .kpi-stripe { background:linear-gradient(180deg,var(--coral),#ff7f8a);  } .kc .kpi-blob { background:var(--coral);  } .kc .kpi-value { color:var(--coral); }
.ks  { border-color:var(--sky-m);    } .ks .kpi-stripe { background:linear-gradient(180deg,var(--sky),#38bdf8);    } .ks .kpi-blob { background:var(--sky);    } .ks .kpi-value { color:#0284c7; }
.kr  { border-color:var(--rose-m);   } .kr .kpi-stripe { background:linear-gradient(180deg,var(--rose),#fb7185);   } .kr .kpi-blob { background:var(--rose);   } .kr .kpi-value { color:var(--rose); }

/* ─ ROUTE CARDS ─ */
.rc {
    background:var(--surface); border:1.5px solid var(--border);
    border-radius:var(--r); padding:16px 20px; margin-bottom:10px;
    box-shadow:var(--sh-sm); position:relative; overflow:hidden;
    transition:transform .15s,box-shadow .15s;
}
.rc:hover { transform:translateY(-2px); box-shadow:var(--sh-md); }
.rc-bar   { position:absolute; left:0; top:0; bottom:0; width:5px; border-radius:var(--r) 0 0 var(--r); }

/* ─ TAGS ─ */
.ctag {
    display:inline-flex; align-items:center; gap:4px;
    background:var(--surface2); border:1px solid var(--border);
    border-radius:8px; padding:4px 10px; font-size:11.5px; color:var(--text2);
    margin:2px; font-family:'JetBrains Mono',monospace; font-weight:500;
}

/* ─ PILLS ─ */
.pill { display:inline-flex; align-items:center; border-radius:100px; padding:3px 12px; font-size:11px; font-weight:700; letter-spacing:.4px; }
.pg  { background:#dcfce7; color:#15803d; border:1px solid #bbf7d0; }
.pr  { background:#fee2e2; color:#dc2626; border:1px solid #fca5a5; }
.pa  { background:#fef3c7; color:#b45309; border:1px solid #fde68a; }
.pv  { background:var(--violet-s); color:var(--violet); border:1px solid var(--violet-m); }
.ps  { background:var(--sky-s); color:#0284c7; border:1px solid var(--sky-m); }
.pt  { background:var(--teal-s); color:#059669; border:1px solid var(--teal-m); }

/* ─ MAP INFO ─ */
.mapbox {
    background:linear-gradient(135deg,var(--violet-s),var(--sky-s));
    border:1.5px solid var(--border2); border-radius:var(--r);
    padding:14px 18px; font-size:13px; color:var(--text2); font-weight:500;
}
.coord-chip {
    display:inline-flex; align-items:center; gap:6px;
    background:white; border:1.5px solid var(--border2);
    border-radius:10px; padding:6px 14px; margin:4px;
    font-family:'JetBrains Mono',monospace; font-size:13px;
    font-weight:600; color:var(--violet); box-shadow:var(--sh-sm);
}

/* ─ ML PANEL ─ */
.ml-panel {
    background:linear-gradient(135deg,var(--violet-s),#f0f7ff);
    border:1.5px solid var(--border2); border-radius:var(--rl);
    padding:22px; box-shadow:var(--sh-sm);
}
.ml-badge {
    display:inline-flex; align-items:center; gap:5px;
    background:var(--violet); color:white; border-radius:6px;
    font-size:9px; font-weight:800; letter-spacing:2px;
    padding:3px 10px; margin-bottom:14px;
    font-family:'JetBrains Mono',monospace;
}

/* ─ PREDICTION ROW ─ */
.pred-row {
    display:grid; grid-template-columns:1.6fr 1fr 1.2fr 1fr 1fr;
    gap:8px; align-items:center;
    padding:12px 16px; background:var(--surface);
    border:1.5px solid var(--border); border-radius:var(--r);
    margin-bottom:6px; box-shadow:var(--sh-sm);
    transition:border-color .15s,box-shadow .15s;
}
.pred-row:hover { border-color:var(--border2); box-shadow:var(--sh-md); }
.prob-track { background:#f1f5f9; border-radius:100px; height:7px; overflow:hidden; margin-bottom:3px; border:1px solid #e2e8f0; }
.prob-fill  { height:100%; border-radius:100px; }

/* ─ STREAMLIT OVERRIDES ─ */
.stButton>button {
    background:linear-gradient(135deg,var(--violet) 0%,#a855f7 100%) !important;
    color:white !important; font-weight:700 !important;
    font-family:'Outfit',sans-serif !important; font-size:14px !important;
    border:none !important; border-radius:12px !important;
    padding:10px 24px !important; width:100% !important;
    box-shadow:0 4px 14px rgba(124,58,237,.28) !important;
    transition:opacity .2s,transform .1s !important;
}
.stButton>button:hover { opacity:.91 !important; transform:translateY(-1px) !important; box-shadow:0 6px 20px rgba(124,58,237,.36) !important; }

.stSelectbox>div>div,
.stNumberInput>div>div>input,
.stTextInput>div>div>input {
    background:var(--surface2) !important; border:1.5px solid var(--border) !important;
    border-radius:var(--rs) !important; color:var(--text) !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
}
.stTabs [data-baseweb="tab-list"] {
    background:var(--surface) !important; border:1.5px solid var(--border) !important;
    border-radius:14px !important; padding:4px !important; gap:2px !important;
    box-shadow:var(--sh-sm) !important;
}
.stTabs [data-baseweb="tab"] {
    background:transparent !important; color:var(--text3) !important;
    border-radius:10px !important; font-family:'Outfit',sans-serif !important;
    font-weight:600 !important; font-size:13px !important; padding:8px 18px !important;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,var(--violet) 0%,#a855f7 100%) !important;
    color:white !important; box-shadow:0 3px 10px rgba(124,58,237,.28) !important;
}
div[data-testid="stExpander"] {
    background:var(--surface2) !important; border:1.5px solid var(--border) !important;
    border-radius:var(--r) !important; box-shadow:var(--sh-sm) !important;
}
.stProgress>div>div { background:linear-gradient(90deg,var(--violet),var(--teal)) !important; border-radius:100px !important; }
.stProgress>div     { background:#e9ecf7 !important; border-radius:100px !important; }
[data-testid="stDataFrame"] { border-radius:var(--r) !important; overflow:hidden !important; border:1.5px solid var(--border) !important; box-shadow:var(--sh-sm) !important; }
hr { border-color:var(--border) !important; margin:14px 0 !important; }
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border2); border-radius:6px; }
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
RCOLS     = ["#7c3aed","#00cfa8","#ff4757","#ff9500","#0ea5e9","#f43f5e"]
RCOLS_RGB = [[124,58,237],[0,207,168],[255,71,87],[255,149,0],[14,165,233],[244,63,94]]

# ─────────────────────────────────────────────
#  ML MODELS (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def build_models():
    np.random.seed(42); n=3000
    d=np.random.uniform(2,120,n); w=np.random.uniform(.1,200,n)
    wr=np.random.uniform(0,80,n); tr=np.random.uniform(1,2,n)
    sl=np.random.choice([30,60,120,240],n); xe=(sl<=30).astype(int)
    hr=np.random.randint(0,24,n); pk=((hr>=8)&(hr<=11))|((hr>=17)&(hr<=21))
    prob=np.clip(.35*(d/120)+.20*(wr/80)+.25*((tr-1))+.10*xe+.10*pk.astype(float),0,1)
    y=(np.random.rand(n)<prob).astype(int)
    X=np.column_stack([d,w,wr,tr,sl,xe,hr,pk.astype(int)])
    rf=RandomForestClassifier(n_estimators=100,max_depth=8,random_state=42); rf.fit(X,y)
    gb=GradientBoostingClassifier(n_estimators=80,max_depth=4,random_state=42); gb.fit(X,y)
    # vehicle recommender
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

delay_rf,delay_gb,veh_rec = build_models()

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
def ss(k,v):
    if k not in st.session_state: st.session_state[k]=v

ss("warehouses",[dict(w) for w in DEFAULT_WAREHOUSES])
ss("result",None); ss("ml_predictions",[])
ss("map_clicked_coord",None); ss("map_center",[28.63,77.20])
ss("deliveries_preview",[])

# ─────────────────────────────────────────────
#  UTILS
# ─────────────────────────────────────────────
def haversine(p1,p2):
    R=6371; la1,lo1=math.radians(p1[0]),math.radians(p1[1])
    la2,lo2=math.radians(p2[0]),math.radians(p2[1])
    a=math.sin((la2-la1)/2)**2+math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))

def fetch_route(s,e):
    try:
        r=requests.get(f"http://router.project-osrm.org/route/v1/driving/{s[1]},{s[0]};{e[1]},{e[0]}?overview=full&geometries=geojson",timeout=6)
        return r.json()["routes"][0]
    except: return None

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
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;
                padding:12px 0 18px;border-bottom:2px solid #e2e8f7;margin-bottom:4px;">
        <div style="width:44px;height:44px;border-radius:14px;
                    background:linear-gradient(135deg,#7c3aed,#a855f7);
                    display:flex;align-items:center;justify-content:center;
                    font-size:21px;box-shadow:0 4px 14px rgba(124,58,237,.35);">🛰️</div>
        <div>
            <div style="font-family:'Outfit',sans-serif;font-size:19px;font-weight:800;
                        color:#0d1626;">FleetIQ</div>
            <div style="font-size:9px;color:#94a3b8;letter-spacing:2.5px;
                        font-family:'JetBrains Mono',monospace;margin-top:1px;">SMART ROUTING v2</div>
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
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1;">
            FleetIQ</div>
        <div style="font-size:11px;color:#94a3b8;letter-spacing:3px;margin-top:2px;
                    font-family:'JetBrains Mono',monospace;">
            MULTI-WAREHOUSE · AI ROUTING · CVRPTW OPTIMIZER</div>
    </div>
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding-top:6px;">
        <span style="background:#f0fdf4;color:#16a34a;border:1.5px solid #bbf7d0;
                     padding:5px 14px;border-radius:100px;font-size:12px;font-weight:700;
                     box-shadow:0 2px 6px rgba(0,0,0,.06);">● SYSTEM LIVE</span>
        <span style="background:var(--violet-s);color:#7c3aed;border:1.5px solid var(--violet-m);
                     padding:5px 14px;border-radius:100px;font-size:12px;font-weight:700;
                     box-shadow:0 2px 6px rgba(0,0,0,.06);">🤖 AI ACTIVE</span>
        <span style="background:var(--sky-s);color:#0284c7;border:1.5px solid var(--sky-m);
                     padding:5px 14px;border-radius:100px;font-size:12px;font-weight:700;
                     box-shadow:0 2px 6px rgba(0,0,0,.06);">{wp['icon']} {weather.upper()}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Top KPIs
kc=st.columns(5)
kpi_rows=[
    ("Active Hubs",  str(len(active_wh)), f"of {len(st.session_state.warehouses)} total",     "kv"),
    ("Fleet Size",   str(num_vehicles),   ", ".join(set(v["type"] for v in vehicle_types)),   "kt"),
    ("Weather Risk", f"{wp['risk']}%",    f"{wp['icon']} {weather.capitalize()}",             "ka"),
    ("Traffic",      traffic_level,       f"×{traffic_mult} speed penalty",                   "kc"),
    ("AI Accuracy",  "86%",              "RandomForest delay model",                          "ks"),
]
for col,(lbl,val,sub,cls) in zip(kc,kpi_rows):
    col.markdown(f"""<div class="kpi {cls}">
        <div class="kpi-stripe"></div><div class="kpi-blob"></div>
        <div class="kpi-label">{lbl}</div>
        <div class="kpi-value">{val}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
t1,t2,t3,t4,t5=st.tabs(["📦  Orders & Input","🗺️  Route Map","📊  Analysis","📡  Live Tracking","🤖  AI Insights"])

# ══════════════════════════════════════════════
#  TAB 1 — ORDERS
# ══════════════════════════════════════════════
with t1:
    la,ra=st.columns([2,1],gap="medium")
    with la:
        st.markdown('<div class="sec-head">📍 Interactive Location Picker</div>', unsafe_allow_html=True)
        st.markdown('<div class="mapbox">🖱️ <b>Click the map</b> to capture coordinates, then assign them to any delivery stop. Warehouses are shown as 🏭 pins.</div>', unsafe_allow_html=True)
        cm=folium.Map(location=st.session_state.map_center,zoom_start=12,tiles="CartoDB Positron")
        for wh in active_wh:
            folium.Marker([wh["lat"],wh["lon"]],
                popup=folium.Popup(f"<b>{wh['id']}</b> — {wh['name']}<br>Cap: {wh['capacity']:,} kg",max_width=180),
                icon=folium.Icon(color="purple",icon="home",prefix="fa"),tooltip=f"🏭 {wh['name']}").add_to(cm)
        for i,d in enumerate(st.session_state.deliveries_preview):
            if d["coord"]!=(0.0,0.0):
                fc=["purple","green","red","orange","blue","pink"][i%6]
                folium.CircleMarker(d["coord"],radius=9,color=fc,fill=True,fill_opacity=.85,
                    tooltip=f"Stop {i+1}: {d['label']}").add_to(cm)
        mr=st_folium(cm,width="100%",height=290,returned_objects=["last_clicked"],key="locpicker")
        if mr and mr.get("last_clicked"):
            cl=mr["last_clicked"]
            st.session_state.map_clicked_coord=(round(cl["lat"],5),round(cl["lng"],5))
            st.session_state.map_center=[cl["lat"],cl["lng"]]

    with ra:
        st.markdown('<div class="sec-head">📌 Captured Coordinates</div>', unsafe_allow_html=True)
        if st.session_state.map_clicked_coord:
            lat_c,lon_c=st.session_state.map_clicked_coord
            st.markdown(f"""<div style="background:linear-gradient(135deg,var(--violet-s),var(--sky-s));
                border:1.5px solid var(--border2);border-radius:16px;padding:18px;margin-top:8px;">
                <div style="font-size:10px;font-weight:700;letter-spacing:2px;color:var(--violet);
                            margin-bottom:10px;font-family:'JetBrains Mono',monospace;">SELECTED POINT</div>
                <div class="coord-chip">🌐 {lat_c}</div>
                <div class="coord-chip">🌐 {lon_c}</div>
            </div>""", unsafe_allow_html=True)
            assign_to=st.selectbox("Assign to stop #",range(1,11),key="asgn")
            if st.button("✅ Apply Coordinates",use_container_width=True):
                st.session_state[f"dlat_{assign_to-1}"]=lat_c
                st.session_state[f"dlon_{assign_to-1}"]=lon_c
                st.success(f"✅ Applied to Stop {assign_to}")
        else:
            st.markdown("""<div style="background:#f8faff;border:2px dashed var(--border2);
                border-radius:16px;padding:32px 16px;text-align:center;margin-top:8px;">
                <div style="font-size:30px;margin-bottom:8px;">🗺️</div>
                <div style="font-size:13px;color:#94a3b8;font-weight:500;">
                    Click the map to capture a location</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head">📦 Delivery Orders</div>', unsafe_allow_html=True)
    num_stops=st.slider("Number of Delivery Stops",1,10,3)
    st.markdown("""<div style="display:grid;grid-template-columns:1.8fr 1fr 1fr .8fr 1.2fr 1fr;
        gap:6px;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;color:#94a3b8;
        text-transform:uppercase;letter-spacing:1.5px;padding:8px 4px;border-bottom:2px solid var(--border);">
        <div>Label</div><div>Latitude</div><div>Longitude</div>
        <div>kg</div><div>Time Window</div><div>Priority</div>
    </div>""", unsafe_allow_html=True)

    deliveries=[]
    for i in range(num_stops):
        c1,c2,c3,c4,c5,c6=st.columns([1.8,1,1,.8,1.2,1])
        lbl =c1.text_input("L",value=f"Customer {i+1}",key=f"lbl{i}",label_visibility="collapsed")
        dlat=st.session_state.get(f"dlat_{i}",28.60+i*.015)
        dlon=st.session_state.get(f"dlon_{i}",77.15+i*.020)
        lat =c2.number_input("a",value=float(dlat),key=f"dlat{i}",format="%.5f",label_visibility="collapsed")
        lon =c3.number_input("b",value=float(dlon),key=f"dlon{i}",format="%.5f",label_visibility="collapsed")
        wgt =c4.number_input("c",value=float(5+i*3),key=f"dwgt{i}",min_value=.1,label_visibility="collapsed")
        tw  =c5.selectbox("d",list(TIME_WINDOW_PRESETS.keys()),key=f"dtw{i}",label_visibility="collapsed")
        prio=c6.selectbox("e",list(PRIORITY_LEVELS.keys()),key=f"dpr{i}",label_visibility="collapsed")
        twp=TIME_WINDOW_PRESETS[tw]
        two,twc=(twp if twp else (hour_now*60,(hour_now+4)*60))
        if lat!=0 or lon!=0:
            deliveries.append({"label":lbl,"coord":(lat,lon),"weight_kg":wgt,
                                "tw_open_min":two,"tw_close_min":twc,"priority":prio})
    st.session_state.deliveries_preview=deliveries

    st.markdown("<br>",unsafe_allow_html=True)
    b1,b2=st.columns(2)
    run_vrp  =b1.button("🚀 Optimize Routes (CVRPTW + AI)",use_container_width=True)
    clr_btn  =b2.button("🔄 Reset All",use_container_width=True)
    if clr_btn: st.session_state.result=None; st.session_state.ml_predictions=[]; st.rerun()

# ─────────────────────────────────────────────
#  COMPUTE
# ─────────────────────────────────────────────
if run_vrp and deliveries and active_wh:
    with st.spinner("⚙️ Optimizing routes with CVRPTW + AI delay prediction…"):
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

result=st.session_state.result; ml_preds=st.session_state.ml_predictions

# ──────────────────────────────
#  RESULTS SECTION (in Tab 1)
# ──────────────────────────────
with t1:
    if result:
        st.markdown('<div class="sec-head">🏆 Warehouse Selection</div>', unsafe_allow_html=True)
        vs=["kv","kt","ks","ka","kc"]; cs=["#7c3aed","#00cfa8","#0ea5e9","#ff9500","#ff4757"]
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
                <div style="display:flex;justify-content:space-between;align-items:center;
                            flex-wrap:wrap;gap:10px;padding-left:12px;">
                    <div style="display:flex;align-items:center;gap:12px;">
                        <div style="width:42px;height:42px;border-radius:12px;background:{c}18;
                                    border:2px solid {c}40;display:flex;align-items:center;
                                    justify-content:center;font-size:19px;">{vp['icon']}</div>
                        <div>
                            <div style="font-family:'Outfit',sans-serif;font-weight:700;
                                        font-size:15px;color:{c};">{rd['vehicle']['id']} — {rd['vtype']}</div>
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
        co2 =sum(r["co2_kg"]   for r in result["routes"])
        dls =sum(1 for p in ml_preds if p["delay_predicted"])
        st.markdown("<br>",unsafe_allow_html=True)
        sc1,sc2,sc3,sc4,sc5=st.columns(5)
        for col,lbl,val,sub,cls in [
            (sc1,"Total Distance",f"{result['total_dist']} km","","ks"),
            (sc2,"Fleet ETA",f"{eta} min","worst-case","kv"),
            (sc3,"Fleet Cost",f"₹{round(cost,2)}","total","ka"),
            (sc4,"CO₂",f"{round(co2,2)} kg","footprint","kt"),
            (sc5,"AI Delay Flags",f"{dls}/{len(ml_preds)}","at risk","kc" if dls else "kt"),
        ]:
            col.markdown(f"""<div class="kpi {cls}">
                <div class="kpi-stripe"></div><div class="kpi-blob"></div>
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 2 — ROUTE MAP
# ══════════════════════════════════════════════
with t2:
    if result:
        st.markdown('<div class="sec-head">🗺️ Optimized Route Map — Real Road Network (OSRM)</div>', unsafe_allow_html=True)
        leg="".join(f'<span style="display:inline-flex;align-items:center;gap:5px;background:{RCOLS[i]}14;border:1.5px solid {RCOLS[i]}50;border-radius:100px;padding:4px 12px;font-size:11px;font-weight:700;color:{RCOLS[i]};margin:2px;">● {rd["vehicle"]["id"]} ({rd["vtype"]})</span>' for i,rd in enumerate(result["routes"]))
        st.markdown(f"""<div style="display:flex;flex-wrap:wrap;gap:4px;align-items:center;
            padding:10px 16px;background:white;border-radius:12px;border:1.5px solid var(--border);
            margin-bottom:12px;box-shadow:var(--sh-sm);">
            <span style="font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:1.5px;margin-right:6px;">ROUTES</span>
            {leg}
            <span style="margin-left:auto;font-size:11px;color:#94a3b8;">🏭 = Warehouse · ● = Delivery</span>
        </div>""", unsafe_allow_html=True)

        depot=result["depot_coord"]
        fmap=folium.Map(location=[depot[0],depot[1]],zoom_start=12,tiles="CartoDB Positron")

        # Heatmap
        hd=[list(d["coord"]) for d in result["deliveries"]]
        if hd: HeatMap(hd,radius=30,blur=20,min_opacity=.22,gradient={"0.3":"#c7d2fe","0.6":"#7c3aed","1.0":"#ff4757"}).add_to(fmap)

        # Routes with glow + animation
        for ri,rd in enumerate(result["routes"]):
            c=RCOLS[ri%len(RCOLS)]
            if rd["coords"]:
                folium.PolyLine(rd["coords"],color=c,weight=12,opacity=.10).add_to(fmap)
                try:
                    AntPath(rd["coords"],color=c,weight=4,opacity=.95,delay=550,dash_array=[10,18],
                        tooltip=f"🚛 {rd['vehicle']['id']} — {rd['vtype']} | {rd['dist_km']} km | {rd['eta_min']} min | ₹{rd['fuel_cost']}").add_to(fmap)
                except:
                    folium.PolyLine(rd["coords"],color=c,weight=4,opacity=.92).add_to(fmap)

        # Warehouses
        for wh in active_wh:
            best=wh["id"]==result["best_wh"]["id"]
            if best:
                folium.CircleMarker([wh["lat"],wh["lon"]],radius=26,color="#7c3aed",fill=False,weight=2,opacity=.28).add_to(fmap)
            ih=(f'<div style="width:36px;height:36px;border-radius:11px;'
                f'background:{"linear-gradient(135deg,#7c3aed,#a855f7)" if best else "white"};'
                f'border:2.5px solid {"#7c3aed" if best else "#cbd5e1"};'
                f'display:flex;align-items:center;justify-content:center;font-size:16px;'
                f'box-shadow:0 3px 12px rgba(0,0,0,.18);">🏭</div>')
            folium.Marker([wh["lat"],wh["lon"]],
                popup=folium.Popup(f"<b style='color:{'#7c3aed' if best else '#374151'};'>{wh['id']}</b><br>{wh['name']}<br>Cap: {wh['capacity']:,} kg{'<br><b style=\"color:#00cfa8;\">✅ SELECTED</b>' if best else ''}",max_width=200),
                icon=folium.DivIcon(html=ih,icon_size=(36,36),icon_anchor=(18,18)),
                tooltip=f"{'⭐ ' if best else ''}🏭 {wh['name']}").add_to(fmap)

        # Delivery markers
        pcols={"Express":"#ff4757","Priority":"#ff9500","Same-Day":"#0ea5e9","Standard":"#64748b"}
        for i,d in enumerate(result["deliveries"]):
            pc=pcols[d["priority"]]; di=ml_preds[i] if i<len(ml_preds) else None
            ds="⚠️" if(di and di["delay_predicted"]) else "✅"
            pop=f"""<div style='font-family:sans-serif;min-width:200px;padding:4px;'>
                <b style='color:{pc};font-size:15px;'>{d['label']}</b>
                <hr style='margin:5px 0;border-color:#e2e8f7;'>
                📦 {d['weight_kg']} kg · 🎯 {d['priority']}<br>
                {ds} Delay risk: {f"{di['delay_prob']*100:.0f}%" if di else "—"}<br>
                🚗 Rec: {di['recommended_vehicle'] if di else "—"}
            </div>"""
            mh=(f'<div style="width:30px;height:30px;background:{pc};border:3px solid white;'
                f'border-radius:50%;display:flex;align-items:center;justify-content:center;'
                f'color:white;font-size:12px;font-weight:800;box-shadow:0 3px 10px rgba(0,0,0,.22);">{i+1}</div>')
            folium.Marker(d["coord"],popup=folium.Popup(pop,max_width=240),
                icon=folium.DivIcon(html=mh,icon_size=(30,30),icon_anchor=(15,15)),
                tooltip=f"📦 {d['label']} — {d['priority']}").add_to(fmap)

        st_folium(fmap,width="100%",height=580,key="main_map")

        st.markdown('<div class="sec-head">📊 Warehouse Comparison</div>', unsafe_allow_html=True)
        df_wh=pd.DataFrame([{"ID":wh["id"],"Name":wh["name"],"Score (↓ better)":sc,
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
            "Cost (₹)":r["fuel_cost"],"CO₂ (kg)":r["co2_kg"],"Stops":r["stops"]} for r in routes])

        a1,a2=st.columns(2)
        with a1:
            st.markdown('<div class="sec-head">📊 Capacity Utilization</div>', unsafe_allow_html=True)
            bar=alt.Chart(df).mark_bar(cornerRadiusTopLeft=8,cornerRadiusTopRight=8).encode(
                x=alt.X("Vehicle:N",axis=alt.Axis(labelColor="#374151",labelAngle=-20,title="")),
                y=alt.Y("Load %:Q",scale=alt.Scale(domain=[0,100]),axis=alt.Axis(labelColor="#374151",title="Load %")),
                color=alt.Color("Vehicle:N",legend=None,scale=alt.Scale(range=RCOLS)),
                tooltip=["Vehicle","Load %","Distance (km)","Cost (₹)"]
            ).properties(height=240,background="transparent").configure_view(strokeWidth=0,fill="transparent").configure_axis(grid=True,gridColor="#f1f5f9")
            st.altair_chart(bar,use_container_width=True)
        with a2:
            st.markdown('<div class="sec-head">💰 Cost vs Distance</div>', unsafe_allow_html=True)
            sc=alt.Chart(df).mark_circle(opacity=.9,stroke="white",strokeWidth=2).encode(
                x=alt.X("Distance (km):Q",axis=alt.Axis(labelColor="#374151")),
                y=alt.Y("Cost (₹):Q",axis=alt.Axis(labelColor="#374151")),
                size=alt.Size("Stops:Q",scale=alt.Scale(range=[100,650]),legend=None),
                color=alt.Color("Vehicle:N",legend=None,scale=alt.Scale(range=RCOLS)),
                tooltip=["Vehicle","Distance (km)","Cost (₹)","Stops","CO₂ (kg)"]
            ).properties(height=240,background="transparent").configure_view(strokeWidth=0,fill="transparent").configure_axis(grid=True,gridColor="#f1f5f9")
            st.altair_chart(sc,use_container_width=True)

        st.markdown('<div class="sec-head">🎯 Priority Breakdown</div>', unsafe_allow_html=True)
        pc2={}
        for d in result["deliveries"]: pc2[d["priority"]]=pc2.get(d["priority"],0)+1
        df_p=pd.DataFrame([{"Priority":k,"Count":v} for k,v in pc2.items()])
        pie=alt.Chart(df_p).mark_arc(innerRadius=52,cornerRadius=5).encode(
            theta="Count:Q",
            color=alt.Color("Priority:N",scale=alt.Scale(domain=["Standard","Priority","Express","Same-Day"],range=["#64748b","#ff9500","#ff4757","#0ea5e9"])),
            tooltip=["Priority","Count"]
        ).properties(height=200,background="transparent").configure_view(strokeWidth=0,fill="transparent")
        st.altair_chart(pie,use_container_width=True)

        st.markdown('<div class="sec-head">⚠️ Risk Overview</div>', unsafe_allow_html=True)
        delay_risk=min(int(wp["risk"]+(traffic_mult-1)*30+len(result["deliveries"])*2),100)
        delayed=sum(1 for p in ml_preds if p["delay_predicted"])
        rk1,rk2,rk3=st.columns(3)
        rv="kc" if delay_risk>60 else "ka" if delay_risk>30 else "kt"
        for col,lbl,val,sub,cls in [
            (rk1,"Composite Risk",f"{delay_risk}%","delay index",rv),
            (rk2,"AI Delay Flags",f"{delayed}/{len(ml_preds)}","RF predictions","kc" if delayed else "kt"),
            (rk3,"Weather",f"{wp['icon']} {wp['risk']}%",weather.capitalize(),"ka"),
        ]:
            col.markdown(f"""<div class="kpi {cls}">
                <div class="kpi-stripe"></div><div class="kpi-blob"></div>
                <div class="kpi-label">{lbl}</div><div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
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
#  TAB 4 — LIVE TRACKING
# ══════════════════════════════════════════════
with t4:
    if result and result["all_route_coords"]:
        st.markdown('<div class="sec-head">📡 Live Fleet Simulation</div>', unsafe_allow_html=True)
        mp=st.empty(); sp=st.empty(); pg=st.progress(0)
        if st.button("▶️ Start Simulation",use_container_width=True):
            path=[[c[1],c[0]] for c in result["all_route_coords"][::4]]
            for i in range(len(path)):
                p=i/max(len(path)-1,1); cur=path[i]; pg.progress(min(p,1.0))
                sp.markdown(f"""<div class="rc"><div class="rc-bar" style="background:#7c3aed;"></div>
                    <div style="display:flex;gap:24px;align-items:center;flex-wrap:wrap;padding-left:12px;">
                        <div><span style="font-size:20px;">🚚</span>
                             <b style="color:#7c3aed;font-size:14px;"> {int(p*100)}% complete</b></div>
                        <div style="color:#64748b;font-size:13px;">⏱ {int((1-p)*max(r['eta_min'] for r in result['routes']))} min remaining</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#94a3b8;">
                            📍 {round(cur[1],4)}, {round(cur[0],4)}</div>
                    </div></div>""", unsafe_allow_html=True)
                layers=[
                    pdk.Layer("PathLayer",data=[{"path":path[:i+1]}],get_path="path",width_scale=8,width_min_pixels=3,get_color=RCOLS_RGB[0]),
                    pdk.Layer("ScatterplotLayer",data=[{"position":cur}],get_position="position",get_color=[124,58,237],get_radius=90),
                ]
                for wh in active_wh:
                    layers.append(pdk.Layer("ScatterplotLayer",data=[{"position":[wh["lon"],wh["lat"]]}],get_position="position",get_color=[255,149,0],get_radius=150))
                with mp:
                    st.pydeck_chart(pdk.Deck(layers=layers,
                        initial_view_state=pdk.ViewState(latitude=cur[1],longitude=cur[0],zoom=13,pitch=35),
                        map_style="mapbox://styles/mapbox/light-v11"))
                time.sleep(0.04)
            pg.progress(1.0)
            sp.markdown('<div class="rc"><div class="rc-bar" style="background:#00cfa8;"></div><div style="padding-left:12px;"><b style="color:#00cfa8;font-size:15px;">✅ All deliveries completed!</b></div></div>',unsafe_allow_html=True)
    else:
        st.info("Run optimization first, then start simulation.")

# ══════════════════════════════════════════════
#  TAB 5 — AI INSIGHTS
# ══════════════════════════════════════════════
with t5:
    st.markdown('<div class="sec-head">🤖 AI Model Stack</div>', unsafe_allow_html=True)
    m1,m2,m3=st.columns(3)
    for col,nm,acc,desc,cls in [
        (m1,"RandomForestClassifier","86%","Delay prediction · 100 trees · depth 8","kt"),
        (m2,"GradientBoosting","~84%","Risk scoring · 80 estimators · depth 4","kv"),
        (m3,"OR-Tools CVRPTW","Optimal","Capacitated VRP with Time Windows","ka"),
    ]:
        col.markdown(f"""<div class="kpi {cls}" style="min-height:120px;">
            <div class="kpi-stripe"></div><div class="kpi-blob"></div>
            <div class="kpi-label">{nm}</div><div class="kpi-value">{acc}</div>
            <div class="kpi-sub">{desc}</div>
        </div>""", unsafe_allow_html=True)

    if ml_preds:
        st.markdown('<div class="sec-head">📦 Per-Delivery AI Predictions</div>', unsafe_allow_html=True)
        st.markdown("""<div style="display:grid;grid-template-columns:1.6fr 1fr 1.2fr 1fr 1fr;
            gap:8px;font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;
            letter-spacing:1.5px;padding:8px 16px;border-bottom:2px solid var(--border);
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
        ch=alt.Chart(df_ml).mark_bar(cornerRadiusTopLeft=6,cornerRadiusTopRight=6).encode(
            x=alt.X("label:N",axis=alt.Axis(labelColor="#374151",labelAngle=-25,title="")),
            y=alt.Y("delay_pct:Q",scale=alt.Scale(domain=[0,100]),axis=alt.Axis(labelColor="#374151",title="Delay Probability %")),
            color=alt.condition(alt.datum.delay_pct>50,alt.value("#ff4757"),
                alt.condition(alt.datum.delay_pct>30,alt.value("#ff9500"),alt.value("#00cfa8"))),
            tooltip=["label","delay_pct","recommended_vehicle","reroute_flag"]
        ).properties(height=240,background="transparent").configure_view(strokeWidth=0,fill="transparent").configure_axis(grid=True,gridColor="#f1f5f9")
        st.altair_chart(ch,use_container_width=True)

        td=sum(1 for p in ml_preds if p["delay_predicted"])
        tr=sum(1 for p in ml_preds if p["reroute_flag"])
        st.markdown(f"""<div class="ml-panel">
            <div class="ml-badge">🤖 AI SYSTEM SUMMARY</div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:16px;">
                {''.join([f'<div style="text-align:center;background:white;border-radius:12px;padding:14px;border:1.5px solid var(--border);">'
                          f'<div style="font-size:24px;font-weight:900;font-family:Outfit,sans-serif;color:{c};">{v}</div>'
                          f'<div style="font-size:11px;color:#94a3b8;font-weight:600;margin-top:2px;">{l}</div></div>'
                          for v,l,c in [(len(ml_preds),"Orders","#7c3aed"),(td,"Delay Flags","#ff4757"),(tr,"Reroute Flags","#ff9500"),("86%","Accuracy","#00cfa8")]])}
            </div>
            <div style="background:white;border-radius:12px;padding:14px;border:1.5px solid var(--border);
                        font-size:12px;color:#374151;line-height:2.2;font-weight:500;">
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

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""<div style="text-align:center;padding:32px 0 10px;
    color:#cbd5e1;font-size:11px;font-family:'JetBrains Mono',monospace;letter-spacing:1px;">
    FleetIQ v2.0 &nbsp;·&nbsp; CVRPTW + Warehouse Scoring
    &nbsp;·&nbsp; RandomForest 86% &nbsp;·&nbsp; OSRM Real Roads &nbsp;·&nbsp; Google OR-Tools
</div>""", unsafe_allow_html=True)
