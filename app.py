import streamlit as st
import numpy as np
from scipy.stats import poisson
import unicodedata
import pandas as pd
from datetime import date
import requests
from bs4 import BeautifulSoup
import re

st.set_page_config(
    page_title="Mundial 2026 · Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================
# DATOS BASE — 49 selecciones
# =====================================================================
DATOS_INICIAL = {
    'Canadá':               {'ataque': 1.06, 'defensa': 0.94, 'factor': 1.1},
    'Estados Unidos':       {'ataque': 1.12, 'defensa': 0.88, 'factor': 1.1},
    'México':               {'ataque': 1.05, 'defensa': 0.92, 'factor': 1.1},
    'Argentina':            {'ataque': 1.45, 'defensa': 0.65, 'factor': 1.3},
    'Brasil':               {'ataque': 1.35, 'defensa': 0.72, 'factor': 1.3},
    'Colombia':             {'ataque': 1.22, 'defensa': 0.74, 'factor': 1.0},
    'Ecuador':              {'ataque': 1.08, 'defensa': 0.82, 'factor': 0.9},
    'Paraguay':             {'ataque': 0.88, 'defensa': 0.85, 'factor': 0.9},
    'Uruguay':              {'ataque': 1.25, 'defensa': 0.75, 'factor': 1.1},
    'Alemania':             {'ataque': 1.30, 'defensa': 0.78, 'factor': 1.2},
    'Austria':              {'ataque': 1.08, 'defensa': 0.91, 'factor': 1.0},
    'Bélgica':              {'ataque': 1.18, 'defensa': 0.85, 'factor': 1.0},
    'Bosnia y Herzegovina': {'ataque': 0.96, 'defensa': 1.02, 'factor': 0.9},
    'República Checa':      {'ataque': 1.02, 'defensa': 0.96, 'factor': 0.9},
    'Croacia':              {'ataque': 1.12, 'defensa': 0.82, 'factor': 1.2},
    'Escocia':              {'ataque': 0.94, 'defensa': 1.04, 'factor': 0.8},
    'España':               {'ataque': 1.40, 'defensa': 0.70, 'factor': 1.2},
    'Francia':              {'ataque': 1.42, 'defensa': 0.68, 'factor': 1.2},
    'Inglaterra':           {'ataque': 1.35, 'defensa': 0.72, 'factor': 1.2},
    'Noruega':              {'ataque': 1.14, 'defensa': 0.95, 'factor': 0.9},
    'Países Bajos':         {'ataque': 1.20, 'defensa': 0.80, 'factor': 1.1},
    'Portugal':             {'ataque': 1.32, 'defensa': 0.75, 'factor': 1.1},
    'Suecia':               {'ataque': 1.08, 'defensa': 0.92, 'factor': 1.0},
    'Suiza':                {'ataque': 1.05, 'defensa': 0.88, 'factor': 1.0},
    'Turquía':              {'ataque': 1.05, 'defensa': 0.95, 'factor': 0.9},
    'Curazao':              {'ataque': 0.90, 'defensa': 1.02, 'factor': 0.7},
    'Haití':                {'ataque': 0.88, 'defensa': 1.06, 'factor': 0.7},
    'Panamá':               {'ataque': 0.95, 'defensa': 1.00, 'factor': 0.8},
    'Argelia':              {'ataque': 1.05, 'defensa': 0.92, 'factor': 0.9},
    'Cabo Verde':           {'ataque': 0.92, 'defensa': 0.98, 'factor': 0.8},
    'Costa de Marfil':      {'ataque': 1.08, 'defensa': 0.88, 'factor': 1.0},
    'Egipto':               {'ataque': 1.04, 'defensa': 0.84, 'factor': 0.9},
    'Ghana':                {'ataque': 0.98, 'defensa': 1.02, 'factor': 1.0},
    'Marruecos':            {'ataque': 1.15, 'defensa': 0.72, 'factor': 1.1},
    'R. D. del Congo':      {'ataque': 0.94, 'defensa': 0.96, 'factor': 0.8},
    'Senegal':              {'ataque': 1.12, 'defensa': 0.82, 'factor': 0.95},
    'Sudáfrica':            {'ataque': 0.95, 'defensa': 0.96, 'factor': 0.8},
    'Túnez':                {'ataque': 0.92, 'defensa': 0.90, 'factor': 0.9},
    'Arabia Saudita':       {'ataque': 0.96, 'defensa': 1.02, 'factor': 0.8},
    'Australia':            {'ataque': 1.08, 'defensa': 0.90, 'factor': 0.9},
    'Corea del Sur':        {'ataque': 1.15, 'defensa': 0.88, 'factor': 0.9},
    'Irán':                 {'ataque': 1.10, 'defensa': 0.86, 'factor': 0.9},
    'Irak':                 {'ataque': 0.98, 'defensa': 0.96, 'factor': 0.8},
    'Japón':                {'ataque': 1.25, 'defensa': 0.78, 'factor': 1.0},
    'Jordania':             {'ataque': 0.94, 'defensa': 1.04, 'factor': 0.7},
    'Catar':                {'ataque': 0.94, 'defensa': 1.04, 'factor': 0.8},
    'Uzbekistán':           {'ataque': 0.95, 'defensa': 0.92, 'factor': 0.8},
    'Nueva Zelanda':        {'ataque': 0.88, 'defensa': 0.96, 'factor': 0.7},
}

# =====================================================================
# GRUPOS (12 grupos de 4 equipos — formato 2026)
# =====================================================================
GRUPOS = {
    'A': ['México', 'Corea del Sur', 'República Checa', 'Sudáfrica'],
    'B': ['Estados Unidos', 'Paraguay', 'Australia', 'Turquía'],
    'C': ['Canadá', 'Bosnia y Herzegovina', 'Suiza', 'Catar'],
    'D': ['Brasil', 'Marruecos', 'Escocia', 'Haití'],
    'E': ['Alemania', 'Curazao', 'Costa de Marfil', 'Ecuador'],
    'F': ['Países Bajos', 'Suecia', 'Japón', 'Túnez'],
    'G': ['Bélgica', 'Irán', 'Nueva Zelanda', 'Egipto'],
    'H': ['España', 'Uruguay', 'Arabia Saudita', 'Cabo Verde'],
    'I': ['Francia', 'Noruega', 'Irak', 'Senegal'],
    'J': ['Argentina', 'Austria', 'Argelia', 'Jordania'],
    'K': ['Portugal', 'Colombia', 'Uzbekistán', 'R. D. del Congo'],
    'L': ['Inglaterra', 'Croacia', 'Ghana', 'Panamá'],
}

# Mapeo inglés → español para el fixture
EN_ES = {
    "South Korea": "Corea del Sur", "Czech Republic": "República Checa",
    "Mexico": "México", "South Africa": "Sudáfrica",
    "United States": "Estados Unidos", "Canada": "Canadá",
    "Bosnia and Herzegovina": "Bosnia y Herzegovina",
    "Australia": "Australia", "Turkey": "Turquía", "Türkiye": "Turquía", "Turkiye": "Turquía", "Qatar": "Catar",
    "Switzerland": "Suiza", "Haiti": "Haití", "Brazil": "Brasil",
    "Morocco": "Marruecos", "Sweden": "Suecia", "Tunisia": "Túnez",
    "Germany": "Alemania", "Curaçao": "Curazao",
    "Netherlands": "Países Bajos", "Japan": "Japón",
    "Ivory Coast": "Costa de Marfil", "Ecuador": "Ecuador",
    "Belgium": "Bélgica", "Egypt": "Egipto", "Iran": "Irán",
    "New Zealand": "Nueva Zelanda", "Spain": "España",
    "Cape Verde": "Cabo Verde", "Saudi Arabia": "Arabia Saudita",
    "Uruguay": "Uruguay", "Austria": "Austria", "Jordan": "Jordania",
    "Argentina": "Argentina", "Algeria": "Argelia", "Iraq": "Irak",
    "Norway": "Noruega", "France": "Francia", "Senegal": "Senegal",
    "Uzbekistan": "Uzbekistán", "Colombia": "Colombia",
    "Portugal": "Portugal", "DR Congo": "R. D. del Congo",
    "England": "Inglaterra", "Croatia": "Croacia",
    "Ghana": "Ghana", "Panama": "Panamá", "Scotland": "Escocia",
    "Paraguay": "Paraguay",
}

def t(nombre_en):
    return EN_ES.get(nombre_en, nombre_en)

# =====================================================================
# FIXTURE OFICIAL — 72 partidos fase de grupos
# =====================================================================
FIXTURE_2026 = [
    ("2026-06-11","South Korea","Czech Republic",2,1),
    ("2026-06-11","Mexico","South Africa",2,0),
    ("2026-06-12","United States","Paraguay",4,1),
    ("2026-06-12","Canada","Bosnia and Herzegovina",1,1),
    ("2026-06-13","Australia","Turkey",2,0),
    ("2026-06-13","Qatar","Switzerland",1,1),
    ("2026-06-13","Haiti","Scotland",0,1),
    ("2026-06-13","Brazil","Morocco",1,1),
    ("2026-06-14","Sweden","Tunisia",5,1),
    ("2026-06-14","Germany","Curaçao",7,1),
    ("2026-06-14","Netherlands","Japan",2,2),
    ("2026-06-14","Ivory Coast","Ecuador",1,0),
    ("2026-06-15","Belgium","Egypt",1,1),
    ("2026-06-15","Iran","New Zealand",2,2),
    ("2026-06-15","Spain","Cape Verde",0,0),
    ("2026-06-15","Saudi Arabia","Uruguay",1,1),
    ("2026-06-16","Austria","Jordan",3,1),
    ("2026-06-16","Argentina","Algeria",3,0),
    ("2026-06-16","Iraq","Norway",1,4),
    ("2026-06-16","France","Senegal",3,1),
    ("2026-06-17","Uzbekistan","Colombia",1,3),
    ("2026-06-17","Portugal","DR Congo",1,1),
    ("2026-06-17","England","Croatia",4,2),
    ("2026-06-17","Ghana","Panama",1,0),
    ("2026-06-18","Canada","Qatar",6,0),
    ("2026-06-18","Switzerland","Bosnia and Herzegovina",4,1),
    ("2026-06-18","Mexico","South Korea",0,1),
    ("2026-06-18","Czech Republic","South Africa",1,1),
    ("2026-06-19","United States","Australia",None,None),
    ("2026-06-19","Turkey","Paraguay",None,None),
    ("2026-06-19","Scotland","Morocco",None,None),
    ("2026-06-19","Brazil","Haiti",None,None),
    ("2026-06-20","Tunisia","Japan",None,None),
    ("2026-06-20","Netherlands","Sweden",None,None),
    ("2026-06-20","Ecuador","Curaçao",None,None),
    ("2026-06-20","Germany","Ivory Coast",None,None),
    ("2026-06-21","New Zealand","Egypt",None,None),
    ("2026-06-21","Belgium","Iran",None,None),
    ("2026-06-21","Spain","Saudi Arabia",None,None),
    ("2026-06-21","Uruguay","Cape Verde",None,None),
    ("2026-06-22","Jordan","Algeria",None,None),
    ("2026-06-22","Argentina","Austria",None,None),
    ("2026-06-22","France","Iraq",None,None),
    ("2026-06-22","Norway","Senegal",None,None),
    ("2026-06-23","Colombia","DR Congo",None,None),
    ("2026-06-23","Portugal","Uzbekistan",None,None),
    ("2026-06-23","Panama","Croatia",None,None),
    ("2026-06-23","England","Ghana",None,None),
    ("2026-06-24","Switzerland","Canada",None,None),
    ("2026-06-24","Bosnia and Herzegovina","Qatar",None,None),
    ("2026-06-24","Czech Republic","Mexico",None,None),
    ("2026-06-24","South Africa","South Korea",None,None),
    ("2026-06-24","Morocco","Haiti",None,None),
    ("2026-06-24","Scotland","Brazil",None,None),
    ("2026-06-25","Paraguay","Australia",None,None),
    ("2026-06-25","Turkey","United States",None,None),
    ("2026-06-25","Japan","Sweden",None,None),
    ("2026-06-25","Tunisia","Netherlands",None,None),
    ("2026-06-25","Curaçao","Ivory Coast",None,None),
    ("2026-06-25","Ecuador","Germany",None,None),
    ("2026-06-26","New Zealand","Belgium",None,None),
    ("2026-06-26","Egypt","Iran",None,None),
    ("2026-06-26","Uruguay","Spain",None,None),
    ("2026-06-26","Cape Verde","Saudi Arabia",None,None),
    ("2026-06-26","Senegal","Iraq",None,None),
    ("2026-06-26","Norway","France",None,None),
    ("2026-06-27","Jordan","Argentina",None,None),
    ("2026-06-27","Algeria","Austria",None,None),
    ("2026-06-27","DR Congo","Uzbekistan",None,None),
    ("2026-06-27","Colombia","Portugal",None,None),
    ("2026-06-27","Croatia","Ghana",None,None),
    ("2026-06-27","Panama","England",None,None),
]

PROMEDIO_GOL = 1.25
CASAS = ["Betano","Jugabet","bet365","1xbet","Tonybet","Coolbet","Betsson","Tikitaka"]
MESES = ["","enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","octubre","noviembre","diciembre"]

# =====================================================================
# BRACKET OFICIAL R32 — Copa Mundial 2026
# pos: 'W'=primero, 'R'=segundo, 'T'=mejor tercero
# terceros_grupos: grupos válidos para ese slot de tercero
# =====================================================================
BRACKET_R32 = [
    # (match, pos1, grupo1, pos2, grupo2_o_None, terceros_grupos_o_None)
    (73, 'R','A',  'R','B',  None),
    (74, 'W','E',  'T', None, ['A','B','C','D','F']),
    (75, 'W','F',  'R','C',  None),
    (76, 'W','C',  'R','F',  None),
    (77, 'W','I',  'T', None, ['C','D','F','G','H']),
    (78, 'R','E',  'R','I',  None),
    (79, 'W','A',  'T', None, ['C','E','F','H','I']),
    (80, 'W','L',  'T', None, ['E','H','I','J','K']),
    (81, 'W','D',  'T', None, ['B','E','F','I','J']),
    (82, 'W','G',  'T', None, ['A','E','H','I','J']),
    (83, 'R','K',  'R','L',  None),
    (84, 'W','H',  'R','J',  None),
    (85, 'W','B',  'T', None, ['E','F','G','I','J']),
    (86, 'W','J',  'R','H',  None),
    (87, 'W','K',  'T', None, ['D','E','I','J','L']),
    (88, 'R','D',  'R','G',  None),
]
# Bracket R16: ganadores de pares de partidos R32 (índices 0-15)
BRACKET_R16 = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)]
BRACKET_QF  = [(0,1),(2,3),(4,5),(6,7)]
BRACKET_SF  = [(0,1),(2,3)]
BRACKET_F   = [(0,1)]

# =====================================================================
# SESSION STATE
# =====================================================================
def init_state():
    if 'datos' not in st.session_state:
        datos = {k: {**v, 'g_favor':0,'g_contra':0,'PJ':0} for k,v in DATOS_INICIAL.items()}
        for _,local_en,visita_en,gl,gv in FIXTURE_2026:
            if gl is None: continue
            loc, vis = t(local_en), t(visita_en)
            if loc in datos and vis in datos:
                datos[loc]['g_favor'] += gl; datos[loc]['g_contra'] += gv; datos[loc]['PJ'] += 1
                datos[vis]['g_favor'] += gv; datos[vis]['g_contra'] += gl; datos[vis]['PJ'] += 1
        st.session_state.datos = datos
    if 'resultados_extra' not in st.session_state:
        st.session_state.resultados_extra = []
    if 'odds_api_key' not in st.session_state:
        st.session_state.odds_api_key = ""
    if 'espn_fetched' not in st.session_state:
        st.session_state.espn_fetched = False

init_state()

# =====================================================================
# HELPERS — modelo
# =====================================================================
def limpiar(texto):
    s = texto.strip().lower().replace(".","").replace(" ","")
    return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')

ATAJOS = {
    "arabia":"Arabia Saudita","arabiasaudi":"Arabia Saudita",
    "chequia":"República Checa","republicacheca":"República Checa",
    "congo":"R. D. del Congo","rdcongo":"R. D. del Congo",
    "qatar":"Catar","holanda":"Países Bajos","paisesbajos":"Países Bajos",
    "usa":"Estados Unidos","eeuu":"Estados Unidos","estadosunidos":"Estados Unidos",
    "bosnia":"Bosnia y Herzegovina","corea":"Corea del Sur","coreadelsur":"Corea del Sur",
    "costademarfil":"Costa de Marfil","nuevazelanda":"Nueva Zelanda",
}

def nombre_oficial(inp):
    k = limpiar(inp)
    if k in ATAJOS: return ATAJOS[k]
    for n in st.session_state.datos:
        if limpiar(n)==k: return n
    return None

# Mapa inverso equipo → grupo
EQUIPO_GRUPO = {eq: g for g, eqs in GRUPOS.items() for eq in eqs}

# Fechas de 3era jornada de grupos
FECHAS_J3 = {
    "2026-06-24","2026-06-25","2026-06-26","2026-06-27"
}

def tabla_grupo(letra):
    equipos = GRUPOS[letra]
    tab = {e:{'PJ':0,'PG':0,'PE':0,'PP':0,'GF':0,'GC':0,'Pts':0} for e in equipos}
    fijos = {}
    for _,local_en,visita_en,gl,gv in FIXTURE_2026:
        if gl is None: continue
        loc,vis = t(local_en),t(visita_en)
        if loc in tab and vis in tab: fijos[(loc,vis)]=(gl,gv)
    for r in st.session_state.resultados_extra:
        if r['local'] in tab and r['visita'] in tab:
            fijos[(r['local'],r['visita'])]=(r['gl'],r['gv'])
    for (loc,vis),(gl,gv) in fijos.items():
        tab[loc]['PJ']+=1; tab[loc]['GF']+=gl; tab[loc]['GC']+=gv
        tab[vis]['PJ']+=1; tab[vis]['GF']+=gv; tab[vis]['GC']+=gl
        if gl>gv:   tab[loc]['PG']+=1; tab[loc]['Pts']+=3; tab[vis]['PP']+=1
        elif gl==gv: tab[loc]['PE']+=1; tab[loc]['Pts']+=1; tab[vis]['PE']+=1; tab[vis]['Pts']+=1
        else:        tab[vis]['PG']+=1; tab[vis]['Pts']+=3; tab[loc]['PP']+=1
    orden = sorted(equipos, key=lambda e:(tab[e]['Pts'],tab[e]['GF']-tab[e]['GC'],tab[e]['GF']), reverse=True)
    return orden, tab

def factor_urgencia(equipo, grupo):
    """
    Retorna (f_atq, f_def) según necesidad de clasificación en 3era fecha.
    Lógica:
      - Clasificado seguro      → ligero repliegue (0.96, 1.04)
      - Necesita solo empatar   → equilibrado ofensivo (1.06, 1.04)
      - Necesita ganar          → máxima urgencia (1.15, 0.91)
      - Eliminado sin opciones  → baja motivación (0.92, 0.96)
    """
    try:
        orden, tab = tabla_grupo(grupo)
        pts    = tab[equipo]['Pts']
        pos    = orden.index(equipo) + 1
        pts_2do = tab[orden[1]]['Pts']
        pts_3ro = tab[orden[2]]['Pts']
        pts_max = pts + 3  # máximo posible ganando

        # Clasificado matemáticamente (no puede ser alcanzado por 3ro)
        if pos <= 2 and pts > pts_3ro + 3:
            return 0.96, 1.04

        # Eliminado (ni ganando llega al 2do y ni aspirar a mejor 3ro con ≥4 pts)
        if pts_max < pts_2do and pts_max < 4:
            return 0.92, 0.96

        # Empate alcanza para igualar o superar al 2do
        if pts + 1 >= pts_2do:
            return 1.06, 1.04

        # Necesita ganar sí o sí
        return 1.15, 0.91

    except Exception:
        return 1.0, 1.0

def lambdas(local, visita, fecha=None):
    d = st.session_state.datos
    # Factor de rendimiento simétrico — misma fórmula para ambos (cancha neutral)
    fl = 1+(d[local]['g_favor'] *0.12) if d[local]['PJ'] >0 else 1.0
    fv = 1+(d[visita]['g_favor']*0.12) if d[visita]['PJ']>0 else 1.0
    ll = d[local]['ataque']*d[visita]['defensa']*PROMEDIO_GOL*d[local]['factor']*fl
    lv = d[visita]['ataque']*d[local]['defensa']*PROMEDIO_GOL*d[visita]['factor']*fv

    # Aplicar presión de clasificación en 3era fecha
    if fecha in FECHAS_J3:
        grp_l = EQUIPO_GRUPO.get(local)
        grp_v = EQUIPO_GRUPO.get(visita)
        if grp_l:
            fa_l, fd_l = factor_urgencia(local, grp_l)
            ll *= fa_l
            lv *= fd_l  # defensa rival se ajusta por urgencia local
        if grp_v:
            fa_v, fd_v = factor_urgencia(visita, grp_v)
            lv *= fa_v
            ll *= fd_v

    return ll, lv

def poisson_probs(ll, lv):
    mat = np.zeros((7,7))
    pl=pe=pv=0
    for i in range(7):
        for j in range(7):
            p = poisson.pmf(i,ll)*poisson.pmf(j,lv)
            mat[i,j]=p
            if i>j: pl+=p
            elif i==j: pe+=p
            else: pv+=p
    gi,gj = np.unravel_index(np.argmax(mat),mat.shape)
    return pl,pe,pv,int(gi),int(gj),mat

def top_marcadores(mat, n=3):
    """Devuelve los n marcadores más probables (gl, gv, %) de la matriz Poisson."""
    flat = sorted([(mat[i,j]*100, i, j) for i in range(7) for j in range(7)], reverse=True)
    return [(i, j, p) for p, i, j in flat[:n]]

def cuotas_simuladas(local, visita):
    d = st.session_state.datos
    atk,dfs = d[local]['ataque'],d[visita]['defensa']
    prob_l = atk/(atk+dfs)
    bl = max(1.20,min(8.00,1/prob_l))
    bv = max(1.20,min(8.00,1/(1-prob_l)))
    np.random.seed(int(len(local)+len(visita)))
    return {c:(round(bl+np.random.uniform(-0.10,0.10),2),
               round(3.30+np.random.uniform(-0.05,0.05),2),
               round(bv+np.random.uniform(-0.12,0.12),2)) for c in CASAS}

def cuotas_reales(local, visita):
    """Fetch real odds from The Odds API if key is configured."""
    key = st.session_state.odds_api_key.strip()
    if not key:
        return None, "Sin API key"
    try:
        resp = requests.get(
            "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds/",
            params={"apiKey":key,"regions":"eu","markets":"h2h","oddsFormat":"decimal"},
            timeout=8
        )
        if resp.status_code != 200:
            return None, f"Error API: {resp.status_code}"
        eventos = resp.json()
        # Comparar sin tildes, en cualquier idioma
        loc_k = limpiar(local); vis_k = limpiar(visita)
        # También mapeo ES→EN via ESPN_MAP inverso
        ES_EN = {v: limpiar(k) for k, v in ESPN_MAP.items()}
        loc_k2 = ES_EN.get(local, loc_k)
        vis_k2 = ES_EN.get(visita, vis_k)
        for ev in eventos:
            home_k = limpiar(ev.get("home_team",""))
            away_k = limpiar(ev.get("away_team",""))
            if not ((home_k in (loc_k, loc_k2) and away_k in (vis_k, vis_k2)) or
                    (home_k in (vis_k, vis_k2) and away_k in (loc_k, loc_k2))):
                continue
            cuotas = {}
            for book in ev.get("bookmakers",[]):
                nombre_book = book["title"]
                for mkt in book.get("markets",[]):
                    if mkt["key"]=="h2h":
                        odds = {o["name"]:o["price"] for o in mkt["outcomes"]}
                        cL = odds.get(ev["home_team"],0)
                        cV = odds.get(ev["away_team"],0)
                        cE = odds.get("Draw",3.30)
                        if cL and cV:
                            cuotas[nombre_book] = (cL,cE,cV)
            return cuotas if cuotas else None, "Sin cuotas disponibles"
        return None, "Partido no encontrado en la API"
    except Exception as e:
        return None, str(e)

def calcular_prediccion(local, visita, usar_api=False, fecha=None):
    ll, lv = lambdas(local, visita, fecha=fecha)
    pl, pe, pv, gi, gj, mat = poisson_probs(ll, lv)

    if usar_api:
        cuotas, msg = cuotas_reales(local, visita)
        fuente = "reales (The Odds API)" if cuotas else f"simuladas ({msg})"
    else:
        cuotas = None
        fuente = "simuladas"

    if cuotas is None:
        cuotas = cuotas_simuladas(local, visita)

    probs_l,probs_e,probs_v=[],[],[]
    for cL,cE,cV in cuotas.values():
        m=1/cL+1/cE+1/cV
        probs_l.append((1/cL)/m); probs_e.append((1/cE)/m); probs_v.append((1/cV)/m)
    ml,me,mv = np.mean(probs_l),np.mean(probs_e),np.mean(probs_v)

    return {
        'local':local,'visita':visita,
        'prob_l': pl*0.6+ml*0.4, 'prob_e': pe*0.6+me*0.4, 'prob_v': pv*0.6+mv*0.4,
        'goles_l':gi,'goles_v':gj,'cuotas':cuotas,'fuente_cuotas':fuente,
        'lam_l':ll,'lam_v':lv,'matriz':mat,
    }

# =====================================================================
# WIKIPEDIA SCRAPER
# =====================================================================
# =====================================================================
# ESPN API — nombres en inglés → nombres internos
# =====================================================================
ESPN_MAP = {
    "Czechia":"República Checa","Czech Republic":"República Checa",
    "South Africa":"Sudáfrica","Mexico":"México","South Korea":"Corea del Sur",
    "United States":"Estados Unidos","Canada":"Canadá",
    "Bosnia-Herzegovina":"Bosnia y Herzegovina","Bosnia and Herzegovina":"Bosnia y Herzegovina",
    "Australia":"Australia","Turkey":"Turquía","Türkiye":"Turquía","Turkiye":"Turquía","Qatar":"Catar",
    "Switzerland":"Suiza","Haiti":"Haití","Brazil":"Brasil",
    "Morocco":"Marruecos","Sweden":"Suecia","Tunisia":"Túnez",
    "Germany":"Alemania","Curacao":"Curazao","Curaçao":"Curazao",
    "Netherlands":"Países Bajos","Japan":"Japón",
    "Ivory Coast":"Costa de Marfil","Cote d'Ivoire":"Costa de Marfil",
    "Ecuador":"Ecuador","Belgium":"Bélgica","Egypt":"Egipto",
    "Iran":"Irán","New Zealand":"Nueva Zelanda","Spain":"España",
    "Cape Verde":"Cabo Verde","Saudi Arabia":"Arabia Saudita",
    "Uruguay":"Uruguay","Austria":"Austria","Jordan":"Jordania",
    "Argentina":"Argentina","Algeria":"Argelia","Iraq":"Irak",
    "Norway":"Noruega","France":"Francia","Senegal":"Senegal",
    "Uzbekistan":"Uzbekistán","Colombia":"Colombia","Portugal":"Portugal",
    "DR Congo":"R. D. del Congo","Congo DR":"R. D. del Congo",
    "England":"Inglaterra","Croatia":"Croacia","Ghana":"Ghana",
    "Panama":"Panamá","Scotland":"Escocia","Paraguay":"Paraguay",
    "Serbia":"Serbia","Kosovo":"Kosovo",
}

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"

def espn_nombre(nombre_espn):
    if nombre_espn in ESPN_MAP: return ESPN_MAP[nombre_espn]
    # Búsqueda tolerante sin tildes
    def _norm(s):
        s = s.strip().lower().replace(".","").replace(" ","")
        return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')
    k = _norm(nombre_espn)
    for en,es in ESPN_MAP.items():
        if _norm(en)==k: return es
    return None

def fetch_espn_fecha(fecha_str):
    """Fetch completed matches for a given date YYYYMMDD from ESPN API."""
    try:
        resp = requests.get(ESPN_URL, params={"dates": fecha_str}, timeout=8)
        if resp.status_code != 200: return []
        data = resp.json()
        resultados = []
        for evento in data.get("events", []):
            for comp in evento.get("competitions", []):
                estado = comp.get("status",{}).get("type",{}).get("name","")
                if estado not in ("STATUS_FINAL","STATUS_FULL_TIME"): continue
                competidores = comp.get("competitors",[])
                if len(competidores) < 2: continue
                home = next((c for c in competidores if c.get("homeAway")=="home"), None)
                away = next((c for c in competidores if c.get("homeAway")=="away"), None)
                if not home or not away: continue
                loc_en = home.get("team",{}).get("displayName","")
                vis_en = away.get("team",{}).get("displayName","")
                loc = espn_nombre(loc_en)
                vis = espn_nombre(vis_en)
                try:
                    gl = int(home.get("score",0))
                    gv = int(away.get("score",0))
                except: continue
                if loc and vis:
                    resultados.append((loc, vis, gl, gv))
        return resultados
    except: return []

@st.cache_data(ttl=60, show_spinner=False)
def fetch_espn_estado(fecha_str):
    """Devuelve dict {(local, visita): {'estado', 'gl', 'gv', 'minuto', 'hora_utc', 'orden'}}"""
    try:
        resp = requests.get(ESPN_URL, params={"dates": fecha_str}, timeout=8)
        if resp.status_code != 200: return {}
        data = resp.json()
        estados = {}
        for idx, evento in enumerate(data.get("events", [])):
            hora_utc = evento.get("date","")  # ISO 8601 UTC
            for comp in evento.get("competitions", []):
                estado_raw = comp.get("status",{}).get("type",{}).get("name","")
                clock = comp.get("status",{}).get("displayClock","")
                competidores = comp.get("competitors",[])
                if len(competidores) < 2: continue
                home = next((c for c in competidores if c.get("homeAway")=="home"), None)
                away = next((c for c in competidores if c.get("homeAway")=="away"), None)
                if not home or not away: continue
                loc = espn_nombre(home.get("team",{}).get("displayName",""))
                vis = espn_nombre(away.get("team",{}).get("displayName",""))
                if not loc or not vis: continue
                try: gl = int(home.get("score",0)); gv = int(away.get("score",0))
                except: gl = gv = 0
                estado = ("final"    if estado_raw in ("STATUS_FINAL","STATUS_FULL_TIME")
                     else "en_juego" if estado_raw == "STATUS_IN_PROGRESS"
                     else "por_jugar")
                estados[(loc,vis)] = {
                    'estado': estado, 'gl': gl, 'gv': gv,
                    'minuto': clock, 'hora_utc': hora_utc, 'orden': idx
                }
        return estados
    except: return {}

def auto_fetch_espn():
    """Fetch all results since June 18 (hardcoded period ends June 17)."""
    from datetime import date, timedelta
    inicio = date(2026, 6, 18)
    hoy = date.today()
    nuevos = []
    ya_en_fixture = {(t(le), t(ve)) for _,le,ve,gl,gv in FIXTURE_2026 if gl is not None}
    ya_registrados = {(r['local'],r['visita']) for r in st.session_state.resultados_extra}

    d = inicio
    while d <= hoy:
        fecha_str = d.strftime("%Y%m%d")
        for loc, vis, gl, gv in fetch_espn_fecha(fecha_str):
            par = (loc, vis)
            if par not in ya_en_fixture and par not in ya_registrados:
                nuevos.append({'local':loc,'visita':vis,'gl':gl,'gv':gv})
                ya_registrados.add(par)
        d += timedelta(days=1)
    return nuevos

# Auto-fetch ESPN al iniciar (una sola vez por sesión)
if not st.session_state.espn_fetched:
    _nuevos = auto_fetch_espn()
    if _nuevos:
        _datos = st.session_state.datos
        for r in _nuevos:
            loc, vis, gl, gv = r['local'], r['visita'], r['gl'], r['gv']
            if loc in _datos and vis in _datos:
                _datos[loc]['g_favor']+=gl; _datos[loc]['g_contra']+=gv; _datos[loc]['PJ']+=1
                _datos[vis]['g_favor']+=gv; _datos[vis]['g_contra']+=gl; _datos[vis]['PJ']+=1
                st.session_state.resultados_extra.append(r)
    st.session_state.espn_fetched = True

WIKI_URL = "https://es.wikipedia.org/wiki/Anexo:Calendario_de_la_Copa_Mundial_de_F%C3%BAtbol_de_2026"

# Map Wikipedia team names (Spanish) → our internal names
WIKI_MAP = {
    "Corea del Sur":"Corea del Sur","República Checa":"República Checa",
    "México":"México","Sudáfrica":"Sudáfrica","Estados Unidos":"Estados Unidos",
    "Canadá":"Canadá","Bosnia y Herzegovina":"Bosnia y Herzegovina",
    "Australia":"Australia","Turquía":"Turquía","Catar":"Catar",
    "Suiza":"Suiza","Haití":"Haití","Brasil":"Brasil","Marruecos":"Marruecos",
    "Suecia":"Suecia","Túnez":"Túnez","Alemania":"Alemania","Curazao":"Curazao",
    "Países Bajos":"Países Bajos","Japón":"Japón","Costa de Marfil":"Costa de Marfil",
    "Ecuador":"Ecuador","Bélgica":"Bélgica","Egipto":"Egipto","Irán":"Irán",
    "Nueva Zelanda":"Nueva Zelanda","España":"España","Cabo Verde":"Cabo Verde",
    "Arabia Saudita":"Arabia Saudita","Uruguay":"Uruguay","Austria":"Austria",
    "Jordania":"Jordania","Argentina":"Argentina","Argelia":"Argelia",
    "Irak":"Irak","Noruega":"Noruega","Francia":"Francia","Senegal":"Senegal",
    "Uzbekistán":"Uzbekistán","Colombia":"Colombia","Portugal":"Portugal",
    "República Democrática del Congo":"R. D. del Congo","R. D. del Congo":"R. D. del Congo",
    "Inglaterra":"Inglaterra","Croacia":"Croacia","Ghana":"Ghana","Panamá":"Panamá",
    "Escocia":"Escocia","Paraguay":"Paraguay",
}

def normalizar_wiki(nombre):
    nombre = nombre.strip()
    if nombre in WIKI_MAP: return WIKI_MAP[nombre]
    n = limpiar(nombre)
    for k,v in WIKI_MAP.items():
        if limpiar(k)==n: return v
    return None

def fetch_wiki_results():
    """Scrape Wikipedia for completed match results."""
    try:
        resp = requests.get(WIKI_URL, headers={"User-Agent":"Mozilla/5.0"}, timeout=12)
        soup = BeautifulSoup(resp.text, "html.parser")

        resultados = {}
        # Wikipedia football tables use specific span classes for team names
        # and score format "X–Y" with en-dash
        # Pattern: look for table cells with score pattern near team names
        for table in soup.find_all("table", class_=re.compile(r"wikitable|vevent")):
            text = table.get_text(" ", strip=True)
            # Find score patterns: digits, en-dash (–), digits
            scores = re.findall(r'(\d)\s*[–-]\s*(\d)', text)
            if scores:
                # Try to find team names in the table rows
                rows = table.find_all("tr")
                for row in rows:
                    cells = row.find_all(["td","th"])
                    row_text = " ".join(c.get_text(strip=True) for c in cells)
                    m = re.search(r'(\d)\s*[–-]\s*(\d)', row_text)
                    if m:
                        gl,gv = int(m.group(1)),int(m.group(2))
                        # Look for team names in adjacent cells
                        parts = re.split(r'\d\s*[–-]\s*\d', row_text, maxsplit=1)
                        if len(parts)==2:
                            left = parts[0].strip().split()
                            right = parts[1].strip().split()
                            local_cand = " ".join(left[-3:]) if left else ""
                            visita_cand = " ".join(right[:3]) if right else ""
                            loc = normalizar_wiki(local_cand)
                            vis = normalizar_wiki(visita_cand)
                            if loc and vis and loc!=vis:
                                resultados[(loc,vis)] = (gl,gv)

        # Fallback: try to match against known fixture to find new results
        nuevos = []
        for fecha,local_en,visita_en,gl_known,gv_known in FIXTURE_2026:
            if gl_known is not None: continue
            loc, vis = t(local_en), t(visita_en)
            if (loc,vis) in resultados:
                gl,gv = resultados[(loc,vis)]
                nuevos.append((fecha,loc,vis,gl,gv))
            elif (vis,loc) in resultados:
                gv,gl = resultados[(vis,loc)]
                nuevos.append((fecha,loc,vis,gl,gv))

        return nuevos, None
    except Exception as e:
        return [], str(e)

# =====================================================================
# MONTE CARLO
# =====================================================================
def sim_partido_mc(local, visita, datos):
    """Quick Poisson simulation for Monte Carlo (no session state overhead)."""
    d = datos
    ll = d[local]['ataque']*d[visita]['defensa']*PROMEDIO_GOL*d[local]['factor']
    lv = d[visita]['ataque']*d[local]['defensa']*PROMEDIO_GOL*d[visita]['factor']
    return int(np.random.poisson(ll)), int(np.random.poisson(lv))

def simular_grupo_mc(equipos, resultados_fijos, datos):
    pts = {e:0 for e in equipos}
    gf  = {e:0 for e in equipos}
    gc  = {e:0 for e in equipos}
    for i,e1 in enumerate(equipos):
        for e2 in equipos[i+1:]:
            if (e1,e2) in resultados_fijos:
                g1,g2 = resultados_fijos[(e1,e2)]
            elif (e2,e1) in resultados_fijos:
                g2,g1 = resultados_fijos[(e2,e1)]
            else:
                g1,g2 = sim_partido_mc(e1,e2,datos)
            gf[e1]+=g1; gc[e1]+=g2; gf[e2]+=g2; gc[e2]+=g1
            if g1>g2: pts[e1]+=3
            elif g1==g2: pts[e1]+=1; pts[e2]+=1
            else: pts[e2]+=3
    orden = sorted(equipos, key=lambda e:(pts[e],gf[e]-gc[e],gf[e]), reverse=True)
    return orden, pts, gf, gc

def sim_partido_ko(e1, e2, datos):
    """KO: en empate, desempate aleatorio basado en probs."""
    g1,g2 = sim_partido_mc(e1,e2,datos)
    if g1!=g2: return e1 if g1>g2 else e2
    # Penales simplificados: 55/45 favor al mejor ataque
    atk1 = datos[e1]['ataque']; atk2 = datos[e2]['ataque']
    return e1 if np.random.random() < atk1/(atk1+atk2) else e2

def _build_bracket_r32(clasi_por_grupo, terceros_clasificados):
    """Arma los 16 partidos del R32 usando el bracket oficial FIFA 2026."""
    W = {g: clasi_por_grupo[g][0] for g in GRUPOS}
    R = {g: clasi_por_grupo[g][1] for g in GRUPOS}
    usados = set()

    def get_tercero(grupos_validos):
        for eq, grupo, *_ in terceros_clasificados:
            if grupo in grupos_validos and eq not in usados:
                usados.add(eq); return eq
        for eq, grupo, *_ in terceros_clasificados:   # fallback
            if eq not in usados:
                usados.add(eq); return eq
        return None

    partidos = []
    for _, pos1, g1, pos2, g2, tg in BRACKET_R32:
        t1 = W[g1] if pos1=='W' else R[g1]
        t2 = (get_tercero(tg) if pos2=='T'
              else (W[g2] if pos2=='W' else R[g2]))
        partidos.append((t1, t2 or t1))   # si no hay tercero, bye
    return partidos   # lista de 16 pares

def _sim_ronda_ko(partidos, datos, counts, nombre_ronda):
    ganadores = []
    for t1, t2 in partidos:
        g = sim_partido_ko(t1, t2, datos)
        counts[g][nombre_ronda] += 1
        ganadores.append(g)
    return ganadores

@st.cache_data(show_spinner=False, ttl=300)
def monte_carlo(n_sims, _datos_hash):
    datos = st.session_state.datos

    fijos = {}
    for _,local_en,visita_en,gl,gv in FIXTURE_2026:
        if gl is None: continue
        fijos[(t(local_en),t(visita_en))] = (gl,gv)
    for r in st.session_state.resultados_extra:
        fijos[(r['local'],r['visita'])] = (r['gl'],r['gv'])

    rondas = ['R32','R16','QF','SF','Final','Campeon']
    counts = {eq:{r:0 for r in ['Grupos']+rondas} for eq in datos}

    for _ in range(n_sims):
        clasi_por_grupo = {}
        terceros = []

        for letra, equipos in GRUPOS.items():
            orden, pts, gf, gc = simular_grupo_mc(equipos, fijos, datos)
            clasi_por_grupo[letra] = orden[:2]
            for e in orden[:2]: counts[e]['Grupos'] += 1
            t3 = orden[2]
            terceros.append((t3, letra, pts[t3], gf[t3]-gc[t3], gf[t3]))

        # 8 mejores terceros (ordenados por Pts, DG, GF)
        terceros.sort(key=lambda x:(x[2],x[3],x[4]), reverse=True)
        for eq,*_ in terceros[:8]: counts[eq]['Grupos'] += 1

        # Bracket real
        r32 = _build_bracket_r32(clasi_por_grupo, terceros[:8])
        w32 = _sim_ronda_ko(r32, datos, counts, 'R32')

        r16 = [(w32[i],w32[j]) for i,j in BRACKET_R16]
        w16 = _sim_ronda_ko(r16, datos, counts, 'R16')

        qf = [(w16[i],w16[j]) for i,j in BRACKET_QF]
        wqf = _sim_ronda_ko(qf, datos, counts, 'QF')

        sf = [(wqf[i],wqf[j]) for i,j in BRACKET_SF]
        wsf = _sim_ronda_ko(sf, datos, counts, 'SF')

        final = [(wsf[0], wsf[1])]
        wf = _sim_ronda_ko(final, datos, counts, 'Final')

        counts[wf[0]]['Campeon'] += 1

    return {eq:{k:v/n_sims*100 for k,v in rnd.items()} for eq,rnd in counts.items()}

# =====================================================================
# CSS
# =====================================================================
st.markdown("""
<style>
.block-container{padding:1rem 1.5rem;max-width:960px;margin:auto}
.card{background:#1e1e2e;border-radius:14px;padding:1.1rem 1.4rem;
      margin-bottom:.8rem;border:1px solid #313244}
.tag{display:inline-block;padding:2px 9px;border-radius:20px;
     font-size:.7rem;font-weight:700;letter-spacing:.05em}
.tag-l{background:#1a472a;color:#69db7c}
.tag-e{background:#3b3b00;color:#ffd43b}
.tag-v{background:#3b0000;color:#ff6b6b}
.bar-wrap{height:10px;border-radius:5px;background:#313244;overflow:hidden;margin:3px 0 10px}
.bar{height:100%;border-radius:5px}
.marcador{font-size:2.6rem;font-weight:800;text-align:center;color:#cba6f7;margin:.4rem 0}
.fixture-row{display:flex;align-items:center;justify-content:space-between;
  padding:7px 12px;border-radius:10px;margin-bottom:3px;
  background:#1e1e2e;border:1px solid #313244;font-size:.9rem}
.fixture-score{font-weight:700;font-size:1.05rem;color:#cba6f7;min-width:64px;text-align:center}
.fixture-pred{font-weight:600;font-size:.82rem;color:#89b4fa;min-width:64px;text-align:center}
.fixture-team{flex:1}.fixture-team.right{text-align:right}
.date-hdr{color:#6c7086;font-size:.78rem;margin:10px 0 3px;text-transform:uppercase;letter-spacing:.08em}
.today-badge{display:inline-block;background:#fab005;color:#1e1e2e;
  padding:1px 8px;border-radius:20px;font-size:.68rem;font-weight:700;margin-left:6px}
h1{color:#cba6f7!important}h2,h3{color:#89b4fa!important}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# HEADER
# =====================================================================
st.markdown("# ⚽ Mundial 2026 · Predictor")
st.markdown("---")

tab_fix, tab_grupos, tab_mc, tab_pred, tab_reg, tab_stats = st.tabs([
    "📅 Fixture","🏆 Grupos","🎲 Monte Carlo","🔮 Predictor","📋 Registrar","📊 Stats"
])

# ─────────────────────────────────────────────────────────────────────
# TAB 1: FIXTURE
# ─────────────────────────────────────────────────────────────────────
with tab_fix:
    col_h1, col_h2 = st.columns([3,1])
    with col_h1:
        st.markdown("### Fase de grupos — 72 partidos")
        st.caption("Resultados reales cargados automáticamente desde ESPN.")
    with col_h2:
        if st.button("🔄 Actualizar resultados", use_container_width=True):
            st.session_state.espn_fetched = False
            st.session_state.resultados_extra = []
            if 'datos' in st.session_state: del st.session_state['datos']
            if 'mc_result' in st.session_state: del st.session_state['mc_result']
            st.rerun()

    hoy = date.today().isoformat()
    extra_lkp = {(r['local'],r['visita']):(r['gl'],r['gv']) for r in st.session_state.resultados_extra}

# Caché de estados ESPN por fecha (también busca en día siguiente UTC para partidos nocturnos)
    _espn_cache = {}
    def get_espn_fecha(f):
        if f not in _espn_cache:
            from datetime import datetime, timedelta as td
            d = datetime.strptime(f, "%Y-%m-%d")
            dia_actual  = dict(fetch_espn_estado(d.strftime("%Y%m%d")))
            dia_sig     = fetch_espn_estado((d + td(days=1)).strftime("%Y%m%d"))
            # Fusionar: solo incluir del día siguiente los partidos que son antes de 06:00 UTC
            # (= antes de 02:00 Chile, misma jornada nocturna)
            for k, v in dia_sig.items():
                hora_utc = v.get('hora_utc','')
                if hora_utc:
                    try:
                        hora = datetime.fromisoformat(hora_utc.replace("Z","+00:00"))
                        if hora.hour < 6:  # antes de 06:00 UTC = antes de 02:00 Chile
                            dia_actual.setdefault(k, v)
                    except: pass
            _espn_cache[f] = dia_actual
        return _espn_cache[f]

    # Agrupar fixture por fecha, ordenado por hora ESPN si disponible
    from itertools import groupby
    fixture_por_fecha = {}
    for item in FIXTURE_2026:
        fixture_por_fecha.setdefault(item[0], []).append(item)

    UTC_OFFSET = -4  # Chile en junio (UTC-4)

    for fecha in sorted(fixture_por_fecha.keys()):
        partidos_dia = fixture_por_fecha[fecha]
        estados_dia = get_espn_fecha(fecha)

        # Ordenar por hora ESPN si hay datos
        def sort_key(item):
            loc, vis = t(item[1]), t(item[2])
            info = estados_dia.get((loc,vis))
            return info['orden'] if info else 999

        partidos_dia = sorted(partidos_dia, key=sort_key)

        dt = pd.to_datetime(fecha)
        badge = '<span class="today-badge">HOY</span>' if fecha==hoy else ""
        st.markdown(f'<div class="date-hdr">{dt.day} de {MESES[dt.month]}{badge}</div>',
                    unsafe_allow_html=True)

        for _fecha,local_en,visita_en,gl,gv in partidos_dia:
            local,visita = t(local_en),t(visita_en)
            if (local,visita) in extra_lkp: gl,gv = extra_lkp[(local,visita)]
            espn_info = estados_dia.get((local,visita))

            # Hora local Chile (UTC-4)
            hora_str = ""
            if espn_info and espn_info.get('hora_utc'):
                try:
                    from datetime import datetime, timedelta as td
                    utc_dt = datetime.fromisoformat(espn_info['hora_utc'].replace("Z","+00:00"))
                    local_dt = utc_dt + td(hours=UTC_OFFSET)
                    hora_str = local_dt.strftime("%H:%M")
                except: pass

            # Badge de medianoche si arranca entre 23:00 y 05:59 hora Chile
            luna = ""
            if hora_str:
                try:
                    h = int(hora_str.split(":")[0])
                    if h >= 23 or h < 6:
                        luna = ' <span title="Partido de madrugada" style="font-size:.75rem">🌙</span>'
                except: pass
            hora_badge = (f'<span style="color:#6c7086;font-size:.72rem;min-width:42px;display:inline-block">{hora_str}{luna}</span>'
                          if hora_str else '<span style="min-width:42px;display:inline-block"></span>')

            if gl is not None:
                st.markdown(f"""<div class="fixture-row">
                  {hora_badge}
                  <span class="fixture-team">{local}</span>
                  <span class="fixture-score">{gl} – {gv}</span>
                  <span class="fixture-team right">{visita}</span>
                  <span style="color:#6c7086;font-size:.72rem;margin-left:8px">✅ Final</span>
                  </div>""", unsafe_allow_html=True)

            elif espn_info and espn_info['estado']=='final':
                gl_e, gv_e = espn_info['gl'], espn_info['gv']
                st.markdown(f"""<div class="fixture-row">
                  {hora_badge}
                  <span class="fixture-team">{local}</span>
                  <span class="fixture-score">{gl_e} – {gv_e}</span>
                  <span class="fixture-team right">{visita}</span>
                  <span style="color:#6c7086;font-size:.72rem;margin-left:8px">✅ Final</span>
                  </div>""", unsafe_allow_html=True)

            elif espn_info and espn_info['estado']=='en_juego':
                gl_e, gv_e = espn_info['gl'], espn_info['gv']
                min_txt = f" {espn_info['minuto']}" if espn_info['minuto'] else ""
                st.markdown(f"""<div class="fixture-row" style="border-color:#fa5252">
                  {hora_badge}
                  <span class="fixture-team">{local}</span>
                  <span class="fixture-score" style="color:#fa5252">{gl_e} – {gv_e}</span>
                  <span class="fixture-team right">{visita}</span>
                  <span style="color:#fa5252;font-size:.72rem;margin-left:8px">🔴 En juego{min_txt}</span>
                  </div>""", unsafe_allow_html=True)

            elif local in st.session_state.datos and visita in st.session_state.datos:
                r = calcular_prediccion(local, visita, fecha=fecha)
                if r['prob_l']>r['prob_v'] and r['prob_l']>r['prob_e']:
                    fav=f"▶ {local} ({r['prob_l']*100:.0f}%)"
                elif r['prob_v']>r['prob_l'] and r['prob_v']>r['prob_e']:
                    fav=f"▶ {visita} ({r['prob_v']*100:.0f}%)"
                else:
                    fav=f"▶ Empate ({r['prob_e']*100:.0f}%)"
                estado_badge = '<span style="color:#6c7086;font-size:.72rem;margin-left:8px">🕐 Por jugar</span>'
                top3 = top_marcadores(r['matriz'])
                alt_txt = "  ".join([f"{gl}–{gv} <span style='color:#6c7086'>({p:.1f}%)</span>" for gl,gv,p in top3])

                # Badge urgencia J3
                urgencia_txt = ""
                if fecha in FECHAS_J3:
                    def _urg_label(eq):
                        grp = EQUIPO_GRUPO.get(eq)
                        if not grp: return ""
                        fa, _ = factor_urgencia(eq, grp)
                        if fa >= 1.14:   return f"🔥 {eq} necesita ganar"
                        elif fa >= 1.05: return f"⚡ {eq} necesita al menos empatar"
                        elif fa <= 0.93: return f"💤 {eq} sin opciones"
                        else:            return f"✅ {eq} clasificado"
                    ul = _urg_label(local); uv = _urg_label(visita)
                    partes = [x for x in [ul, uv] if x]
                    if partes:
                        urgencia_txt = f'<div style="font-size:.68rem;color:#fab005;padding:0 8px 2px 52px">{" · ".join(partes)}</div>'

                st.markdown(f"""<div class="fixture-row">
                  {hora_badge}
                  <span class="fixture-team">{local}</span>
                  <span class="fixture-pred" title="{fav}">~ {r['goles_l']}–{r['goles_v']} ~</span>
                  <span class="fixture-team right">{visita}</span>{estado_badge}
                  </div>
                  {urgencia_txt}
                  <div style="font-size:.70rem;color:#a6adc8;padding:2px 8px 6px 52px">
                    Más probables: {alt_txt}
                  </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────
# TAB 2: GRUPOS
# ─────────────────────────────────────────────────────────────────────
with tab_grupos:
    st.markdown("### Tablas de posiciones — 12 grupos")

    # Mostrar grupos en 3 columnas
    letras = list(GRUPOS.keys())
    for fila in range(0, len(letras), 3):
        cols = st.columns(3)
        for ci, letra in enumerate(letras[fila:fila+3]):
            with cols[ci]:
                orden, tab = tabla_grupo(letra)
                st.markdown(f"**Grupo {letra}**")
                rows = []
                for pos, eq in enumerate(orden, 1):
                    v = tab[eq]
                    dg = v['GF']-v['GC']
                    clasificado = "✅" if pos<=2 else ("🟡" if pos==3 else "❌")
                    rows.append({
                        '': clasificado, 'Equipo': eq,
                        'PJ':v['PJ'],'Pts':v['Pts'],
                        'GF':v['GF'],'GC':v['GC'],
                        'DG':f"{'+' if dg>=0 else ''}{dg}"
                    })
                df_g = pd.DataFrame(rows).set_index('')
                st.dataframe(df_g, use_container_width=True, hide_index=False)

    st.caption("✅ clasificado directo · 🟡 candidato a mejor tercero · ❌ eliminado")

# TAB 3: MONTE CARLO
# ─────────────────────────────────────────────────────────────────────
with tab_mc:
    st.markdown("### Simulación del torneo completo")
    st.caption("Corre el torneo miles de veces y muestra la probabilidad de cada equipo de llegar a cada ronda.")

    col_mc1, col_mc2 = st.columns([2,1])
    with col_mc1:
        n_sims = st.select_slider("Número de simulaciones",
            options=[1000,2000,5000,10000], value=5000)
    with col_mc2:
        correr = st.button("▶ Simular", use_container_width=True, type="primary")

    if correr:
        # Cache key = hash of datos state
        datos_hash = str(sorted(
            (k,v['PJ'],v['g_favor']) for k,v in st.session_state.datos.items()
        ))
        with st.spinner(f"Simulando {n_sims:,} torneos..."):
            resultado_mc = monte_carlo(n_sims, datos_hash)
        st.session_state.mc_result = resultado_mc

    if 'mc_result' in st.session_state:
        mc = st.session_state.mc_result

        rondas = ['Grupos','R32','R16','QF','SF','Final','Campeon']
        rows = []
        for eq in st.session_state.datos:
            if eq in mc:
                row = {'Equipo':eq}
                row.update({r:round(mc[eq][r],1) for r in rondas})
                rows.append(row)

        df_mc = (pd.DataFrame(rows)
                   .set_index('Equipo')
                   .sort_values('Campeon',ascending=False))

        st.markdown("#### % probabilidad por ronda")
        st.dataframe(
            df_mc.style
                .background_gradient(subset=['Grupos'], cmap='Blues')
                .background_gradient(subset=['R32','R16','QF','SF'], cmap='Purples')
                .background_gradient(subset=['Final','Campeon'], cmap='Oranges')
                .format("{:.1f}%"),
            use_container_width=True, height=500
        )

        st.markdown("#### Top 10 candidatos al título")
        top10 = df_mc.head(10)
        chart_data = top10[['Campeon','Final','SF']].rename(columns={
            'Campeon':'Campeón','Final':'Final','SF':'Semis'
        })
        st.bar_chart(chart_data)

# ─────────────────────────────────────────────────────────────────────
# TAB 3: PREDICTOR MANUAL
# ─────────────────────────────────────────────────────────────────────
with tab_pred:
    st.markdown("### Predice cualquier partido")

    # API Key config
    with st.expander("⚙️ Cuotas reales — The Odds API (opcional)"):
        st.markdown("""
        Obtén una API key gratis en **[the-odds-api.com](https://the-odds-api.com)** (500 requests/mes gratuitos).
        Con la key activa, las cuotas de Betano, bet365, etc. serán **reales** en vez de simuladas.
        """)
        key_input = st.text_input("API Key", value=st.session_state.odds_api_key,
                                   type="password", placeholder="abc123...")
        if key_input != st.session_state.odds_api_key:
            st.session_state.odds_api_key = key_input
        usar_api = bool(st.session_state.odds_api_key)
        if usar_api:
            st.success("✅ API key configurada — se usarán cuotas reales cuando estén disponibles.")
            if st.button("🔍 Probar conexión API"):
                cuotas_test, msg_test = cuotas_reales("Argentina", "Austria")
                if cuotas_test:
                    st.success(f"Conexión OK — {len(cuotas_test)} casas disponibles para Argentina vs Austria")
                else:
                    st.warning(f"Respuesta API: {msg_test}")

    equipos = sorted(st.session_state.datos.keys())
    c1,c2 = st.columns(2)
    with c1:
        local = st.selectbox("🏠 Local", equipos, index=equipos.index("México"), key="sel_local")
    with c2:
        opts=[e for e in equipos if e!=local]
        visita = st.selectbox("✈️ Visita", opts,
            index=opts.index("Corea del Sur") if "Corea del Sur" in opts else 0,
            key="sel_visita")

    aplicar_j3 = st.checkbox("🏁 Aplicar presión de clasificación (3era fecha grupos)", value=False)
    fecha_pred = "2026-06-24" if aplicar_j3 else None  # activa el factor J3

    if st.button("⚡ Predecir", use_container_width=True, type="primary"):
        with st.spinner("Calculando..."):
            st.session_state.ultima_pred = calcular_prediccion(local, visita, usar_api=usar_api, fecha=fecha_pred)

    if 'ultima_pred' in st.session_state:
        r = st.session_state.ultima_pred
        loc,vis = r['local'],r['visita']
        pL,pE,pV = r['prob_l']*100,r['prob_e']*100,r['prob_v']*100

        st.markdown(f"""<div class="card">
          <div style="text-align:center;color:#a6adc8;font-size:.8rem;margin-bottom:4px">MARCADOR ESTIMADO</div>
          <div class="marcador">{loc} {r['goles_l']} – {r['goles_v']} {vis}</div>
          <div style="text-align:center;color:#6c7086;font-size:.78rem">
            λ {loc}={r['lam_l']:.2f} · λ {vis}={r['lam_v']:.2f}
          </div></div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="card">
          <div style="color:#a6adc8;font-size:.76rem;margin-bottom:10px">
            PROBABILIDADES — 60% Poisson · 40% mercado
            <span style="color:#6c7086;font-size:.72rem"> (cuotas {r['fuente_cuotas']})</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:2px">
            <span><span class="tag tag-l">LOCAL</span> &nbsp;{loc}</span>
            <strong style="color:#69db7c">{pL:.1f}% &nbsp; cuota {1/r['prob_l']:.2f}</strong>
          </div>
          <div class="bar-wrap"><div class="bar" style="width:{pL}%;background:#40c057"></div></div>
          <div style="display:flex;justify-content:space-between;margin-bottom:2px">
            <span><span class="tag tag-e">EMPATE</span></span>
            <strong style="color:#ffd43b">{pE:.1f}% &nbsp; cuota {1/r['prob_e']:.2f}</strong>
          </div>
          <div class="bar-wrap"><div class="bar" style="width:{pE}%;background:#fab005"></div></div>
          <div style="display:flex;justify-content:space-between;margin-bottom:2px">
            <span><span class="tag tag-v">VISITA</span> &nbsp;{vis}</span>
            <strong style="color:#ff6b6b">{pV:.1f}% &nbsp; cuota {1/r['prob_v']:.2f}</strong>
          </div>
          <div class="bar-wrap"><div class="bar" style="width:{pV}%;background:#fa5252"></div></div>
        </div>""", unsafe_allow_html=True)

        with st.expander(f"📋 Cuotas por casa ({r['fuente_cuotas']})"):
            rows_c = [{'Casa':c,f'Local ({loc})':cL,'Empate':cE,f'Visita ({vis})':cV}
                      for c,(cL,cE,cV) in r['cuotas'].items()]
            st.dataframe(pd.DataFrame(rows_c).set_index('Casa'), use_container_width=True)

        top3 = top_marcadores(r['matriz'])
        top3_html = "  ·  ".join([f"<strong>{gl}–{gv}</strong> ({p:.1f}%)" for gl,gv,p in top3])
        st.markdown(f"""<div class="card" style="padding:12px 16px">
          <div style="color:#a6adc8;font-size:.75rem;margin-bottom:6px">🎯 MARCADORES MÁS PROBABLES</div>
          <div style="font-size:.95rem">{top3_html}</div>
        </div>""", unsafe_allow_html=True)

        with st.expander("🔢 Matriz de marcadores (% probabilidad)"):
            st.caption("Filas = goles local · Columnas = goles visita")
            df_mat = pd.DataFrame(
                r['matriz']*100,
                index=[f"{loc} {i}" for i in range(7)],
                columns=[f"{vis} {j}" for j in range(7)]
            ).round(2)
            st.dataframe(df_mat.style.background_gradient(cmap='Blues'), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# TAB 4: REGISTRAR
# ─────────────────────────────────────────────────────────────────────
with tab_reg:
    st.markdown("### Registrar resultado real")
    st.caption("El modelo se actualiza automáticamente y mejora las siguientes predicciones.")

    equipos = sorted(st.session_state.datos.keys())
    c1,c2,c3,c4 = st.columns([3,1,1,3])
    with c1: r_local  = st.selectbox("Local",  equipos, key="reg_local")
    with c2: g_local  = st.number_input("Goles",0,20,0,key="reg_gl")
    with c3: g_visita = st.number_input("Goles",0,20,0,key="reg_gv")
    with c4: r_visita = st.selectbox("Visita",[e for e in equipos if e!=r_local],key="reg_visita")

    if st.button("✅ Registrar",use_container_width=True,type="primary"):
        d=st.session_state.datos
        d[r_local]['g_favor']+=g_local;  d[r_local]['g_contra']+=g_visita;  d[r_local]['PJ']+=1
        d[r_visita]['g_favor']+=g_visita; d[r_visita]['g_contra']+=g_local; d[r_visita]['PJ']+=1
        st.session_state.resultados_extra.append({'local':r_local,'visita':r_visita,'gl':g_local,'gv':g_visita})
        if 'ultima_pred' in st.session_state: del st.session_state['ultima_pred']
        if 'mc_result' in st.session_state: del st.session_state['mc_result']
        st.success(f"Registrado: {r_local} {g_local} – {g_visita} {r_visita}")
        st.rerun()

    if st.session_state.resultados_extra:
        st.markdown("#### Resultados de esta sesión")
        df=pd.DataFrame([{'Partido':f"{r['local']} vs {r['visita']}",'Resultado':f"{r['gl']} – {r['gv']}"}
                          for r in st.session_state.resultados_extra[::-1]])
        st.dataframe(df,use_container_width=True,hide_index=True)
        if st.button("🗑️ Reiniciar todo"):
            for k in ['datos','resultados_extra','ultima_pred','mc_result']:
                if k in st.session_state: del st.session_state[k]
            st.rerun()

# ─────────────────────────────────────────────────────────────────────
# TAB 5: ESTADÍSTICAS
# ─────────────────────────────────────────────────────────────────────
with tab_stats:
    datos = st.session_state.datos
    st.markdown("### Rendimiento real en el torneo")
    jugados = {k:v for k,v in datos.items() if v['PJ']>0}
    if jugados:
        rows=[{'Equipo':eq,'PJ':v['PJ'],'GF':v['g_favor'],'GC':v['g_contra'],
               'DG':f"{'+' if v['g_favor']>=v['g_contra'] else ''}{v['g_favor']-v['g_contra']}",
               'Prom GF':round(v['g_favor']/v['PJ'],2)}
              for eq,v in sorted(jugados.items(),key=lambda x:-(x[1]['g_favor']-x[1]['g_contra']))]
        st.dataframe(pd.DataFrame(rows).set_index('Equipo'),use_container_width=True)
    else:
        st.info("Sin partidos registrados.")

    # ── Precisión del modelo ──────────────────────────────────────────
    st.markdown("### 🎯 Precisión del modelo")
    todos_cerrados = []
    for _,loc_en,vis_en,gl,gv in FIXTURE_2026:
        if gl is None: continue
        loc2, vis2 = t(loc_en), t(vis_en)
        if loc2 in DATOS_INICIAL and vis2 in DATOS_INICIAL:
            todos_cerrados.append((loc2, vis2, gl, gv))
    for r2 in st.session_state.resultados_extra:
        if r2['local'] in DATOS_INICIAL and r2['visita'] in DATOS_INICIAL:
            todos_cerrados.append((r2['local'], r2['visita'], r2['gl'], r2['gv']))

    if todos_cerrados:
        ac_ganador = ac_exacto = 0
        filas_acc = []
        for loc2, vis2, gl_r, gv_r in todos_cerrados:
            d0 = DATOS_INICIAL
            ll0 = d0[loc2]['ataque']*d0[vis2]['defensa']*PROMEDIO_GOL*d0[loc2]['factor']
            lv0 = d0[vis2]['ataque']*d0[loc2]['defensa']*PROMEDIO_GOL*d0[vis2]['factor']
            pl0,pe0,pv0,gi0,gj0,_ = poisson_probs(ll0, lv0)
            if pl0>=pe0 and pl0>=pv0: pred_res='L'
            elif pe0>=pl0 and pe0>=pv0: pred_res='E'
            else: pred_res='V'
            if gl_r>gv_r: real_res='L'
            elif gl_r==gv_r: real_res='E'
            else: real_res='V'
            ok_g = pred_res==real_res
            ok_e = (gi0==gl_r and gj0==gv_r)
            if ok_g: ac_ganador+=1
            if ok_e: ac_exacto+=1
            filas_acc.append({
                'Partido': f"{loc2} vs {vis2}",
                'Real': f"{gl_r}–{gv_r}",
                'Predicho': f"{gi0}–{gj0}",
                '¿Ganador OK?': '✅' if ok_g else '❌',
                '¿Exacto?': '✅' if ok_e else '❌',
            })

        n = len(todos_cerrados)
        pct_g = ac_ganador/n*100
        pct_e = ac_exacto/n*100
        c1,c2,c3 = st.columns(3)
        c1.metric("Partidos analizados", n)
        c2.metric("Acierto ganador", f"{pct_g:.0f}%", f"{ac_ganador}/{n}")
        c3.metric("Acierto marcador exacto", f"{pct_e:.0f}%", f"{ac_exacto}/{n}")
        with st.expander("Ver detalle partido a partido"):
            st.dataframe(pd.DataFrame(filas_acc).set_index('Partido'), use_container_width=True)
    else:
        st.info("Aún no hay partidos cerrados para evaluar.")

    st.markdown("### Ratings base — todos los equipos")
    rows_b=[{'Equipo':eq,'Ataque':v['ataque'],'Defensa':v['defensa'],'Factor':v['factor']}
            for eq,v in sorted(datos.items(),key=lambda x:-x[1]['ataque'])]
    st.dataframe(
        pd.DataFrame(rows_b).set_index('Equipo')
        .style.background_gradient(subset=['Ataque'],cmap='Greens')
               .background_gradient(subset=['Defensa'],cmap='Reds_r'),
        use_container_width=True
    )
