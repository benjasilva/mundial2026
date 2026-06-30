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
    ("2026-06-18","Mexico","South Korea",1,0),
    ("2026-06-18","Czech Republic","South Africa",1,1),
    ("2026-06-19","United States","Australia",2,0),
    ("2026-06-19","Turkey","Paraguay",0,1),
    ("2026-06-19","Scotland","Morocco",0,1),
    ("2026-06-19","Brazil","Haiti",3,0),
    ("2026-06-20","Tunisia","Japan",0,4),
    ("2026-06-20","Netherlands","Sweden",5,1),
    ("2026-06-20","Ecuador","Curaçao",0,0),
    ("2026-06-20","Germany","Ivory Coast",2,1),
    ("2026-06-21","New Zealand","Egypt",1,3),
    ("2026-06-21","Belgium","Iran",0,0),
    ("2026-06-21","Spain","Saudi Arabia",4,0),
    ("2026-06-21","Uruguay","Cape Verde",2,2),
    ("2026-06-22","Jordan","Algeria",1,2),
    ("2026-06-22","Argentina","Austria",2,0),
    ("2026-06-22","France","Iraq",3,0),
    ("2026-06-22","Norway","Senegal",3,2),
    ("2026-06-23","Colombia","DR Congo",1,0),
    ("2026-06-23","Portugal","Uzbekistan",5,0),
    ("2026-06-23","Panama","Croatia",0,1),
    ("2026-06-23","England","Ghana",0,0),
    ("2026-06-24","Switzerland","Canada",2,1),
    ("2026-06-24","Bosnia and Herzegovina","Qatar",3,1),
    ("2026-06-24","Czech Republic","Mexico",0,3),
    ("2026-06-24","South Africa","South Korea",1,0),
    ("2026-06-24","Morocco","Haiti",4,2),
    ("2026-06-24","Scotland","Brazil",0,3),
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
UTC_OFFSET = -4  # Chile continental en junio/julio (UTC-4)

# =====================================================================
# FIXTURE FASE ELIMINATORIA
# Llenar equipos a medida que se confirman; None = por definir
# =====================================================================
# R32 — dieciseisavos de final
# Formato: (fecha_chile, local, visita, goles_local, goles_visita)
# Dejar local/visita en None mientras no se confirmen equipos
FIXTURE_R32 = {
    # Fuente: FIFA oficial / Wikipedia 2026 FIFA World Cup knockout stage
    # Fechas en hora Chile (UTC-4)
    73: ("2026-06-28", "Sudáfrica",           "Canadá",               0,    1),     # → Canadá 1-0
    74: ("2026-06-29", "Ecuador",             "Alemania",             2,    1),     # → Ecuador 2-1 (Forza)
    75: ("2026-06-29", "Países Bajos",        "Marruecos",            1,    1),     # → 1-1 (pen: Marruecos 3-2)
    76: ("2026-06-29", "Brasil",              "Francia",              None, None),  # → pendiente confirmar rival
    77: ("2026-06-30", "Japón",               "Suecia",               1,    1),     # → 1-1 TC (Forza)
    78: ("2026-06-30", "Costa de Marfil",     "Noruega",              None, None),  # en curso (Forza)
    79: ("2026-06-30", "México",              "Chequia",              3,    0),     # → México 3-0 en curso (Forza)
    80: ("2026-07-01", "Inglaterra",           "R. D. del Congo",      None, None),  # 12:00
    81: ("2026-07-01", "Estados Unidos",       "Bosnia y Herzegovina", None, None),  # 20:00
    82: ("2026-07-01", "Bélgica",              "Senegal",              None, None),  # 16:00
    83: ("2026-07-02", "Portugal",             "Croacia",              None, None),  # 19:00
    84: ("2026-07-02", "España",               "Austria",              None, None),  # 15:00
    85: ("2026-07-02", "Suiza",                "Argelia",              None, None),  # 23:00
    86: ("2026-07-03", "Argentina",            "Cabo Verde",           None, None),  # 18:00
    87: ("2026-07-03", "Colombia",             "Ghana",                None, None),  # 21:30
    88: ("2026-07-03", "Australia",            "Egipto",               None, None),  # 14:00
}
# R16 — octavos de final
FIXTURE_R16 = {
    89:  ("2026-07-06", None, None, None, None),
    90:  ("2026-07-06", None, None, None, None),
    91:  ("2026-07-07", None, None, None, None),
    92:  ("2026-07-07", None, None, None, None),
    93:  ("2026-07-08", None, None, None, None),
    94:  ("2026-07-08", None, None, None, None),
    95:  ("2026-07-09", None, None, None, None),
    96:  ("2026-07-09", None, None, None, None),
}
# QF — cuartos de final
FIXTURE_QF = {
    97:  ("2026-07-11", None, None, None, None),
    98:  ("2026-07-11", None, None, None, None),
    99:  ("2026-07-12", None, None, None, None),
    100: ("2026-07-12", None, None, None, None),
}
# SF — semifinales
FIXTURE_SF = {
    101: ("2026-07-14", None, None, None, None),
    102: ("2026-07-15", None, None, None, None),
}
# Final
FIXTURE_FINAL = {
    103: ("2026-07-19", None, None, None, None),
}

# Bracket de R16/QF/SF/Final (qué partidos enfrentan ganadores de qué matches)
BRACKET_R16_LABELS = [(89,"M73","M74"),(90,"M75","M76"),(91,"M77","M78"),(92,"M79","M80"),
                      (93,"M81","M82"),(94,"M83","M84"),(95,"M85","M86"),(96,"M87","M88")]
BRACKET_QF_LABELS  = [(97,"M89","M90"),(98,"M91","M92"),(99,"M93","M94"),(100,"M95","M96")]
BRACKET_SF_LABELS  = [(101,"M97","M98"),(102,"M99","M100")]
BRACKET_F_LABEL    = [(103,"M101","M102")]

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
    if 'alineaciones' not in st.session_state:
        # {key: {'aus_local': [], 'aus_visita': []}}  key = "Local_vs_Visita"
        st.session_state.alineaciones = {}

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
            lv *= fd_l
        if grp_v:
            fa_v, fd_v = factor_urgencia(visita, grp_v)
            lv *= fa_v
            ll *= fd_v

    # Aplicar factor de estadísticas reales (disparos al arco)
    ts = build_team_stats()
    if local in ts:
        ll *= ts[local]['f_atq']
        lv *= ts[local]['f_def']
    if visita in ts:
        lv *= ts[visita]['f_atq']
        ll *= ts[visita]['f_def']

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
    # Fuerza relativa simétrica (cancha neutral)
    sl = d[local]['ataque']  / d[visita]['defensa']
    sv = d[visita]['ataque'] / d[local]['defensa']
    prob_l = sl / (sl + sv)
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
            # Detectar si los equipos están invertidos respecto al orden del usuario
            api_home_k = limpiar(ev.get("home_team",""))
            invertidos = api_home_k not in (loc_k, loc_k2)
            cuotas = {}
            for book in ev.get("bookmakers",[]):
                nombre_book = book["title"]
                for mkt in book.get("markets",[]):
                    if mkt["key"]=="h2h":
                        odds = {o["name"]:o["price"] for o in mkt["outcomes"]}
                        c_home = odds.get(ev["home_team"],0)
                        c_away = odds.get(ev["away_team"],0)
                        cE = odds.get("Draw",3.30)
                        if c_home and c_away:
                            # Asignar cL/cV según el orden elegido por el usuario
                            cL, cV = (c_away, c_home) if invertidos else (c_home, c_away)
                            cuotas[nombre_book] = (cL,cE,cV)
            return cuotas if cuotas else None, "Sin cuotas disponibles"
        return None, "Partido no encontrado en la API"
    except Exception as e:
        return None, str(e)

# =====================================================================
# FACTOR ESTRELLA — jugadores clave por selección
# impacto_atq: fracción de reducción del lambda ofensivo si el jugador no juega
# =====================================================================
ESTRELLAS = {
    'Argentina':    [{'nombre':'Lionel Messi',        'impacto_atq':0.28},
                     {'nombre':'Lautaro Martínez',    'impacto_atq':0.14}],
    'Brasil':       [{'nombre':'Vinicius Jr.',         'impacto_atq':0.24},
                     {'nombre':'Rodrygo',              'impacto_atq':0.12}],
    'Colombia':     [{'nombre':'Luis Díaz',            'impacto_atq':0.20},
                     {'nombre':'James Rodríguez',      'impacto_atq':0.16}],
    'Uruguay':      [{'nombre':'Darwin Núñez',         'impacto_atq':0.22},
                     {'nombre':'Federico Valverde',    'impacto_atq':0.14}],
    'Ecuador':      [{'nombre':'Enner Valencia',       'impacto_atq':0.20}],
    'Paraguay':     [{'nombre':'Miguel Almirón',       'impacto_atq':0.18}],
    'México':       [{'nombre':'Raúl Jiménez',         'impacto_atq':0.20},
                     {'nombre':'Hirving Lozano',       'impacto_atq':0.14}],
    'Estados Unidos':[{'nombre':'Christian Pulisic',  'impacto_atq':0.24},
                      {'nombre':'Gio Reyna',           'impacto_atq':0.12}],
    'Canadá':       [{'nombre':'Alphonso Davies',      'impacto_atq':0.22},
                     {'nombre':'Jonathan David',       'impacto_atq':0.18}],
    'Panamá':       [{'nombre':'Rolando Blackburn',    'impacto_atq':0.16}],
    'Francia':      [{'nombre':'Kylian Mbappé',        'impacto_atq':0.30},
                     {'nombre':'Antoine Griezmann',    'impacto_atq':0.14}],
    'España':       [{'nombre':'Lamine Yamal',         'impacto_atq':0.18},
                     {'nombre':'Álvaro Morata',        'impacto_atq':0.14},
                     {'nombre':'Dani Olmo',            'impacto_atq':0.12}],
    'Portugal':     [{'nombre':'Cristiano Ronaldo',    'impacto_atq':0.22},
                     {'nombre':'Bruno Fernandes',      'impacto_atq':0.18}],
    'Alemania':     [{'nombre':'Florian Wirtz',        'impacto_atq':0.20},
                     {'nombre':'Jamal Musiala',        'impacto_atq':0.18}],
    'Inglaterra':   [{'nombre':'Harry Kane',           'impacto_atq':0.24},
                     {'nombre':'Jude Bellingham',      'impacto_atq':0.20}],
    'Países Bajos': [{'nombre':'Cody Gakpo',           'impacto_atq':0.18},
                     {'nombre':'Memphis Depay',        'impacto_atq':0.14}],
    'Noruega':      [{'nombre':'Erling Haaland',       'impacto_atq':0.38}],
    'Bélgica':      [{'nombre':'Kevin De Bruyne',      'impacto_atq':0.22},
                     {'nombre':'Romelu Lukaku',        'impacto_atq':0.16}],
    'Croacia':      [{'nombre':'Luka Modrić',          'impacto_atq':0.18},
                     {'nombre':'Ivan Perišić',         'impacto_atq':0.14}],
    'Suiza':        [{'nombre':'Xherdan Shaqiri',      'impacto_atq':0.14},
                     {'nombre':'Breel Embolo',         'impacto_atq':0.12}],
    'Turquía':      [{'nombre':'Hakan Çalhanoğlu',     'impacto_atq':0.18},
                     {'nombre':'Burak Yılmaz',         'impacto_atq':0.14}],
    'Austria':      [{'nombre':'Marcel Sabitzer',      'impacto_atq':0.16},
                     {'nombre':'Marko Arnautović',     'impacto_atq':0.14}],
    'Escocia':      [{'nombre':'Andy Robertson',       'impacto_atq':0.12},
                     {'nombre':'Lyndon Dykes',         'impacto_atq':0.14}],
    'República Checa':[{'nombre':'Patrik Schick',      'impacto_atq':0.18},
                       {'nombre':'Tomáš Souček',       'impacto_atq':0.14}],
    'Bosnia y Herzegovina':[{'nombre':'Edin Džeko',    'impacto_atq':0.22}],
    'Japón':        [{'nombre':'Kaoru Mitoma',         'impacto_atq':0.18},
                     {'nombre':'Takumi Minamino',      'impacto_atq':0.14}],
    'Corea del Sur':[{'nombre':'Son Heung-min',        'impacto_atq':0.26},
                     {'nombre':'Hwang Hee-chan',       'impacto_atq':0.12}],
    'Australia':    [{'nombre':'Mathew Leckie',        'impacto_atq':0.16},
                     {'nombre':'Mitchell Duke',        'impacto_atq':0.12}],
    'Irán':         [{'nombre':'Mehdi Taremi',         'impacto_atq':0.22}],
    'Arabia Saudita':[{'nombre':'Salem Al-Dawsari',    'impacto_atq':0.20}],
    'Uzbekistán':   [{'nombre':'Eldor Shomurodov',     'impacto_atq':0.20}],
    'Marruecos':    [{'nombre':'Achraf Hakimi',        'impacto_atq':0.16},
                     {'nombre':'Hakim Ziyech',         'impacto_atq':0.18},
                     {'nombre':'Youssef En-Nesyri',    'impacto_atq':0.18}],
    'Senegal':      [{'nombre':'Sadio Mané',           'impacto_atq':0.26}],
    'Argelia':      [{'nombre':'Riyad Mahrez',         'impacto_atq':0.22}],
    'Ghana':        [{'nombre':'Jordan Ayew',          'impacto_atq':0.16},
                     {'nombre':'André Ayew',           'impacto_atq':0.14}],
    'Costa de Marfil':[{'nombre':'Sébastien Haller',  'impacto_atq':0.18},
                       {'nombre':'Nicolas Pépé',       'impacto_atq':0.14}],
    'Egipto':       [{'nombre':'Mohamed Salah',        'impacto_atq':0.32}],
    'R. D. del Congo':[{'nombre':'Dodi Lukébakio',     'impacto_atq':0.18}],
    'Sudáfrica':    [{'nombre':'Percy Tau',            'impacto_atq':0.18}],
    'Túnez':        [{'nombre':'Wahbi Khazri',         'impacto_atq':0.16}],
    'Cabo Verde':   [{'nombre':'Garry Rodrigues',      'impacto_atq':0.20}],
    'Jordania':     [{'nombre':'Musa Al-Taamari',      'impacto_atq':0.16}],
    'Irak':         [{'nombre':'Mohanad Ali',          'impacto_atq':0.16}],
    'Catar':        [{'nombre':'Almoez Ali',           'impacto_atq':0.16}],
    'Nueva Zelanda':[{'nombre':'Chris Wood',           'impacto_atq':0.20}],
    'Curazao':      [{'nombre':'Leandro Bacuna',       'impacto_atq':0.16}],
    'Haití':        [{'nombre':'Duckens Nazon',        'impacto_atq':0.14}],
}

# =====================================================================
# ONCE TITULAR — 11 titulares habituales por selección
# Base: lineups de este Mundial. impacto_atq = reducción lambda si no juega.
# POS: POR · DEF · MED · DEL
# =====================================================================
ONCE_TITULAR = {
 'Argentina': {'formacion':'4-3-3','jugadores':[
    {'nombre':'Emiliano Martínez', 'pos':'POR','impacto_atq':0.02},
    {'nombre':'Nahuel Molina',     'pos':'DEF','impacto_atq':0.06},
    {'nombre':'Cristian Romero',   'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Lisandro Martínez', 'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Nicolás Tagliafico','pos':'DEF','impacto_atq':0.05},
    {'nombre':'Rodrigo De Paul',   'pos':'MED','impacto_atq':0.09},
    {'nombre':'Enzo Fernández',    'pos':'MED','impacto_atq':0.09},
    {'nombre':'Alexis Mac Allister','pos':'MED','impacto_atq':0.09},
    {'nombre':'Lionel Messi',      'pos':'DEL','impacto_atq':0.28},
    {'nombre':'Lautaro Martínez',  'pos':'DEL','impacto_atq':0.14},
    {'nombre':'Julián Álvarez',    'pos':'DEL','impacto_atq':0.09},
 ]},
 'Australia': {'formacion':'4-3-3','jugadores':[
    {'nombre':'Mat Ryan',          'pos':'POR','impacto_atq':0.01},
    {'nombre':'Nathaniel Atkinson','pos':'DEF','impacto_atq':0.05},
    {'nombre':'Harry Souttar',     'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Kye Rowles',        'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Aziz Behich',       'pos':'DEF','impacto_atq':0.05},
    {'nombre':'Jackson Irvine',    'pos':'MED','impacto_atq':0.07},
    {'nombre':'Riley McGree',      'pos':'MED','impacto_atq':0.08},
    {'nombre':"Aiden O'Neill",     'pos':'MED','impacto_atq':0.06},
    {'nombre':'Mathew Leckie',     'pos':'DEL','impacto_atq':0.16},
    {'nombre':'Mitchell Duke',     'pos':'DEL','impacto_atq':0.12},
    {'nombre':'Marco Tilio',       'pos':'DEL','impacto_atq':0.08},
 ]},
 'Austria': {'formacion':'4-2-3-1','jugadores':[
    {'nombre':'Patrick Pentz',         'pos':'POR','impacto_atq':0.01},
    {'nombre':'Stefan Lainer',         'pos':'DEF','impacto_atq':0.05},
    {'nombre':'Philipp Lienhart',      'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Kevin Danso',           'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Phillip Mwene',         'pos':'DEF','impacto_atq':0.05},
    {'nombre':'Nicolas Seiwald',       'pos':'MED','impacto_atq':0.07},
    {'nombre':'Florian Grillitsch',    'pos':'MED','impacto_atq':0.07},
    {'nombre':'Marcel Sabitzer',       'pos':'MED','impacto_atq':0.14},
    {'nombre':'Christoph Baumgartner', 'pos':'MED','impacto_atq':0.10},
    {'nombre':'Marko Arnautović',      'pos':'DEL','impacto_atq':0.14},
    {'nombre':'Michael Gregoritsch',   'pos':'DEL','impacto_atq':0.08},
 ]},
 'Bélgica': {'formacion':'4-2-3-1','jugadores':[  # Forza: XI vs Nueva Zelanda
    {'nombre':'Thibaut Courtois',   'pos':'POR','impacto_atq':0.02},
    {'nombre':'Timothy Castagne',   'pos':'DEF','impacto_atq':0.07},
    {'nombre':'Brandon Mechele',    'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Arthur Theate',      'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Maxim De Cuyper',    'pos':'DEF','impacto_atq':0.06},
    {'nombre':'Hans Vanaken',       'pos':'MED','impacto_atq':0.08},
    {'nombre':'Youri Tielemans',    'pos':'MED','impacto_atq':0.09},
    {'nombre':'Leandro Trossard',   'pos':'MED','impacto_atq':0.12},
    {'nombre':'Kevin De Bruyne',    'pos':'MED','impacto_atq':0.22},
    {'nombre':'Jérémy Doku',        'pos':'MED','impacto_atq':0.14},
    {'nombre':'Charles De Ketelaere','pos':'DEL','impacto_atq':0.14},
 ]},
 'Bosnia y Herzegovina': {'formacion':'4-2-3-1','jugadores':[
    {'nombre':'Vedran Kjosak',       'pos':'POR','impacto_atq':0.01},
    {'nombre':'Sasa Kolašinac',      'pos':'DEF','impacto_atq':0.04},
    {'nombre':'Ermin Bičakčić',      'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Amer Govedarica',     'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Anel Ahmedhodžić',    'pos':'DEF','impacto_atq':0.05},
    {'nombre':'Miralem Pjanić',      'pos':'MED','impacto_atq':0.12},
    {'nombre':'Armin Hodžić',        'pos':'MED','impacto_atq':0.07},
    {'nombre':'Sead Kolašinac',      'pos':'MED','impacto_atq':0.06},
    {'nombre':'Haris Hajradinović',  'pos':'MED','impacto_atq':0.09},
    {'nombre':'Edin Džeko',          'pos':'DEL','impacto_atq':0.22},
    {'nombre':'Anel Ahmedhodžić',    'pos':'DEL','impacto_atq':0.07},
 ]},
 'Brasil': {'formacion':'4-3-3','jugadores':[  # XI real vs Japón R32
    {'nombre':'Alisson',           'pos':'POR','impacto_atq':0.02},
    {'nombre':'Danilo',            'pos':'DEF','impacto_atq':0.05},
    {'nombre':'Marquinhos',        'pos':'DEF','impacto_atq':0.04},
    {'nombre':'Gabriel Magalhães', 'pos':'DEF','impacto_atq':0.04},
    {'nombre':'Douglas Santos',    'pos':'DEF','impacto_atq':0.05},
    {'nombre':'Casemiro',          'pos':'MED','impacto_atq':0.06},
    {'nombre':'Bruno Guimarães',   'pos':'MED','impacto_atq':0.10},
    {'nombre':'Lucas Paquetá',     'pos':'MED','impacto_atq':0.12},
    {'nombre':'Rayan',             'pos':'DEL','impacto_atq':0.10},
    {'nombre':'Matheus Cunha',     'pos':'DEL','impacto_atq':0.12},
    {'nombre':'Vinicius Jr.',      'pos':'DEL','impacto_atq':0.24},
 ]},
 'Canadá': {'formacion':'4-3-3','jugadores':[  # XI real vs Sudáfrica R32 (Davies en banca)
    {'nombre':'Milan Crepeau',     'pos':'POR','impacto_atq':0.01},
    {'nombre':'Alistair Johnston', 'pos':'DEF','impacto_atq':0.06},
    {'nombre':'Moïse Bombito',     'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Derek Cornelius',   'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Richie Laryea',     'pos':'DEF','impacto_atq':0.07},
    {'nombre':'Tajon Buchanan',    'pos':'MED','impacto_atq':0.12},
    {'nombre':'Stephen Eustáquio', 'pos':'MED','impacto_atq':0.09},
    {'nombre':'Liam Millar',       'pos':'MED','impacto_atq':0.07},
    {'nombre':'Jonathan David',    'pos':'DEL','impacto_atq':0.18},
    {'nombre':'Luca Oluwaseyi',    'pos':'DEL','impacto_atq':0.10},
    {'nombre':'Alphonso Davies',   'pos':'DEF','impacto_atq':0.22},  # banca en R32, disponible
 ]},
 'Colombia': {'formacion':'4-2-3-1','jugadores':[
    {'nombre':'Camilo Vargas',     'pos':'POR','impacto_atq':0.02},
    {'nombre':'Daniel Muñoz',      'pos':'DEF','impacto_atq':0.07},
    {'nombre':'Dávinson Sánchez',  'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Yerry Mina',        'pos':'DEF','impacto_atq':0.04},
    {'nombre':'Johan Mojica',      'pos':'DEF','impacto_atq':0.06},
    {'nombre':'Wilmar Barrios',    'pos':'MED','impacto_atq':0.05},
    {'nombre':'Richard Ríos',      'pos':'MED','impacto_atq':0.09},
    {'nombre':'James Rodríguez',   'pos':'MED','impacto_atq':0.16},
    {'nombre':'Luis Díaz',         'pos':'DEL','impacto_atq':0.20},
    {'nombre':'Cucho Hernández',   'pos':'DEL','impacto_atq':0.12},
    {'nombre':'Rafael Santos Borré','pos':'DEL','impacto_atq':0.10},
 ]},
 'Costa de Marfil': {'formacion':'4-3-3','jugadores':[  # Forza: XI vs Noruega R32
    {'nombre':'Yahia Fofana',      'pos':'POR','impacto_atq':0.01},  # #1
    {'nombre':'Guela Doué',        'pos':'DEF','impacto_atq':0.06},  # #17 RB
    {'nombre':'Emmanuel Agbadou',  'pos':'DEF','impacto_atq':0.04},  # #20
    {'nombre':'Odilon Kossounou',  'pos':'DEF','impacto_atq':0.04},  # #7
    {'nombre':'Ghislain Konan',    'pos':'DEF','impacto_atq':0.06},  # #3
    {'nombre':'Christ Inao Oulaï', 'pos':'MED','impacto_atq':0.07},  # #26
    {'nombre':'Franck Kessié',     'pos':'MED','impacto_atq':0.08},  # #8
    {'nombre':'Ibrahim Sangaré',   'pos':'MED','impacto_atq':0.07},  # #18
    {'nombre':'Nicolas Pépé',      'pos':'DEL','impacto_atq':0.14},  # #19
    {'nombre':'Ange-Yoan Bonny',   'pos':'DEL','impacto_atq':0.14},  # #9 (titular real)
    {'nombre':'Yan Diomande',      'pos':'DEL','impacto_atq':0.14},  # #11
 ]},
 'Croacia': {'formacion':'4-2-3-1','jugadores':[
    {'nombre':'Dominik Livaković', 'pos':'POR','impacto_atq':0.01},
    {'nombre':'Josip Stanišić',    'pos':'DEF','impacto_atq':0.05},
    {'nombre':'Joško Gvardiol',    'pos':'DEF','impacto_atq':0.06},
    {'nombre':'Dario Španić',      'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Borna Sosa',        'pos':'DEF','impacto_atq':0.06},
    {'nombre':'Marcelo Brozović',  'pos':'MED','impacto_atq':0.07},
    {'nombre':'Mateo Kovačić',     'pos':'MED','impacto_atq':0.10},
    {'nombre':'Luka Modrić',       'pos':'MED','impacto_atq':0.18},
    {'nombre':'Ivan Perišić',      'pos':'DEL','impacto_atq':0.14},
    {'nombre':'Andrej Kramarić',   'pos':'DEL','impacto_atq':0.14},
    {'nombre':'Bruno Petković',    'pos':'DEL','impacto_atq':0.08},
 ]},
 'Ecuador': {'formacion':'4-4-2','jugadores':[  # Forza: XI vs Alemania R32
    {'nombre':'Hernán Galíndez',   'pos':'POR','impacto_atq':0.01},  # #1
    {'nombre':'Alan Franco',       'pos':'DEF','impacto_atq':0.06},  # #21
    {'nombre':'Joel Ordóñez',      'pos':'DEF','impacto_atq':0.04},  # #4
    {'nombre':'Willian Pacho',     'pos':'DEF','impacto_atq':0.04},  # #6
    {'nombre':'Piero Hincapié',    'pos':'DEF','impacto_atq':0.06},  # #3
    {'nombre':'John Yeboah',       'pos':'MED','impacto_atq':0.08},  # #9
    {'nombre':'Moisés Caicedo',    'pos':'MED','impacto_atq':0.12},  # #23
    {'nombre':'Pedro Vite',        'pos':'MED','impacto_atq':0.09},  # #15
    {'nombre':'Nilson Angulo',     'pos':'MED','impacto_atq':0.08},  # #20
    {'nombre':'Gonzalo Plata',     'pos':'DEL','impacto_atq':0.10},  # #19
    {'nombre':'Enner Valencia',    'pos':'DEL','impacto_atq':0.20},  # #13
 ]},
 'Egipto': {'formacion':'4-2-3-1','jugadores':[
    {'nombre':'Mohamed El-Shenawy','pos':'POR','impacto_atq':0.01},
    {'nombre':'Ahmed Hegazi',      'pos':'DEF','impacto_atq':0.04},
    {'nombre':'Omar Kamal',        'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Ahmed Fatouh',      'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Omar Abdel Hakem',  'pos':'DEF','impacto_atq':0.05},
    {'nombre':'Tarek Hamed',       'pos':'MED','impacto_atq':0.05},
    {'nombre':'Hamdi Fathi',       'pos':'MED','impacto_atq':0.07},
    {'nombre':'Trézéguet',         'pos':'MED','impacto_atq':0.10},
    {'nombre':'Omar Marmoush',     'pos':'DEL','impacto_atq':0.14},
    {'nombre':'Mohamed Salah',     'pos':'DEL','impacto_atq':0.32},
    {'nombre':'Mostafa Mohamed',   'pos':'DEL','impacto_atq':0.12},
 ]},
 'España': {'formacion':'4-3-3','jugadores':[
    {'nombre':'Unai Simón',        'pos':'POR','impacto_atq':0.01},
    {'nombre':'Dani Carvajal',     'pos':'DEF','impacto_atq':0.07},
    {'nombre':'Pau Cubarsí',       'pos':'DEF','impacto_atq':0.04},
    {'nombre':'Aymeric Laporte',   'pos':'DEF','impacto_atq':0.04},
    {'nombre':'Alejandro Grimaldo','pos':'DEF','impacto_atq':0.07},
    {'nombre':'Rodri',             'pos':'MED','impacto_atq':0.10},
    {'nombre':'Fabián Ruiz',       'pos':'MED','impacto_atq':0.10},
    {'nombre':'Dani Olmo',         'pos':'MED','impacto_atq':0.12},
    {'nombre':'Lamine Yamal',      'pos':'DEL','impacto_atq':0.18},
    {'nombre':'Álvaro Morata',     'pos':'DEL','impacto_atq':0.14},
    {'nombre':'Nico Williams',     'pos':'DEL','impacto_atq':0.14},
 ]},
 'Estados Unidos': {'formacion':'4-3-3','jugadores':[  # Forza: XI vs Turquía
    {'nombre':'Matt Turner',        'pos':'POR','impacto_atq':0.01},  # #1
    {'nombre':'Joe Scally',         'pos':'DEF','impacto_atq':0.06},  # #23
    {'nombre':'Miles Robinson',     'pos':'DEF','impacto_atq':0.04},  # #12
    {'nombre':'Mark McKenzie',      'pos':'DEF','impacto_atq':0.03},  # #22
    {'nombre':'Auston Trusty',      'pos':'DEF','impacto_atq':0.04},  # #6
    {'nombre':'Weston McKennie',    'pos':'MED','impacto_atq':0.10},  # #8
    {'nombre':'Sebastian Berhalter','pos':'MED','impacto_atq':0.07},  # #14
    {'nombre':'Giovanni Reyna',     'pos':'MED','impacto_atq':0.12},  # #7
    {'nombre':'Brenden Aaronson',   'pos':'DEL','impacto_atq':0.09},  # #11
    {'nombre':'Ricardo Pepi',       'pos':'DEL','impacto_atq':0.14},  # #9
    {'nombre':'Tim Weah',           'pos':'DEL','impacto_atq':0.12},  # #21
 ]},
 'Francia': {'formacion':'4-2-3-1','jugadores':[  # Forza: XI real vs Noruega
    {'nombre':'Mike Maignan',       'pos':'POR','impacto_atq':0.01},  # #16
    {'nombre':'Jules Koundé',       'pos':'DEF','impacto_atq':0.07},  # #5
    {'nombre':'Dayot Upamecano',    'pos':'DEF','impacto_atq':0.03},  # #4
    {'nombre':'William Saliba',     'pos':'DEF','impacto_atq':0.03},  # #17 CB
    {'nombre':'Théo Hernández',     'pos':'DEF','impacto_atq':0.07},  # #19 LB
    {'nombre':'Manu Koné',          'pos':'MED','impacto_atq':0.08},  # #6 (no Rabiot)
    {'nombre':'Aurélien Tchouaméni','pos':'MED','impacto_atq':0.07},  # #8
    {'nombre':'Ousmane Dembélé',    'pos':'DEL','impacto_atq':0.14},  # #7
    {'nombre':'Michael Olise',      'pos':'DEL','impacto_atq':0.14},  # #11
    {'nombre':'Désiré Doué',        'pos':'DEL','impacto_atq':0.12},  # #20
    {'nombre':'Kylian Mbappé',      'pos':'DEL','impacto_atq':0.30},  # #10
 ]},
 'Ghana': {'formacion':'4-2-3-1','jugadores':[
    {'nombre':'Lawrence Ati-Zigi', 'pos':'POR','impacto_atq':0.01},
    {'nombre':'Tariqe Fosu',       'pos':'DEF','impacto_atq':0.05},
    {'nombre':'Alexander Djiku',   'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Daniel Amartey',    'pos':'DEF','impacto_atq':0.04},
    {'nombre':'Gideon Mensah',     'pos':'DEF','impacto_atq':0.05},
    {'nombre':'Thomas Partey',     'pos':'MED','impacto_atq':0.08},
    {'nombre':'Iddrisu Baba',      'pos':'MED','impacto_atq':0.06},
    {'nombre':'Kudus Mohammed',    'pos':'MED','impacto_atq':0.14},
    {'nombre':'Jordan Ayew',       'pos':'DEL','impacto_atq':0.16},
    {'nombre':'André Ayew',        'pos':'DEL','impacto_atq':0.14},
    {'nombre':'Antoine Semenyo',   'pos':'DEL','impacto_atq':0.10},
 ]},
 'Inglaterra': {'formacion':'4-2-3-1','jugadores':[  # Forza: XI vs Panamá
    {'nombre':'Jordan Pickford',   'pos':'POR','impacto_atq':0.01},  # #1
    {'nombre':'Ezri Konsa',        'pos':'DEF','impacto_atq':0.06},  # #2 RB
    {'nombre':'Jarell Quansah',    'pos':'DEF','impacto_atq':0.03},  # #26 CB
    {'nombre':'Marc Guéhi',        'pos':'DEF','impacto_atq':0.03},  # #6 CB
    {'nombre':"Nico O'Reilly",     'pos':'DEF','impacto_atq':0.06},  # #3 LB
    {'nombre':'Elliot Anderson',   'pos':'MED','impacto_atq':0.08},  # #8
    {'nombre':'Jude Bellingham',   'pos':'MED','impacto_atq':0.20},  # #10
    {'nombre':'Bukayo Saka',       'pos':'DEL','impacto_atq':0.16},  # #7
    {'nombre':'Morgan Rogers',     'pos':'DEL','impacto_atq':0.12},  # #17
    {'nombre':'Marcus Rashford',   'pos':'DEL','impacto_atq':0.12},  # #11
    {'nombre':'Harry Kane',        'pos':'DEL','impacto_atq':0.24},  # #9
 ]},
 'Japón': {'formacion':'3-4-2-1','jugadores':[  # Forza: XI vs Suecia R32 (Kubo lesión)
    {'nombre':'Zion Suzuki',       'pos':'POR','impacto_atq':0.01},  # #1
    {'nombre':'Kou Itakura',       'pos':'DEF','impacto_atq':0.04},  # #4 CB
    {'nombre':'Ayumu Seko',        'pos':'DEF','impacto_atq':0.04},  # #20 CB
    {'nombre':'Junya Ito',         'pos':'DEF','impacto_atq':0.05},  # #14 CB (3-back)
    {'nombre':'Yukinari Sugawara', 'pos':'MED','impacto_atq':0.07},  # #2 WB
    {'nombre':'Daichi Kamada',     'pos':'MED','impacto_atq':0.10},  # #15
    {'nombre':'Ao Tanaka',         'pos':'MED','impacto_atq':0.08},  # #7
    {'nombre':'Keito Nakamura',    'pos':'MED','impacto_atq':0.09},  # #13 WB
    {'nombre':'Ritsu Doan',        'pos':'DEL','impacto_atq':0.12},  # #10
    {'nombre':'Daizen Maeda',      'pos':'DEL','impacto_atq':0.10},  # #11
    {'nombre':'Ayase Ueda',        'pos':'DEL','impacto_atq':0.14},  # #18 ST
 ]},
 'Marruecos': {'formacion':'4-2-3-1','jugadores':[  # XI real vs Países Bajos R32
    {'nombre':'Yassine Bounou',     'pos':'POR','impacto_atq':0.01},
    {'nombre':'Achraf Hakimi',      'pos':'DEF','impacto_atq':0.16},
    {'nombre':'Issa Diop',          'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Chadi Riad',         'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Noussair Mazraoui',  'pos':'DEF','impacto_atq':0.07},
    {'nombre':'Ayyoub Bouaddi',     'pos':'MED','impacto_atq':0.07},
    {'nombre':'Neil El Aynaoui',    'pos':'MED','impacto_atq':0.07},
    {'nombre':'Brahim Díaz',        'pos':'MED','impacto_atq':0.14},
    {'nombre':'Azzedine Ounahi',    'pos':'MED','impacto_atq':0.09},
    {'nombre':'Bilal El Khannouss', 'pos':'MED','impacto_atq':0.12},
    {'nombre':'Ismael Saibari',     'pos':'DEL','impacto_atq':0.14},
 ]},
 'México': {'formacion':'4-3-3','jugadores':[  # Forza: XI vs Chequia R32
    {'nombre':'Raúl Rangel',       'pos':'POR','impacto_atq':0.01},  # #1 (no Ochoa)
    {'nombre':'Mateo Chávez',      'pos':'DEF','impacto_atq':0.06},  # #20
    {'nombre':'Israel Reyes',      'pos':'DEF','impacto_atq':0.04},  # #15
    {'nombre':'César Montes',      'pos':'DEF','impacto_atq':0.03},  # #3
    {'nombre':'Jorge Sánchez',     'pos':'DEF','impacto_atq':0.05},  # #2
    {'nombre':'Luis Romo',         'pos':'MED','impacto_atq':0.08},  # #7
    {'nombre':'Edson Álvarez',     'pos':'MED','impacto_atq':0.07},  # #4
    {'nombre':'Gilberto Mora',     'pos':'MED','impacto_atq':0.07},  # #19
    {'nombre':'Julián Quiñones',   'pos':'DEL','impacto_atq':0.10},  # #16
    {'nombre':'Guillermo Martínez','pos':'DEL','impacto_atq':0.12},  # #22
    {'nombre':'Roberto Alvarado',  'pos':'DEL','impacto_atq':0.10},  # #25
 ]},
 'Noruega': {'formacion':'4-3-3','jugadores':[  # Forza: XI vs Costa de Marfil R32
    {'nombre':'Ørjan Nyland',           'pos':'POR','impacto_atq':0.02},  # #1
    {'nombre':'David Møller Wolfe',     'pos':'DEF','impacto_atq':0.05},  # #5
    {'nombre':'Kristoffer Ajer',        'pos':'DEF','impacto_atq':0.04},  # #3
    {'nombre':'Torbjørn Heggem',        'pos':'DEF','impacto_atq':0.04},  # #17
    {'nombre':'Marcus Holmgren Pedersen','pos':'DEF','impacto_atq':0.06},  # #16
    {'nombre':'Sander Berge',           'pos':'MED','impacto_atq':0.08},  # #8
    {'nombre':'Patrick Berg',           'pos':'MED','impacto_atq':0.07},  # #6
    {'nombre':'Martin Ødegaard',        'pos':'MED','impacto_atq':0.18},  # #10
    {'nombre':'Antonio Nusa',           'pos':'DEL','impacto_atq':0.12},  # #20
    {'nombre':'Erling Haaland',         'pos':'DEL','impacto_atq':0.38},  # #9
    {'nombre':'Alexander Sørloth',      'pos':'DEL','impacto_atq':0.14},  # #7
 ]},
 'Países Bajos': {'formacion':'4-3-3','jugadores':[
    {'nombre':'Bart Verbruggen',    'pos':'POR','impacto_atq':0.01},
    {'nombre':'Denzel Dumfries',    'pos':'DEF','impacto_atq':0.09},
    {'nombre':'Jan Paul van Hecke', 'pos':'DEF','impacto_atq':0.03},  # de Ligt (lesión) out; Aké fuera del XI
    {'nombre':'Virgil van Dijk',    'pos':'DEF','impacto_atq':0.05},
    {'nombre':'Micky van de Ven',   'pos':'DEF','impacto_atq':0.06},
    {'nombre':'Ryan Gravenberch',   'pos':'MED','impacto_atq':0.09},
    {'nombre':'Tijjani Reijnders',  'pos':'MED','impacto_atq':0.10},
    {'nombre':'Frenkie de Jong',    'pos':'MED','impacto_atq':0.10},
    {'nombre':'Donyell Malen',      'pos':'DEL','impacto_atq':0.12},  # Xavi Simons (ACL) out
    {'nombre':'Cody Gakpo',         'pos':'DEL','impacto_atq':0.18},
    {'nombre':'Memphis Depay',      'pos':'DEL','impacto_atq':0.14},
 ]},
 'Paraguay': {'formacion':'4-3-3','jugadores':[  # XI real vs Alemania R32
    {'nombre':'Alfredo Aguilar',   'pos':'POR','impacto_atq':0.01},
    {'nombre':'Júnior Alonso',     'pos':'DEF','impacto_atq':0.04},
    {'nombre':'José María Canale', 'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Gustavo Gómez',     'pos':'DEF','impacto_atq':0.04},
    {'nombre':'Santiago Cáceres',  'pos':'DEF','impacto_atq':0.05},
    {'nombre':'Andrés Cubas',      'pos':'MED','impacto_atq':0.07},
    {'nombre':'Mathías Villasanti','pos':'MED','impacto_atq':0.07},
    {'nombre':'Richard Bobadilla', 'pos':'MED','impacto_atq':0.07},
    {'nombre':'Miguel Almirón',    'pos':'DEL','impacto_atq':0.18},
    {'nombre':'Julio Enciso',      'pos':'DEL','impacto_atq':0.14},
    {'nombre':'Antonio Sanabria',  'pos':'DEL','impacto_atq':0.10},
 ]},
 'Portugal': {'formacion':'4-3-3','jugadores':[
    {'nombre':'Diogo Costa',       'pos':'POR','impacto_atq':0.01},
    {'nombre':'João Cancelo',      'pos':'DEF','impacto_atq':0.09},
    {'nombre':'Rúben Dias',        'pos':'DEF','impacto_atq':0.04},
    {'nombre':'Gonçalo Inácio',    'pos':'DEF','impacto_atq':0.04},
    {'nombre':'Nuno Mendes',       'pos':'DEF','impacto_atq':0.07},
    {'nombre':'João Palhinha',     'pos':'MED','impacto_atq':0.06},
    {'nombre':'Bruno Fernandes',   'pos':'MED','impacto_atq':0.18},
    {'nombre':'Vitinha',           'pos':'MED','impacto_atq':0.10},
    {'nombre':'Rafael Leão',       'pos':'DEL','impacto_atq':0.14},
    {'nombre':'Cristiano Ronaldo', 'pos':'DEL','impacto_atq':0.22},
    {'nombre':'Diogo Jota',        'pos':'DEL','impacto_atq':0.14},
 ]},
 'R. D. del Congo': {'formacion':'4-3-3','jugadores':[  # Forza: XI vs Uzbekistán
    {'nombre':'Lionel Mpasi',      'pos':'POR','impacto_atq':0.01},  # #1
    {'nombre':'Aaron Wan-Bissaka', 'pos':'DEF','impacto_atq':0.06},  # #2
    {'nombre':'Axel Tuanzebe',     'pos':'DEF','impacto_atq':0.03},  # #4
    {'nombre':'Chancel Mbemba',    'pos':'DEF','impacto_atq':0.04},  # #22
    {'nombre':'Arthur Masuaku',    'pos':'DEF','impacto_atq':0.06},  # #26
    {'nombre':'Samuel Moutoussamy','pos':'MED','impacto_atq':0.07},  # #8
    {'nombre':'Noah Sadiki',       'pos':'MED','impacto_atq':0.07},  # #14
    {'nombre':'Brian Cipenga',     'pos':'MED','impacto_atq':0.08},  # #9
    {'nombre':'Yoane Wissa',       'pos':'DEL','impacto_atq':0.16},  # #20
    {'nombre':'Cédric Bakambu',    'pos':'DEL','impacto_atq':0.14},  # #17
    {'nombre':'Théo Bongonda',     'pos':'DEL','impacto_atq':0.12},  # #10
 ]},
 'Senegal': {'formacion':'4-2-3-1','jugadores':[  # Forza: XI vs Irak (Mendy lesionado)
    {'nombre':'Yehvann Diouf',  'pos':'POR','impacto_atq':0.01},  # #1 (Mendy out)
    {'nombre':'Ismail Jakobs',  'pos':'DEF','impacto_atq':0.06},  # #14
    {'nombre':'Moussa Niakhaté','pos':'DEF','impacto_atq':0.03},  # #19
    {'nombre':'Abdoulaye Seck', 'pos':'DEF','impacto_atq':0.03},  # #4
    {'nombre':'Krepin Diatta',  'pos':'DEF','impacto_atq':0.05},  # #15
    {'nombre':'Habib Diarra',   'pos':'MED','impacto_atq':0.06},  # #21
    {'nombre':'Idrissa Gueye',  'pos':'MED','impacto_atq':0.07},  # #5
    {'nombre':'Lamine Camara',  'pos':'MED','impacto_atq':0.08},  # #8
    {'nombre':'Ibrahim Mbaye',  'pos':'MED','impacto_atq':0.10},  # #20
    {'nombre':'Sadio Mané',     'pos':'DEL','impacto_atq':0.26},  # #10
    {'nombre':'Ismaila Sarr',   'pos':'DEL','impacto_atq':0.14},  # #18
 ]},
 'Sudáfrica': {'formacion':'4-3-3','jugadores':[
    {'nombre':'Ronwen Williams',   'pos':'POR','impacto_atq':0.02},
    {'nombre':'Reeve Frosler',     'pos':'DEF','impacto_atq':0.06},
    {'nombre':'Rushine De Reuck',  'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Mothobi Mvala',     'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Innocent Maela',    'pos':'DEF','impacto_atq':0.05},
    {'nombre':'Teboho Mokoena',    'pos':'MED','impacto_atq':0.10},
    {'nombre':'Ethan Nkosi',       'pos':'MED','impacto_atq':0.07},
    {'nombre':'Themba Zwane',      'pos':'MED','impacto_atq':0.10},
    {'nombre':'Percy Tau',         'pos':'DEL','impacto_atq':0.18},
    {'nombre':'Lyle Foster',       'pos':'DEL','impacto_atq':0.14},
    {'nombre':'Bongokuhle Hlongwane','pos':'DEL','impacto_atq':0.10},
 ]},
 'Suecia': {'formacion':'3-4-3','jugadores':[  # Forza: XI real vs Japón R32
    {'nombre':'Jacob Widell Zetterström','pos':'POR','impacto_atq':0.01},  # #1
    {'nombre':'Gabriel Gudmundsson',    'pos':'DEF','impacto_atq':0.05},  # #5 CB
    {'nombre':'Isak Hien',              'pos':'DEF','impacto_atq':0.03},  # #4 CB
    {'nombre':'Gustaf Lagerbielke',     'pos':'DEF','impacto_atq':0.03},  # #2 CB
    {'nombre':'Elliot Stroud',          'pos':'MED','impacto_atq':0.06},  # #24 WB
    {'nombre':'Yasin Ayari',            'pos':'MED','impacto_atq':0.07},  # #18
    {'nombre':'Victor Lindelöf',        'pos':'MED','impacto_atq':0.05},  # #3
    {'nombre':'Alexander Bernhardsson', 'pos':'MED','impacto_atq':0.07},  # #21 WB
    {'nombre':'Alexander Isak',         'pos':'DEL','impacto_atq':0.22},  # #9
    {'nombre':'Viktor Gyökeres',        'pos':'DEL','impacto_atq':0.18},  # #17
    {'nombre':'Anthony Elanga',         'pos':'DEL','impacto_atq':0.12},  # #11
 ]},
 'Suiza': {'formacion':'4-2-3-1','jugadores':[
    {'nombre':'Yann Sommer',       'pos':'POR','impacto_atq':0.01},
    {'nombre':'Silvan Widmer',     'pos':'DEF','impacto_atq':0.07},
    {'nombre':'Nico Elvedi',       'pos':'DEF','impacto_atq':0.03},
    {'nombre':'Manuel Akanji',     'pos':'DEF','impacto_atq':0.04},
    {'nombre':'Ricardo Rodríguez', 'pos':'DEF','impacto_atq':0.06},
    {'nombre':'Remo Freuler',      'pos':'MED','impacto_atq':0.07},
    {'nombre':'Granit Xhaka',      'pos':'MED','impacto_atq':0.10},
    {'nombre':'Xherdan Shaqiri',   'pos':'MED','impacto_atq':0.14},
    {'nombre':'Ruben Vargas',      'pos':'DEL','impacto_atq':0.12},
    {'nombre':'Ardon Jashari',     'pos':'MED','impacto_atq':0.09},
    {'nombre':'Breel Embolo',      'pos':'DEL','impacto_atq':0.12},
 ]},
 'Alemania': {'formacion':'4-2-3-1','jugadores':[
    {'nombre':'Manuel Neuer',      'pos':'POR','impacto_atq':0.02},
    {'nombre':'Joshua Kimmich',    'pos':'DEF','impacto_atq':0.10},
    {'nombre':'Antonio Rüdiger',   'pos':'DEF','impacto_atq':0.04},
    {'nombre':'Jonathan Tah',      'pos':'DEF','impacto_atq':0.04},
    {'nombre':'Maximilian Mittelstädt','pos':'DEF','impacto_atq':0.06},
    {'nombre':'Toni Kroos',        'pos':'MED','impacto_atq':0.12},
    {'nombre':'Robert Andrich',    'pos':'MED','impacto_atq':0.07},
    {'nombre':'Jamal Musiala',     'pos':'MED','impacto_atq':0.18},
    {'nombre':'Leroy Sané',        'pos':'DEL','impacto_atq':0.14},
    {'nombre':'Kai Havertz',       'pos':'DEL','impacto_atq':0.14},
    {'nombre':'Florian Wirtz',     'pos':'DEL','impacto_atq':0.20},
 ]},
}

def _get_jugadores(equipo):
    """Devuelve la lista de jugadores con impacto. Prioriza ONCE_TITULAR, fallback a ESTRELLAS."""
    if equipo in ONCE_TITULAR:
        return ONCE_TITULAR[equipo]['jugadores']
    return ESTRELLAS.get(equipo, [])

def lambdas_adj(local, visita, fecha=None, aus_local=None, aus_visita=None):
    """Lambda base + descuento por ausencia de titulares."""
    ll, lv = lambdas(local, visita, fecha=fecha)
    if aus_local:
        impacto = sum(j['impacto_atq'] for j in _get_jugadores(local)
                      if j['nombre'] in aus_local)
        ll *= max(0.55, 1.0 - impacto)
    if aus_visita:
        impacto = sum(j['impacto_atq'] for j in _get_jugadores(visita)
                      if j['nombre'] in aus_visita)
        lv *= max(0.55, 1.0 - impacto)
    return ll, lv

def calcular_prediccion(local, visita, usar_api=False, fecha=None, aus_local=None, aus_visita=None):
    ll, lv = (lambdas_adj(local, visita, fecha=fecha,
                          aus_local=aus_local, aus_visita=aus_visita)
              if (aus_local or aus_visita)
              else lambdas(local, visita, fecha=fecha))
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

ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_event_ids(fecha_str):
    """Devuelve lista de (event_id, loc, vis) para partidos terminados en esa fecha."""
    try:
        resp = requests.get(ESPN_URL, params={"dates": fecha_str}, timeout=8)
        if resp.status_code != 200: return []
        ids = []
        for ev in resp.json().get("events", []):
            for comp in ev.get("competitions", []):
                estado = comp.get("status",{}).get("type",{}).get("name","")
                if estado not in ("STATUS_FINAL","STATUS_FULL_TIME"): continue
                competidores = comp.get("competitors",[])
                if len(competidores) < 2: continue
                h = next((c for c in competidores if c.get("homeAway")=="home"), None)
                a = next((c for c in competidores if c.get("homeAway")=="away"), None)
                if h and a:
                    loc = espn_nombre(h.get("team",{}).get("displayName",""))
                    vis = espn_nombre(a.get("team",{}).get("displayName",""))
                    if loc and vis:
                        ids.append((ev["id"], loc, vis))
        return ids
    except: return []

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_match_stats(event_id):
    """Devuelve {local: {sot, shots, poss}, visita: {...}} para un partido."""
    try:
        resp = requests.get(ESPN_SUMMARY_URL, params={"event": event_id}, timeout=8)
        if resp.status_code != 200: return None
        data = resp.json()
        teams = data.get("boxscore",{}).get("teams",[])
        if len(teams) < 2: return None
        resultado = {}
        for td in teams:
            nombre = espn_nombre(td.get("team",{}).get("displayName",""))
            if not nombre: continue
            stats = {s["name"]: s.get("value", 0) for s in td.get("statistics",[])}
            resultado[nombre] = {
                'sot':   float(stats.get("shotsOnTarget", 0)),
                'shots': float(stats.get("totalShots", 0)),
                'poss':  float(stats.get("possessionPct", 50)),
            }
        return resultado if len(resultado)==2 else None
    except: return None

def build_team_stats():
    """Construye promedios de disparos al arco y posesión por equipo en el torneo."""
    from datetime import date, timedelta
    if 'team_stats_cache' in st.session_state:
        return st.session_state.team_stats_cache

    acum = {}  # equipo → {sot_for:[], sot_against:[], poss:[]}
    inicio = date(2026, 6, 11)
    hoy = date.today()
    d = inicio
    while d <= hoy:
        for eid, loc, vis in fetch_event_ids(d.strftime("%Y%m%d")):
            stats = fetch_match_stats(eid)
            if not stats or loc not in stats or vis not in stats:
                continue  # no avanzar fecha aquí — era el bug original
            for eq, rival in [(loc, vis), (vis, loc)]:
                sot_eq = stats[eq]['sot']
                sot_rv = stats[rival]['sot']
                # Solo acumular si hay datos reales (sot > 0 en al menos un equipo)
                if sot_eq > 0 or sot_rv > 0:
                    if eq not in acum:
                        acum[eq] = {'sot_for':[], 'sot_against':[], 'poss':[]}
                    acum[eq]['sot_for'].append(sot_eq)
                    acum[eq]['sot_against'].append(sot_rv)
                    acum[eq]['poss'].append(stats[eq]['poss'])
        d += timedelta(days=1)

    # Calcular promedios y factor relativo al promedio del torneo
    promedios = {}
    if acum:
        all_sot = [v for eq in acum for v in acum[eq]['sot_for'] if v > 0]
        avg_sot = sum(all_sot) / max(len(all_sot), 1) if all_sot else 3.0
        for eq, v in acum.items():
            sot_for  = sum(v['sot_for'])  / max(len(v['sot_for']), 1)
            sot_ag   = sum(v['sot_against']) / max(len(v['sot_against']), 1)
            poss_avg = sum(v['poss']) / max(len(v['poss']), 1)
            # Si sot_for=0 (sin datos útiles), factor neutro = 1.0
            f_atq = (sot_for / avg_sot) ** 0.25 if sot_for > 0 and avg_sot > 0 else 1.0
            f_def = (avg_sot / sot_ag)  ** 0.20 if sot_ag  > 0 and avg_sot > 0 else 1.0
            promedios[eq] = {
                'sot_for': round(sot_for, 1),
                'sot_ag':  round(sot_ag, 1),
                'poss':    round(poss_avg, 1),
                'f_atq':   round(f_atq, 3),
                'f_def':   round(f_def, 3),
            }
    st.session_state.team_stats_cache = promedios
    return promedios

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
# BRACKET R32 — armar cruces desde standings actuales
# =====================================================================
def build_r32_bracket():
    """Devuelve lista de (match_num, equipo1, equipo2) usando standings reales."""
    W, R2 = {}, {}
    terceros = []
    for letra in GRUPOS:
        orden, tab = tabla_grupo(letra)
        W[letra]  = orden[0] if orden          else "Por confirmar"
        R2[letra] = orden[1] if len(orden) > 1 else "Por confirmar"
        if len(orden) >= 3:
            t3 = orden[2]; v = tab[t3]
            terceros.append((t3, letra, v['Pts'], v['GF']-v['GC'], v['GF']))
    terceros.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
    top8 = terceros[:8]; usados = set()

    def get_t(gv):
        for eq, gr, *_ in top8:
            if gr in gv and eq not in usados: usados.add(eq); return eq
        for eq, gr, *_ in top8:
            if eq not in usados: usados.add(eq); return eq
        return "Por confirmar"

    result = []
    for mnum, pos1, g1, pos2, g2, tg in BRACKET_R32:
        t1 = W[g1] if pos1 == 'W' else R2[g1]
        t2 = get_t(tg) if pos2 == 'T' else (W[g2] if pos2 == 'W' else R2[g2])
        result.append((mnum, t1, t2))
    return result

# =====================================================================
# HELPERS DE RENDERIZADO — reutilizables en todas las rondas
# =====================================================================
def _hora_badge_html(espn_info):
    """Genera el badge HTML de hora en Chile desde el campo hora_utc de ESPN."""
    hora_str = ""
    if espn_info and espn_info.get('hora_utc'):
        try:
            from datetime import datetime, timedelta as td
            utc_dt = datetime.fromisoformat(espn_info['hora_utc'].replace("Z", "+00:00"))
            hora_str = (utc_dt + td(hours=UTC_OFFSET)).strftime("%H:%M")
        except: pass
    luna = ""
    if hora_str:
        try:
            if int(hora_str[:2]) >= 23 or int(hora_str[:2]) < 6:
                luna = ' <span title="Madrugada" style="font-size:.75rem">🌙</span>'
        except: pass
    return (f'<span style="color:#6c7086;font-size:.72rem;min-width:42px;display:inline-block">{hora_str}{luna}</span>'
            if hora_str else '<span style="min-width:42px;display:inline-block"></span>')

def _render_match_row(local, visita, fecha, gl=None, gv=None, espn_info=None):
    """
    Renderiza una fila de partido con resultado real, estado en vivo o predicción.
    Usado en todas las rondas (grupos y eliminatoria).
    """
    hb = _hora_badge_html(espn_info)
    por_confirmar = local in (None, "Por confirmar") or visita in (None, "Por confirmar")

    if por_confirmar:
        loc_lbl = local  or "Por confirmar"
        vis_lbl = visita or "Por confirmar"
        st.markdown(f"""<div class="fixture-row" style="opacity:.42">
          {hb}<span class="fixture-team">{loc_lbl}</span>
          <span class="fixture-pred">?</span>
          <span class="fixture-team right">{vis_lbl}</span>
          <span style="color:#6c7086;font-size:.72rem;margin-left:8px">⏳ Por definir</span>
          </div>""", unsafe_allow_html=True)
        return

    # Resultado hardcodeado
    if gl is not None:
        st.markdown(f"""<div class="fixture-row">
          {hb}<span class="fixture-team">{local}</span>
          <span class="fixture-score">{gl} – {gv}</span>
          <span class="fixture-team right">{visita}</span>
          <span style="color:#6c7086;font-size:.72rem;margin-left:8px">✅ Final</span>
          </div>""", unsafe_allow_html=True)
        return

    # Resultado ESPN (final)
    if espn_info and espn_info['estado'] == 'final':
        g1e, g2e = espn_info['gl'], espn_info['gv']
        st.markdown(f"""<div class="fixture-row">
          {hb}<span class="fixture-team">{local}</span>
          <span class="fixture-score">{g1e} – {g2e}</span>
          <span class="fixture-team right">{visita}</span>
          <span style="color:#6c7086;font-size:.72rem;margin-left:8px">✅ Final</span>
          </div>""", unsafe_allow_html=True)
        return

    # En juego
    if espn_info and espn_info['estado'] == 'en_juego':
        g1e, g2e = espn_info['gl'], espn_info['gv']
        min_txt = f" {espn_info['minuto']}" if espn_info['minuto'] else ""
        st.markdown(f"""<div class="fixture-row" style="border-color:#fa5252">
          {hb}<span class="fixture-team">{local}</span>
          <span class="fixture-score" style="color:#fa5252">{g1e} – {g2e}</span>
          <span class="fixture-team right">{visita}</span>
          <span style="color:#fa5252;font-size:.72rem;margin-left:8px">🔴 En juego{min_txt}</span>
          </div>""", unsafe_allow_html=True)
        return

    # Predicción
    if local in st.session_state.datos and visita in st.session_state.datos:
        r = calcular_prediccion(local, visita, fecha=fecha)
        pl, pe, pv = r['prob_l']*100, r['prob_e']*100, r['prob_v']*100
        if pl > pv and pl > pe:   fav = f"▶ {local} ({pl:.0f}%)"
        elif pv > pl and pv > pe: fav = f"▶ {visita} ({pv:.0f}%)"
        else:                      fav = f"▶ Empate ({pe:.0f}%)"
        top3 = top_marcadores(r['matriz'])
        alt_txt = "  ".join([f"{a}–{b} <span style='color:#6c7086'>({p:.1f}%)</span>" for a,b,p in top3])
        st.markdown(f"""<div class="fixture-row">
          {hb}<span class="fixture-team">{local}</span>
          <span class="fixture-pred" title="{fav}">~ {r['goles_l']}–{r['goles_v']} ~</span>
          <span class="fixture-team right">{visita}</span>
          <span style="color:#6c7086;font-size:.72rem;margin-left:8px">🕐 Por jugar</span>
          </div>
          <div style="font-size:.70rem;color:#a6adc8;padding:2px 8px 6px 52px">
            Más probables: {alt_txt}
          </div>""", unsafe_allow_html=True)

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
.date-hdr{color:#cba6f7;font-size:.82rem;font-weight:700;letter-spacing:.10em;text-transform:uppercase;
  margin:22px 0 6px;padding:6px 12px;
  border-left:3px solid #cba6f7;background:rgba(203,166,247,.07);border-radius:0 6px 6px 0}
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
        st.markdown("### Fixture — Copa Mundial 2026")
        st.caption("Resultados en tiempo real · Predicciones para partidos por jugar · Hora Chile (UTC-4)")
    with col_h2:
        if st.button("🔄 Actualizar resultados", use_container_width=True):
            st.session_state.espn_fetched = False
            st.session_state.resultados_extra = []
            if 'datos' in st.session_state: del st.session_state['datos']
            if 'mc_result' in st.session_state: del st.session_state['mc_result']
            st.rerun()

    hoy_str = date.today().isoformat()
    extra_lkp = {(r['local'],r['visita']):(r['gl'],r['gv']) for r in st.session_state.resultados_extra}

    _espn_cache = {}
    def get_espn_fecha(f):
        if f not in _espn_cache:
            from datetime import datetime, timedelta as td
            d = datetime.strptime(f, "%Y-%m-%d")
            dia_actual = dict(fetch_espn_estado(d.strftime("%Y%m%d")))
            for k, v in fetch_espn_estado((d + td(days=1)).strftime("%Y%m%d")).items():
                hora_utc = v.get('hora_utc','')
                if hora_utc:
                    try:
                        hora = datetime.fromisoformat(hora_utc.replace("Z","+00:00"))
                        if hora.hour < 6:
                            dia_actual.setdefault(k, v)
                    except: pass
            _espn_cache[f] = dia_actual
        return _espn_cache[f]

    def _date_header(fecha):
        dt = pd.to_datetime(fecha)
        badge = '<span class="today-badge">HOY</span>' if fecha == hoy_str else ""
        st.markdown(f'<div class="date-hdr">{dt.day} de {MESES[dt.month]}{badge}</div>',
                    unsafe_allow_html=True)

    # ── 🔔 ALERTAS PRÓXIMOS PARTIDOS (≤ 30 min) ──────────────────────
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    _now = _dt.now(_tz.utc)
    _r32_tmp = build_r32_bracket()
    for _mnum, _t1, _t2 in _r32_tmp:
        if _t1 in (None,"Por confirmar") or _t2 in (None,"Por confirmar"):
            continue
        _f32 = FIXTURE_R32[_mnum][0]
        _ei  = get_espn_fecha(_f32).get((_t1,_t2)) or get_espn_fecha(_f32).get((_t2,_t1))
        if not _ei or _ei['estado'] != 'por_jugar': continue
        _hu  = _ei.get('hora_utc','')
        if not _hu: continue
        try:
            _mt  = _dt.fromisoformat(_hu.replace("Z","+00:00"))
            _dif = (_mt - _now).total_seconds() / 60
            if 0 < _dif <= 30:
                _hora_cl = (_mt + _td(hours=UTC_OFFSET)).strftime("%H:%M")
                st.warning(
                    f"⚠️ **Faltan {int(_dif)} min** — **{_t1} vs {_t2}** · {_hora_cl} hrs Chile  "
                    f"· Abre el **Predictor** y cargá las alineaciones antes de que empiece."
                )
        except: pass

    # ── ⚔️ DIECISEISAVOS DE FINAL — R32 (expanded) ───────────────────
    st.markdown("## ⚔️ Dieciseisavos de Final")
    st.caption("Ronda de 32 · 16 partidos · Cruces calculados desde standings finales de grupos")

    r32_bracket = build_r32_bracket()
    # Sobreescribir equipos con cualquier dato manual en FIXTURE_R32
    r32_map = {mnum: (loc, vis, gl, gv)
               for mnum, (_, loc, vis, gl, gv) in FIXTURE_R32.items()
               if loc is not None}

    # Agrupar por fecha
    r32_por_fecha = {}
    for mnum, t1, t2 in r32_bracket:
        fecha_r32 = FIXTURE_R32[mnum][0]
        # Resultado manual tiene precedencia
        if mnum in r32_map:
            loc_m, vis_m, gl_m, gv_m = r32_map[mnum]
            r32_por_fecha.setdefault(fecha_r32, []).append((mnum, loc_m, vis_m, gl_m, gv_m))
        else:
            r32_por_fecha.setdefault(fecha_r32, []).append((mnum, t1, t2, None, None))

    for fecha in sorted(r32_por_fecha.keys()):
        _date_header(fecha)
        estados_dia = get_espn_fecha(fecha)
        for mnum, t1, t2, gl, gv in r32_por_fecha[fecha]:
            # Buscar resultado en extra_lkp
            if gl is None: gl, gv = extra_lkp.get((t1, t2), (None, None))
            espn_info = estados_dia.get((t1, t2)) or (estados_dia.get((t2, t1)) if t2 else None)
            lbl = f'<span style="color:#6c7086;font-size:.68rem;min-width:28px;display:inline-block">M{mnum}</span>'
            st.markdown(lbl, unsafe_allow_html=True)
            _render_match_row(t1, t2, fecha, gl=gl, gv=gv, espn_info=espn_info)

    st.markdown("---")

    # ── 🔵 OCTAVOS DE FINAL — R16 ────────────────────────────────────
    with st.expander("🔵 Octavos de Final — 8 partidos  ·  ~6–9 jul", expanded=False):
        st.caption("Ganadores de los dieciseisavos. Equipos se confirman a medida que avance el torneo.")
        r16_por_fecha = {}
        for mnum, lbl1, lbl2 in BRACKET_R16_LABELS:
            fecha_r16, loc_m, vis_m, gl_m, gv_m = FIXTURE_R16[mnum]
            t1 = loc_m or f"Gan. {lbl1}"
            t2 = vis_m or f"Gan. {lbl2}"
            # si hay resultado en extra_lkp (cuando teams ya son reales)
            gl_r, gv_r = extra_lkp.get((t1, t2), (gl_m, gv_m))
            r16_por_fecha.setdefault(fecha_r16, []).append((mnum, t1, t2, gl_r, gv_r))
        for fecha in sorted(r16_por_fecha.keys()):
            _date_header(fecha)
            estados_dia = get_espn_fecha(fecha)
            for mnum, t1, t2, gl, gv in r16_por_fecha[fecha]:
                espn_info = estados_dia.get((t1,t2)) or estados_dia.get((t2,t1))
                st.markdown(f'<span style="color:#6c7086;font-size:.68rem;min-width:28px;display:inline-block">M{mnum}</span>', unsafe_allow_html=True)
                _render_match_row(t1, t2, fecha, gl=gl, gv=gv, espn_info=espn_info)

    # ── 🟣 CUARTOS DE FINAL — QF ──────────────────────────────────────
    with st.expander("🟣 Cuartos de Final — 4 partidos  ·  ~11–12 jul", expanded=False):
        st.caption("Ganadores de los octavos.")
        qf_por_fecha = {}
        for mnum, lbl1, lbl2 in BRACKET_QF_LABELS:
            fecha_qf, loc_m, vis_m, gl_m, gv_m = FIXTURE_QF[mnum]
            t1 = loc_m or f"Gan. {lbl1}"
            t2 = vis_m or f"Gan. {lbl2}"
            gl_r, gv_r = extra_lkp.get((t1, t2), (gl_m, gv_m))
            qf_por_fecha.setdefault(fecha_qf, []).append((mnum, t1, t2, gl_r, gv_r))
        for fecha in sorted(qf_por_fecha.keys()):
            _date_header(fecha)
            estados_dia = get_espn_fecha(fecha)
            for mnum, t1, t2, gl, gv in qf_por_fecha[fecha]:
                espn_info = estados_dia.get((t1,t2)) or estados_dia.get((t2,t1))
                st.markdown(f'<span style="color:#6c7086;font-size:.68rem;min-width:28px;display:inline-block">M{mnum}</span>', unsafe_allow_html=True)
                _render_match_row(t1, t2, fecha, gl=gl, gv=gv, espn_info=espn_info)

    # ── 🔴 SEMIFINALES ────────────────────────────────────────────────
    with st.expander("🔴 Semifinales — 2 partidos  ·  ~14–15 jul", expanded=False):
        for mnum, lbl1, lbl2 in BRACKET_SF_LABELS:
            fecha_sf, loc_m, vis_m, gl_m, gv_m = FIXTURE_SF[mnum]
            t1 = loc_m or f"Gan. {lbl1}"
            t2 = vis_m or f"Gan. {lbl2}"
            gl_r, gv_r = extra_lkp.get((t1, t2), (gl_m, gv_m))
            estados_dia = get_espn_fecha(fecha_sf)
            espn_info = estados_dia.get((t1,t2)) or estados_dia.get((t2,t1))
            _date_header(fecha_sf)
            st.markdown(f'<span style="color:#6c7086;font-size:.68rem;min-width:28px;display:inline-block">M{mnum}</span>', unsafe_allow_html=True)
            _render_match_row(t1, t2, fecha_sf, gl=gl_r, gv=gv_r, espn_info=espn_info)

    # ── 🏆 FINAL ─────────────────────────────────────────────────────
    with st.expander("🏆 Final — 19 julio 2026 · MetLife Stadium, Nueva Jersey", expanded=False):
        mnum, lbl1, lbl2 = BRACKET_F_LABEL[0]
        fecha_f, loc_m, vis_m, gl_m, gv_m = FIXTURE_FINAL[mnum]
        t1 = loc_m or f"Gan. {lbl1}"
        t2 = vis_m or f"Gan. {lbl2}"
        gl_r, gv_r = extra_lkp.get((t1, t2), (gl_m, gv_m))
        estados_dia = get_espn_fecha(fecha_f)
        espn_info = estados_dia.get((t1,t2)) or estados_dia.get((t2,t1))
        _date_header(fecha_f)
        _render_match_row(t1, t2, fecha_f, gl=gl_r, gv=gv_r, espn_info=espn_info)

    st.markdown("---")

    # ── 📋 FASE DE GRUPOS (colapsada) ────────────────────────────────
    with st.expander("📋 Fase de Grupos — 72 partidos  ·  11–27 jun", expanded=False):
        fixture_por_fecha = {}
        for item in FIXTURE_2026:
            fixture_por_fecha.setdefault(item[0], []).append(item)

        for fecha in sorted(fixture_por_fecha.keys()):
            partidos_dia = fixture_por_fecha[fecha]
            estados_dia  = get_espn_fecha(fecha)

            def sort_key(item):
                loc, vis = t(item[1]), t(item[2])
                info = estados_dia.get((loc, vis))
                return info['orden'] if info else 999
            partidos_dia = sorted(partidos_dia, key=sort_key)

            _date_header(fecha)

            for _fecha, local_en, visita_en, gl, gv in partidos_dia:
                local, visita = t(local_en), t(visita_en)
                if (local, visita) in extra_lkp: gl, gv = extra_lkp[(local, visita)]
                espn_info = estados_dia.get((local, visita))

                # Badge urgencia J3 (debajo de la fila de predicción)
                urgencia_txt = ""
                if _fecha in FECHAS_J3:
                    def _urg_label(eq):
                        grp = EQUIPO_GRUPO.get(eq)
                        if not grp: return ""
                        fa, _ = factor_urgencia(eq, grp)
                        if fa >= 1.14:   return f"🔥 {eq} necesita ganar"
                        elif fa >= 1.05: return f"⚡ {eq} necesita empatar"
                        elif fa <= 0.93: return f"💤 {eq} sin opciones"
                        else:            return f"✅ {eq} clasificado"
                    ul = _urg_label(local); uv = _urg_label(visita)
                    partes = [x for x in [ul, uv] if x]
                    if partes:
                        urgencia_txt = f'<div style="font-size:.68rem;color:#fab005;padding:0 8px 4px 52px">{" · ".join(partes)}</div>'

                _render_match_row(local, visita, _fecha, gl=gl, gv=gv, espn_info=espn_info)
                if urgencia_txt:
                    st.markdown(urgencia_txt, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────
# TAB 2: GRUPOS
# ─────────────────────────────────────────────────────────────────────
with tab_grupos:
    col_tit, col_btn = st.columns([3,1])
    with col_tit:
        st.markdown("### Tablas de posiciones — 12 grupos")
    with col_btn:
        if st.button("🔄 Actualizar resultados", use_container_width=True, key="btn_refresh_grupos"):
            st.session_state.espn_fetched = False
            st.rerun()

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

    # ── Ranking mejores terceros ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🥉 Ranking mejores terceros — 8 clasifican")
    st.caption("Se comparan todos los 3eros de los 12 grupos. Clasifican los 8 con más puntos (desempate: DG → GF).")

    terceros = []
    for letra in letras:
        orden, tab = tabla_grupo(letra)
        if len(orden) >= 3:
            eq = orden[2]  # 3er lugar
            v  = tab[eq]
            dg = v['GF'] - v['GC']
            terceros.append({
                'Grupo': letra,
                'Equipo': eq,
                'PJ': v['PJ'],
                'Pts': v['Pts'],
                'GF': v['GF'],
                'GC': v['GC'],
                'DG': dg,
            })

    # Criterio oficial FIFA 2026: Pts → DG → GF → Fair Play → Ranking FIFA
    # (Fair Play y Ranking FIFA no disponibles automáticamente — desempate manual si aplica)
    terceros_ord = sorted(terceros, key=lambda x: (x['Pts'], x['DG'], x['GF']), reverse=True)

    rows_t = []
    for i, r in enumerate(terceros_ord, 1):
        pasa = i <= 8
        dg_str = f"{'+' if r['DG']>=0 else ''}{r['DG']}"
        rows_t.append({
            'Pos': i,
            '': '✅' if pasa else '❌',
            'Grupo': r['Grupo'],
            'Equipo': r['Equipo'],
            'PJ': r['PJ'],
            'Pts': r['Pts'],
            'GF': r['GF'],
            'GC': r['GC'],
            'DG': dg_str,
        })

    df_t = pd.DataFrame(rows_t).set_index('Pos')
    st.dataframe(df_t, use_container_width=True, hide_index=False)
    st.caption("Desempate oficial FIFA 2026: Pts → DG → GF → Fair Play (tarjetas) → Ranking FIFA. "
               "Los últimos dos criterios requieren datos manuales — si hay empate en GF, verificar manualmente.")

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

    # ── API Key ───────────────────────────────────────────────────────
    with st.expander("⚙️ Cuotas reales — The Odds API (opcional)"):
        st.markdown("Obtén una API key gratis en **[the-odds-api.com](https://the-odds-api.com)** (500 req/mes).")
        key_input = st.text_input("API Key", value=st.session_state.odds_api_key,
                                   type="password", placeholder="abc123...")
        if key_input != st.session_state.odds_api_key:
            st.session_state.odds_api_key = key_input
        usar_api = bool(st.session_state.odds_api_key)
        if usar_api:
            st.success("✅ API key configurada.")
            if st.button("🔍 Probar conexión API"):
                cuotas_test, msg_test = cuotas_reales("Argentina", "Austria")
                if cuotas_test:
                    st.success(f"Conexión OK — {len(cuotas_test)} casas")
                else:
                    st.warning(f"Respuesta: {msg_test}")

    # ── Selección de equipos ──────────────────────────────────────────
    equipos = sorted(st.session_state.datos.keys())
    c1, c2 = st.columns(2)
    with c1:
        local = st.selectbox("Local", equipos, index=equipos.index("Argentina"), key="sel_local")
    with c2:
        opts = [e for e in equipos if e != local]
        visita = st.selectbox("Visita", opts,
            index=opts.index("Francia") if "Francia" in opts else 0, key="sel_visita")

    aplicar_j3 = st.checkbox("🏁 Aplicar presión de clasificación (3era fecha grupos)", value=False)
    es_ko       = st.checkbox("⚔️ Partido de fase eliminatoria", value=True,
                               help="Activa nota sobre resultado: cuenta hasta 120 min, penales no incluidos.")
    fecha_pred  = "2026-06-24" if aplicar_j3 else None
    match_key   = f"{local}_vs_{visita}"

    if es_ko:
        st.markdown("""<div style="background:#1a1a2e;border-left:3px solid #fab005;
          border-radius:6px;padding:8px 12px;font-size:.78rem;color:#a6adc8;margin-bottom:4px">
          ⚔️ <strong>Fase eliminatoria</strong> — El resultado que predice el modelo es a los 90 min regulares.<br>
          En esta fase el empate es válido: significa que el partido se definió en alargue (sin gol extra) o penales.<br>
          Si hubo gol en prórroga (al 120 min), ese marcador también es correcto como resultado final.
        </div>""", unsafe_allow_html=True)

    # ── ⭐ ONCE TITULAR — seleccionar bajas ───────────────────────────
    st.markdown("---")
    st.markdown("##### ⭐ Alineación titular — marcá las bajas")
    st.caption("Por defecto todos los titulares están disponibles. Marcá como baja a quien no juega (lesión, suspensión, etc.).")

    jug_local  = _get_jugadores(local)
    jug_visita = _get_jugadores(visita)
    alin_prev  = st.session_state.alineaciones.get(match_key, {'aus_local':[], 'aus_visita':[]})

    def _xi_html(jugadores, aus_prev, prefix):
        """Renderiza el once como tarjetas con botón de baja. Devuelve HTML + lista actual bajas."""
        pos_order = ['POR', 'DEF', 'MED', 'DEL']
        por_pos = {p: [] for p in pos_order}
        for j in jugadores:
            p = j.get('pos', 'DEL')
            por_pos.setdefault(p, []).append(j)
        rows = []
        for pos in pos_order:
            for j in por_pos.get(pos, []):
                rows.append((pos, j))
        return rows

    def _render_xi_col(jugadores, aus_prev, side_key, equipo_nombre):
        """Muestra el XI con multiselect de bajas y visual HTML."""
        nombres = [j['nombre'] for j in jugadores]
        bajas = st.multiselect(
            f"Bajas {equipo_nombre}",
            options=nombres,
            default=[n for n in aus_prev if n in nombres],
            key=f"bajas_{side_key}_{match_key}",
            placeholder="Ninguno (todos disponibles)",
            label_visibility="collapsed"
        )
        # Mostrar XI visual
        pos_order = ['POR', 'DEF', 'MED', 'DEL']
        pos_label = {'POR':'🥅','DEF':'🛡️','MED':'⚙️','DEL':'⚡'}
        por_pos = {p: [] for p in pos_order}
        for j in jugadores:
            por_pos.setdefault(j.get('pos','DEL'), []).append(j)
        html_rows = []
        for pos in pos_order:
            pjs = por_pos.get(pos, [])
            if not pjs: continue
            for j in pjs:
                es_baja = j['nombre'] in bajas
                imp = int(j['impacto_atq']*100)
                if es_baja:
                    estilo = "color:#f38ba8;text-decoration:line-through;opacity:.65"
                    badge = f'<span style="background:#f38ba8;color:#1e1e2e;font-size:.6rem;padding:1px 5px;border-radius:4px;margin-left:4px">BAJA ⬇{imp}%</span>'
                else:
                    estilo = "color:#cdd6f4"
                    badge = f'<span style="color:#a6adc8;font-size:.68rem;margin-left:4px">⬇{imp}%</span>' if imp >= 8 else ''
                html_rows.append(
                    f'<div style="display:flex;align-items:center;padding:3px 0;border-bottom:1px solid #313244">'
                    f'<span style="color:#6c7086;font-size:.65rem;min-width:26px">{pos_label.get(pos,"")}</span>'
                    f'<span style="{estilo};font-size:.8rem">{j["nombre"]}</span>'
                    f'{badge}'
                    f'</div>'
                )
        st.markdown(
            f'<div style="background:#1e1e2e;border:1px solid #313244;border-radius:8px;padding:8px 10px;margin-top:4px">'
            + ''.join(html_rows) +
            '</div>',
            unsafe_allow_html=True
        )
        return bajas

    col_e1, col_e2 = st.columns(2)
    aus_local_new, aus_visita_new = [], []

    with col_e1:
        st.markdown(f"**{local}**")
        if jug_local:
            aus_local_new = _render_xi_col(jug_local, alin_prev['aus_local'], 'loc', local)
        else:
            st.caption("Sin titulares definidos")

    with col_e2:
        st.markdown(f"**{visita}**")
        if jug_visita:
            aus_visita_new = _render_xi_col(jug_visita, alin_prev['aus_visita'], 'vis', visita)
        else:
            st.caption("Sin titulares definidos")

    if st.button("↩️ Restablecer alineación titular", key=f"reset_xi_{match_key}"):
        st.session_state.alineaciones[match_key] = {'aus_local': [], 'aus_visita': []}
        st.rerun()

    # Guardar alineación en session_state en tiempo real
    st.session_state.alineaciones[match_key] = {
        'aus_local': list(aus_local_new),
        'aus_visita': list(aus_visita_new),
    }
    hay_ajuste = bool(aus_local_new or aus_visita_new)

    # ── Botón predecir ────────────────────────────────────────────────
    st.markdown("---")
    if st.button("⚡ Predecir", use_container_width=True, type="primary"):
        with st.spinner("Calculando..."):
            st.session_state.pred_base = calcular_prediccion(
                local, visita, usar_api=usar_api, fecha=fecha_pred)
            if hay_ajuste:
                st.session_state.pred_adj = calcular_prediccion(
                    local, visita, usar_api=usar_api, fecha=fecha_pred,
                    aus_local=aus_local_new, aus_visita=aus_visita_new)
            elif 'pred_adj' in st.session_state:
                del st.session_state['pred_adj']
        st.session_state.ultima_pred = st.session_state.pred_base  # compatibilidad

    # ── Helper: renderizar un bloque de predicción ────────────────────
    def _pred_card(r, titulo, color_titulo="#cba6f7"):
        loc2, vis2 = r['local'], r['visita']
        pL, pE, pV = r['prob_l']*100, r['prob_e']*100, r['prob_v']*100
        top3 = top_marcadores(r['matriz'])
        top3_html = "  ·  ".join([f"<strong>{a}–{b}</strong> ({p:.1f}%)" for a,b,p in top3])
        st.markdown(f"""<div class="card">
          <div style="color:{color_titulo};font-size:.75rem;font-weight:700;margin-bottom:6px">{titulo}</div>
          <div style="text-align:center;font-size:2rem;font-weight:800;color:#cba6f7;margin:.3rem 0">
            {loc2} {r['goles_l']} – {r['goles_v']} {vis2}
          </div>
          <div style="text-align:center;color:#6c7086;font-size:.72rem;margin-bottom:10px">
            λ={r['lam_l']:.2f} · λ={r['lam_v']:.2f}
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:2px">
            <span><span class="tag tag-l">L</span> {loc2}</span>
            <strong style="color:#69db7c">{pL:.1f}%</strong>
          </div>
          <div class="bar-wrap"><div class="bar" style="width:{pL}%;background:#40c057"></div></div>
          <div style="display:flex;justify-content:space-between;margin-bottom:2px">
            <span><span class="tag tag-e">E</span></span>
            <strong style="color:#ffd43b">{pE:.1f}%</strong>
          </div>
          <div class="bar-wrap"><div class="bar" style="width:{pE}%;background:#fab005"></div></div>
          <div style="display:flex;justify-content:space-between;margin-bottom:2px">
            <span><span class="tag tag-v">V</span> {vis2}</span>
            <strong style="color:#ff6b6b">{pV:.1f}%</strong>
          </div>
          <div class="bar-wrap"><div class="bar" style="width:{pV}%;background:#fa5252"></div></div>
          <div style="margin-top:10px;font-size:.78rem;color:#a6adc8">
            🎯 <strong>Top 3 marcadores</strong><br>{top3_html}
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Resultados ────────────────────────────────────────────────────
    if 'pred_base' in st.session_state and st.session_state.pred_base['local'] == local:
        r_base = st.session_state.pred_base
        r_adj  = st.session_state.get('pred_adj')

        if r_adj:
            # Mostrar lado a lado: base vs ajustada
            ausencias_txt = []
            if aus_local_new:  ausencias_txt.append(f"Sin {', '.join(aus_local_new)}")
            if aus_visita_new: ausencias_txt.append(f"Sin {', '.join(aus_visita_new)}")
            col_b, col_a = st.columns(2)
            with col_b:
                _pred_card(r_base, "📊 MODELO BASE — sin ajuste", color_titulo="#89b4fa")
            with col_a:
                nota = " · ".join(ausencias_txt)
                _pred_card(r_adj, f"⭐ CON FACTOR ESTRELLA — {nota}", color_titulo="#fab005")

            # Diferencia de lambdas
            dl = r_adj['lam_l'] - r_base['lam_l']
            dv = r_adj['lam_v'] - r_base['lam_v']
            st.markdown(f"""<div class="card" style="padding:10px 14px;margin-top:0">
              <div style="color:#a6adc8;font-size:.73rem;margin-bottom:4px">IMPACTO DE LAS AUSENCIAS</div>
              <div style="font-size:.85rem">
                λ {local}: {r_base['lam_l']:.2f} → {r_adj['lam_l']:.2f}
                &nbsp;<span style="color:{'#fa5252' if dl<0 else '#69db7c'}">{dl:+.2f}</span>
                &emsp;|&emsp;
                λ {visita}: {r_base['lam_v']:.2f} → {r_adj['lam_v']:.2f}
                &nbsp;<span style="color:{'#fa5252' if dv<0 else '#69db7c'}">{dv:+.2f}</span>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            _pred_card(r_base, "📊 PREDICCIÓN — modelo base")

        # Cuotas detalladas (siempre del base)
        with st.expander(f"📋 Cuotas por casa ({r_base['fuente_cuotas']})"):
            rows_c = [{'Casa':c, f'L ({local})':cL, 'E':cE, f'V ({visita})':cV}
                      for c,(cL,cE,cV) in r_base['cuotas'].items()]
            st.dataframe(pd.DataFrame(rows_c).set_index('Casa'), use_container_width=True)

        with st.expander("🔢 Matriz de marcadores base"):
            st.caption("Filas = goles local · Columnas = goles visita")
            df_mat = pd.DataFrame(
                r_base['matriz']*100,
                index=[f"{local} {i}" for i in range(7)],
                columns=[f"{visita} {j}" for j in range(7)]
            ).round(2)
            st.dataframe(df_mat.style.background_gradient(cmap='Blues'), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# TAB 4: REGISTRAR
# ─────────────────────────────────────────────────────────────────────
with tab_reg:
    st.markdown("### Registrar resultado real")
    st.caption("El modelo se actualiza automáticamente y mejora las siguientes predicciones.")

    # Regla de resultado según fase
    with st.expander("📋 ¿Qué resultado debo cargar?", expanded=False):
        st.markdown("""
**Fase de grupos** → resultado al final de los 90 min + adicionado. Sin más.

**Fase eliminatoria (dieciseisavos en adelante):**
- Partido termina en 90 min con diferencia de goles → cargá ese marcador ✅
- Partido va a alargue (prórroga): terminó empate a 90 min →
  - Si algún equipo gana en los 30 min extra → cargá el marcador al final del alargue ✅
  - Si sigue empatado tras 120 min y se va a penales → cargá el **empate** (el marcador a los 120 min, SIN contar los penales) ✅

> **Los penales no se cuentan.** Un empate a los 120 min es un empate para el tracker de precisión aunque luego lo defina un penal.
        """)

    fase_ko = st.checkbox("⚽ Partido de fase eliminatoria (dieciseisavos en adelante)", value=True)

    equipos = sorted(st.session_state.datos.keys())
    c1,c2,c3,c4 = st.columns([3,1,1,3])
    with c1: r_local  = st.selectbox("Local",  equipos, key="reg_local")
    with c2: g_local  = st.number_input("Goles",0,20,0,key="reg_gl")
    with c3: g_visita = st.number_input("Goles",0,20,0,key="reg_gv")
    with c4: r_visita = st.selectbox("Visita",[e for e in equipos if e!=r_local],key="reg_visita")

    if fase_ko:
        if g_local == g_visita:
            st.info("ℹ️ Marcador empatado → se registra como **empate** (el partido se fue a penales o no terminó con diferencia).")
        else:
            ganador = r_local if g_local > g_visita else r_visita
            st.info(f"ℹ️ **{ganador}** gana — este resultado puede ser al 90 min o al final del alargue (120 min).")

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
            # Usar el marcador predicho (moda) para determinar resultado, no las probs acumuladas
            if gi0 > gj0: pred_res='L'
            elif gi0 == gj0: pred_res='E'
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

    # ── Estadísticas reales del torneo (ESPN) ────────────────────────
    st.markdown("### 📊 Estadísticas reales del torneo")
    st.caption("Disparos al arco, posesión y factores de ajuste calculados desde los partidos jugados.")
    with st.spinner("Cargando estadísticas ESPN..."):
        ts = build_team_stats()
    if ts:
        rows_ts = []
        for eq, v in sorted(ts.items(), key=lambda x: -x[1]['sot_for']):
            rows_ts.append({
                'Equipo': eq,
                'Disp. arco / partido': v['sot_for'],
                'Disp. recibidos / partido': v['sot_ag'],
                'Posesión %': v['poss'],
                'Factor ataque': v['f_atq'],
                'Factor defensa': v['f_def'],
            })
        df_ts = pd.DataFrame(rows_ts).set_index('Equipo')
        st.dataframe(
            df_ts.style
                .background_gradient(subset=['Disp. arco / partido'], cmap='Greens')
                .background_gradient(subset=['Disp. recibidos / partido'], cmap='Reds_r')
                .background_gradient(subset=['Posesión %'], cmap='Blues')
                .format({'Factor ataque':'{:.3f}','Factor defensa':'{:.3f}','Posesión %':'{:.1f}'}),
            use_container_width=True
        )
    else:
        st.info("No hay estadísticas disponibles aún.")

    st.markdown("### Ratings base — todos los equipos")
    rows_b=[{'Equipo':eq,'Ataque':v['ataque'],'Defensa':v['defensa'],'Factor':v['factor']}
            for eq,v in sorted(datos.items(),key=lambda x:-x[1]['ataque'])]
    st.dataframe(
        pd.DataFrame(rows_b).set_index('Equipo')
        .style.background_gradient(subset=['Ataque'],cmap='Greens')
               .background_gradient(subset=['Defensa'],cmap='Reds_r'),
        use_container_width=True
    )
