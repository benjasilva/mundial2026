import streamlit as st
import numpy as np
import pandas as pd
import requests
from datetime import datetime

st.set_page_config(
    page_title="Ligas · Combinada",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================
# LIGAS SOPORTADAS — nombre visible -> sport key de The Odds API
# =====================================================================
LIGAS = {
    "España":      {"sport_key": "soccer_spain_la_liga",         "emoji": "🇪🇸"},
    "Inglaterra":  {"sport_key": "soccer_epl",                   "emoji": "🏴"},
    "Holanda":     {"sport_key": "soccer_netherlands_eredivisie","emoji": "🇳🇱"},
    "Italia":      {"sport_key": "soccer_italy_serie_a",         "emoji": "🇮🇹"},
    "Portugal":    {"sport_key": "soccer_portugal_primeira_liga","emoji": "🇵🇹"},
    "Francia":     {"sport_key": "soccer_france_ligue_one",      "emoji": "🇫🇷"},
    "Chile":       {"sport_key": "soccer_chile_campeonato",      "emoji": "🇨🇱"},
    "Brasil":      {"sport_key": "soccer_brazil_campeonato",     "emoji": "🇧🇷"},
    "Alemania":    {"sport_key": "soccer_germany_bundesliga",    "emoji": "🇩🇪"},
}

# Equipos y ratings de referencia para el modo demo (sin API key).
# El rating es solo un orden relativo de fuerza para generar cuotas de ejemplo.
DEMO_EQUIPOS = {
    "España":     ["Real Madrid", "Barcelona", "Atlético de Madrid", "Athletic Club", "Real Sociedad", "Real Betis"],
    "Inglaterra": ["Manchester City", "Arsenal", "Liverpool", "Chelsea", "Tottenham", "Manchester United"],
    "Holanda":    ["Ajax", "PSV Eindhoven", "Feyenoord", "AZ Alkmaar", "FC Twente", "FC Utrecht"],
    "Italia":     ["Inter de Milán", "Juventus", "AC Milan", "Nápoli", "AS Roma", "Atalanta"],
    "Portugal":   ["Benfica", "Porto", "Sporting CP", "Braga", "Vitória de Guimarães", "Famalicão"],
    "Francia":    ["Paris Saint-Germain", "Mónaco", "Marsella", "Lyon", "Lille", "Niza"],
    "Chile":      ["Colo-Colo", "Universidad de Chile", "Universidad Católica", "Palestino", "Cobresal", "Huachipato"],
    "Brasil":     ["Flamengo", "Palmeiras", "São Paulo", "Corinthians", "Botafogo", "Atlético Mineiro"],
    "Alemania":   ["Bayern Múnich", "Bayer Leverkusen", "Borussia Dortmund", "RB Leipzig", "Stuttgart", "Eintracht Frankfurt"],
}


def init_state():
    if "odds_api_key" not in st.session_state:
        st.session_state.odds_api_key = ""


# =====================================================================
# CUOTAS REALES — The Odds API
# =====================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_odds_api(sport_key, api_key):
    resp = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/",
        params={"apiKey": api_key, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _devig(avg_home, avg_draw, avg_away):
    """Quita el margen de la casa promediando cuotas y normalizando a 100%."""
    inv = {"L": 1 / avg_home, "E": 1 / avg_draw, "V": 1 / avg_away}
    overround = sum(inv.values())
    return {k: v / overround for k, v in inv.items()}


def _parse_evento(ev):
    home, away = ev.get("home_team"), ev.get("away_team")
    cuotas = {"L": [], "E": [], "V": []}
    for book in ev.get("bookmakers", []):
        for mkt in book.get("markets", []):
            if mkt["key"] != "h2h":
                continue
            precios = {o["name"]: o["price"] for o in mkt["outcomes"]}
            ch, cd, ca = precios.get(home), precios.get("Draw"), precios.get(away)
            if ch and cd and ca:
                cuotas["L"].append(ch)
                cuotas["E"].append(cd)
                cuotas["V"].append(ca)
    if not cuotas["L"]:
        return None
    avg = {k: sum(v) / len(v) for k, v in cuotas.items()}
    best = {k: max(v) for k, v in cuotas.items()}
    probs = _devig(avg["L"], avg["E"], avg["V"])
    fecha = ev.get("commence_time", "")[:16].replace("T", " ")
    return {
        "local": home, "visita": away, "fecha": fecha,
        "prob": probs, "cuota": best, "n_casas": len(cuotas["L"]), "fuente": "API",
    }


def obtener_partidos_liga(nombre_liga, api_key):
    """Devuelve (lista_partidos, mensaje). Usa cuotas reales si hay API key, si no cae a modo demo."""
    sport_key = LIGAS[nombre_liga]["sport_key"]
    if api_key:
        try:
            eventos = _fetch_odds_api(sport_key, api_key)
            partidos = [p for p in (_parse_evento(ev) for ev in eventos) if p]
            if partidos:
                return partidos, None
            return partidos_demo(nombre_liga), "Sin cuotas disponibles ahora mismo para esta liga — mostrando demo."
        except Exception as e:
            return partidos_demo(nombre_liga), f"Error consultando API ({e}) — mostrando demo."
    return partidos_demo(nombre_liga), None


def partidos_demo(nombre_liga):
    """Fixtures y cuotas simuladas (no reales) para poder explorar la app sin API key."""
    equipos = DEMO_EQUIPOS[nombre_liga]
    ratings = {eq: 1.0 - i * 0.07 for i, eq in enumerate(equipos)}
    partidos = []
    for i in range(0, len(equipos) - 1, 2):
        local, visita = equipos[i], equipos[i + 1]
        sh, sv = ratings[local] * 1.12, ratings[visita]  # ventaja de local
        total = sh + sv
        p_local_raw, p_visita_raw = sh / total, sv / total
        p_empate = 0.24 + 0.06 * (1 - abs(p_local_raw - p_visita_raw))
        p_local = p_local_raw * (1 - p_empate)
        p_visita = p_visita_raw * (1 - p_empate)
        s = p_local + p_empate + p_visita
        probs = {"L": p_local / s, "E": p_empate / s, "V": p_visita / s}
        margen = 1.06
        cuota = {k: round(margen / v, 2) for k, v in probs.items()}
        partidos.append({
            "local": local, "visita": visita, "fecha": "próxima jornada",
            "prob": probs, "cuota": cuota, "n_casas": 0, "fuente": "Demo",
        })
    return partidos


# =====================================================================
# UI
# =====================================================================
init_state()

st.title("🎯 Ligas · Probabilidades y combinada")
st.caption("España · Inglaterra · Holanda · Italia · Portugal · Francia · Chile · Brasil · Alemania")

with st.expander("⚙️ Configuración — The Odds API (opcional pero recomendado)"):
    st.markdown(
        "Cuotas reales vía **[the-odds-api.com](https://the-odds-api.com)** (plan gratis: 500 solicitudes/mes). "
        "Sin API key la app muestra partidos de **demostración** con cuotas simuladas, claramente marcados."
    )
    key_input = st.text_input("API Key", value=st.session_state.odds_api_key, type="password", placeholder="abc123...")
    if key_input != st.session_state.odds_api_key:
        st.session_state.odds_api_key = key_input
        st.cache_data.clear()

api_key = st.session_state.odds_api_key.strip()

ligas_sel = st.multiselect("Ligas a incluir", list(LIGAS.keys()), default=list(LIGAS.keys()))
n_combinada = st.slider("Tamaño de la combinada (número de partidos)", min_value=2, max_value=8, value=4)

st.markdown("---")

if not ligas_sel:
    st.warning("Elegí al menos una liga.")
    st.stop()

datos_por_liga = {}
avisos = []
with st.spinner("Consultando partidos y cuotas..."):
    for liga in ligas_sel:
        partidos, aviso = obtener_partidos_liga(liga, api_key)
        datos_por_liga[liga] = partidos
        if aviso:
            avisos.append(f"**{liga}**: {aviso}")

for a in avisos:
    st.info(a)

# ── Tablas por liga ──────────────────────────────────────────────────
tabs = st.tabs([f"{LIGAS[l]['emoji']} {l}" for l in ligas_sel])
for tab, liga in zip(tabs, ligas_sel):
    with tab:
        partidos = datos_por_liga[liga]
        if not partidos:
            st.warning("No hay partidos disponibles para esta liga.")
            continue
        filas = []
        for p in partidos:
            filas.append({
                "Partido": f"{p['local']} vs {p['visita']}",
                "Fecha": p["fecha"],
                "Local %": round(p["prob"]["L"] * 100, 1),
                "Empate %": round(p["prob"]["E"] * 100, 1),
                "Visita %": round(p["prob"]["V"] * 100, 1),
                "Cuota L": p["cuota"]["L"],
                "Cuota E": p["cuota"]["E"],
                "Cuota V": p["cuota"]["V"],
                "Fuente": p["fuente"],
            })
        df = pd.DataFrame(filas)
        st.dataframe(
            df.style.background_gradient(subset=["Local %", "Empate %", "Visita %"], cmap="Blues"),
            use_container_width=True, hide_index=True,
        )

# =====================================================================
# COMBINADA — top N picks por mayor probabilidad, un pick por partido
# =====================================================================
st.markdown("---")
st.markdown("## 🔮 Combinada sugerida")
st.caption("Se elige, por cada partido, el resultado (Local/Empate/Visita) con mayor probabilidad, y se arma la combinada con los N picks más probables entre todas las ligas seleccionadas.")

candidatos = []
for liga, partidos in datos_por_liga.items():
    for p in partidos:
        mejor = max(p["prob"], key=p["prob"].get)
        etiqueta = {"L": f"Gana {p['local']}", "E": "Empate", "V": f"Gana {p['visita']}"}[mejor]
        candidatos.append({
            "Liga": liga,
            "Partido": f"{p['local']} vs {p['visita']}",
            "Fecha": p["fecha"],
            "Pronóstico": etiqueta,
            "Probabilidad": p["prob"][mejor],
            "Cuota": p["cuota"][mejor],
            "Fuente": p["fuente"],
        })

candidatos.sort(key=lambda c: c["Probabilidad"], reverse=True)
elegidos = candidatos[:n_combinada]

if len(elegidos) < n_combinada:
    st.warning(f"Solo hay {len(elegidos)} partidos disponibles entre las ligas elegidas.")

if elegidos:
    df_comb = pd.DataFrame(elegidos)
    df_comb_view = df_comb.copy()
    df_comb_view["Probabilidad"] = (df_comb_view["Probabilidad"] * 100).round(1).astype(str) + "%"
    st.dataframe(df_comb_view, use_container_width=True, hide_index=True)

    prob_combinada = np.prod([c["Probabilidad"] for c in elegidos])
    cuota_combinada = np.prod([c["Cuota"] for c in elegidos])
    usa_demo = any(c["Fuente"] == "Demo" for c in elegidos)

    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidad conjunta", f"{prob_combinada * 100:.1f}%")
    c2.metric("Cuota combinada", f"{cuota_combinada:.2f}")
    c3.metric("Retorno por $100", f"${cuota_combinada * 100:.0f}")

    if usa_demo:
        st.caption("⚠️ Incluye partidos en modo demo (cuotas simuladas, no reales).")

    st.caption(
        "La probabilidad conjunta es el producto de probabilidades individuales — a más partidos en la combinada, "
        "menor la probabilidad de acertarla completa aunque la cuota suba. Esta información es orientativa, no una "
        "recomendación de apuesta. Jugá responsable."
    )
else:
    st.warning("No hay partidos disponibles para armar una combinada.")
