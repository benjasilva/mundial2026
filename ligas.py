import streamlit as st
import numpy as np
import requests
from datetime import datetime

st.set_page_config(
    page_title="Ligas · Combinada",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# El borde por defecto de st.container(border=True) tiene muy poco contraste
# en tema oscuro — se refuerza con fondo y borde propios, válidos en ambos temas.
st.markdown("""
<style>
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(150, 150, 150, 0.45) !important;
    border-radius: 12px;
    padding: 0.5rem 1rem 1rem 1rem;
    background-color: rgba(150, 150, 150, 0.08);
}
.pick-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0.8rem;
    margin-bottom: 0.4rem;
    border: 1px solid rgba(150, 150, 150, 0.3);
    border-radius: 8px;
    background-color: rgba(150, 150, 150, 0.05);
}
.pick-row .pr-main { font-size: 0.85rem; line-height: 1.35; }
.pick-row .pr-main .pr-sub { opacity: 0.7; }
.pick-row .pr-stats { font-size: 0.8rem; line-height: 1.35; opacity: 0.9; text-align: right; white-space: nowrap; }
</style>
""", unsafe_allow_html=True)


def _pick_row_html(main_html, stats_html):
    return f'<div class="pick-row"><div class="pr-main">{main_html}</div><div class="pr-stats">{stats_html}</div></div>'

# =====================================================================
# LIGAS SOPORTADAS — nombre visible -> sport key de The Odds API
# =====================================================================
LIGAS = {
    "España":             {"sport_key": "soccer_spain_la_liga",              "emoji": "🇪🇸"},
    "Inglaterra":         {"sport_key": "soccer_epl",                        "emoji": "🏴"},
    "Holanda":            {"sport_key": "soccer_netherlands_eredivisie",     "emoji": "🇳🇱"},
    "Italia":             {"sport_key": "soccer_italy_serie_a",              "emoji": "🇮🇹"},
    "Portugal":           {"sport_key": "soccer_portugal_primeira_liga",     "emoji": "🇵🇹"},
    "Francia":            {"sport_key": "soccer_france_ligue_one",           "emoji": "🇫🇷"},
    "Chile":              {"sport_key": "soccer_chile_campeonato",           "emoji": "🇨🇱"},
    "Brasil":             {"sport_key": "soccer_brazil_campeonato",          "emoji": "🇧🇷"},
    "Alemania":           {"sport_key": "soccer_germany_bundesliga",         "emoji": "🇩🇪"},
    "Champions League":   {"sport_key": "soccer_uefa_champs_league",         "emoji": "🏆"},
    "Europa League":      {"sport_key": "soccer_uefa_europa_league",         "emoji": "🇪🇺"},
    "Copa Libertadores":  {"sport_key": "soccer_conmebol_copa_libertadores", "emoji": "🌎"},
    "Copa Sudamericana":  {"sport_key": "soccer_conmebol_copa_sudamericana", "emoji": "🥈"},
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
    "Champions League":  ["Real Madrid", "Manchester City", "Bayern Múnich", "Paris Saint-Germain", "Inter de Milán", "Liverpool"],
    "Europa League":     ["Tottenham", "Atalanta", "Bayer Leverkusen", "AS Roma", "Ajax", "Lazio"],
    "Copa Libertadores": ["Flamengo", "River Plate", "Palmeiras", "Boca Juniors", "Colo-Colo", "Fluminense"],
    "Copa Sudamericana": ["Independiente", "Universidad Católica", "Corinthians", "Racing Club", "LDU Quito", "Nacional"],
}


def init_state():
    if "odds_api_key" not in st.session_state:
        try:
            st.session_state.odds_api_key = st.secrets.get("ODDS_API_KEY", "")
        except Exception:
            st.session_state.odds_api_key = ""


def fmt_pct(x):
    """Formatea un porcentaje con más decimales cuando es muy chico para no mostrar '0.0%' engañoso."""
    if x == 0:
        return "0%"
    if x < 0.1:
        return f"{x:.3f}%"
    if x < 1:
        return f"{x:.2f}%"
    return f"{x:.1f}%"


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
    commence_time = ev.get("commence_time", "")
    fecha = commence_time[:16].replace("T", " ")
    try:
        fecha_dt = datetime.strptime(commence_time, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        fecha_dt = None
    return {
        "local": home, "visita": away, "fecha": fecha, "fecha_dt": fecha_dt,
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
            "local": local, "visita": visita, "fecha": "(demo, sin fecha real)", "fecha_dt": None,
            "prob": probs, "cuota": cuota, "n_casas": 0, "fuente": "Demo",
        })
    return partidos


def filtra_por_fecha(partidos, filtro):
    """Filtra partidos por día de semana según fecha_dt real. Los sin fecha real (demo) se excluyen si hay filtro activo."""
    if filtro == "Todos":
        return partidos
    out = []
    for p in partidos:
        dt = p.get("fecha_dt")
        if dt is None:
            continue
        wd = dt.weekday()  # lunes=0 ... domingo=6
        if filtro.startswith("Fin de semana") and wd in (4, 5, 6, 0):  # vie, sáb, dom, lun
            out.append(p)
        elif filtro.startswith("Entre semana") and wd in (1, 2, 3):  # mar, mié, jue
            out.append(p)
    return out


# =====================================================================
# UI
# =====================================================================
init_state()

st.title("🎯 Ligas · Probabilidades y combinada")
st.caption("España · Inglaterra · Holanda · Italia · Portugal · Francia · Chile · Brasil · Alemania · Champions League · Europa League · Copa Libertadores · Copa Sudamericana")

with st.expander("⚙️ Configuración — The Odds API (opcional pero recomendado)"):
    st.markdown(
        "Cuotas reales vía **[the-odds-api.com](https://the-odds-api.com)** (plan gratis: 500 solicitudes/mes). "
        "Sin API key la app muestra partidos de **demostración** con cuotas simuladas, claramente marcados."
    )
    key_input = st.text_input("API Key", value=st.session_state.odds_api_key, type="password", placeholder="abc123...")
    if key_input != st.session_state.odds_api_key:
        st.session_state.odds_api_key = key_input
        st.cache_data.clear()

    if st.session_state.odds_api_key:
        st.success("✅ API key configurada.")
        if st.button("🔍 Probar conexión y validar ligas"):
            try:
                resp = requests.get(
                    "https://api.the-odds-api.com/v4/sports/",
                    params={"apiKey": st.session_state.odds_api_key.strip(), "all": "true"},
                    timeout=8,
                )
                if resp.status_code == 200:
                    deportes = resp.json()
                    restantes = resp.headers.get("x-requests-remaining", "?")
                    st.success(f"Conexión OK — {len(deportes)} deportes totales (activos e inactivos). Solicitudes restantes este mes: {restantes}")
                    info_por_key = {d["key"]: d for d in deportes}
                    inexistentes, inactivas = [], []
                    for nombre, info in LIGAS.items():
                        d = info_por_key.get(info["sport_key"])
                        if d is None:
                            inexistentes.append(nombre)
                        elif not d.get("active", True):
                            inactivas.append(nombre)
                    if inexistentes:
                        st.error("Estas ligas tienen un sport_key inválido (no existe en la API) — avísame para corregirlo: " + ", ".join(inexistentes))
                    if inactivas:
                        st.info("Estas ligas/copas existen pero están fuera de temporada ahora mismo (van a mostrar demo hasta que arranque el torneo): " + ", ".join(inactivas))
                    if not inexistentes and not inactivas:
                        st.success("Los sport_key de las 13 ligas/copas son válidos y están activos.")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                st.error(f"No se pudo conectar: {e}")

api_key = st.session_state.odds_api_key.strip()

with st.expander(f"🏆 Ligas ({len(LIGAS)} disponibles)", expanded=False):
    ligas_sel = st.multiselect("Ligas", list(LIGAS.keys()), default=list(LIGAS.keys()), label_visibility="collapsed")

estrategia = st.radio(
    "Estrategia", ["Más seguro", "Value bets", "Manual"], horizontal=True,
    help="Más seguro: resultado más probable de cada partido. Value bets: mejor cuota vs. consenso del "
         "mercado (necesita varias casas de apuestas para tener señal real). Manual: eliges cada resultado.",
)
usar_manual = estrategia == "Manual"
if not usar_manual:
    n_combinada = st.slider("Tamaño de la combinada", min_value=2, max_value=10, value=4,
                             help="Número de partidos en la combinada. Xperto (Polla) usa entre 3 y 10.")
filtro_fecha = st.radio(
    "Fecha", ["Todos", "Fin de semana", "Entre semana"], horizontal=True,
    help="Fin de semana = vie-lun. Entre semana = mar-jue. Solo funciona con fecha real (API); en demo no hay fecha.",
)

st.markdown("---")

if not ligas_sel:
    st.warning("Elige al menos una liga.")
    st.stop()

if not api_key:
    st.warning(
        "⚠️ **No hay API key configurada — estás viendo partidos de DEMOSTRACIÓN.** Son equipos de ejemplo "
        "emparejados en orden fijo, no el fixture real de esta semana (el texto 'próxima jornada' es un marcador "
        "genérico, no una fecha real). Configura tu API key en '⚙️ Configuración' arriba para ver los partidos y "
        "cuotas reales."
    )

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

if filtro_fecha != "Todos":
    if not api_key:
        st.warning("El filtro de fecha necesita partidos con fecha real (API) — en modo demo no hay fecha, así que no va a quedar ningún partido.")
    datos_por_liga = {liga: filtra_por_fecha(partidos, filtro_fecha) for liga, partidos in datos_por_liga.items()}

# =====================================================================
# OPCIONES — cada partido expandido en sus 3 resultados posibles
# (se calcula antes del layout para poder mostrarla en la columna derecha)
# =====================================================================
usar_value = estrategia.startswith("Value")

todas_opciones = []
for liga, partidos in datos_por_liga.items():
    for p in partidos:
        etiqueta = {"L": f"Gana {p['local']}", "E": "Empate", "V": f"Gana {p['visita']}"}
        edges = {k: p["prob"][k] * p["cuota"][k] - 1 for k in ("L", "E", "V")}
        for k in ("L", "E", "V"):
            todas_opciones.append({
                "id_": f"{liga}||{p['local']} vs {p['visita']}||{k}",
                "Liga": liga,
                "Partido": f"{p['local']} vs {p['visita']}",
                "Fecha": p["fecha"],
                "Pronóstico": etiqueta[k],
                "Probabilidad": p["prob"][k],
                "Cuota": p["cuota"][k],
                "Edge %": edges[k] * 100,
                "Fuente": p["fuente"],
            })
todas_opciones.sort(key=lambda c: c["Probabilidad"], reverse=True)

# candidatos = el mejor resultado (L/E/V) de cada partido — se usa para el
# pick individual destacado y para las estrategias automáticas.
candidatos = []
vistos = set()
for c in sorted(todas_opciones, key=lambda c: c["Edge %"] if usar_value else c["Probabilidad"], reverse=True):
    if c["Partido"] in vistos:
        continue
    if usar_value and c["Edge %"] <= 0:
        continue
    vistos.add(c["Partido"])
    candidatos.append(c)

def _label(c):
    return f"{fmt_pct(c['Probabilidad'] * 100)} · {c['Liga']} · {c['Partido']} · {c['Pronóstico']} · cuota {c['Cuota']:.2f}"

if usar_manual:
    # Los picks manuales se guardan en session_state por id_ estable — así
    # sobreviven aunque cambies el filtro de fecha, las ligas o la estrategia
    # (antes se perdían porque dependían de la lista de opciones del momento).
    if "picks_manual" not in st.session_state:
        st.session_state.picks_manual = {}
    if "add_ms_counter" not in st.session_state:
        st.session_state.add_ms_counter = 0
else:
    elegidos = candidatos[:n_combinada]

# =====================================================================
# RESUMEN — la mejor opción, arriba de todo (sin tener que scrollear
# los partidos por liga para llegar a esto, sobre todo en el celular)
# =====================================================================
st.markdown("## 🎯 Resumen")

if usar_manual:
    if candidatos:
        mejor_ref = candidatos[0]
        st.info(f"🥇 **{mejor_ref['Partido']}** — {mejor_ref['Pronóstico']} · {fmt_pct(mejor_ref['Probabilidad'] * 100)}")

    visibles = {c["id_"]: c for c in todas_opciones}
    disponibles = [i for i in visibles if i not in st.session_state.picks_manual]

    add_key = f"add_manual_{st.session_state.add_ms_counter}"
    nuevos = st.multiselect(
        "Agregar partido",
        options=disponibles,
        format_func=lambda i: _label(visibles[i]),
        key=add_key,
        help="Busca por equipo o liga. Se guarda aunque cambies filtro o ligas.",
    )
    if nuevos:
        for i in nuevos:
            st.session_state.picks_manual[i] = visibles[i]
        st.session_state.add_ms_counter += 1
        st.rerun()

    elegidos = list(st.session_state.picks_manual.values())

    if elegidos:
        c_vaciar, c_info = st.columns([1, 4])
        if c_vaciar.button("🗑️ Vaciar"):
            st.session_state.picks_manual = {}
            st.rerun()
        c_info.caption(f"{len(elegidos)} guardado(s).")

if elegidos:
    if not usar_manual:
        mejor = candidatos[0]
        st.info(f"🥇 **{mejor['Partido']}** — {mejor['Pronóstico']} · {fmt_pct(mejor['Probabilidad'] * 100)}")

    prob_combinada = np.prod([c["Probabilidad"] for c in elegidos])
    cuota_combinada = np.prod([c["Cuota"] for c in elegidos])
    usa_demo = any(c["Fuente"] == "Demo" for c in elegidos)

    with st.container(border=True):
        m1, m2 = st.columns(2)
        m1.metric("Probabilidad", fmt_pct(prob_combinada * 100))
        m2.metric("Cuota", f"{cuota_combinada:.2f}")
        monto = st.number_input("Monto ($)", min_value=500, step=500, value=5000)
        retorno = monto * cuota_combinada
        r1, r2 = st.columns(2)
        r1.metric("Retorno", f"${retorno:,.0f}")
        r2.metric("Ganancia", f"${retorno - monto:,.0f}")
        if usa_demo:
            st.caption("⚠️ Modo demo — cuotas simuladas.")

    if usar_manual:
        for id_, c in list(st.session_state.picks_manual.items()):
            cq1, cq2 = st.columns([6, 1])
            with cq1:
                st.markdown(_pick_row_html(
                    c["Partido"], f"<b>{c['Pronóstico']}</b><br>{fmt_pct(c['Probabilidad'] * 100)}",
                ), unsafe_allow_html=True)
            with cq2:
                if st.button("✕", key=f"quitar_{id_}", help="Quitar"):
                    del st.session_state.picks_manual[id_]
                    st.rerun()
    else:
        for c in elegidos:
            st.markdown(_pick_row_html(
                c["Partido"], f"<b>{c['Pronóstico']}</b><br>{fmt_pct(c['Probabilidad'] * 100)}",
            ), unsafe_allow_html=True)
        if len(elegidos) < n_combinada:
            st.caption(f"Solo hay {len(elegidos)} disponibles.")

    with st.expander("ℹ️ Xperto / Polla Gol"):
        st.markdown(
            "**[Xperto](https://xperto.polla.cl)** es cuota fija, igual que este simulador. **Polla Gol** es un "
            "pozo (el premio depende de cuánta gente acertó esa semana, no de una cuota fija) — ahí no aplica la "
            "cuota combinada, pero sí sirven las probabilidades de cada partido."
        )
    st.caption("Más partidos = menor probabilidad de acertar todos. Juega responsable.")
elif usar_manual:
    st.info("Todavía no elegiste ningún resultado — agrégalos arriba.")
else:
    st.warning("No hay partidos disponibles."
               + (" 'Value bets' necesita partidos con edge positivo." if usar_value else ""))

# =====================================================================
# REINVERSIÓN PROGRESIVA — apostar en cadena, más seguro primero
# (colapsado por defecto: es una vista alternativa/avanzada, no la respuesta principal)
# =====================================================================
st.markdown("---")
with st.expander("🔁 Reinversión progresiva (apostar en cadena)", expanded=False):
    st.caption("Si ganas, reinviertes en el siguiente paso — puedes cortar en cualquier paso y quedarte con lo ganado.")

    if elegidos:
        cadena = sorted(elegidos, key=lambda c: c["Probabilidad"], reverse=True)
        capital_inicial = monto
        pct_reinversion = st.slider(
            "% que reinviertes en cada paso", min_value=0, max_value=100, value=100, step=10,
            help="100% equivale a la combinada de una sola boleta. Menos baja el riesgo pero también el capital final.",
        )

        en_juego = capital_inicial
        guardado = 0.0
        prob_acum = 1.0
        for i, c in enumerate(cadena, start=1):
            prob_acum *= c["Probabilidad"]
            apostado = en_juego * (pct_reinversion / 100)
            guardado += en_juego - apostado
            en_juego = apostado * c["Cuota"]

            main = (f"<b>Paso {i}</b> · {c['Partido']}<br>"
                    f"<span class='pr-sub'>{c['Pronóstico']} · {fmt_pct(c['Probabilidad'] * 100)} "
                    f"(acumulada {fmt_pct(prob_acum * 100)})</span>")
            stats = f"Apostado ${apostado:,.0f}<br>Si ganas: ${en_juego:,.0f}"
            if guardado > 0.01:
                stats += f"<br>Guardado: ${guardado:,.0f}"
            stats += f"<br><b>Total si cortas: ${en_juego + guardado:,.0f}</b>"
            st.markdown(_pick_row_html(main, stats), unsafe_allow_html=True)
        st.caption(f"Capital inicial ${capital_inicial:,.0f}. Bajar el % de reinversión asegura plata en el camino a costa de un capital final menor.")
    else:
        st.info("Elige partidos para simular la cadena.")

# =====================================================================
# DETALLE POR LIGA — partidos completos de cada liga/copa, colapsado
# (queda abajo de todo: es información de referencia, no la respuesta)
# =====================================================================
st.markdown("---")
with st.expander("📋 Partidos por liga", expanded=False):
    tabs = st.tabs([f"{LIGAS[l]['emoji']} {l}" for l in ligas_sel])
    for tab, liga in zip(tabs, ligas_sel):
        with tab:
            partidos = datos_por_liga[liga]
            if not partidos:
                st.caption("Sin partidos.")
                continue
            for p in partidos:
                main = f"{p['local']} vs {p['visita']}"
                stats = f"L {p['prob']['L'] * 100:.0f}% · E {p['prob']['E'] * 100:.0f}% · V {p['prob']['V'] * 100:.0f}%"
                st.markdown(_pick_row_html(main, stats), unsafe_allow_html=True)
