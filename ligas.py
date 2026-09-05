import streamlit as st
import numpy as np
import requests
from datetime import datetime

st.set_page_config(
    page_title="Ligas · Combinada",
    page_icon="⚽",
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
.stat-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin: 0.3rem 0;
}
.stat-chip {
    flex: 1 1 0;
    min-width: 5.5rem;
    text-align: center;
    padding: 0.35rem 0.4rem;
    border-radius: 8px;
    background-color: rgba(34, 139, 230, 0.12);
}
.stat-chip .stat-label { font-size: 0.68rem; opacity: 0.75; }
.stat-chip .stat-value { font-size: 1.15rem; font-weight: 700; color: #1c7ed6; }
</style>
""", unsafe_allow_html=True)


def _pick_row_html(main_html, stats_html):
    return f'<div class="pick-row"><div class="pr-main">{main_html}</div><div class="pr-stats">{stats_html}</div></div>'


def _stat_chip(label, value):
    return f'<div class="stat-chip"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>'


def _stat_grid(*chips):
    return f'<div class="stat-grid">{"".join(chips)}</div>'

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

st.markdown("### ⚽ Combinada")

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
    "Estrategia", ["Más seguro", "Value bets"], horizontal=True,
    help="Más seguro: resultado más probable de cada partido. Value bets: mejor cuota vs. consenso del "
         "mercado (necesita varias casas de apuestas para tener señal real). Los partidos sugeridos vienen "
         "marcados abajo, en 'Todos los partidos' — desmarca o marca los que quieras.",
)
n_combinada = st.slider("Tamaño de la combinada", min_value=2, max_value=10, value=4,
                         help="Número de partidos sugeridos. Xperto (Polla) usa entre 3 y 10.")
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
# PARTIDOS — un registro por partido, con su resultado más probable
# (se calcula antes del layout para poder usarlo en el Resumen)
# =====================================================================
usar_value = estrategia.startswith("Value")

todos_partidos = []
ids_usados = set()
for liga, partidos in datos_por_liga.items():
    for p in partidos:
        etiqueta = {"L": f"Gana {p['local']}", "E": "Empate", "V": f"Gana {p['visita']}"}
        mejor_k = max(p["prob"], key=p["prob"].get)
        edge = p["prob"][mejor_k] * p["cuota"][mejor_k] - 1
        id_ = f"{liga}||{p['local']} vs {p['visita']}||{p['fecha']}"
        if id_ in ids_usados:
            id_ = f"{id_}||{len(ids_usados)}"  # salvaguarda si dos partidos generan el mismo id
        ids_usados.add(id_)
        todos_partidos.append({
            "id_": id_,
            "Liga": liga,
            "Partido": f"{p['local']} vs {p['visita']}",
            "fecha_dt": p.get("fecha_dt"),
            "Fecha": p["fecha"],
            "Pronóstico": etiqueta[mejor_k],
            "Probabilidad": p["prob"][mejor_k],
            "Cuota": p["cuota"][mejor_k],
            "Edge %": edge * 100,
            "Fuente": p["fuente"],
        })
todos_partidos.sort(key=lambda c: c["Probabilidad"], reverse=True)

# sugeridos = los que vienen pre-marcados en la lista de abajo, según la estrategia.
if usar_value:
    ranking = sorted([c for c in todos_partidos if c["Edge %"] > 0], key=lambda c: c["Edge %"], reverse=True)
else:
    ranking = todos_partidos
sugeridos_ids = {c["id_"] for c in ranking[:n_combinada]}

# elegidos = lo que esté marcado ahora mismo (leído de session_state antes de
# dibujar las casillas, así el Resumen puede ir arriba de todo en la página).
elegidos = [c for c in todos_partidos if st.session_state.get(f"chk_{c['id_']}", c["id_"] in sugeridos_ids)]

# =====================================================================
# RESUMEN — la mejor opción, arriba de todo (sin tener que scrollear
# los partidos por liga para llegar a esto, sobre todo en el celular)
# =====================================================================
st.markdown("## ⚽ Resumen")

if todos_partidos:
    mejor = todos_partidos[0]
    st.info(f"🥇 **{mejor['Partido']}** — {mejor['Pronóstico']} · {fmt_pct(mejor['Probabilidad'] * 100)}")

if elegidos:
    prob_combinada = np.prod([c["Probabilidad"] for c in elegidos])
    cuota_combinada = np.prod([c["Cuota"] for c in elegidos])
    usa_demo = any(c["Fuente"] == "Demo" for c in elegidos)

    with st.container(border=True):
        st.markdown(_stat_grid(
            _stat_chip("Probabilidad", fmt_pct(prob_combinada * 100)),
            _stat_chip("Cuota", f"{cuota_combinada:.2f}"),
        ), unsafe_allow_html=True)
        monto = st.number_input("Monto ($)", min_value=500, step=500, value=5000)
        retorno = monto * cuota_combinada
        st.markdown(_stat_grid(
            _stat_chip("Retorno", f"${retorno:,.0f}"),
            _stat_chip("Ganancia", f"${retorno - monto:,.0f}"),
        ), unsafe_allow_html=True)
        if usa_demo:
            st.caption("⚠️ Modo demo — cuotas simuladas.")

    for c in elegidos:
        st.markdown(_pick_row_html(
            c["Partido"], f"<b>{c['Pronóstico']}</b><br>{fmt_pct(c['Probabilidad'] * 100)}",
        ), unsafe_allow_html=True)
    st.caption("Marca o desmarca partidos en '📋 Todos los partidos' (abajo) para ajustar la combinada.")

    with st.expander("ℹ️ Xperto / Polla Gol"):
        st.markdown(
            "**[Xperto](https://xperto.polla.cl)** es cuota fija, igual que este simulador. **Polla Gol** es un "
            "pozo (el premio depende de cuánta gente acertó esa semana, no de una cuota fija) — ahí no aplica la "
            "cuota combinada, pero sí sirven las probabilidades de cada partido."
        )
    st.caption("Más partidos = menor probabilidad de acertar todos. Juega responsable.")
else:
    st.warning("No hay partidos marcados."
               + (" 'Value bets' necesita partidos con edge positivo." if usar_value else " Marca alguno en '📋 Todos los partidos' (abajo)."))

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
# TODOS LOS PARTIDOS — agrupados por fecha, con casilla para marcar o
# desmarcar cada uno de la combinada. Los sugeridos vienen pre-marcados;
# lo que se lea de acá abajo actualiza el Resumen de arriba en el próximo rerun.
# =====================================================================
st.markdown("---")
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

def _grupo_fecha(c):
    dt = c["fecha_dt"]
    if dt is None:
        return "Sin fecha (demo)"
    return f"{DIAS[dt.weekday()]} {dt.day:02d}/{dt.month:02d}"

with st.expander(f"📋 Todos los partidos ({len(todos_partidos)})", expanded=False):
    st.caption("Pre-marcados = sugeridos por la estrategia elegida. Marca o desmarca los que quieras.")
    grupos = {}
    for c in todos_partidos:
        grupos.setdefault(_grupo_fecha(c), []).append(c)

    def _orden_grupo(nombre):
        dts = [c["fecha_dt"] for c in grupos[nombre] if c["fecha_dt"]]
        return (1, None) if not dts else (0, min(dts))

    for nombre_grupo in sorted(grupos, key=_orden_grupo):
        st.markdown(f"**{nombre_grupo}**")
        for c in grupos[nombre_grupo]:
            emoji = LIGAS[c["Liga"]]["emoji"]
            label = f"{emoji} {c['Partido']} — **{c['Pronóstico']} {fmt_pct(c['Probabilidad'] * 100)}**"
            st.checkbox(label, value=(c["id_"] in sugeridos_ids), key=f"chk_{c['id_']}")
