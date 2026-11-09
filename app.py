"""Bündeln und seine Grenzen: wann reicht Zusammenlegen, und wann nicht? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Lauffähig mit: streamlit run app.py
"""

import math
import time

import streamlit as st

import bue_algorithm as A
import bue_constants as C
import bue_evaluation as ev
from bue_layout import dunkles_design, kopf_css
from bue_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from bue_visualization import build_correlation, build_exp1, build_exponent, build_main, build_service

st.set_page_config(page_title="Bündeln und seine Grenzen – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _curve(kind, param, q):
    d = A.Dist(kind, param)
    return ev.advantage_curve(d, q)


@st.cache_data(show_spinner=False)
def _sweep(q):
    return ev.sweep_distributions(q)


@st.cache_data(show_spinner=False)
def _service(n):
    return ev.service_matrix(n)


@st.cache_data(show_spinner=False)
def _exponents():
    return ev.exponent_table(ev.exponent_dists())


@st.cache_data(show_spinner=False)
def _mc_table():
    return ev.load_mc_table()


def _kind_label(kind):
    return C.KIND_LABELS[kind]


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in bue_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Bündeln und seine Grenzen: wann reicht Zusammenlegen, und wann nicht?")
st.markdown(
    """
**n Standorte** halten je ein Lager für Ersatzteile. Entweder jeder hält seinen eigenen **Sicherheitsbestand**, oder ein **gemeinsames Lager** bedient alle (eine **Allianz**, hier ein reiner Modellbegriff).
Der **Bündelungsvorteil V** sagt, um welchen Faktor der gemeinsame Bestand kleiner ist als die Summe der Einzelbestände. Die Wurzel-Regel behauptet **V = √n**; sie gilt exakt bei normalverteilter, unabhängiger Nachfrage.
Diese Demo misst, wie weit sie bei **schweren Tails**, **hohem Servicegrad** und **Korrelation** reicht, und wo nicht.

Weiter unten: fünf **🔬 Experimente**, **🚧 Wo die Annahmen enden**, **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Bündeln", expanded=True):
    st.markdown(
        """
1. **Einzeln:** Jeder Standort hat wöchentliche Nachfrage mit Mittelwert 100 und Skala 20 und hält als Sicherheitsbestand das **Quantil des Servicegrads minus den Mittelwert**.
2. **Gebündelt:** Das gemeinsame Lager bedient die Summe der n Nachfragen; sein Sicherheitsbestand ist das Quantil der **Summe** minus n mal Mittelwert.
3. **Vorteil:** V = n · (Einzelbestand) / (gemeinsamer Bestand). Die Summe wird auf zwei Wegen berechnet: durch **Faltung** auf einem feinen Gitter (FFT) und durch **Monte Carlo** (SplitMix64).
4. **Wo die Wurzel-Regel herkommt:** Bei Normalverteilung wächst die Streuung der Summe nur mit √n. Bei schweren Tails bestimmen vermutlich wenige große Ausschläge das Quantil der Summe, die sich weniger wegmitteln (Deutung, nicht gemessen).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Nachfrage je Standort**")
    kind = st.selectbox("Verteilung", C.KIND_ORDER, key="kind_select", format_func=_kind_label,
                        help="Unabhängige, gleich verteilte wöchentliche Nachfrage; Mittelwert 100, Skala 20.")
    nu = st.slider("Freiheitsgrade ν (nur Student-t)", *bounds("nu_slider"), step=0.1, key="nu_slider", disabled=kind != "t",
                   help="Je kleiner ν, desto schwerer der Tail. Bei ν ≤ 1 gäbe es keinen Mittelwert.")
    alpha = st.slider("Tail-Index α (nur Pareto)", *bounds("alpha_slider"), step=0.1, key="alpha_slider", disabled=kind != "pareto",
                      help="P(Nachfrage > x) fällt wie x^(-α). Unter α = 2 ist die Varianz unendlich.")
    sigma = st.slider("Streuung σ (nur Lognormal)", *bounds("sigma_slider"), step=0.1, key="sigma_slider", disabled=kind != "lognormal",
                      help="Standardabweichung des Logarithmus der Nachfrage.")
    st.markdown("**Allianz**")
    n = st.slider("Standorte n", *bounds("n_slider"), key="n_slider", help="Wie viele Standorte ihre Bestände zusammenlegen.")
    q = st.select_slider("Servicegrad (%)", C.Q_OPTIONS, key="q_select", help="Wahrscheinlichkeit, dass die Nachfrage der Woche aus dem Bestand gedeckt wird.")
    rho = st.slider("Gleichkorrelation ρ (nur Normal)", *bounds("rho_slider"), step=0.05, key="rho_slider", disabled=kind != "normal",
                    help="Korrelation zwischen je zwei Standorten. Nur bei Normalverteilung geschlossen berechnet; bei den anderen Verteilungen gilt ρ = 0.")
    st.markdown("**Zufall der Monte-Carlo-Nachprüfung**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Zufalls-Seed würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"kind_select": kind, "nu_slider": float(nu), "alpha_slider": float(alpha), "sigma_slider": float(sigma), "q_select": float(q),
                   "n_slider": int(n), "rho_slider": float(rho), "seed_input": int(seed)})

settings = ev.Settings(kind, float(nu), float(alpha), float(sigma), float(q), int(n), float(rho), int(seed))
d = ev.dist_of(settings)
with st.spinner("Faltung läuft ..."):
    a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
show_v = a.reliable and math.isfinite(a.v)  # unzuverlässiges Gitter: keine Zahl, wie das Verdikt es sagt
show_sbn = a.reliable and math.isfinite(a.sbn)
m1.metric("Bündelungsvorteil V", "–" if not show_v else ev.fmt(a.v), delta=f"{ev.fmt(a.v - a.sqrt_n)} gegen die Wurzel-Regel" if show_v else None, delta_color="off")
m2.metric("Wurzel-Regel", ev.fmt(a.sqrt_n), delta="Referenz", delta_color="off")
m3.metric("Sicherheitsbestand je Standort", f"{ev.fmt(a.sb1, 1)} Stück")
m4.metric("Gemeinsames Lager", "–" if not show_sbn else f"{ev.fmt(a.sbn, 1)} Stück",
          delta=None if not show_sbn else f"{ev.fmt(a.saving, 1)} Stück weniger als {a.n} Einzelbestände", delta_color="off")

if d.kind == "normal":
    curve = {k: A.advantage_normal(k, settings.rho) for k in C.N_SWEEP}
else:
    with st.spinner("Faltung für alle n ..."):
        curve = _curve(d.kind, d.param, settings.q / 100.0)["v"]

mc_state = st.session_state.get("mc_result")
mc_ok = mc_state is not None and mc_state["key"] == (d, settings.q, settings.n, settings.seed)
can_mc = d.kind != "normal" or settings.rho == 0.0
mc_clicked = st.button(f"🎲 Monte Carlo nachprüfen ({C.MC_DRAWS:,} Ziehungen je Zelle)".replace(",", "."), disabled=not can_mc or settings.n > C.MC_N_MAX,
                       help=f"Rechnet V für die aktuelle Einstellung ein zweites Mal mit Zufallsziehungen (SplitMix64). Nur bis n = {C.MC_N_MAX}, nur ohne Korrelation.")
if mc_clicked:
    t0 = time.perf_counter()
    with st.spinner(f"Monte Carlo läuft: {settings.n} mal {C.MC_DRAWS:,} Ziehungen (einige Sekunden) ...".replace(",", ".")):
        v_mc, se_mc = A.mc_advantage_with_error(d, settings.n, settings.q / 100.0, C.MC_DRAWS, settings.seed)
    st.session_state["mc_result"] = {"key": (d, settings.q, settings.n, settings.seed), "v": v_mc, "se": se_mc, "secs": time.perf_counter() - t0}
    mc_state, mc_ok = st.session_state["mc_result"], True
if not can_mc:
    st.caption("Mit Korrelation ist nur die geschlossene Rechnung verfügbar, die Nachprüfung ist aus.")
elif settings.n > C.MC_N_MAX:
    st.caption(f"Die Monte-Carlo-Nachprüfung läuft bis n = {C.MC_N_MAX}; bei größerem n rechnet nur die Faltung.")

st.plotly_chart(
    build_main(curve, ev.dist_label(d), settings.n, a.v, (settings.n, mc_state["v"], mc_state["se"]) if mc_ok else None),
    width="stretch", key="main_chart",
)
st.caption("Beide Achsen logarithmisch. Gestrichelt: Wurzel-Regel. Linie: Faltung für n = 2 bis 256 beim gewählten Servicegrad; gelb: aktuelles n.")
if mc_ok:
    diff = mc_state["v"] - a.v
    st.info(f"Monte Carlo: V = {ev.fmt(mc_state['v'], 3)} ± {ev.fmt(mc_state['se'], 3)} (Standardfehler aus {C.MC_BATCHES} Teilstichproben, Rechenzeit {ev.fmt(mc_state['secs'], 1)} s); "
            f"Faltung: V = {ev.fmt(a.v, 3)}; Unterschied {ev.fmt(diff, 3)}, das sind {ev.fmt(abs(diff) / mc_state['se'], 1) if mc_state['se'] > 0 else '–'} Standardfehler.")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie viel Sicherheitsbestand spart eine Allianz, und wovon hängt es ab?")
kind_v, text = ev.verdict_kernfrage(a)
getattr(st, kind_v)(text)
if d.kind != "normal":
    st.caption("Faltung der Gitterverteilung (Mittelwert erhaltend); Gittergröße und Randfehler siehe Expander „Annahmen, Grenzen und bekannte Probleme“.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

mc_table = _mc_table()

st.subheader("🔬 Experiment 1: Vorteil über n gegen die Wurzel-Regel")
st.caption(f"Acht feste Verteilungen beim Servicegrad {str(settings.q).replace('.', ',')} % (Seitenleiste). Linien: Faltung. Rauten mit Spanne: Monte Carlo, Median und Minimum bis Maximum über "
           f"{len(mc_table['seeds'])} Zufalls-Seeds mit je {mc_table['draws']:,} Ziehungen (vorgerechnet, nur bei den Servicegraden 95 % und 99 %, n bis {C.MC_N_MAX}).".replace(",", "."))
with st.spinner("Faltung für acht Verteilungen ..."):
    rows = _sweep(settings.q / 100.0)
mc_bands = {}
if settings.q in (95.0, 99.0):
    for r in rows:
        band = {n_: ev.mc_band(mc_table, r["dist"], settings.q / 100.0, n_) for n_ in C.N_MC_TABLE}
        mc_bands[r["label"]] = {k: v for k, v in band.items() if v}
st.plotly_chart(build_exp1(rows, mc_bands), width="stretch", key="exp1_chart")
st.dataframe(
    [{"Verteilung": r["label"], **{f"n = {k}": round(float(r['v'][k]), 2) for k in C.N_SWEEP}} for r in rows] + [{"Verteilung": "Wurzel-Regel √n", **{f"n = {k}": round(math.sqrt(k), 2) for k in C.N_SWEEP}}],
    hide_index=True, width="stretch",
)
kind_v, text = ev.verdict_exp1(rows, settings.n if settings.n in C.N_SWEEP else 32)
getattr(st, kind_v)(text)

st.markdown("---")
st.subheader("🔬 Experiment 2: Welche Verteilung bremst das Bündeln?")
n2 = settings.n if settings.n in C.N_SWEEP else 32
st.caption(f"Rangfolge bei n = {n2} und Servicegrad {str(settings.q).replace('.', ',')} % (bei n außerhalb der Tabelle gilt n = 32).")
kind_v, text = ev.verdict_exp2(rows, n2, len(mc_table["seeds"]))
getattr(st, kind_v)(text)
if settings.q in (95.0, 99.0) and n2 in C.N_MC_TABLE:
    st.dataframe(
        [{"Verteilung": r["label"], "Faltung V": round(float(r["v"][n2]), 3),
          "Monte Carlo Minimum": round(ev.mc_band(mc_table, r["dist"], settings.q / 100.0, n2)[0], 3),
          "Monte Carlo Median": round(ev.mc_band(mc_table, r["dist"], settings.q / 100.0, n2)[1], 3),
          "Monte Carlo Maximum": round(ev.mc_band(mc_table, r["dist"], settings.q / 100.0, n2)[2], 3)} for r in sorted(rows, key=lambda r: r["v"][n2])],
        hide_index=True, width="stretch",
    )

st.markdown("---")
st.subheader("🔬 Experiment 3: Wächst der Tail-Effekt mit dem Servicegrad?")
st.caption(f"V je Verteilung und Servicegrad bei n = {settings.n} (Faltung).")
with st.spinner("Faltung für acht Verteilungen ..."):
    service = _service(settings.n)
st.plotly_chart(build_service(service), width="stretch", key="service_chart")
kind_v, text = ev.verdict_exp3(service)
getattr(st, kind_v)(text)
st.caption("V steht im Verhältnis zum Einzelbestand desselben Servicegrads. Ein hoher Servicegrad macht bei schwerem Tail schon den Einzelbestand groß; dann ist der Faktor V größer; "
           "der gemeinsame Bestand in Stück ist aber ebenfalls größer. Das ist eine Deutung der Definition, keine eigene Messung.")

st.markdown("---")
st.subheader("🔬 Experiment 4: Korrelation (Normal, geschlossen)")
st.caption("Bei Gleichkorrelation ρ gilt V = √n / √(1 + (n − 1)ρ), unabhängig vom Servicegrad. Die Messreihe zur Korrelation steht in der Demo „standort-bestand-demo“ "
           "(Lagerstandorte mit Korrelation), hier wird sie nicht wiederholt.")
st.plotly_chart(build_correlation(ev.correlation_table()), width="stretch", key="corr_chart")
kind_v, text = ev.verdict_exp4(settings.n, settings.rho)
getattr(st, kind_v)(text)

st.markdown("---")
st.subheader("🔬 Experiment 5: Wächst V wie n^((α−1)/α)?")
st.caption("Gemessene Steigung von log V gegen log n zwischen n = 64 und n = 256 (Faltung), je Verteilung und Servicegrad, gegen den Exponenten (a − 1)/a für Tail-Index a unter 2 "
           "und 1/2 bei endlicher Varianz. Die Spalte „Niveau“ ist V(256) geteilt durch 256^Exponent, also gegen eine Potenzfunktion, die bei V(1) = 1 beginnt.")
with st.spinner("Faltung für sieben Verteilungen und drei Servicegrade ..."):
    ex_rows = _exponents()
st.plotly_chart(build_exponent(ex_rows), width="stretch", key="exp_chart")
st.dataframe(
    [{"Verteilung": r["label"], "Servicegrad": f"{str(r['q'] * 100).replace('.0', '').replace('.', ',')} %", "Steigung n = 16 bis 64": round(r["e_low"], 3),
      "Steigung n = 64 bis 256": round(r["e_high"], 3), "asymptotisch": round(r["asymptote"], 3), "Niveau V(256) / 256^e": round(r["level"], 2),
      "V(256) / √256": round(r["factor"], 2),
      "Grenzwert V / √n (n → ∞)": "–" if not math.isfinite(r["limit_factor"]) else ev.fmt(r["limit_factor"])} for r in ex_rows],
    hide_index=True, width="stretch",
)
kind_v, text = ev.verdict_exp5(ex_rows)
getattr(st, kind_v)(text)
st.caption("Der Exponent gilt hier für die Steigung, nicht für das Niveau: bei Tail-Index unter 2 liegt V(256) deutlich unter der Potenzfunktion, die bei V(1) = 1 beginnt (Spalte „Niveau“). "
           "Bei endlicher Varianz geht V/√n für n gegen unendlich gegen den Grenzwert der letzten Spalte (Zentraler Grenzwertsatz); im Bereich n ≤ 256 liegt die gemessene Steigung "
           "bei den Servicegraden 99 % und 99,9 % über 1/2. Deutung (nicht gemessen): Der Grenzwert von V/√n liegt über 1, und das Summenquantil nähert sich der Normal-Näherung erst langsam.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Unabhängige Nachfrage der Standorte** | Bei Normalverteilung und Gleichkorrelation ρ sinkt V auf √n / √(1 + (n − 1)ρ), für viele Standorte höchstens bis 1/√ρ (Experiment 4). | [standort-bestand-demo](https://github.com/sebastian-hanisch/standort-bestand-demo) (Risk Pooling mit Korrelation) |
| **Gleiche Standorte** | Ungleiche Standorte (verschiedene Mittelwerte und Streuungen) werden hier nicht gerechnet. | – |
| **Leichter Tail** | Bei Tail-Index unter 2 liegt V unter √n; bei höherem Index hängt es vom Servicegrad ab (Experimente 1 bis 3). Asymptotisch, für stabile Nachfrage, wächst V bei Tail-Index unter 2 wie n^((α−1)/α) (Experiment 5); bei n ≤ 256 liegt V teils bei 56 bis 97 % der Potenzfunktion. | – |
| **Keine Kosten des Zusammenlegens** | Umlagerung, Transport und Lieferzeit gehen nicht ein; ein gemeinsames Lager ist hier gratis erreichbar. | [standort-bestand-demo](https://github.com/sebastian-hanisch/standort-bestand-demo) (Standortwahl mit Bestandskosten) |
| **Bestand statt Kapazität** | Zusammenlegen von Bedienkapazität statt Beständen (gemeinsame gegen getrennte Schlangen) ist ein eigenes Modell. | [mmc-queue-demo](https://github.com/sebastian-hanisch/mmc-queue-demo), [kreuzschulung-demo](https://github.com/sebastian-hanisch/kreuzschulung-demo) (Flexibilität) |
| **Nachfrage unabhängig von der Zeit** | Autokorrelation über die Wochen ist nicht modelliert; Prognose und Bestand zusammen: | [forecast-inventory-demo](https://github.com/sebastian-hanisch/forecast-inventory-demo) |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen**
- n Standorte mit unabhängiger, gleich verteilter wöchentlicher Nachfrage (Mittelwert 100, Skala 20); nur bei Normalverteilung optional Gleichkorrelation ρ.
- Sicherheitsbestand = Quantil des Servicegrads minus Mittelwert; V = n · SB₁ / SBₙ. Das ist eine Definition: Mit der Standardabweichung als Maß (SB = z · σ mit gleichem z einzeln und gebündelt) wäre V bei unabhängiger Nachfrage mit endlicher Varianz per Konstruktion √n.
- Ein gemeinsames Lager ohne Kosten und Verzögerung des Zusammenlegens.

**Grenzen**
- Die Faltung rechnet auf einem Gitter. Bei Pareto und Lognormal (nicht negativ) ändert eine feinere Gitterweite die Quantile unterhalb der Abschneidegrenze kaum (nachgerechnet: V unter 0,01 %); die verlorene Masse oberhalb der Grenze ist höchstens die Hälfte von 1 − Servicegrad, sonst wird die Grenze verdoppelt.
  Bei Normal und Student-t rechnet die FFT zyklisch; als Randfehler gilt n · P(|Z| > L), das höchstens ein Zehntel von 1 − Servicegrad sein soll. Reicht das Gitter nicht, zeigen die Kennzahlen kein Ergebnis.
- Monte Carlo: Standardfehler aus 10 Teilstichproben, bei Servicegrad 99,9 % liegen nur 400 der 400.000 Ziehungen oberhalb des Quantils, der Standardfehler ist dort groß.
- Der Einzelbestand SB₁ ist exakt (Quantilfunktion), nicht simuliert.
- „V < 1“ ist möglich: Es tritt auf, wenn der Einzelbestand sehr klein ist (Einzelquantil nur knapp über dem Mittelwert, Pareto 1,2, 90 %: 0,81 gegen 6,0) und der gemeinsame Bestand mit n schneller wächst als n · SB₁ (Ursache nicht gemessen).

**Bekannte Probleme und Unsicherheiten**
- Alle Befunde gelten für die gewählten Verteilungsfamilien und Skala 20; echte Nachfrage ist nur an erfundenen Beispielen geprüft.
- Die Literaturaussage (n^((α−1)/α)) ist nur aus der Kurzfassung des Aufsatzes bestätigt, der Aufsatz selbst wurde nicht gelesen.
- Nicht mit echtem Mausklick getestet: Hervorhebung des aktuellen n im Diagramm folgt aus dem Regler.
        """
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Standorte mit unabhängiger, gleich verteilter Nachfrage $D_i$ (Mittelwert $\mu$) und ein Servicegrad $q$. Gesucht ist der Bündelungsvorteil $V(n)$.

| Symbol | Bedeutung |
|---|---|
| $n$ | Zahl der Standorte |
| $q$ | Servicegrad (Quantil) |
| $\mu$ | Mittelwert der Nachfrage eines Standorts |
| $F$ | Verteilungsfunktion der Nachfrage eines Standorts |
| $F_n$ | Verteilungsfunktion der Summe von $n$ Nachfragen |
| $SB_1$, $SB_n$ | Sicherheitsbestand einzeln bzw. gebündelt |
| $V$ | Bündelungsvorteil |
| $\rho$ | Gleichkorrelation (nur Normal) |
| $\alpha$ | Tail-Index: $P(D > x) \sim x^{-\alpha}$ |

**Sicherheitsbestand und Vorteil.**

$$SB_1 = F^{-1}(q) - \mu$$

$$SB_n = F_n^{-1}(q) - n\mu$$

$$V(n) = \frac{n \, SB_1}{SB_n}$$

**Normalverteilung.** Die Summe ist normalverteilt mit Standardabweichung $\sigma\sqrt{n}$, also

$$V(n) = \sqrt{n}$$

Bei Gleichkorrelation $\rho$ ist die Varianz der Summe $n\sigma^2(1 + (n-1)\rho)$, also

$$V(n) = \frac{\sqrt{n}}{\sqrt{1 + (n-1)\rho}} \;\le\; \frac{1}{\sqrt{\rho}}$$

**Schwere Tails.** Für stabile Nachfrage mit Tail-Index $\alpha \in (1, 2)$ wächst die Summe wie $n^{1/\alpha}$ (Quantil minus $n\mu$), der Einzelbestand ist konstant, also

$$V(n) \sim n^{1 - 1/\alpha} = n^{(\alpha-1)/\alpha}$$

Das ist eine asymptotische Aussage für $n \to \infty$; die Demo misst, wie weit sie für $n \le 256$ reicht (Experiment 5). Bei endlicher Varianz gilt für $n \to \infty$ mit dem Normal-Quantil $z_q$ und der Standardabweichung $\sigma$

$$\frac{V(n)}{\sqrt{n}} \to \frac{SB_1}{z_q \, \sigma}$$

**Faltung auf dem Gitter.** Die Zellen $[jh, (j+1)h)$ geben ihre Masse $m_j$ und ihren Mittelwert an die Knoten $jh$ und $(j+1)h$ weiter, so dass Masse und Mittelwert erhalten bleiben; mit dem Teilmoment $M(x) = \int_{-\infty}^{x} t f(t)\,dt$ ist der Anteil auf dem rechten Knoten

$$w_j = \frac{M((j+1)h) - M(jh) - jh\, m_j}{h}$$

Die Verteilung der Summe ist die $n$-fache Faltung dieser Gittermassen $p$ (per FFT, bei nicht negativen Verteilungen durch wiederholtes Quadrieren mit Abschneiden):

$$p^{*n} = \mathcal{F}^{-1}\!\left(\mathcal{F}(p)^{n}\right)$$

**Literatur.** K. Bimpikis, M. G. Markakis: *Inventory Pooling Under Heavy-Tailed Demand*, Management Science 62(6), 2016, S. 1800 bis 1813 (Bibliographische Angaben und Kernaussage zu n^((α−1)/α) nur aus der Kurzfassung bestätigt; Aufsatz nicht gelesen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
