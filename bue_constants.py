"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# Nachfrage: Mittelwert 100, Skala 20 (Stück je Woche); V hängt von beidem nicht ab
MEAN_DEMAND, SCALE = 100.0, 20.0

KIND_LABELS = {"normal": "Normal", "t": "Student-t", "pareto": "Pareto", "lognormal": "Lognormal"}
KIND_ORDER = tuple(KIND_LABELS)

# Regler. Komplexität am Maximum: Faltung O(M log M) mit M <= FFT_POINTS_MAX Gitterpunkten je Verteilung und n (Quadrieren), Monte Carlo O(n * N).
N_MIN, N_MAX, DEFAULT_N = 2, 256, 8
NU_MIN, NU_MAX, DEFAULT_NU = 1.2, 30.0, 3.0
ALPHA_MIN, ALPHA_MAX, DEFAULT_ALPHA = 1.2, 4.0, 1.5
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA = 0.2, 2.0, 1.0
RHO_MIN, RHO_MAX, DEFAULT_RHO = 0.0, 1.0, 0.0
Q_OPTIONS = (90.0, 95.0, 99.0, 99.9)   # Servicegrad in Prozent
DEFAULT_Q = 95.0
SEED_MAX, DEFAULT_SEED = 999999, 1
DEFAULT_KIND = "normal"

# Faltung
FFT_H_TARGET = 0.02          # Wunsch-Gitterweite (standardisierte Nachfrage, Skala 1)
FFT_POINTS_MAX = 1 << 18     # höchstens so viele Gitterpunkte bei nicht negativen Verteilungen (n-fache Faltung durch Quadrieren)
FFT_POINTS_MAX_SIGNED = 1 << 20   # bei Normal und Student-t (eine FFT, Potenz im Frequenzraum)
FFT_TAIL_TOLERANCE = 0.1     # vorzeichenbehaftet: Randfehler-Schranke höchstens dieser Anteil von 1 - q
FFT_TAIL_TOLERANCE_EXACT = 0.5   # nicht negativ: verlorene Masse höchstens dieser Anteil von 1 - q (die Quantile unterhalb der Grenze sind dann exakt)

# Monte Carlo
MC_DRAWS = 400_000           # Ziehungen je Zelle
MC_BATCHES = 10              # Teilstichproben für den Standardfehler
MC_N_MAX = 32                # größtes n der Nachprüfung in der App (n * 400.000 Ziehungen)
SWEEP_SEEDS = (100000, 100001, 100002, 100003, 100004)   # feste Zufalls-Seeds für Mehr-Seed-Bänder
N_SWEEP = (2, 4, 8, 16, 32, 64, 128, 256)                # n der Vorteils-Kurven (Faltung); Monte Carlo bis MC_N_MAX
N_MC_TABLE = (2, 4, 8, 16, 32)

COLOR_FFT = "#2F6B65"
COLOR_SQRT = "#9a9a9a"
COLOR_MC = "#d6732b"
COLOR_LINE = "#4c78a8"
KIND_COLORS = {"normal": "#2F6B65", "t": "#4c78a8", "pareto": "#c0504d", "lognormal": "#8c6bb1"}

# Presets (Schlüssel: kind, nu, alpha, sigma, q, n, rho, seed)
_BASE = {"kind": DEFAULT_KIND, "nu": DEFAULT_NU, "alpha": DEFAULT_ALPHA, "sigma": DEFAULT_SIGMA, "q": DEFAULT_Q, "n": DEFAULT_N, "rho": DEFAULT_RHO, "seed": DEFAULT_SEED}
PRESETS = {
    "Normal, 8 Standorte": dict(_BASE),
    "Schwerer Tail (Pareto 1,5), 32 Standorte": {**_BASE, "kind": "pareto", "alpha": 1.5, "n": 32},
    "Student-t mit 3 Freiheitsgraden": {**_BASE, "kind": "t", "nu": 3.0, "n": 16},
    "Hoher Servicegrad 99 %": {**_BASE, "kind": "t", "nu": 3.0, "q": 99.0, "n": 16},
    "Gleichkorrelation 0,5": {**_BASE, "rho": 0.5},
}
PRESET_HELP = {
    "Normal, 8 Standorte": "Der Ausgangsfall: normalverteilte, unabhängige Nachfrage, 8 Standorte, Servicegrad 95 %. Hier gilt die Wurzel-Regel exakt.",
    "Schwerer Tail (Pareto 1,5), 32 Standorte": "Pareto mit Tail-Index 1,5 (Mittelwert endlich, Varianz unendlich), 32 Standorte, Servicegrad 95 %. Die Anzeige vergleicht V mit der Wurzel-Regel.",
    "Student-t mit 3 Freiheitsgraden": "Student-t mit 3 Freiheitsgraden (endliche Varianz, schwerer Tail), 16 Standorte, Servicegrad 95 %.",
    "Hoher Servicegrad 99 %": "Wie das Student-t-Beispiel (3 Freiheitsgrade, 16 Standorte), aber Servicegrad 99 %: das Quantil liegt tiefer im Tail.",
    "Gleichkorrelation 0,5": "Normalverteilte Nachfrage, 8 Standorte, Gleichkorrelation 0,5: die geschlossene Formel mit Korrelation statt der Wurzel-Regel.",
}
