# Bündeln und seine Grenzen: wann reicht Zusammenlegen, und wann nicht? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-buendelung-demo.streamlit.app/)**

Stück der Linie **Gemeinsam gewinnen** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Wurzel der Linie ist `kuchen-vergroessern-demo` (noch ohne Link); diese Demo ist ihr Nachfolger mit der Kante *idee*: eine andere Quelle des Überschusses, nämlich der Größenvorteil des Zusammenlegens statt verschieden gewichteter Punkte.

Vehikel: n Standorte mit je einem Lager für Ersatzteile und unabhängiger wöchentlicher Nachfrage (Mittelwert 100, Skala 20). Entweder hält jeder einen eigenen Sicherheitsbestand, oder ein gemeinsames Lager bedient alle. „Allianz“ ist hier ein reiner Modellbegriff.

## Kernfrage

*Wie viel Sicherheitsbestand spart eine Allianz, wenn n Standorte ihre Bestände zusammenlegen, und wovon hängt es ab?* Leitidee: Der Vorteil wächst bei leichtem Tail wie √n; bei Tail-Index unter 2 wächst er langsamer, bei endlicher Varianz und hohem Servicegrad (oder schiefer Verteilung wie Lognormal) kann er im getesteten Bereich über √n liegen; Korrelation und Servicegrad ändern ihn.

## Modell

Sicherheitsbestand einzeln SB₁ = Quantil(Servicegrad) − Mittelwert; gebündelt SBₙ = Quantil der Summe der n Nachfragen − n · Mittelwert; **Vorteil V(n) = n · SB₁ / SBₙ**. Verteilungen je Standort (unabhängig, gleich verteilt, Mittelwert 100, Skala 20): Normal, Student-t (ν ab 1,2), Pareto (α von 1,2 bis 4), Lognormal (σ von 0,2 bis 2). Servicegrad 90, 95, 99 oder 99,9 % (die 99,9 % sind eine Erweiterung gegenüber dem Auftrag, um den Tail zu erreichen). Bei Normalverteilung gilt V = √n exakt, mit Gleichkorrelation ρ gilt V = √n / √(1 + (n − 1)ρ). Formeln, Symboltabelle und Literatur stehen im 📐-Expander der App. Keine Umlagerungs- oder Transportkosten, keine Standortwahl.

## Methodik

Die Summe von n Nachfragen wird auf **zwei unabhängigen Wegen** berechnet: (1) **Faltung** der Verteilung auf einem feinen Gitter per FFT (die Gitterdarstellung erhält Masse und Mittelwert jeder Zelle; bei Pareto und Lognormal sind die Quantile unterhalb der Abschneidegrenze exakt bis auf die Gitterdarstellung, bei Normal und Student-t rechnet die FFT zyklisch mit ausgewiesener Randfehler-Schranke) und (2) **Monte Carlo** mit SplitMix64 (400.000 Ziehungen je Zelle, Standardfehler aus 10 Teilstichproben). Der Einzelbestand SB₁ kommt exakt aus der Quantilfunktion. Der **Wachstumsexponent** e(n₁, n₂) ist die Steigung von log V gegen log n; verglichen wird er mit 1/2 (Normal) und (α − 1)/α (Tail-Index α zwischen 1 und 2). Die Aussage n^((α−1)/α) für stabile Nachfrage steht in K. Bimpikis, M. G. Markakis, *Inventory Pooling Under Heavy-Tailed Demand*, Management Science 62(6), 2016, S. 1800 bis 1813; bibliographische Angaben und Kernaussage sind nur anhand der Kurzfassung bestätigt, der Aufsatz selbst wurde nicht gelesen, Zahlen daraus werden nicht übernommen.

## Befunde (gemessen, keine Behauptungen)

Faltung: ein deterministischer Lauf je Zelle. Monte Carlo: Tabelle `bue_mc_tabelle.json`, 5 Zufalls-Seeds (100000 bis 100004), 400.000 Ziehungen je Zelle, erzeugt mit `tools/make_mc_table.py`.

| Frage | Befund (Voraussetzung) | Test |
|---|---|---|
| Normal, Servicegrad 95 % | V = √n für n = 2 bis 256 (Faltung innerhalb 0,3 %) | `test_b01` |
| Schwerer Tail, Pareto α = 1,5, n = 32, Servicegrad 95 % | V = 2,19 statt √32 = 5,66, das sind 39 % davon; Monte Carlo Minimum bis Maximum über 5 Seeds 2,17 bis 2,20 | `test_b02` |
| Student-t ν = 3, n = 32, Servicegrad 95 % | V = 4,87; Monte Carlo 4,86 bis 4,90 | `test_b03` |
| Wie viele der 8 festen Verteilungen haben einen Vorteil V unter 0,95 · √n (mehr als 5 % unter dem Wert √n)? (Servicegrad 95 %) | 5 von 8 bei n = 8, n = 32 und n = 256 (nicht: Normal, Student-t ν = 10, Lognormal σ = 0,5); am niedrigsten Pareto α = 1,5 mit V = 4,12 bei n = 256 (26 % von √256); die Zahl hängt an der Schwelle 5 % und am Servicegrad 95 %, bei 99 % sind es 2 von 8 | `test_b04` |
| Rangfolge bei n = 32, Servicegrad 95 % (aufsteigend) | Pareto 1,5 (2,19) < Pareto 2 (3,34) < Pareto 3 (4,64) < Student-t 3 (4,87) < Lognormal σ = 1 (5,12) < Student-t 10 (5,58) < Normal (5,66) < Lognormal σ = 0,5 (6,20) | `test_b05`, `test_b06` |
| Faltung gegen Monte Carlo | In allen 80 Zellen (8 Verteilungen, Servicegrad 95 und 99 %, n = 2 bis 32) liegt der Monte-Carlo-Median höchstens 1,5 % neben der Faltung und die Faltung innerhalb der Spanne Minimum bis Maximum der 5 Seeds | `test_b07`, `test_b08` |
| Servicegrad (n = 32, 8 Verteilungen) | V steigt von 90 auf 99,9 % bei 7 von 8 Verteilungen; bei Normal gleich. Pareto 1,5: 1,62 auf 3,09; Student-t 3: 4,45 auf 8,45 | `test_b09` |
| Korrelation (Normal) | n = 8, ρ = 0,5: V = 1,33 (statt 2,83); für beliebig viele Standorte höchstens 1/√0,5 = 1,41; ρ = 1: V = 1 | `test_b10` |
| Steigung n = 64 bis 256 gegen (α − 1)/α | Bei Tail-Index unter 2 (Pareto α = 1,2 / 1,5 / 1,8, Student-t ν = 1,5; Servicegrad 95 / 99 / 99,9 %) liegt die Steigung in **12 von 12 Zellen** höchstens 0,05 neben dem Exponenten; Pareto 1,5: 0,309 bei 95 %, 0,332 bei 99,9 % (asymptotisch 0,333); bei 95 % liegt die Steigung bis 0,048 neben dem Exponenten, bei 99,9 % höchstens 0,015 | `test_b11` |
| Niveau gegen die Potenzfunktion durch V(1) = 1 | V(256) liegt bei Tail-Index unter 2 in allen 12 Zellen unter 256^e, in 6 von 12 unter 90 % davon (Pareto 1,5, Servicegrad 95 %: 65 %; 99,9 %: 97 %) | `test_b12` |
| Endliche Varianz, hoher Servicegrad | Pareto α = 3, Servicegrad 99,9 %: Steigung 0,63 (über 1/2), V(256)/√256 = 1,97, Grenzwert für n → ∞ 3,18; Student-t 3, n = 16, 99 %: V = 4,31 > √16. In 6 von 6 Zellen (Pareto 3, Student-t 3, Lognormal σ = 1 bei 99 und 99,9 %) liegt die Steigung im Bereich n = 64 bis 256 über 1/2, bei Student-t 3 und 99 % nur knapp (0,508) | `test_b13`, `test_b18` |
| V unter 1 | Pareto α = 1,2, Servicegrad 90 %, n = 16: V = 0,55 (gemeinsamer Bestand größer als alle Einzelbestände zusammen) | `test_b14` |
| Standardfall (Normal, n = 8, 95 %) | Einzelbestände zusammen 263,2 Stück, gemeinsamer Bestand 93,0 Stück | `test_b15` |
| Gitterfeinheit | Pareto 1,5 / Student-t 1,5 / Lognormal 1 bei n = 64, 99 %: V ändert sich zwischen 2^16 und 2^19 Gitterpunkten um höchstens 0,3 %; bei Student-t 1,5 und Lognormal 1 ist das Gitter in beiden Einstellungen dasselbe, nur bei Pareto 1,5 wird es feiner (nachgerechnet: eine Verfeinerung der Gitterweite von 0,08 auf 0,01 ändert V in den vier geprüften Fällen um weniger als 0,01 %) | `test_b16`, `test_b17` |

Die Zahlen dieser Tabelle stehen als Bänder in `tests/test_claims.py` (Testname in der letzten Spalte).

## Befunde und Korrekturen gegenüber dem Plan

- **Der Hook bestätigt sich für die Steigung, nicht für das Niveau.** Die Vormessung (Pareto α = 1,5, n = 32: 2,17; asymptotisch „(α−1)/α“ hätte 3,17 gegeben) verglich das Niveau mit 32^(1/3). Die gemessene Steigung zwischen n = 64 und n = 256 liegt dagegen bei Tail-Index unter 2 in 12 von 12 Zellen nahe am asymptotischen Exponenten (bei den Pareto-Zellen am nächsten bei 99,9 %). Der Abstand von 3,17 zu 2,19 ist ein Niveauunterschied: V(256) liegt bei Pareto 1,5 und 95 % bei 65 % der Potenzfunktion, die bei V(1) = 1 beginnt.
- **Die Definition des Vorteils zählt.** Mit Quantil minus Mittelwert als Sicherheitsbestand ist V bei endlicher Varianz für große n etwa √n mal SB₁/(z · σ); der Faktor liegt bei hohem Servicegrad und bei schiefer Verteilung über 1 (Pareto 3, 99,9 %: Grenzwert 3,18; Lognormal σ = 0,5 schon bei 95 %: 1,15). V kann also über √n liegen (Lognormal σ = 0,5 bei n = 32, Student-t 3 bei 99 %); das heißt nicht, dass der gemeinsame Bestand klein ist, nur dass er im Verhältnis zum großen Einzelbestand kleiner ausfällt. Und V kann unter 1 liegen.
- **Abweichung vom Auftrag:** Servicegrad 99,9 % als vierte Stufe; `scipy` (nur `scipy.special`) im Kern für Quantilfunktionen und Verteilungsfunktionen (in `requirements.txt`); die Faltung ist die Hauptrechnung der App, Monte Carlo die Nachprüfung auf Klick (n ≤ 32, ohne Korrelation) und die vorgerechnete Tabelle.

## Ehrliche Grenzen

Nur die gewählten Verteilungsfamilien, Skala 20, gleiche Standorte, unabhängige Wochen. Die Korrelation ist nur für Normal gerechnet. Der gemeinsame Bestand ist kostenlos erreichbar (keine Umlagerung). Die Literaturaussage ist nicht an der Quelle geprüft. Deutungen („weil“) stehen in der App als Deutung. Monte Carlo bei 99,9 % hat nur 400 Ziehungen oberhalb des Quantils je Zelle und einen großen Standardfehler. Nur an erfundenen Verteilungen geprüft, nicht an echten Nachfragedaten.

## Tests

138 Tests (`pytest --collect-only -q tests`): Einheiten gegen Handrechnung (Mini-Beispiel: diskrete Nachfrage 0 bis 3, n = 2, Servicegrad 0,9, V = 1,5), **Orakel** (`tests/test_oracle_buendelung.py`: Normal gegen √n, Summe schwerer Verteilungen gegen die Faltung der `scipy.stats`-Dichten und gegen Monte Carlo, Gleichkorrelation gegen simulierte Normalvektoren mit `scipy.stats.multivariate_normal`, Quantilfunktion und exakte Faltung gegen `numpy.quantile` auf der Aufzählung), Randfälle (n = 1, n = 2, ungültige Parameter, nicht positiver gemeinsamer Bestand, Reglerränder), Verdikte (jede Schwelle, jedes Preset), Zahlen (`test_claims.py`), kopierter Zufallsbaustein, Layout, Permalink, AppTest.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `bue_algorithm.py` | Kern: Verteilungen, Gitter, Faltung, Quantile, Monte Carlo, Kennzahlen |
| `bue_constants.py` | Regler-Grenzen, Gitter- und Monte-Carlo-Parameter, Presets |
| `bue_evaluation.py` | Analyse, Experimente, Verdikte |
| `bue_layout.py` | Fester Kopf (Stylesheet) |
| `bue_presets.py` | Permalink (Begrenzen, Einrasten), Presets, Zufalls-Seed-Button |
| `bue_rng.py` | Wörtliche Kopie von `ab-test-demo/abt_rng.py` (SplitMix64) |
| `bue_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `bue_mc_tabelle.json` | Vorgerechnete Monte-Carlo-Tabelle (5 Seeds, 400.000 Ziehungen) |
| `tools/make_mc_table.py` | Erzeugt die Tabelle |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Einheiten, Orakel, Zahlen, Randfälle, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Kopietest |

## Bewusst nicht umgesetzt

Umlagerungs- und Transportkosten, Standortwahl, ungleiche Standorte, zeitliche Abhängigkeit (Autokorrelation), Lieferzeit-Unsicherheit, Korrelation bei schweren Tails. Zur Korrelation von Lagerstandorten gibt es die Messreihe in [standort-bestand-demo](https://github.com/sebastian-hanisch/standort-bestand-demo); gemeinsame gegen getrennte Schlangen zeigt [mmc-queue-demo](https://github.com/sebastian-hanisch/mmc-queue-demo), Flexibilität [kreuzschulung-demo](https://github.com/sebastian-hanisch/kreuzschulung-demo), Prognose und Bestand zusammen [forecast-inventory-demo](https://github.com/sebastian-hanisch/forecast-inventory-demo).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die Monte-Carlo-Tabelle neu erzeugen: `python tools/make_mc_table.py` (rund 1 Minute).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
