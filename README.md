# Posterior-Sampling: Wann konvergiert die Kette? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-posterior-sampling-demo.streamlit.app/)**

Achtes Stück der Linie **Inverse Probleme und Regularisierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Es baut auf [bayes-map-demo](https://github.com/sebastian-hanisch/bayes-map-demo), [sparsity-tv-demo](https://github.com/sebastian-hanisch/sparsity-tv-demo), [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo) und [schlecht-gestellt-demo](https://github.com/sebastian-hanisch/schlecht-gestellt-demo) auf und nutzt dieselbe Welt: ein Phantom aus zwei Hügeln und einem Kasten, gemessen mit einer Gauß-Unschärfe und weißem Rauschen. Die Wahrheit ist bekannt, weil das Phantom selbst die Wahrheit ist; sie dient hier nur als Maßstab.

## Kernfrage

Im Gaußschen Fall ist der Posterior in geschlossener Form bekannt (Stück 7). Sonst bleibt nur **Sampling**. Woran erkennt man, dass eine Kette stimmt? Wie hängt die Wahl des Samplers von der Information in den Daten und von der Dimension ab? Und wo trägt die **Laplace-Näherung** (Gaußsche Näherung am MAP), die den Sampler vorkonditioniert?

## Modell

Generator `inverse_generator.py`, wortgleich in allen Stücken der Linie (Phantom, analytische Unschärfe, Gitter, Rauschen):

- **Daten** b = A x + e, e ~ N(0, σ² I); Standardinstanz n = 48, w = 0,03, Rauschen 0,1 % des Effektivwerts (absolut 3,4·10⁻⁴), Kondition 1,26·10⁴. Zwei Welten: das Phantom und drei **Spitzen**.
- **Priors:** Gauß mit exponentiellem Kern (ℓ = 0,12, τ = 0,5; seine Präzisionsmatrix ist gut konditioniert), **Laplace** (|x|₁/b, das MAP ist L1 aus Stück 6) und **pseudo-Huber** (glatte Näherung des Betrags, mit Glättung ε).
- **Sampler:** pCN (x′ = √(1−β²) x + β ξ, ξ aus dem Prior, nur Gauß), **Zufallsschritt-Metropolis** isotrop und **vorkonditioniert** (Vorschläge mit der Kovarianz des Posteriors beziehungsweise der Laplace-Näherung, Schritt 2,38/√n). Vektorisiert über viele Ketten.
- **Diagnostik:** Annahmerate, Autokorrelationszeit (Sokal-Fenster), effektive Stichprobengröße, R̂ (Gelman-Rubin) und der Fehler gegen den exakten Gauß-Posterior (Mittel und Standardabweichung in sd).

Alles ist erfunden; die Wahrheit ist das simulierte Phantom oder die Spitzen.

## Methodik

- **Start:** im MAP (um die Näherung gestreut, Standardabweichung mal zwei) oder aus dem Prior; die ersten 20 % jeder Kette werden verworfen.
- **Abstimmung:** der Schritt von pCN und isotropem RWM kommt aus einem Gitter, dessen Pilot-Annahmerate dem Ziel (0,3 beziehungsweise 0,234) am nächsten liegt; das Gitter reicht bis 10⁻⁴·⁵, damit auch informative Daten überdeckt sind.
- **Experimente** laufen über feste Zufallsströme (SplitMix64); nie ein einzelner Seed. Bei Autokorrelationszeiten um 1.000 geben 20.000 Schritte nur etwa 20 unabhängige Werte, die sd-Fehler solcher Ketten sind entsprechend unsicher (und zeigen die stehende Kette, nicht die Verteilung).

## Befunde (gemessen, keine Behauptungen)

Standardinstanz (n = 48, w = 0,03), Phantom, wenn nicht anders genannt. Jede Zahl steht in `tests/test_claims.py`.

| Frage | Befund | Test |
|---|---|---|
| Welcher Sampler bei wachsender Information (Experiment 1, 20.000 Schritte, Start im Mittel)? | Rauschen 10 / 1 / 0,1 %: **pCN** braucht β = 0,030 / 0,0031 / 0,00031 (Annahme 0,23), die Autokorrelationszeit des Punktwerts ist 1.131 / 1.316 / 1.339 und der größte sd-Fehler **0,65 / 0,95 / 0,99** (die Kette steht). **Isotropes RWM**: Schritt 0,019 / 0,0027 / 0,00037, IAT bis 2.161. **RWM mit Posterior-Kovarianz**: Schritt 0,34 (= 2,38/√48), Annahme 0,24, IAT **111 / 199 / 109** (Punkt), größter sd-Fehler 0,09 bis 0,13. | `test_information_numbers` |
| Und bei wachsender Dimension (m = 24 Messwerte, Rauschen 10 %, n = 24 … 384)? | **pCN** behält β = 0,03 und die Annahme **0,38 bis 0,40**; die IAT des Mittelwerts bleibt 17 bis 36. **Isotropes RWM** mit dem Schritt von n = 24: Annahme 0,12 bei n = 24, **0,00** bei n = 192 und 384; neu abgestimmt schrumpft der Schritt (0,036 → 0,0098) und die IAT des Mittelwerts steigt bis auf **1.169** (n = 384). Für Punktwert und raue Richtung sind beide Sampler langsam (IAT 200 bis 2.250). | `test_dimension_numbers` |
| Wie wichtig ist der Start (Experiment 2, 4 Ketten × 8.000 Schritte, RWM mit Kovarianz, Rauschen 0,1 %)? | Start um das MAP: R̂ **1,01** (Punkt) und 1,00 (Mittelwert) schon ohne Verwerfen. Start aus dem Prior: R̂ **2,19 / 22,4** ohne Verwerfen und nach 4.000 verworfenen Schritten **3,49 / 43,5**: die Ketten erreichen den Posterior nicht. | `test_burnin_numbers` |
| Was bestimmen die Daten (Experiment 3, Richtungen mit prior-gewichtetem Hesse-Eigenwert über 1, von 48)? | Rauschen 10 / 1 / 0,1 / 0,01 %: Prior der App (exponentiell) **21 / 30 / 38 / 44**; quadratisch-exponentiell (ℓ = 0,05) 19 / 24 / 29 / 33; weißer Prior 25 / 34 / 41 / 48. Auch bei 0,1 % Rauschen bleiben 10 von 48 Richtungen vom Prior bestimmt. | `test_identification_numbers` |
| MAP gegen Posterior-Mittel beim Laplace-Prior (Experiment 4, n = 24, Spitzen, 20 Ketten × 10.000 Schritte)? | Fehler MAP / Mittel / Tikhonov (bester λ): Rauschen 1 %, b = 0,1: 1,4 / 1,4 / 1,5 %; **3 %, b = 0,03: 2,1 / 3,0 / 4,6 %** (14 Werte des MAP ungleich null, das Mittel dicht besetzt mit 23 von 24); 3 %, b = 0,1: 3,3 / 3,9 / 4,6 %. Das MAP liegt nie über dem Mittel. | `test_laplace_prior_numbers` |
| Wo trägt die Laplace-Näherung (pseudo-Huber, 20 Ketten × 20.000 Schritte)? | sd-Verhältnis (Näherung / Kette), Median und Bereich: ε = 0,02, b = 0,1, 3 %: **0,98** (0,94 bis 1,00); ε = 0,002, b = 0,03, 3 %: **0,85** (0,50 bis 1,24); 10 %: **0,46** (0,27 bis 1,43); b = 0,01, 10 %: **0,37** (0,32 bis 0,93). Größter \|MAP − Mittel\| 0,08 / 0,58 / 0,70 / 0,85 sd. | `test_laplace_approximation_numbers` |
| Einzelstichproben (Presets) | Standard (RWM mit Kovarianz): Annahme 0,24, R̂ ≤ 1,01, ESS ≥ 229, Mittelwertfehler 0,22 sd, sd-Fehler 11 %. **pCN bei 0,1 %**: β = 3,1·10⁻⁴, R̂ bis 429, sd-Fehler 269 %. **Isotropes RWM**: Annahme 0,12, R̂ bis 80. **Start aus dem Prior**: R̂ bis 38,6, Mittelwertfehler 21,6 sd. **Laplace-Prior** (40.000 Schritte): R̂ ≤ 1,04, MAP 2,7 % gegen Mittel 5,4 %. **Dünner pseudo-Huber**: sd-Verhältnis 0,38. | `test_standard_preset_numbers` u. a. |
| Mini-Instanz von Hand | 3 × 3-Matrix (τ = 1, σ = 0,1): Posterior-Mittel (1,153, 1,653, 1,153), Eigenwerte 1/101, 1/26, 1/2; Φ(1, 2, 1) = 0 und Φ(0) = 267; R̂ von (0, 2, 0, 2) und (1, 3, 1, 3) = √1,125; Identifikations-Eigenwerte (25, 6,25, 0,25) bei σ = 0,2. | `tests/test_model.py` |

## Befunde und Korrekturen gegenüber dem Plan

- **pCN ist dimensionsrobust, nicht informationsrobust.** Die Gliederung nannte pCN als Kern; die Messung zeigt, dass es bei informativen Daten (der Normalfall der Linie) mit winzigen Schritten kriecht und ohne Vorkonditionierung nicht konvergiert. Der Weg Laplace-Näherung → vorkonditionierter Sampler ist der eigentliche Zusammenhang.
- **Eine stehende Kette sieht ruhig aus.** Mit einem zu groben pCN-Gitter (β ≥ 0,02) kam in der Vormessung „Annahme 0,00, sd-Fehler 1,00“ heraus: die Kette bewegte sich nicht. Das Verdikt der App kommt aus R̂, effektiver Stichprobengröße und dem Fehler gegen die Referenz, nicht aus der Spur.
- **Der Start ist ein eigener Befund.** Ein Entwurf las R̂ = 44 als Fehler des Samplers; die Ursache war der Start aus dem Prior (siehe Experiment 2).
- **Das Posterior-Mittel ist nicht der bessere Punktschätzer in der Spitzen-Welt:** MAP 2,1 % gegen Mittel 3,0 % bei passendem Prior, weil die Wahrheit keine Prior-Ziehung ist (Stück 7 sagt: im Modell ist das Mittel optimal).
- **Das exakte Gibbs-Verfahren dient als Orakel, nicht als Löser:** mit scipy-Quantilen brauchte es 16 Minuten für neun Zellen; das vektorisierte RWM trifft dieselben Mittel (1,4 / 3,0 / 3,9 % gegen 1,5 / 3,0 / 3,9 %) in Sekunden.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Schritt passt zur Geometrie** | pCN bei Rauschen 0,1 %: Schritt 3,1·10⁻⁴, IAT 1.339 für den Punktwert, sd-Fehler 99 %. | Hamilton-Monte-Carlo (nicht gebaut) |
| **Der Start liegt im Posterior** | Aus dem Prior bleibt R̂ nach 4.000 verworfenen Schritten bei 3,49 / 43,5. | – |
| **Die Kette ist lang genug** | Bei Autokorrelationszeiten um 1.000 bis 2.000 geben 20.000 Schritte nur etwa 10 bis 20 unabhängige Werte. | – |
| **Die Laplace-Näherung trägt** | Beim dünnen Prior liegt ihre Standardabweichung beim 0,37-Fachen der Kette. | Stück 9 |
| **Es gibt eine Referenz** | Beim Laplace-Prior nicht; die Prüfung beruht auf R̂, effektiver Stichprobengröße und Wiederholung mit anderem Start. | – |
| **Ein Prior (exponentiell), zwei Welten, eine Matrix** | Die Zahlen gelten für dieses Beispiel; kein Hamilton-Monte-Carlo, keine Zwischenverteilungen. | Stück 10, 15 |
| **Ketten endlicher Länge** | Alle Fehler gegen die Referenz sind Fehler von Monte-Carlo-Schätzungen. | – |

Die Zahlen gelten für diese erfundene Welt und zeigen nichts über echte Messungen.

## Bezug zu den Nachbarn im Portfolio

- [bayes-map-demo](https://github.com/sebastian-hanisch/bayes-map-demo): Stück 7; der exakte Gauß-Posterior ist dort das Ergebnis, hier der Maßstab für die Sampler.
- [sparsity-tv-demo](https://github.com/sebastian-hanisch/sparsity-tv-demo): Stück 6; der Laplace-Prior und sein MAP (L1) aus dem exakten Aktivmengenlöser.
- [kalman-filter-demo](https://github.com/sebastian-hanisch/kalman-filter-demo): der Gaußsche Posterior als Filterschritt.

## Tests

Mini-Instanz von Hand (3 × 3: Posterior, Potentiale, MAP, Hesse-Matrix, Funktionale, Annahme bei Schritt 0 und riesigem Schritt, Burn-in, R̂, IAT, Identifikation), vorkonditioniertes RWM und pCN gegen den exakten Gauß-Posterior (Mittel, Kovarianz, Kolmogorov-Abstand der Randverteilungen), Laplace-Prior-Mittel gegen das exakte Gibbs-Verfahren (scipy), IAT gegen AR(1)-Ketten, R̂, Detailed Balance der Annahmeregeln, ein eindimensionaler nicht-Gaußscher Posterior gegen die numerische Integration, Ketten aus weiter Ferne, Generator, Auswertung, jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler-Grenzen, jede Welt, jeder Prior und Sampler, Start, Spuren-Auswahl, PDF, Permalink), Footer, Presets und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `inverse_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Phantom, analytische Unschärfe, Operator, Daten) |
| `smp_model.py` | Ziele, MAP, Laplace-Näherung, Sampler (pCN, RWM), Diagnostik, Identifikation |
| `smp_evaluation.py` | Welten, Kennzahlen, Experimente, Verdikte |
| `smp_visualization.py`, `smp_pdf_export.py` | Figuren, PDF |
| `smp_constants.py`, `smp_presets.py` | Regler, Presets, Permalink |
| `tests/` | Mini-Instanz, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Hamilton-Monte-Carlo und Langevin-Verfahren:** die naheliegende Antwort auf die Geometrie, ein eigenes Stück an Aufwand.
- **Parallel Tempering, Sequential Monte Carlo:** nur nötig bei mehrgipfligen Posteriors, die diese Welt nicht hat.
- **Gaußsche Priors mit schlecht konditionierter Präzisionsmatrix (SE, Matérn):** die RWM-Dichte braucht sie; die Aussage hängt nicht von der Familie ab.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Inverse Probleme: vom Rauschen zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-inverse-probleme.html).
