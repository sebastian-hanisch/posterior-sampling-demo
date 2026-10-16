"""Posterior-Sampling: Sampler, Diagnostik und Laplace-Näherung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Achtes Stück der Linie Inverse Probleme und Regularisierung. Auf der Welt aus Stück 1 bis 7 (Phantom, Gauß-Unschärfe, Rauschen) wird der Posterior nicht mehr in
geschlossener Form gelesen, sondern gesampelt: pCN, Zufallsschritt-Metropolis mit und ohne Posterior-Kovarianz, dazu Diagnostik (R̂, Autokorrelationszeit) und die
Laplace-Näherung. Im Gaußschen Fall ist der exakte Posterior der Maßstab. Kern: smp_model.py, Generator: inverse_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import smp_constants as C
import smp_model as model
from smp_evaluation import (
    Settings,
    analyse,
    burnin_experiment,
    de,
    err_pct,
    iat_text,
    identification_experiment,
    nongauss_experiment,
    noise_label,
    num_text,
    sampler_experiment,
    sci,
    share,
    step_text,
    verdict_burnin,
    verdict_identification,
    verdict_instance,
    verdict_nongauss,
    verdict_sampler,
)
from smp_pdf_export import build_pdf
from smp_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from smp_visualization import (
    build_burnin_chart,
    build_dimension_chart,
    build_identification_chart,
    build_laplace_chart,
    build_nongauss_chart,
    build_recon_chart,
    build_sampler_info_chart,
    build_sd_chart,
    build_table,
    build_trace_chart,
    style_table,
)

st.set_page_config(page_title="Posterior-Sampling – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _sampler():
    return sampler_experiment()


@st.cache_data(show_spinner=SPINNER)
def _burnin():
    return burnin_experiment()


@st.cache_data(show_spinner=SPINNER)
def _identification():
    return identification_experiment()


@st.cache_data(show_spinner=SPINNER)
def _nongauss():
    return nongauss_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🎰 Posterior-Sampling: Wann konvergiert die Kette?")
st.markdown(
    """
**Ein Sampler muss zur Geometrie des Posteriors passen, sonst „konvergiert“ er, ohne es zu zeigen.** Dies ist das achte Stück der Linie **Inverse Probleme und Regularisierung**
und baut auf **Bayes-Sicht: MAP** und **Sparsity und TV** auf: dasselbe Phantom, dieselbe Gauß-Unschärfe, dasselbe Rauschen. Außerhalb des Gaußschen Falls ist der Posterior nur
noch durch **Sampling** zugänglich (Zufallsschritt-Metropolis, pCN). Im Gaußschen Fall kennt man den Posterior exakt und kann die Ketten daran messen. Die Frage ist, **woran man
erkennt, dass eine Kette stimmt**, wie die Wahl des Samplers von der Information in den Daten und der Dimension abhängt und wo die **Laplace-Näherung** (Gaußsche Näherung am MAP)
trägt. Die Welt ist simuliert; die Wahrheit ist bekannt und dient nur als Maßstab.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Wie man eine Kette prüft", expanded=True):
    st.markdown(
        r"""
1. **Sampler.** *pCN* schlägt $x' = \sqrt{1-\beta^2}\,x + \beta\xi$ mit $\xi \sim \mathcal N(0, \tau^2 C)$ vor und nimmt nach der Likelihood an: das Verhältnis hängt nicht von der Dimension ab.
   *Zufallsschritt-Metropolis (RWM)* schlägt $x' = x + h\,L z$ vor; $L = I$ ist *isotrop*, $L L^\top$ = Kovarianz des Posteriors (Laplace-Näherung) ist *vorkonditioniert*.
2. **Annahmerate.** Ein Schritt, der zu groß ist, wird fast immer abgelehnt (Kette steht), einer, der zu klein ist, fast immer angenommen (Kette kriecht). Ziel: rund 0,25 bis 0,3.
3. **Autokorrelationszeit und ESS.** Aufeinanderfolgende Werte sind abhängig: die effektive Stichprobengröße ist $N/\tau_{\mathrm{int}}$ je Funktional.
4. **R̂.** Mehrere Ketten aus verschiedenen Startpunkten sollen dieselbe Verteilung zeigen; $\hat R$ nahe 1 (unter 1,1) heißt: Varianz zwischen und innerhalb der Ketten stimmen überein.
5. **Maßstab.** Im Gaußschen Fall der exakte Posterior (Mittel und Standardabweichung), beim pseudo-Huber-Prior die Laplace-Näherung als Vergleich; beim Laplace-Prior gibt es keine Referenz.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt und Messung**")
    world = st.selectbox("Welche Wahrheit?", C.WORLDS, key="world_select", format_func=lambda v: C.WORLD_LABELS[v],
                         help="Das Phantom ist die Welt der Linie; die Spitzen passen zu den dünn besetzten Priors.")
    n = st.slider("Gitterpunkte n", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Wie viele Messpunkte und Unbekannte.")
    w = st.slider("Breite der Unschärfe w", *bounds("w_slider"), step=C.W_STEP, key="w_slider", format="%.3f",
                  help="Breite des Gauß-Kerns als Anteil der Länge 1. Je breiter, desto stärker fallen die Singulärwerte.")
    noise = st.select_slider("Rauschen (relativ)", options=C.NOISE_CHOICES, key="noise_select", format_func=noise_label,
                             help="Standardabweichung des weißen Rauschens, relativ zum Effektivwert der exakten Daten; je kleiner, desto informativer die Daten.")
    st.markdown("**Prior**")
    prior = st.selectbox("Welcher Prior?", C.PRIORS, key="prior_select", format_func=lambda v: C.PRIOR_LABELS[v],
                         help="Gauß: exakter Posterior als Maßstab. Laplace: MAP gleich L1, keine Laplace-Näherung. Pseudo-Huber: glatt, mit Laplace-Näherung.")
    scale = st.select_slider("Skala b (Laplace und pseudo-Huber)", options=C.SCALE_CHOICES, key="scale_select", format_func=lambda v: de(v, 2),
                             help="Typische Größe der Werte unter dem Prior; kleiner heißt dünner. Beim Gaußschen Prior ohne Wirkung.")
    eps = st.select_slider("Glättung ε (pseudo-Huber)", options=C.EPS_CHOICES, key="eps_select", format_func=lambda v: de(v, 3),
                           help="Je kleiner, desto näher am Betrag und desto weiter vom Gaußschen Fall. Nur beim pseudo-Huber-Prior.")
    st.markdown("**Sampler**")
    sampler = st.selectbox("Welcher Sampler?", C.SAMPLERS, key="sampler_select", format_func=lambda v: C.SAMPLER_LABELS[v],
                           help="pCN braucht einen Gaußschen Prior (sonst läuft RWM mit Kovarianz). Die Schrittweite von pCN und isotropem RWM wird aus einem Gitter so gewählt, dass die Annahmerate nahe dem Ziel liegt.")
    steps = st.select_slider("Schritte je Kette", options=C.STEPS_CHOICES, key="steps_select", help="Die ersten 20 % jeder Kette werden verworfen.")
    chains = st.slider("Ketten", *bounds("chains_slider"), step=1, key="chains_slider", help="Mehrere Ketten aus verschiedenen Startpunkten braucht R̂.")
    start = st.selectbox("Start", C.STARTS, key="start_select", format_func=lambda v: C.START_LABELS[v],
                         help="Im MAP: um die Näherung gestreut (Standardabweichung mal zwei). Aus dem Prior: weit weg vom Posterior, wenn die Daten informativ sind.")
    seed = st.number_input("Zufalls-Seed des Rauschens und der Ketten", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "w_slider": round(float(w), 3), "noise_select": float(noise), "world_select": world, "prior_select": prior, "sampler_select": sampler,
                   "steps_select": int(steps), "chains_slider": int(chains), "start_select": start, "scale_select": float(scale), "eps_select": float(eps), "seed_input": int(seed)})

settings = Settings(int(n), round(float(w), 3), float(noise), world, prior, sampler, int(steps), int(chains), start, float(scale), float(eps), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
if a.singular:
    st.warning(f"Bei dieser Wahl von n und w ist die Matrix in doppelter Genauigkeit numerisch singulär (Kondition {sci(a.cond)}): Die kleinen Singulärwerte sind "
               "Rundungsrauschen, und die Zahlen unten sind nur noch Zahlen.")
if a.fallback:
    st.info("pCN braucht einen Gaußschen Prior; bei diesem Prior läuft stattdessen der Zufallsschritt-Metropolis-Sampler mit der Kovarianz der Näherung.")
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{C.SAMPLER_SHORT[a.sampler_used]}: Annahmerate", de(a.acc, 2), delta=f"Schritt {step_text(a.step)}", delta_color="off",
          help="Anteil angenommener Vorschläge, Mittel über die Ketten; Ziel rund 0,25 bis 0,3.")
c2.metric("R̂ (größter der drei Funktionale)", num_text(max(a.rhats)), delta="Soll: unter 1,1", delta_color="off", help="Gelman-Rubin über die Ketten nach dem Verwerfen.")
c3.metric("Effektive Stichprobengröße (kleinste)", de(min(a.ess), 0), delta=f"IAT bis {iat_text(max(a.iats))}", delta_color="off", help="Summe über die Ketten von N/IAT, je Funktional; die kleinste zählt.")
if a.ref_kind == "exakt":
    c4.metric("Größter Mittelwertfehler", f"{num_text(a.mean_err_sd)} sd", delta=f"sd-Fehler bis {share(a.sd_err)}", delta_color="off", help="Kettenmittel gegen den exakten Gauß-Posterior, in dessen Standardabweichungen.")
elif a.ref_kind:
    c4.metric("Laplace-Näherung: sd-Verhältnis", de(a.sd_ratio_median, 2), delta="Soll: 1", delta_color="off", help="Median von sd der Laplace-Näherung geteilt durch sd der Ketten.")
else:
    c4.metric("Referenz", "keine", delta="Hesse-Matrix des Betrags ist null", delta_color="off", help="Beim Laplace-Prior gibt es weder eine geschlossene Form noch eine Laplace-Näherung.")
st.caption(f"Fehler gegen die Wahrheit: MAP {err_pct(a.err_map)}, Kettenmittel {err_pct(a.err_mean)}; Werte ungleich null: MAP {a.nnz_map} von {settings.n}, Kettenmittel {a.nnz_mean}.")

tab_rec, tab_trace, tab_sd, tab_tab = st.tabs(["Rekonstruktion", "Spuren", "Unsicherheit", "Tabelle"])
with tab_rec:
    st.plotly_chart(build_recon_chart(a), width="stretch", key="recon_chart")
    st.caption("Grün die Wahrheit, orange gestrichelt das MAP, blau das Kettenmittel mit 95-%-Intervall aus den Ketten, schwarz punktiert die Referenz (±1,96 sd). Die Achse ist fest.")
with tab_trace:
    f = st.radio("Funktional", (0, 1, 2), horizontal=True, format_func=lambda i: model.FUNCTIONAL_NAMES[i], key="trace_f")
    st.plotly_chart(build_trace_chart(a, f), width="stretch", key="trace_chart")
    st.caption(f"Autokorrelationszeit dieses Funktionals: {iat_text(a.iats[f])} Schritte, R̂ {num_text(a.rhats[f])}, effektive Stichprobengröße {de(a.ess[f], 0)}. "
               "Eine ruhig aussehende Spur ist kein Beleg: Die Ketten sollen sich überlappen.")
with tab_sd:
    st.plotly_chart(build_sd_chart(a), width="stretch", key="sd_chart")
    st.caption("Standardabweichung aus den Ketten gegen die Referenz je Gitterpunkt (logarithmisch).")
with tab_tab:
    st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch", height=min(40 + 35 * min(settings.n, 40), 560))
    st.caption("Die ersten Punkte der Instanz.")

if st.button("📄 PDF der aktuellen Instanz erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="posterior-sampling.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Ist die Kette konvergiert?")
verdict(verdict_instance(a))
st.caption("Das Urteil kommt aus R̂, der effektiven Stichprobengröße und dem Fehler gegen die exakte Referenz, nicht aus dem Aussehen der Spur.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Welcher Sampler für welche Daten?")
res1 = _sampler()
st.caption(f"Gaußscher Posterior (Prior exponentiell, ℓ = 0,12, τ = 0,5), {res1['steps']} Schritte, eine Kette im Posterior-Mittel; Teil 1 bei n = {res1['n']} mit wachsender Information, "
           f"Teil 2 mit {res1['m']} Messwerten und Rauschen 10 % bei wachsender Dimension.")
st.plotly_chart(build_sampler_info_chart(res1), width="stretch", key="sampler_info_chart")
verdict(verdict_sampler(res1))
st.dataframe(
    [{"Rauschen": noise_label(r["noise"]), "Sampler": C.SAMPLER_SHORT[r["sampler"]], "Schritt": step_text(r["step"]), "Annahme": de(r["acc"], 2),
      "IAT Mittelwert": iat_text(r["iats"][0]), "IAT Punkt": iat_text(r["iats"][1]), "IAT raue Richtung": iat_text(r["iats"][2]), "Mittelwertfehler (sd)": de(r["mean_err"], 2),
      "größter sd-Fehler": share(r["sd_err"])} for r in res1["info"]],
    hide_index=True, width="stretch", height=360)
st.caption("pCN braucht bei informativen Daten winzige Schritte und bewegt sich dann kaum (sd-Fehler bis 0,99: die Kette steht); der isotrope Sampler ebenso. Mit der Kovarianz des Posteriors "
           "als Vorschlag liegt die Autokorrelationszeit bei jedem Rauschen um 100 bis 350.")
st.plotly_chart(build_dimension_chart(res1), width="stretch", key="dimension_chart")
st.dataframe(
    [{"n": d["n"], "pCN Schritt": step_text(d["pcn"]["step"]), "pCN Annahme": de(d["pcn"]["acc"], 2), "pCN IAT Mittelwert": iat_text(d["pcn"]["iats"][0]),
      "RWM Schritt (abgestimmt)": step_text(d["rwm"]["step"]), "RWM Annahme": de(d["rwm"]["acc"], 2), "RWM IAT Mittelwert": iat_text(d["rwm"]["iats"][0]),
      "Annahme mit Schritt von n = 24: pCN": de(d["fixed_pcn_acc"], 2), "RWM": de(d["fixed_rwm_acc"], 2)} for d in res1["dims"]],
    hide_index=True, width="stretch")
st.caption("pCN ist dimensionsrobust (derselbe Schritt, dieselbe Annahme), der isotrope Sampler nicht: sein Schritt muss schrumpfen, und die Autokorrelationszeit des Mittelwerts wächst. "
           "Für Punktwert und raue Richtung sind beide langsam.")

st.subheader("🔬 Experiment 2: Burn-in und Start")
res2 = _burnin()
st.caption(f"Standardinstanz (n = {res2['n']}, Rauschen 0,1 %), Zufallsschritt-Metropolis mit der exakten Kovarianz, vier Ketten à {res2['steps']} Schritte; R̂ nach dem Verwerfen der ersten Schritte.")
st.plotly_chart(build_burnin_chart(res2), width="stretch", key="burnin_chart")
verdict(verdict_burnin(res2))
st.dataframe(
    [{"verworfen": r["discard"], "Start um das MAP: R̂ Punkt": num_text(r["rhat_point"]), "R̂ Mittelwert": num_text(r["rhat_mean"]),
      "Start aus dem Prior: R̂ Punkt": num_text(p["rhat_point"]), "R̂ Mittelwert ": num_text(p["rhat_mean"])} for r, p in zip(res2["rows"]["map"], res2["rows"]["prior"])],
    hide_index=True, width="stretch")
st.caption("Aus dem Prior startende Ketten müssen durch ein Gebiet mit riesiger Likelihood-Strafe; das Verwerfen der ersten Schritte hilft nicht, weil sie den Posterior nicht erreichen. "
           "Das MAP als Start (oder ein Optimierungsschritt davor) ist die praktische Regel.")

st.subheader("🔬 Experiment 3: Was die Daten bestimmen")
res3 = _identification()
st.caption(f"n = {res3['n']}, Prior-Standardabweichung τ = 0,5: Zahl der Richtungen, in denen der Posterior (nach Gewichtung mit der Prior-Kovarianz) schmaler als der halbe Prior ist, "
           "also der Eigenwert λ der prior-gewichteten Hesse-Matrix über 1 liegt.")
st.plotly_chart(build_identification_chart(res3), width="stretch", key="identification_chart")
verdict(verdict_identification(res3))
st.dataframe([{"Prior": r["label"], **{f"Rauschen {noise_label(nz)}": f"{c} von {res3['n']} (Varianzreduktion {de(sv, 1)})" for nz, c, sv in zip(res3["noise"], r["counts"], r["sums"])}}
              for r in res3["rows"]], hide_index=True, width="stretch")
st.caption("Identifikation heißt: Die Daten bestimmen die Richtung stärker als der Prior. Alles andere ist Prior-Schluss, und das gilt auch bei kleinem Rauschen für rund ein Fünftel der "
           "Richtungen. Sampling macht diese Richtungen sichtbar (breite Randverteilungen), ohne sie zu beseitigen.")

st.subheader("🔬 Experiment 4: Nicht-Gaußsche Posteriors")
res4 = _nongauss()
st.caption(f"Spitzen-Welt, n = {res4['n']}, {res4['chains']} Ketten mit der Kovarianz der Näherung als Vorschlag. Oben: Laplace-Prior (MAP gleich L1, 10.000 Schritte je Kette); unten: "
           "pseudo-Huber-Prior (20.000 Schritte) im Vergleich mit der Laplace-Näherung.")
st.plotly_chart(build_nongauss_chart(res4), width="stretch", key="nongauss_chart")
st.plotly_chart(build_laplace_chart(res4), width="stretch", key="laplace_chart")
verdict(verdict_nongauss(res4))
st.dataframe(
    [{"Rauschen": noise_label(r["noise"]), "Skala b": de(r["scale"], 2), "MAP": err_pct(r["map"]), "Posterior-Mittel": err_pct(r["mean"]), "Tikhonov (bester λ)": err_pct(r["tik"]),
      "Werte ungleich null (MAP / Mittel)": f"{r['nnz_map']} / {r['nnz_mean']} von {res4['n']}", "Annahme": de(r["acc"], 2)} for r in res4["laplace"]],
    hide_index=True, width="stretch")
st.dataframe(
    [{"ε": de(r["eps"], 3), "b": de(r["scale"], 2), "Rauschen": noise_label(r["noise"]), "MAP": err_pct(r["map"]), "Mittel": err_pct(r["mean"]), "sd-Verhältnis (Median)": de(r["ratio_median"], 2),
      "min bis max": f"{de(r['ratio_min'], 2)} bis {de(r['ratio_max'], 2)}", "größter |MAP − Mittel| (sd)": de(r["gap_sd"], 2), "R̂ (größter)": de(r["rhat"], 2)} for r in res4["huber"]],
    hide_index=True, width="stretch")
st.caption("Das Posterior-Mittel eines Laplace-Priors ist dicht besetzt, das MAP nicht; beide liegen nahe beieinander, wenn der Prior mild ist. Die Laplace-Näherung trifft beim glatten Prior "
           "auf zwei Prozent, beim dünnen unterschätzt sie die Unsicherheit um den Faktor zwei bis drei. Das MAP hat in der Spitzen-Welt den kleineren Fehler, obwohl das Mittel der "
           "Bayes-Schätzer wäre: die Wahrheit ist keine Prior-Ziehung.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
rows1 = {(r["noise"], r["sampler"]): r for r in res1["info"]}
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Schritt passt zur Geometrie** | pCN bei Rauschen 0,1 %: Schritt {step_text(rows1[(1e-3, 'pcn')]['step'])}, Autokorrelationszeit {iat_text(rows1[(1e-3, 'pcn')]['iats'][1])} für den Punktwert, sd-Fehler {share(rows1[(1e-3, 'pcn')]['sd_err'])}. | Hamilton-Monte-Carlo (nicht gebaut) |
| **Der Start liegt im Posterior** | Aus dem Prior bleibt R̂ nach 4.000 verworfenen Schritten bei {num_text(res2['rows']['prior'][-1]['rhat_point'])} / {num_text(res2['rows']['prior'][-1]['rhat_mean'])}. | – |
| **Die Kette ist lang genug** | Bei Autokorrelationszeiten um 1.000 bis 2.000 geben 20.000 Schritte nur rund 10 bis 20 unabhängige Werte; die sd-Fehler sind dann zu klein oder zu groß. | – |
| **Die Laplace-Näherung trägt** | Beim dünnen Prior liegt die Standardabweichung der Näherung beim {de(res4['huber'][-1]['ratio_median'], 2)}-Fachen der Kette. | Stück 9 (nichtlineare Probleme) |
| **Es gibt eine Referenz** | Beim Laplace-Prior gibt es keine; die Prüfung beruht auf R̂, effektiver Stichprobengröße und Wiederholung mit anderem Start. | – |
| **Ein Prior (exponentiell), zwei Welten, eine Matrix** | Die Zahlen gelten für dieses Beispiel; kein Hamilton-Monte-Carlo, keine Zwischenverteilungen. | Stück 10 (Ensemble-Kalman), 15 (Diffusions-Prior) |
| **Ketten endlicher Länge** | Alle Fehler gegen die Referenz sind Fehler von Monte-Carlo-Schätzungen. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Posterior.** $b = Ax + e$, $e \sim \mathcal N(0, \sigma^2 I)$; die negative Log-Posterior ist $U(x) = \Phi(x) + R(x)$ mit $\Phi(x) = \|Ax - b\|^2/(2\sigma^2)$ und dem Prior-Term
$R(x) = \tfrac12 x^\top (\tau^2 C)^{-1} x$ (Gauß), $\|x\|_1/b$ (Laplace) oder $\sum_i(\sqrt{x_i^2 + \varepsilon^2} - \varepsilon)/b$ (pseudo-Huber). Im Gaußschen Fall ist der Posterior $\mathcal N(m, \Gamma)$
mit $\Gamma = (A^\top A/\sigma^2 + (\tau^2 C)^{-1})^{-1}$ und $m = \Gamma A^\top b/\sigma^2$ (Stück 7).

**Metropolis-Hastings.** Aus $x$ wird $x'$ vorgeschlagen und mit $\min\{1, \pi(x')q(x'\to x)/(\pi(x)q(x\to x'))\}$ angenommen. *RWM:* $q$ symmetrisch, Annahme $\exp(U(x) - U(x'))$;
mit $x' = x + h L z$, $L L^\top$ = Kovarianz der Laplace-Näherung und $h = 2{,}38/\sqrt n$ ist die Annahmerate für Gaußsche Posteriors nahe 0,234. *pCN:* $x' = \sqrt{1-\beta^2}\,x + \beta\xi$ lässt den Prior invariant,
die Annahme ist $\exp(\Phi(x) - \Phi(x'))$ und hängt nur von der Likelihood ab.

**Laplace-Näherung.** $\pi(x \mid b) \approx \mathcal N(x_{\mathrm{MAP}}, H^{-1})$ mit der Hesse-Matrix $H = \nabla^2 U(x_{\mathrm{MAP}})$; im Gaußschen Fall exakt.

**Diagnostik.** Integrierte Autokorrelationszeit $\tau_{\mathrm{int}} = 1 + 2\sum_{t\ge1}\rho_t$ (Sokal-Fenster), effektive Stichprobengröße $N/\tau_{\mathrm{int}}$. $\hat R = \sqrt{\hat V/W}$ mit
$\hat V = \tfrac{N-1}{N}W + \tfrac1N B$ ($W$ Varianz in den Ketten, $B$ zwischen den Ketten).

**Identifikation.** Die Eigenwerte $\lambda_i$ von $\tau^2 R^\top A^\top A R/\sigma^2$ mit $R R^\top = C$: die Posterior-Varianz der Richtung $i$ ist $\tau^2/(1 + \lambda_i)$; $\lambda_i > 1$ heißt halbiert.

**Literatur.** Das Beispiel ist erfunden; pCN geht auf S. L. Cotter, G. O. Roberts, A. M. Stuart und D. White (2013) zurück, die Skalierung $2{,}38/\sqrt n$ auf G. O. Roberts, A. Gelman und W. R. Gilks (1997),
$\hat R$ auf A. Gelman und D. B. Rubin (1992), die Laplace-Näherung für inverse Probleme auf T. Bui-Thanh, O. Ghattas, J. Martin und G. Stadler (2013)
(alles aus dem Gedächtnis, nicht geprüft; die Zahlen hier sind gemessen, nicht der Literatur entnommen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
