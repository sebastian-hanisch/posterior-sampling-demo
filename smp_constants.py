"""Konstanten: Regler-Grenzen, feste Annahmen, Experimente, Farben und Presets."""

# Regler. Rechenaufwand am Maximum: bis 20 Ketten x 40.000 Schritte, vektorisiert über die Ketten (Python-Schleife über die Schritte, rund 5 s); die Experimente
# nutzen feste Ströme.
N_MIN, N_MAX, N_STEP, DEFAULT_N = 12, 96, 4, 48
W_MIN, W_MAX, W_STEP, DEFAULT_W = 0.01, 0.05, 0.005, 0.03        # Breite der Unschärfe (Anteil der Länge 1)
NOISE_CHOICES = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1)                   # Rauschen relativ zum Effektivwert der exakten Daten
DEFAULT_NOISE = 1e-3
WORLDS = ("phantom", "spitzen")
WORLD_LABELS = {"phantom": "Phantom (zwei glatte Hügel und ein Kasten)", "spitzen": "Spitzen (drei Impulse)"}
WORLD_SHORT = {"phantom": "Phantom", "spitzen": "Spitzen"}
PRIORS = ("gauss", "laplace", "huber")
PRIOR_LABELS = {"gauss": "Gauß (exponentieller Kern, rau)", "laplace": "Laplace (Summe der Beträge)", "huber": "Pseudo-Huber (glatte Näherung des Betrags)"}
PRIOR_SHORT = {"gauss": "Gauß", "laplace": "Laplace", "huber": "Pseudo-Huber"}
SAMPLERS = ("pcn", "rwm_iso", "rwm_cov")
SAMPLER_LABELS = {"pcn": "pCN (Schritt aus dem Prior, nur Gauß)", "rwm_iso": "Zufallsschritt-Metropolis, isotrop", "rwm_cov": "Zufallsschritt-Metropolis mit Kovarianz des Posteriors (Laplace)"}
SAMPLER_SHORT = {"pcn": "pCN", "rwm_iso": "RWM isotrop", "rwm_cov": "RWM mit Kovarianz"}
STARTS = ("map", "prior")
START_LABELS = {"map": "Im MAP (überdispers um die Näherung)", "prior": "Aus dem Prior gezogen"}
SCALE_CHOICES = (0.01, 0.03, 0.1, 0.3)                           # Skala b des Laplace- und pseudo-Huber-Priors
DEFAULT_SCALE = 0.1
EPS_CHOICES = (0.02, 0.002)                                      # Glättung ε des pseudo-Huber-Priors
DEFAULT_EPS = 0.02
STEPS_CHOICES = (1000, 3000, 10000, 20000, 40000)
DEFAULT_STEPS = 10000
CHAINS_MIN, CHAINS_MAX, DEFAULT_CHAINS = 4, 20, 4
SEED_MAX, DEFAULT_SEED = 999999, 11

# feste Modellannahmen (Prior-Kern: exponentiell, gut konditionierte Präzisionsmatrix)
PRIOR_ELL = 0.12                                                 # Korrelationslänge des Gaußschen Priors
PRIOR_TAU = 0.5                                                  # Standardabweichung des Gaußschen Priors
BURN_FRACTION = 0.2                                              # verworfener Anteil jeder Kette
THIN_TARGET = 400                                                # gespeicherte Werte je Kette für Verteilungen
BLOCK = 500                                                      # Schritte je Zufallszahlenblock
PILOT_STEPS, PILOT_CHAINS = 1500, 4
BETAS = tuple(10.0 ** (-4.5 + 4.3 * k / 13.0) for k in range(14))   # pCN: Schrittgitter (1e-4,5 bis 0,63), muss die informativen Daten überdecken
HS = tuple(10.0 ** (-4.0 + 3.7 * k / 13.0) for k in range(14))      # isotropes RWM: Schrittgitter (1e-4 bis 0,5)
TARGET_ACC_PCN, TARGET_ACC_RWM = 0.3, 0.234
RWM_COV_SCALE_FACTOR = 2.38                                      # Skalierung 2,38/sqrt(n) (Roberts, Gelman, Gilks)
SINGULAR_COND = 1e15
SHOW_RANGE = (-1.5, 2.5)
Z_BAND = 1.96

# Verdikt-Schwellen für „konvergiert“ (Gelman und Rubin, effektive Stichprobengröße, Fehler gegen das Orakel)
RHAT_OK = 1.1
ESS_OK = 100.0
SD_ERR_OK = 0.2                                                  # größter relativer sd-Fehler gegen den exakten Posterior
MEAN_ERR_OK = 0.5                                                # größter Mittelwertfehler in sd

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 990000
EXP_STEPS = 20000
EXP_NOISE = (1e-1, 1e-2, 1e-3)                                   # Experiment 1: Information
EXP_DIMS = (24, 48, 96, 192, 384)                                # Experiment 1: Dimension (m = 24 Messwerte)
EXP_M = 24
EXP_DIM_NOISE = 1e-1
BURN_STEPS = 8000
BURN_DISCARD = (0, 500, 1000, 2000, 4000)
ID_NOISE = (1e-1, 1e-2, 1e-3, 1e-4)                              # Experiment 3
NG_CHAINS = 20                                                   # Experiment 4: Ketten
NG_N = 24
NG_LAPLACE = ((1e-2, 0.1), (3e-2, 0.03), (3e-2, 0.1))            # (Rauschen, Skala b)
NG_HUBER = ((0.02, 0.1, 3e-2), (0.002, 0.03, 3e-2), (0.002, 0.03, 1e-1), (0.002, 0.01, 1e-1))   # (ε, b, Rauschen)

COLOR_TRUTH = "#2F6B65"
COLOR_EXACT = "#222222"
COLOR_MEAN = "#4c78a8"
COLOR_BAND = "rgba(76, 120, 168, 0.18)"
COLOR_MAP = "#D68A2E"
COLOR_LAPLACE = "#8e44ad"
COLOR_GREY = "#8a8a8a"
COLOR_CHAINS = ("#4c78a8", "#D68A2E", "#2F6B65", "#8e44ad", "#c0392b", "#7f7f7f")
SAMPLER_COLORS = {"pcn": "#8e44ad", "rwm_iso": "#D68A2E", "rwm_cov": "#4c78a8"}

PRESETS = {
    "Standard": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "world": "phantom", "prior": "gauss", "sampler": "rwm_cov", "steps": 10000, "chains": 4,
                 "start": "map", "scale": DEFAULT_SCALE, "eps": DEFAULT_EPS, "seed": DEFAULT_SEED},
    "pCN bei informativen Daten": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "world": "phantom", "prior": "gauss", "sampler": "pcn", "steps": 10000, "chains": 4,
                                   "start": "map", "scale": DEFAULT_SCALE, "eps": DEFAULT_EPS, "seed": DEFAULT_SEED},
    "Isotropes RWM": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "world": "phantom", "prior": "gauss", "sampler": "rwm_iso", "steps": 10000, "chains": 4,
                      "start": "map", "scale": DEFAULT_SCALE, "eps": DEFAULT_EPS, "seed": DEFAULT_SEED},
    "Start aus dem Prior": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "world": "phantom", "prior": "gauss", "sampler": "rwm_cov", "steps": 10000, "chains": 4,
                            "start": "prior", "scale": DEFAULT_SCALE, "eps": DEFAULT_EPS, "seed": DEFAULT_SEED},
    "Laplace-Prior": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": 1e-2, "world": "spitzen", "prior": "laplace", "sampler": "rwm_cov", "steps": 40000, "chains": 4,
                      "start": "map", "scale": 0.03, "eps": DEFAULT_EPS, "seed": DEFAULT_SEED},
    "Dünner Pseudo-Huber-Prior": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": 3e-2, "world": "spitzen", "prior": "huber", "sampler": "rwm_cov", "steps": 10000, "chains": 4,
                                  "start": "map", "scale": 0.03, "eps": 0.002, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standard": "Gaußscher Posterior, Zufallsschritt-Metropolis mit der exakten Kovarianz als Vorschlag, Start um das MAP: die Kette trifft Mittel und Standardabweichung.",
    "pCN bei informativen Daten": "pCN bei Rauschen 0,1 %: der Schritt muss so klein sein, dass die Kette sich kaum bewegt; der Mittelwert stimmt, die Standardabweichung nicht.",
    "Isotropes RWM": "Derselbe Posterior mit isotropem Schritt: die Autokorrelationszeit ist um ein Vielfaches länger, die Ketten decken den Posterior schlecht ab.",
    "Start aus dem Prior": "Dieselben Ketten, aber aus dem Prior gestartet: R̂ bleibt weit über 1, obwohl die Spur beruhigt aussieht.",
    "Laplace-Prior": "Spitzen-Welt mit Laplace-Prior: das MAP (L1) ist dünn besetzt, das Posterior-Mittel nicht; eine Laplace-Näherung gibt es hier nicht (die Hesse-Matrix des Betrags ist null).",
    "Dünner Pseudo-Huber-Prior": "Glatter, aber dünner Prior (ε = 0,002, b = 0,03) bei 3 % Rauschen: die Laplace-Näherung unterschätzt die Standardabweichung um mehr als ein Zehntel bis zur Hälfte.",
}
