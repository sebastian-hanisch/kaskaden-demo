"""Regler-Grenzen, feste Annahmen, gemessene Werte und Presets."""

SPACING = 1.0                    # Abstand der Kreuzungen im Betriebsnetz-Raster
JITTER = 0.18
SEED_MAX = 999999
DEFAULT_SEED = 35

KINDS = ("city", "ba", "barbell")
KIND_LABELS = {"city": "Betriebsnetz (Raster/Zufallsgraph)", "ba": "Skalenfreies Netz (Barabási-Albert)", "barbell": "Barbell-Lehrbuch"}

# --- Betriebsnetz (wortgleich aus robustheit-demo) ----------------------------------------------------------------------------------------------------
SIDE_MIN, SIDE_MAX, DEFAULT_SIDE = 4, 20, 10
NETTYPES = ("grid", "random")
NETTYPE_LABELS = {"grid": "Raster (Straßennetz)", "random": "Zufallsgraph (gleiche Kantenzahl)"}
BLOCKED_OPTIONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
DEFAULT_BLOCKED = 0.2

# --- Skalenfreies Netz (Barabási und Albert 1999) -----------------------------------------------------------------------------------------------------
N_BA_MIN, N_BA_MAX, DEFAULT_N_BA = 30, 300, 150
M0_BA_MIN, M0_BA_MAX, DEFAULT_M0_BA = 3, 12, 4
M_BA_MIN, DEFAULT_M_BA = 1, 2

# --- Barbell-Lehrbuch (wortgleich aus centrality-demo) ------------------------------------------------------------------------------------------------
BARBELL_K_MIN, BARBELL_K_MAX, DEFAULT_BARBELL_K = 3, 40, 8

# --- Dynamik-Umschalter --------------------------------------------------------------------------------------------------------------------------------
MODES = ("kaskade", "ausbreitung")
MODE_LABELS = {"kaskade": "Kaskade (lastbasierter Ausfall)", "ausbreitung": "Ausbreitung (SIR-Epidemie)"}

# --- Kaskade (Motter & Lai 2002) -----------------------------------------------------------------------------------------------------------------------
ALPHA_MIN, ALPHA_MAX, DEFAULT_ALPHA = 0.0, 3.0, 0.5                       # Regler-Bereich fuer Schritt 1 ("in Aktion") - ein EINZELNES alpha waehlen
ALPHA_STEP = 0.05
INITIAL_STRATEGIES = ("highest_load", "random")
INITIAL_STRATEGY_LABELS = {"highest_load": "höchste Last (Betweenness)", "random": "Zufall"}
# ALPHA_SWEEP: x-Achse der G(alpha)-Kurve (Schritt 2) und Basis fuer critical_tolerance - deutlich WEITER als der Regler oben, weil ein skalenfreier
# Hub-Ausfall erst bei alpha~8-16 aufhoert zu kaskadieren (gemessen, s. tests/test_claims.py), waehrend Betriebsnetz/Barbell schon bei alpha<=4 stoppen.
ALPHA_SWEEP = tuple(round(0.0 + 0.2 * i, 2) for i in range(101))          # 0.0 .. 20.0 in 0.2-Schritten

# --- Ausbreitung (Pastor-Satorras & Vespignani 2001) --------------------------------------------------------------------------------------------------
BETA_MIN, BETA_MAX, DEFAULT_BETA = 0.0, 1.0, 0.15
BETA_STEP = 0.01
MU_MIN, MU_MAX, DEFAULT_MU = 0.05, 1.0, 0.2
MU_STEP = 0.05
N_RUNS_MIN, N_RUNS_MAX, DEFAULT_N_RUNS = 20, 300, 100
MAX_STEPS_SIR = 200                                                       # Sicherheitsgrenze fuer die Zeitschritte einer SIR-Simulation
THRESHOLD_FRAC = 0.05                                                     # endliche-Groesse-Proxy fuer "die Epidemie ist ausgebrochen" (mean_fraction >= THRESHOLD_FRAC), s. kas_evaluation.measured_threshold

STEPS = {1: "1 · Kaskade/Ausbreitung in Aktion", 2: "2 · Kaskadengröße über die Toleranz", 3: "3 · Endgröße der Ausbreitung über β", 4: "4 · Drei Netze im Vergleich"}

# --- Gemessene Werte (Seed 35, sofern nicht anders angegeben; 2026-09-27, alle Werte über ev.*/A.*-Aufrufe nachgerechnet, s. tests/test_claims.py) -----------
# KASKADE, Skalenfreies Netz (n=150, m=296, Hub=Knoten 3): bei der Standard-Toleranz alpha=0.5 fallen 57 von 150 Knoten aus (G=0.38) - das 57-FACHE des einmaligen Ausfalls aus Stück 8 (dort genau
#   1 Knoten). Kritische Toleranz (kein Sekundaerausfall mehr) alpha*=17.0 - ein SEHR grosser Sicherheitsspielraum waere noetig, um diesen Hub-Ausfall folgenlos zu machen.
# KASKADE, Betriebsnetz (n=100, m=144, Raster, 20% gesperrt, Hub=Knoten 34): kritische Toleranz alpha*=2.8 - eine Groessenordnung KLEINER als beim skalenfreien Netz. Schon bei alpha=3.0 (leicht
#   erreichbarer Sicherheitsspielraum) gibt es GAR KEINEN Sekundaerausfall mehr (G=0.01, nur der Anfangsausfall selbst) - waehrend derselbe alpha=3.0 auf dem skalenfreien Netz IMMER NOCH kaskadiert
#   (s. Bericht). Die kritische Toleranz, nicht der Vervielfachungsfaktor bei einem festen alpha, ist die Kennzahl, die die drei Vehikel am klarsten unterscheidet.
# KASKADE, Barbell (n=16, k=8): kritische Toleranz alpha*=0.0 EXAKT (Satz, s. tests/test_cascade.py) - die Bruecke ist nach der ersten Entfernung schon getrennt, keinerlei Toleranz noetig.
# AUSBREITUNG, Barbell (n=16, k=8, patient_zero=14, beta=0.2, mu=0.3, 100 Laeufe): mittlere Endgroesse 9.17, p10=1 (stirbt fast sofort aus), p50=8 (genau EINE Clique voll infiziert, Bruecke NICHT
#   passiert), p90=16 (BEIDE Cliquen). Nur 30% der Laeufe ueberqueren die Bruecke ueberhaupt - ein sichtbarer Flaschenhals-Effekt.
# EPIDEMIESCHWELLE, Betriebsnetz mit Zufallsgraph-Option (n=289, mu=0.3): beta_c=0.0636 (Pastor-Satorras-Formel) gegen gemessen 0.0907 (Abstand +0.0271, endliche Groesse). MIT der Raster-Variante
#   (n=100, Standardeinstellung) ist der Abstand deutlich GROESSER (beta_c=0.0939 gegen gemessen 0.1423) - ein STRUKTURELLES Manko der Mean-Field-Naeherung auf raeumlich strukturierten Netzen, kein
#   Rauschen (s. README "Design-Entscheidung", analog zur PERCOLATION_FRAC_VANISH-Erkenntnis der robustheit-demo).
# EPIDEMIESCHWELLE, Skalenfrei (n=288, mu=0.3): beta_c=0.0330 gegen gemessen 0.0397 (Abstand nur +0.0067, deutlich enger als beim Betriebsnetz) - UND rund halb so gross wie beim Betriebsnetz
#   (Verhaeltnis beta_c: 0.52, gemessen: 0.44) - die Pastor-Satorras-Kernaussage (2001) bestaetigt: MESSBAR niedriger, nicht literarisch "keine Schwelle" (die Formel bleibt bei endlichem n>0).
PRESET_HELP_MEASURED_AT = "2026-09-27"

PRESETS = {
    "Barbell-Kaskade (bricht sofort ab)": {"kind": "barbell", "mode": "kaskade", "kbarbell": 8, "initial": "highest_load", "alpha": 0.5, "seed": 35, "step": 1},
    "Barbell-Ausbreitung (muss die Brücke passieren)": {"kind": "barbell", "mode": "ausbreitung", "kbarbell": 8, "initial": "random", "beta": 0.2, "mu": 0.3, "nruns": 100, "seed": 35, "step": 1},
    "Skalenfrei-Kaskade (ein Hub-Ausfall reißt ein Vielfaches mit)": {"kind": "ba", "mode": "kaskade", "nba": 150, "mba": 2, "m0ba": 4, "initial": "highest_load", "alpha": 0.5, "seed": 35, "step": 2},
    "Kritische Toleranz messen (Barbell: exakt 0)": {"kind": "barbell", "mode": "kaskade", "kbarbell": 8, "initial": "highest_load", "alpha": 0.5, "seed": 35, "step": 2},
    "Betriebsnetz robust gegen Kaskaden": {"kind": "city", "mode": "kaskade", "side": 10, "blocked": 0.2, "nettype": "grid", "initial": "highest_load", "alpha": 3.0, "seed": 35, "step": 2},
    "Epidemieschwelle Betriebsnetz gegen Pastor-Satorras-Formel": {"kind": "city", "mode": "ausbreitung", "side": 17, "blocked": 0.0, "nettype": "random", "initial": "highest_load", "mu": 0.3,
                                                                    "nruns": 150, "seed": 35, "step": 3},
    "Skalenfrei hat eine deutlich niedrigere Schwelle": {"kind": "ba", "mode": "ausbreitung", "nba": 288, "mba": 2, "m0ba": 4, "initial": "highest_load", "mu": 0.3, "nruns": 150, "seed": 35,
                                                          "step": 3},
    "Drei-Netze-Vergleich": {"kind": "city", "mode": "kaskade", "side": 10, "blocked": 0.2, "nettype": "grid", "nba": 150, "mba": 2, "m0ba": 4, "kbarbell": 8, "initial": "highest_load", "mu": 0.3,
                             "nruns": 100, "seed": 35, "step": 4},
}
PRESET_HELP = {
    "Barbell-Kaskade (bricht sofort ab)": "16 Knoten (k=8): der Ausfall eines Brückenendes trennt den Graphen sofort in zwei vollständige Teilgraphen (Größe 7 und 8) - dort trägt kein Knoten "
                                           "mehr Last (jeder kürzeste Weg ist eine direkte Kante), die Kaskade endet nach GENAU Runde 1, unabhängig von α (Satz, s. Tests).",
    "Barbell-Ausbreitung (muss die Brücke passieren)": "Patient Null in einer Clique, β=0.2, μ=0.3: mittlere Endgröße 9.2 von 16, p50=8 (nur EINE Clique infiziert - die Brücke NICHT passiert) "
                                                        "gegen p90=16 (beide Cliquen) - nur 30% der 100 Läufe überqueren die Brücke überhaupt, ein sichtbarer Flaschenhals-Effekt.",
    "Skalenfrei-Kaskade (ein Hub-Ausfall reißt ein Vielfaches mit)": "150 Knoten: bei Standard-Toleranz α=0.5 fallen 57 von 150 Knoten aus (G=0.38) - das 57-FACHE des einmaligen Ausfalls aus "
                                                                      "Stück 8 (dort genau 1 Knoten). Kritische Toleranz α*=17.0 - ein sehr großer Sicherheitsspielraum wäre nötig.",
    "Kritische Toleranz messen (Barbell: exakt 0)": "Die kritische Toleranz des Barbell ist EXAKT 0.0 (Satz) - im Gegensatz dazu braucht das skalenfreie Netz α*=17.0 und das Betriebsnetz α*=2.8, "
                                                     "um denselben Hub-Ausfall folgenlos zu machen. Die kritische Toleranz unterscheidet die drei Vehikel deutlich klarer als der Schaden bei "
                                                     "einem einzelnen festen α.",
    "Betriebsnetz robust gegen Kaskaden": "Kritische Toleranz α*=2.8 - eine Größenordnung kleiner als beim skalenfreien Netz (17.0). Schon bei α=3.0 (leicht erreichbarer Sicherheitsspielraum) "
                                           "gibt es GAR KEINEN Sekundärausfall mehr (G=0.01, nur der Anfangsausfall) - während derselbe α=3.0 auf dem skalenfreien Netz immer noch kaskadiert.",
    "Epidemieschwelle Betriebsnetz gegen Pastor-Satorras-Formel": "289 Knoten (Zufallsgraph-Option, näher an der Mean-Field-Annahme als das reine Raster): β_c=0.0636 (Formel) gegen gemessen "
                                                                   "0.0907 (Abstand 0.027, endliche Größe). Mit dem reinen Straßenraster ist der Abstand deutlich GRÖSSER (0.094 gegen 0.142) - "
                                                                   "ein strukturelles Manko der Näherung auf räumlichen Netzen, s. README.",
    "Skalenfrei hat eine deutlich niedrigere Schwelle": "288 Knoten: β_c=0.0330 gegen gemessen 0.0397 (Abstand nur 0.007, viel enger als beim Betriebsnetz) - und etwa HALB so groß wie beim "
                                                         "Betriebsnetz (Verhältnis 0.52). Die Pastor-Satorras-Kernaussage (2001): messbar niedriger, nicht buchstäblich null.",
    "Drei-Netze-Vergleich": "Beide Dynamiken nebeneinander: das skalenfreie Netz hat sowohl die GRÖSSTE kritische Toleranz (17.0 gegen 2.8 Betriebsnetz, 0.0 Barbell) als auch die NIEDRIGSTE "
                            "Epidemieschwelle (0.037 gegen 0.094 Betriebsnetz, 0.042 Barbell) - für beide Dynamiken das verwundbarste der drei Vehikel.",
}
