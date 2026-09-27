# Kaskaden und Ausbreitung – lastbasierte Kaskadenausfälle, SIR-Epidemieschwelle – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-kaskaden-demo.streamlit.app/)**

Neuntes Stück der **Graphen-und-Netzwerke-Reihe** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Kind von Stück 8 (Robustheit, [robustheit-demo](https://github.com/sebastian-hanisch/robustheit-demo)): dort war jeder Knotenausfall ein EINMALIGES, unabhängiges Ereignis (Zufall oder gezielt) – hier geht es um FOLGEEREIGNISSE, die aus einem einzigen Ausfall eine viel größere Katastrophe machen. Zwei Dynamiken: **Lastbasierte Kaskadenausfälle** (Motter & Lai 2002, Phys. Rev. E 66, 065102(R)): jeder Knoten trägt eine Anfangslast (seine Betweenness) und eine Kapazität = (1+α)·Anfangslast (Toleranz α); fällt ein Knoten aus, verteilen sich die kürzesten Wege neu – manche Knoten tragen jetzt mehr Last als ihre Kapazität und fallen ebenfalls aus, eine Kettenreaktion, die den einmaligen Schaden aus Stück 8 um ein Vielfaches übertreffen kann. **SIR-Ausbreitung** (Pastor-Satorras & Vespignani 2001, Phys. Rev. Lett. 86(14), 3200–3203): eine stochastische Infektion (S→I→R, diskrete Zeit) verbreitet sich über die Kanten – skalenfreie Netze haben eine deutlich niedrigere Epidemieschwelle als homogene Netze (hier an den drei Vehikeln aus Stück 7/8 GEMESSEN, nicht vorausgesetzt).

**Einordnung in die Reihe:** die Reihe hat zwölf Stücke, dies ist das neunte (Details in `graphen-planung/PLAN.md` des Portfolio-Ordners):

```
1 BFS und DFS (Wurzel)                                                        [gebaut: bfs-dfs-demo]
 ├─ 2 Brücken und Artikulationspunkte ─ 4 Euler-Touren                        [gebaut: bridges-demo, euler-tour-demo]
 ├─ 3 Starke Zusammenhangskomponenten, topologische Sortierung                [gebaut: scc-demo]
 ├─ 5 Graphfärbung                                                            [gebaut: graph-coloring-demo]
 ├─ 6 Zentralität ─ 7 Strukturkennzahlen ─ 8 Robustheit ─ 9 Kaskaden/Ausbr.   [gebaut: centrality-demo, strukturkennzahlen-demo, robustheit-demo, kaskaden-demo ─ DIESES STÜCK]
 │                            └─ 10 Kritische Knoten härten                   [nicht gebaut]
 └─ 11 Bandbreite ─ 12 Bandbreite von G(n,k,b) und Cliquenüberdeckung         [nicht gebaut]
```

Ergebnis in Kürze: Die **kritische Toleranz** (kleinstes α, ab dem ein Hub-Ausfall auf dem skalenfreien Netz KEINE Sekundärausfälle mehr auslöst) unterscheidet die drei Vehikel deutlich klarer als der Schaden bei einem festen α: **α\*=17.0 beim skalenfreien Netz** gegen nur **α\*=2.8 beim Betriebsnetz** und **α\*=0.0 exakt beim Barbell** (ein Satz: die Brücke ist nach der ersten Entfernung schon getrennt). Bei der Ausbreitung bestätigt sich die Pastor-Satorras-Kernaussage GEMESSEN: das skalenfreie Netz hat eine etwa **halb so große Epidemieschwelle** wie ein vergleichbar großes homogenes Netz (β_c=0.033 gegen 0.064) – messbar niedriger, aber (bei den hier gewählten, endlichen Größen) NICHT buchstäblich null, wie ein vereinfachtes Schlagwort suggerieren könnte. Eine ehrliche Design-Entscheidung: die Mean-Field-Formel β_c=μ·⟨k⟩/⟨k²⟩ trifft auf einen Zufallsgraph erkennbar besser zu als auf das raumstrukturierte Straßenraster – ein strukturelles, kein zufälliges Manko (s. unten).

## Warum dieses Problem

Ein einzelner Knotenausfall ist selten das ganze Bild: in Stromnetzen, Verkehrsnetzen und im Internet lösen Ausfälle häufig FOLGEEREIGNISSE aus – überlastete Leitungen/Router übernehmen die umgeleitete Last und fallen selbst aus, was wiederum weitere Umleitungen erzwingt (Motter & Lai 2002, ursprünglich am nordamerikanischen Stromnetz-Blackout 1996 motiviert). Ähnlich verbreitet sich eine Infektion nicht gleichmäßig, sondern hängt entscheidend von der Netzstruktur ab: auf einem skalenfreien Kontaktnetz (wenige, aber extrem gut vernetzte "Superspreader"-Knoten) kann sich eine Epidemie schon bei sehr kleiner Ansteckungswahrscheinlichkeit ausbreiten, während ein homogenes Netz eine echte, positive Schwelle hat (Pastor-Satorras & Vespignani 2001). Beide Dynamiken macht diese Demo an denselben drei Vehikeln (Betriebsnetz, skalenfreies Netz, Barbell) messbar und vergleichbar.

## Design-Entscheidung: Mean-Field-Formel auf Raster gegen Zufallsgraph

Die heterogene Mean-Field-Näherung β_c=μ·⟨k⟩/⟨k²⟩ setzt implizit ein "gut gemischtes" Netz ohne starke räumliche Struktur voraus (ein Konfigurationsmodell-artiges Netz). Ein reines Straßenraster ist das Gegenteil: es ist niedrigdimensional und räumlich stark strukturiert (kaum Abkürzungen). Gemessen (n≈289, μ=0.3): auf der **Zufallsgraph-Option** des Betriebsnetzes (Poisson-artige Gradverteilung) liegt die gemessene Schwelle bei β\*=0.091 gegen die Vorhersage β_c=0.064 (Abstand 0.027) – auf dem **reinen Raster** (n=100, Standardeinstellung) ist der Abstand mit β\*=0.142 gegen β_c=0.094 deutlich GRÖSSER (0.048). Das ist – anders als bei bloßem endliche-Größe-Rauschen – ein **struktureller** Effekt: der Abstand wird beim Raster nicht kleiner, wenn n wächst (getestet bei n=144 und n=289, s. `tests/test_evaluation.py`). Für den eigentlichen Formelvergleich verwendet diese Demo darum bewusst die Zufallsgraph-Option; das reine Raster bleibt als Instanz wählbar, zeigt aber ehrlich die größere Abweichung.

## Vorab-Hypothesen (vor der Messung notiert, hier geprüft)

| Hypothese | Ergebnis |
|---|---|
| **H1** Die Kaskaden-Lastneuberechnung je Runde stimmt mit einer unabhängigen Brandes-/networkx-Neuberechnung überein. | ✅ Bestätigt auf >20 kleinen Instanzen (Betriebsnetz, Skalenfrei, Barbell), Runde für Runde. |
| **H2** Ein Hub-Ausfall auf dem skalenfreien Netz löst eine Kaskade aus, die ein Vielfaches des einmaligen Schadens (Stück 8: 1 Knoten) kostet. | ✅ Bestätigt: bei Standard-Toleranz α=0.5 fallen 57 von 150 Knoten aus – das 57-fache. |
| **H3** Die kritische Toleranz unterscheidet die drei Vehikel deutlicher als der Schaden bei einem festen α. | ✅ Bestätigt: α\*=17.0 (Skalenfrei) gegen 2.8 (Betriebsnetz) gegen 0.0 (Barbell) – eine Größenordnung Unterschied, wo der Schaden bei α=0.5 sich nicht annähernd so stark unterscheidet (G=0.38 gegen 0.39 – praktisch GLEICH groß, eine echte Überraschung, s. Befunde). |
| **H4** α→∞ stoppt die Kaskade beim Anfangsausfall (keine Sekundärausfälle), UNABHÄNGIG von der Instanz. | ⚠️ **Mit einer dokumentierten Einschränkung bestätigt:** empirisch auf allen drei Vehikeln und >55 zufälligen Stichproben (0 Ausreißer) – aber ein bekanntes theoretisches Merkmal der Motter-Lai-Formel (Kapazität = (1+α)·Anfangslast) bleibt: ein Knoten mit Anfangslast EXAKT 0 hat Kapazität 0 für JEDES α; trägt er später auch nur minimal Last, fällt er trotzdem aus. Auf den hier verwendeten Instanzen/Größen nicht beobachtet, aber kein Beweis, dass es nie vorkommt (s. Grenzen unten). |
| **H5** Der Barbell zeigt den dramatischsten Kaskaden-Sonderfall: die Kaskade endet exakt nach Runde 1. | ✅ Bestätigt als Satz (nicht nur gemessen): nach Entfernung eines Brückenendes trägt in den beiden entstehenden vollständigen Teilgraphen kein Knoten mehr Last. |
| **H6** SIR degeneriert bei β=1, μ=0 zu einer deterministischen BFS-Wellenfront. | ✅ Bestätigt EXAKT (kein Toleranzband nötig) gegen `bfs_distances` auf allen Testinstanzen. |
| **H7** Das skalenfreie Netz hat eine messbar niedrigere Epidemieschwelle als ein vergleichbar großes homogenes Netz (Pastor-Satorras & Vespignani 2001). | ✅ Bestätigt: β_c=0.033 (Skalenfrei) gegen 0.064 (Betriebsnetz, Zufallsgraph-Option) – etwa halb so groß. NICHT bestätigt: "praktisch keine Schwelle" – bei den hier gewählten endlichen Größen bleibt β_c positiv und die Formel liefert eine sinnvolle, von Null verschiedene Zahl. |
| **H8** Die Mean-Field-Formel trifft auf ALLE Betriebsnetz-Varianten gleich gut zu. | ❌ **Widerlegt:** auf dem reinen Straßenraster ist der Abstand zur Formel STRUKTURELL größer als auf der Zufallsgraph-Option (s. Design-Entscheidung oben) – kein Rauschen. |
| **H9** Die mittlere SIR-Endgröße ist nicht fallend in β (über viele Läufe gemittelt). | ✅ Bestätigt (Mittelwert-Aussage) auf >100 (Instanz, Seed)-Kombinationen, gebändert gegen Stichprobenrauschen. |

## Befunde (gemessen, keine Behauptungen)

Seed 35, Standardeinstellungen sofern nicht anders angegeben; die SIR-Simulation ist stochastisch (Python-`random`, plattformstabile Seed-Formel), die Kaskade und die Instanzen sind deterministisch.

| Frage | Ergebnis |
|---|---|
| **Stimmt das Verfahren?** | ✅ Kaskaden-Lastneuberechnung je Runde == unabhängige Brandes-/networkx-Neuberechnung; Barbell-Sonderfall exakt (Runde 1); SIR-Erhaltungssatz S+I+R=n in JEDEM Zeitschritt; β=1,μ=0 exakt gegen BFS-Erreichbarkeit |
| **Skalenfreies Netz, Kaskade** (150 Knoten, 296 Kanten, Hub-Ausfall) | bei α=0.5: 57 von 150 Knoten fallen aus (G=0.38, **57-facher** Schaden gegenüber dem einmaligen Ausfall aus Stück 8). Kritische Toleranz **α\*=17.0**. |
| **Betriebsnetz, Kaskade** (100 Knoten, 144 Straßen, 20 % gesperrt, Hub-Ausfall) | bei α=0.5: 39 von 100 Knoten fallen aus (G=0.39 – **praktisch derselbe Schaden wie beim skalenfreien Netz bei diesem α**, eine Überraschung). Kritische Toleranz **α\*=2.8** – schon bei α=3.0 GAR KEIN Sekundärausfall mehr. |
| **Barbell, Kaskade** (16 Knoten, k=8) | kritische Toleranz **α\*=0.0 EXAKT** (Satz) – die Brücke trennt sofort, keine Lastumverteilung mehr möglich. |
| **Barbell, Ausbreitung** (β=0.2, μ=0.3, Patient Null in einer Clique, 100 Läufe) | mittlere Endgröße 9.2 von 16; p50=8 (nur EINE Clique, Brücke NICHT passiert) gegen p90=16 (beide Cliquen) – nur **30 % der Läufe** überqueren die Brücke überhaupt. |
| **Epidemieschwelle, Betriebsnetz** (289 Knoten, Zufallsgraph-Option, μ=0.3) | β_c=0.0636 (Formel) gegen gemessen 0.0907 (Abstand +0.027, endliche Größe) |
| **Epidemieschwelle, Betriebsnetz** (100 Knoten, reines Raster) | β_c=0.0939 gegen gemessen 0.1423 (Abstand +0.048 – **deutlich größer** als bei der Zufallsgraph-Option, strukturell, s. Design-Entscheidung) |
| **Epidemieschwelle, Skalenfrei** (288 Knoten, μ=0.3) | β_c=0.0330 gegen gemessen 0.0397 (Abstand nur +0.007, viel enger als beim Betriebsnetz) – **etwa halb so groß** wie beim Betriebsnetz (Verhältnis 0.52) |

Presets (8), alle mit den Zahlen in ihren Hilfetexten (`tests/test_presets.py`):

| Preset | Was es zeigt |
|---|---|
| Barbell-Kaskade (bricht sofort ab) | Bruckenende-Ausfall trennt sofort in K7+K8, Kaskade endet nach Runde 1 exakt |
| Barbell-Ausbreitung (muss die Brücke passieren) | p50=8 (eine Clique), p90=16 (beide) – nur 30 % überqueren die Brücke |
| Skalenfrei-Kaskade (ein Hub-Ausfall reißt ein Vielfaches mit) | 57-facher Schaden bei α=0.5, kritische Toleranz α\*=17.0 |
| Kritische Toleranz messen (Barbell: exakt 0) | α\*=0.0 exakt, im Kontrast zu 17.0 (Skalenfrei) und 2.8 (Betriebsnetz) |
| Betriebsnetz robust gegen Kaskaden | α\*=2.8 – schon bei α=3.0 kein Sekundärausfall mehr |
| Epidemieschwelle Betriebsnetz gegen Pastor-Satorras-Formel | β_c=0.064 gegen gemessen 0.091 (Zufallsgraph-Option, enger als das reine Raster) |
| Skalenfrei hat eine deutlich niedrigere Schwelle | β_c=0.033 gegen 0.064 beim Betriebsnetz – etwa halb so groß |
| Drei-Netze-Vergleich | Beide Dynamiken nebeneinander: Skalenfrei am verwundbarsten in BEIDEN |

## Modell und Verfahren

- **Betriebsnetz, Skalenfreies Netz, Barbell** (`kas_scenario.py`, wortgleich aus `robustheit-demo`): dieselben drei Vehikel wie in Stück 8.
- **Lastbasierte Kaskade** (`motter_lai_cascade`, Motter & Lai 2002): Anfangslast = Betweenness auf dem vollen Graphen (einmal berechnet), Kapazität = (1+α)·Anfangslast; rundenweise Neuberechnung auf dem Restgraphen, bis keine neuen Ausfälle mehr auftreten.
- **Kritische Toleranz** (`critical_tolerance`): kleinstes α aus einem Sweep (0 bis 20), ab dem keine Sekundärausfälle mehr auftreten.
- **SIR-Ausbreitung** (`sir_simulate`, `sir_many_runs`, Pastor-Satorras & Vespignani 2001): diskrete Zeit, S/I/R-Zustände, `random.Random`-Hauskonvention für plattformstabile Reproduzierbarkeit.
- **Epidemieschwelle** (`epidemic_threshold_prediction`): β_c=μ·⟨k⟩/⟨k²⟩ aus den GEMESSENEN Gradmomenten des konkreten Graphen.

## Was die App zeigt

1. **Vier Schritte** (Schritt-Regler): **Kaskade/Ausbreitung in Aktion** (Karte mit Schieberegler über Runden/Zeitschritte) → **Kaskadengröße über die Toleranz** (G(α)-Kurve, kritische Toleranz markiert, Vervielfachungsfaktor-Balken – nur im Kaskade-Modus) → **Endgröße der Ausbreitung über β** (Kurve mit Streuband, Pastor-Satorras-Schwelle gestrichelt – nur im Ausbreitung-Modus) → **Drei Netze im Vergleich** (kritische Toleranz UND Epidemieschwelle je Netz, tabellarisch).
2. **Dynamik-Umschalter** (Kaskade / Ausbreitung): schaltet zwischen den beiden Regler-Sätzen um (Toleranz α ↔ β/μ/Wiederholungen) – Schritt 2 und 3 zeigen einen erklärenden Hinweis statt eines leeren Diagramms, wenn der aktuelle Modus nicht zu ihnen passt.
3. Regler: Instanz (Betriebsnetz / Skalenfreies Netz / Barbell), instanzspezifische Parameter, Startknoten-Strategie (höchste Last / Zufall), Zufalls-Seed (+🎲); Permalink in der Adresszeile.

## Was nicht funktioniert hat / Grenzen

- **Kapazität = (1+α)·Anfangslast wird für Knoten mit Anfangslast 0 problematisch.** Ein Knoten ohne jede Betweenness im Ausgangsnetz (z. B. ein Blatt) hat Kapazität 0 für JEDES α – trägt er nach einem Ausfall anderswo im Netz auch nur minimal Last, fällt er sofort aus, unabhängig von der Toleranz. Auf den drei Vehikeln dieser Demo empirisch nicht beobachtet (0 Ausreißer bei >55 Stichproben), aber ein bekanntes theoretisches Merkmal der Motter-Lai-Formel selbst, keine Eigenheit dieser Implementierung.
- **Mean-Field-Formel auf räumlich strukturierten Netzen.** Die Epidemieschwellen-Formel trifft auf ein reines Straßenraster spürbar schlechter zu als auf einen Zufallsgraph gleicher Kantenzahl – ein strukturelles, kein zufälliges Manko (s. Design-Entscheidung oben).
- **"Kritische Toleranz" ist binär definiert** (keine Sekundärausfälle oder mindestens einer) – nicht als Prozentsatz an Gesamtschaden. Ein Netz mit vielen kleinen, aber knapp unterschwelligen Sekundärausfällen zählt hier noch als "kaskadierend".
- **SIR ohne Immunitätsverlust.** Genesene Knoten können sich nie wieder anstecken – für Krankheiten mit nachlassender Immunität eine Vereinfachung.
- **Synthetische Instanzen.** Betriebsnetz, skalenfreies Netz und Barbell sind erzeugt, keine echten Infrastruktur- oder Kontaktnetzdaten.

## Tests

`tests/test_scenario.py` (Rauchtests der drei Instanzen, wortgleich aus `robustheit-demo`), `tests/test_algorithm_base.py` (wortgleich kopierte Bausteine gegen networkx), `tests/test_cascade.py` (Korrektheits-Kette Punkte 1–4: Lastneuberechnung je Runde gegen unabhängige Brandes-/networkx-Neuberechnung, Monotonie/Terminierung, α→∞-Grenzfall, Barbell-Sonderfall exakt), `tests/test_sir.py` (Korrektheits-Kette Punkte 5–8: Erhaltungssatz jeden Zeitschritt, β=0-Grenzfall, β=1/μ=0 exakt gegen BFS, Mittelwert-Monotonie in β auf >100 Instanzen), `tests/test_evaluation.py` (Korrektheits-Kette Punkte 9–10: Epidemieschwelle gebändert gegen Pastor-Satorras, skalenfrei messbar niedriger, Design-Entscheidung Raster/Zufallsgraph, Sonderfälle), `tests/test_presets.py` (jede Zahl der Hilfetexte), `tests/test_claims.py` (jede README-Zahl über die echten Auswertungsfunktionen), `tests/test_app.py` (AppTest – Voreinstellung, jedes Preset, jeder Schritt für jede Instanzart und Dynamik, bedingte Regler, Permalink-Grenzen, Footer).

```
python -m pytest tests/ -v
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-App |
| `kas_algorithm.py` | Bausteine (wortgleich aus robustheit-demo), Kaskade (Motter & Lai), SIR-Ausbreitung, Epidemieschwelle |
| `kas_scenario.py` | Betriebsnetz, skalenfreies Netz, Barbell, Erdős-Rényi (wortgleich aus robustheit-demo) |
| `kas_evaluation.py` | Analyse, α-Sweep/kritische Toleranz, Vervielfachungsfaktor, β-Sweep/Epidemieschwelle, Netzvergleich |
| `kas_visualization.py` | Plotly-Figuren (Kaskaden-/SIR-Karte, G(α)-Kurve, β-Endgrößen-Kurve, Netzvergleichsbalken) |
| `kas_presets.py`, `kas_constants.py` | Permalink, Presets, gemessene Werte |
| `tests/` | Tests |

## Bewusst nicht umgesetzt

Kritische Knoten härten (Stück 10 der Reihe) – ein eigenes Stück, das gezielte Mitigation/Verstärkung ausgewählter Knoten gegen genau diese beiden Dynamiken behandeln würde.

## Lokal ausführen

```
python -m venv venv
venv\Scripts\pip install -r requirements-dev.txt
venv\Scripts\streamlit run app.py
```

## Literatur

- Motter, A. E., & Lai, Y.-C. (2002). *Cascade-based attacks on complex networks.* Physical Review E 66, 065102(R).
- Pastor-Satorras, R., & Vespignani, A. (2001). *Epidemic spreading in scale-free networks.* Physical Review Letters 86(14), 3200–3203.
- Brandes, U. (2001). *A faster algorithm for betweenness centrality.* Journal of Mathematical Sociology 25(2), 163–177 (Betweenness-Berechnung, wortgleich aus `robustheit-demo`/`centrality-demo`).

Gebaut mit Streamlit, Plotly, NumPy und pandas.
