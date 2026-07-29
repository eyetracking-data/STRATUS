# Revisionsübersicht für die EDBT-Fassung

## Neue wissenschaftliche Aussage

Die Arbeit behauptet nicht mehr, dass zeitliche Persistenz grundsätzlich besser
ist. Stattdessen trennt sie zwei Fragen:

1. Ist die neue Hybridarchitektur besser als die bisherigen diagnostischen
   Modelle?
2. Liefert das HMM innerhalb derselben Hybridarchitektur einen eigenständigen
   Mehrwert?

Dafür wird `STRATUS-H` ausschließlich gegen die identische punktweise Ablation
`Hybrid-P` ausgewertet. Beide besitzen dieselben Merkmale, Zustände, Regeln und
Aktionen. Nur STRATUS-H verwendet gelernte Übergänge und Viterbi-Decodierung.

## Hauptergebnisse

| Modell | MAE | RMSE | Q_local |
|---|---:|---:|---:|
| STRATUS-H | 3.598 | 14.615 | 0.920 |
| Hybrid-P | 3.825 | 14.821 | 0.914 |
| Pointwise-D | 3.566 | 22.703 | 0.882 |
| STRATUS-D | 4.849 | 32.456 | 0.841 |

Der isolierte HMM-Effekt gegenüber Hybrid-P beträgt:

- Q_local: `+0.0057`, 95%-KI `[0.0043, 0.0071]`
- MAE: `-0.227`, 95%-KI `[-0.320, -0.148]`
- RMSE: `-0.206`, 95%-KI `[-0.348, -0.063]`

Alle 14 Testidentifikatoren besitzen mit STRATUS-H einen höheren
teilnehmerbezogenen Q_local-Wert als mit Hybrid-P.

## Offen dargestellte Einschränkungen

- Pointwise-D besitzt im Haupttest weiterhin den geringfügig besseren MAE.
- Die operative Ground Truth wird durch kontrollierte Injektion erzeugt und ist
  keine natürliche Expertenannotation.
- Der feste Outer Split erlaubt genaue gepaarte Vergleiche, aber keine Aussage
  über die Streuung zwischen verschiedenen Outer Splits.
- Der Shift-Test verwendet nur zwei Seeds und ist deshalb eine Sensitivitäts-,
  keine zweite Bestätigungsstudie.
- Rohdaten und koordinatengenaue Referenzfenster werden aus Lizenz- und
  Datenverantwortungsgründen nicht im GitHub-Paket weitergegeben.

## Überarbeitete Teile

- Titel und Abstract
- Forschungsfragen und Beiträge
- Related Work mit moderner Data-Cleaning-/Pipeline-Literatur
- selektive Hybrid-HMM-Architektur
- matched Hybrid-P-Ablation
- Training/Development/Test-Trennung
- primärer Teilnehmer-Bootstrap
- Shift- und Gewichtssensitivität
- Ergebnis-, Diskussions- und Limitationsabschnitte
- neue Haupttabelle und drei aussagekräftige Abbildungen
- aktualisierte Artifacts-Sektion


## Neue qualitative Figure 2

Die abstrakte Ergebnisdarstellung wurde um zwei reale, zurückgehaltene
Zeitreihenbeispiele ergänzt. Dargestellt werden Clean Reference, degradierter
Input, Hybrid-P und STRATUS-H sowie die wahren und vorhergesagten
Regimeverläufe. Dadurch wird sichtbar, dass STRATUS-H nur einen kleinen Anteil
der Positionen verändert, dort aber kurze Zustandswechsel reduziert. Die
Beispiele wurden erst nach der vollständigen Auswertung zur Illustration
ausgewählt und dienen nicht als statistischer Nachweis. Die frühere zweigeteilte Scatter-/Bootstrap-Abbildung wurde vollständig entfernt, weil ihre Aussagen bereits präziser in der Haupttabelle und im Bootstrap-Text enthalten sind. Die bisherige Shift-Abbildung rückt dadurch automatisch zu Figure 3 auf.
