# STRATUS: erste Laufzeitmessung

Durchgeführt am 16.10.2026 auf dem lokalen Windows-Rechner.

## Verarbeitung von 8-Sekunden-Ausschnitten

Median über alle Wiederholungen, Testpersonen und fünf Korruptionsarten pro Datensatz. Zeiten in Millisekunden.

| Datensatz | Hybrid-P | STRATUS-H | Gepaarter Mehraufwand |
|---|---:|---:|---:|
| ETDD70 | 22.08 | 58.54 | +36.20 |
| Autism | 14.86 | 22.30 | +7.21 |

## Training

Gemeinsames Modell für beide Verfahren: Median 3.278 s aus fünf Fits; Spanne 3.183–3.854 s. Vorbereitung der 240 Trainingsfälle separat: 4.378 s.

## Messumfang und Grenzen

- 14 originale Testpersonen; 2-, 4- und 8-Sekunden-Ausschnitte; fünf Korruptionsarten, mittlere Stärke, Seed 0.
- Zehn Wiederholungen je Fall und Methode, Aufwärmläufe, alternierende Methodenreihenfolge.
- Diagnostik, Merkmalsberechnung, Vorhersage und Korrektur enthalten; Dateiimport, Korruption und Qualitätsmetriken ausgeschlossen.
- Veröffentlichtes C=1 und Übergangsstärke=1; Hyperparametersuche ausgeschlossen.
- BLAS auf einen Thread begrenzt. Betriebssystem, Sicherheitsprüfungen und Hintergrundlast bleiben Teil der lokalen Umgebung.
- Offline-Sequenzen: keine Aussage über Streaming-Latenz oder Sequenzen über acht Sekunden.
- Die Qualitätsgewinne wurden in diesem Lauf nicht neu überprüft; dies ist eine Laufzeitmessung.
- Die Zusammenfassung gewichtet Wiederholungen/Korruptionsfälle innerhalb jedes Datensatzes gleich; keine gepoolte datensatzübergreifende Zahl.

Umgebung: Python 3.13.16, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1.
Plattform: Windows-11-10.0.26200-SP0; Prozessor: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD; logische CPUs: 12.

Rohmessungen: results/runtime/timings.csv; Training: training.csv; Metadaten: environment.json.
Die exakte Referenzdatei wurde vor der Messung per SHA-256 geprüft.
