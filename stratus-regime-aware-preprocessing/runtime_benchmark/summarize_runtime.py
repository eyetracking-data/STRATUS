from pathlib import Path
import csv
import json
import statistics as st

root = Path(__file__).resolve().parent
import argparse
parser = argparse.ArgumentParser(description='Rebuild the runtime summary from paired measurements.')
parser.add_argument('--results', type=Path, default=root/'results/runtime')
args = parser.parse_args()
folder = args.results
with (folder/'timings.csv').open(newline='') as f:
    rows = list(csv.DictReader(f))
with (folder/'training.csv').open(newline='') as f:
    fits = [float(r['seconds']) for r in csv.DictReader(f)]
env = json.loads((folder/'environment.json').read_text())
assert len(rows) == 14*3*5*10*2, len(rows)
lines = ['# STRATUS: erste Laufzeitmessung', '',
    'Durchgeführt am 16.10.2026 auf dem lokalen Windows-Rechner.', '',
    '## Verarbeitung von 8-Sekunden-Ausschnitten', '',
    'Median über alle Wiederholungen, Testpersonen und fünf Korruptionsarten pro Datensatz. Zeiten in Millisekunden.', '',
    '| Datensatz | Hybrid-P | STRATUS-H | Gepaarter Mehraufwand |',
    '|---|---:|---:|---:|']
for ds in ('ETDD70','Autism'):
    selected = [r for r in rows if r['dataset']==ds and r['duration_setting_s']=='8']
    med = {m: st.median(float(r['milliseconds']) for r in selected if r['method']==m)
           for m in ('Hybrid-P','STRATUS-H')}
    pairs = {}
    for r in selected:
        key = (r['case_segment'], r['corruption'], r['repeat'])
        pairs.setdefault(key,{})[r['method']] = float(r['milliseconds'])
    delta = st.median(p['STRATUS-H']-p['Hybrid-P'] for p in pairs.values())
    lines.append(f"| {ds} | {med['Hybrid-P']:.2f} | {med['STRATUS-H']:.2f} | {delta:+.2f} |")
lines += ['', '## Training', '',
    f'Gemeinsames Modell für beide Verfahren: Median {st.median(fits):.3f} s aus fünf Fits; '
    f'Spanne {min(fits):.3f}–{max(fits):.3f} s. '
    f'Vorbereitung der {env["training_cases"]} Trainingsfälle separat: {env["training_preparation_seconds"]:.3f} s.', '',
    '## Messumfang und Grenzen', '',
    '- 14 originale Testpersonen; 2-, 4- und 8-Sekunden-Ausschnitte; fünf Korruptionsarten, mittlere Stärke, Seed 0.',
    '- Zehn Wiederholungen je Fall und Methode, Aufwärmläufe, alternierende Methodenreihenfolge.',
    '- Diagnostik, Merkmalsberechnung, Vorhersage und Korrektur enthalten; Dateiimport, Korruption und Qualitätsmetriken ausgeschlossen.',
    '- Veröffentlichtes C=1 und Übergangsstärke=1; Hyperparametersuche ausgeschlossen.',
    '- BLAS auf einen Thread begrenzt. Betriebssystem, Sicherheitsprüfungen und Hintergrundlast bleiben Teil der lokalen Umgebung.',
    '- Offline-Sequenzen: keine Aussage über Streaming-Latenz oder Sequenzen über acht Sekunden.',
    '- Die Qualitätsgewinne wurden in diesem Lauf nicht neu überprüft; dies ist eine Laufzeitmessung.',
    '- Die Zusammenfassung gewichtet Wiederholungen/Korruptionsfälle innerhalb jedes Datensatzes gleich; keine gepoolte datensatzübergreifende Zahl.', '',
    f'Umgebung: Python {env["python"].split()[0]}, NumPy {env["numpy"]}, pandas {env["pandas"]}, scikit-learn {env["sklearn"]}.',
    f'Plattform: {env["platform"]}; Prozessor: {env["processor"]}; logische CPUs: {env["logical_cpus"]}.', '',
    'Rohmessungen: results/runtime/timings.csv; Training: training.csv; Metadaten: environment.json.',
    'Die exakte Referenzdatei wurde vor der Messung per SHA-256 geprüft.']
(folder/'RUNTIME_RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print('\n'.join(lines[:17]))
