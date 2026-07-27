# Upload-Anleitung

1. Das ZIP im Verzeichnis `stratus-regime-aware-preprocessing/` entpacken.
2. Den neu entstandenen Ordner `pointwise-baseline/` vollständig zu GitHub
   hinzufügen.
3. Keine vorhandene Datei ersetzen. Der Ordner ist ein reines Add-on.
4. Die Rohdaten nicht hochladen. Sie bleiben unter `data/raw/` bzw. an den
   über Umgebungsvariablen gesetzten Pfaden.

Optionaler Kontrolllauf aus `stratus-regime-aware-preprocessing/`:

```bash
python pointwise-baseline/run_pointwise_ablation.py
```

Nach dem Upload muss dieser Link erreichbar sein:

<https://github.com/eyetracking-data/STRATUS/tree/main/stratus-regime-aware-preprocessing/pointwise-baseline>
