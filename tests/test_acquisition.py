"""v0.9.7 — Tests fonctionnels de la lecture des drapeaux de repli (audit du 20/10/2026, points F3 et F5).
Lancer depuis la racine : python tests/test_acquisition.py"""
import os, sys, tempfile
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # v0.9.11 : console Windows (cp1252)
import numpy as np, pandas as pd
import acquisition as acq

def _csv(df):
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8"); df.to_csv(f, index=False); f.close(); return f.name

def _stops(df, ids):
    try: acq.read_flags_file(_csv(df), ids)
    except SystemExit: return True
    return False

good = pd.DataFrame(dict(source_id=[1, 2, 3, 4], non_single_star=[0, 1, 0, 0], sea_env_flag=[0, 0, 1, np.nan]))
F = acq.read_flags_file(_csv(good), [1, 2, 3, 4])
ok = acq.flags_ok(F)
assert ok.loc[1] and not ok.loc[2] and not ok.loc[3], "drapeaux 0/1 mal lus"
assert not ok.loc[4], "un drapeau manquant (NaN) ne doit jamais être compté propre"
assert _stops(good.drop(columns="sea_env_flag"), [1, 2]), "colonne absente non détectée"
assert _stops(pd.concat([good, good.iloc[:1]]), [1, 2]), "source_id en double non détecté"
assert _stops(good, [1, 2, 99]), "source_id demandé mais absent non détecté"
assert _stops(good.assign(non_single_star=["0", "x", "0", "0"]), [1, 2]), "valeur non numérique non détectée"
d = tempfile.mkdtemp(); cwd = os.getcwd(); os.chdir(d)
try:
    acq.record("drX", "drapeaux", source="fichier", n=4); acq.record("drX", "epoques", source="DataLink")
    import json; r = json.load(open("ACQUISITION_drX.json", encoding="utf-8"))
    assert set(r) == {"drapeaux", "epoques"} and r["drapeaux"]["n"] == 4, "journal d'acquisition incomplet"
    acq.record("drX", "drapeaux", source="TAP", n=5); r = json.load(open("ACQUISITION_drX.json", encoding="utf-8"))
    assert r["drapeaux"]["n"] == 5 and r["drapeaux"]["historique"][0]["n"] == 4, "historique des réécritures perdu"
    try: acq.require("drX"); raise AssertionError("section « catalogue » manquante non détectée")
    except SystemExit: pass
    acq.record("drX", "catalogue", source="test"); acq.require("drX")
    os.remove("ACQUISITION_drX.json")
    try: acq.require("drX"); raise AssertionError("journal absent non détecté")
    except SystemExit: pass
finally: os.chdir(cwd)
print("tests d'acquisition : OK (9 cas)")
