"""v0.9.13 — Tests déterministes du filtre HR et du multivers (audit du 21/10/2026, P1). Aucune donnée réelle.
Lancer depuis la racine : python tests/test_hr_multiverse.py"""
import os, sys, json
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from wbt_pipeline import hr_band, in_hr
import multivers as mv
# 1. hr_band : médiane de MG − 3,3 (BP−RP) sur toutes les composantes dans les bornes ; une étoile répétée compte deux fois
K = pd.DataFrame(dict(c1=[1.0, 1.0, 2.0, 0.2], MG1=[5.0, 5.0, 9.0, 3.0], c2=[1.5, 3.0, 1.2, 1.0], MG2=[6.5, 13.0, 6.0, 15.0]))
b, s, h = hr_band(K)
r = np.array([5.0 - 3.3, 5.0 - 3.3, 9.0 - 6.6, 6.5 - 4.95, 13.0 - 9.9, 6.0 - 3.96])       # bornes : (0.2, 3.0) et (1.0, 15.0) exclues
assert abs(b - np.median(r)) < 1e-12 and s == 3.3 and h == 0.8, f"hr_band : {b} au lieu de {np.median(r)}"
assert in_hr(np.array([1.0]), np.array([b + 3.3 + 0.39]), b, 3.3, 0.8)[0] and not in_hr(np.array([1.0]), np.array([b + 3.3 + 0.41]), b, 3.3, 0.8)[0]
# 2. ordonnées du multivers
for o, d in (("mediane", 0.0), ("mediane-0.2", -0.2), ("mediane-0.1", -0.1), ("mediane+0.1", 0.1), ("mediane+0.2", 0.2)):
    assert abs(mv.ordonnee(o, 1.861, None) - (1.861 + d)) < 1e-12, o
assert mv.ordonnee("cookson", 1.861, (2.05, 3.4)) == 2.05 and mv.pente("cookson", 3.3, (2.05, 3.4)) == 3.4 and mv.pente("mediane+0.1", 3.3, None) == 3.3
# 3. dénombrement avec la configuration réelle : 396 / 1584 sans Cookson, 468 / 1872 avec
G = json.load(open("config_prereg.json", encoding="utf-8"))["multiverse"]
n = {(rel, hc is not None): len(mv.specs(G, rel, hc)) for rel in ("dr3", "dr4") for hc in (None, 2.0)}
assert n == {("dr3", False): 396, ("dr4", False): 1584, ("dr3", True): 468, ("dr4", True): 1872}, n
# la spécification principale figure une fois, avec l'ordonnée médiane
P = json.load(open("config_prereg.json", encoding="utf-8"))["primary"]
prim = [x for x in mv.specs(G, "dr4") if x[1] == P["hr_height"] and x[2] == "mediane" and x[0]["degroup"] == P["degroup"]
        and x[0]["sph"] == P["sph"] and x[0]["dmax"] == P["dmax"] and x[0]["ruwe"] == P["ruwe"] and x[0]["screen"] == P["screen"] and x[0]["epoch"] == P["epoch"]]
assert len(prim) == 1, f"spécification principale trouvée {len(prim)} fois"
assert P["hr_intercept"] is None, "la spécification principale doit utiliser la médiane (hr_intercept nul)"
# 4. garde de la valeur de Cookson
ok = dict(valeur=2.0, date_limite="2026-11-10", recue_le="2026-11-03", source="courriel archivé", nature="centre")
assert mv.check_cookson(dict(ok, valeur=None)) is None and mv.check_cookson(ok) == (2.0, 3.3)
pts = dict(ok, valeur=None, points=[[0.5, 4.0], [3.5, 14.0]])                  # droite de Cookson, par deux points (courriel du 26/10/2026)
c_, m_ = mv.check_cookson(pts); assert abs(m_ - 10/3) < 1e-12 and abs(c_ - 7/3) < 1e-12, (c_, m_)
assert abs(m_*0.5 + c_ - 4.0) < 1e-12 and abs(m_*3.5 + c_ - 14.0) < 1e-12
for bad in (dict(ok, recue_le="2026-11-12"), dict(ok, recue_le=None), dict(ok, source=None), dict(ok, nature="borne inférieure")):
    try: mv.check_cookson(bad); raise AssertionError(f"valeur de Cookson non refusée : {bad}")
    except SystemExit: pass
print("tests du filtre HR et du multivers : OK (4 groupes, 396/1584 et 468/1872 spécifications)")
