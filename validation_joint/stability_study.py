"""v0.9.1 — Critères de stabilité du test conjoint (amendement du 18/10/2026, fixé avant tout calcul de Λ v0.9.1 sur des
vitesses observées). Simulation seulement : aucune vitesse observée n'entre dans la statistique des jeux simulés ; la géométrie est celle de
l'échantillon principal, dont la sélection utilise des coupes cinématiques (voir validation_joint/common.py).
Les critères portent sur des jeux simulés et non sur Λ observé, pour pouvoir être évalués sur DR4 AVANT la levée, et parce
que DR3 et DR4 observent les mêmes paires (Λ observé sur DR3 ne doit pas entrer dans une décision sur DR4).
(c) dispersion due aux gabarits : pour chaque jeu simulé, écart-type de Λ entre 5 tirages indépendants de gabarits ;
    moyenne quadratique sur 20 jeux (10 Newton central, 10 MOND central) < 1,0 ;
(d) influence d'une paire : pour chaque jeu, max |ΔΛ| en excluant les paires une à une ; 90e centile sur les 20 jeux < 1,5.
À lancer depuis la racine du dépôt après « blind » : python validation_joint/stability_study.py [dr3|dr4]"""
import os, sys, json, time
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
import jointlik as jl

sys.path.insert(0, os.path.join(os.getcwd(), "validation_joint"))
from common import load_inputs, provenance
t0 = time.time(); REL = sys.argv[1] if len(sys.argv) > 1 else "dr3"
I = load_inputs(REL); P, CC, GRID, R, G, zs, a_c, b_c, f, CONT = (I[k] for k in ("P", "CC", "GRID", "R", "G", "zs", "a_c", "b_c", "f", "CONT"))
rng = np.random.default_rng(910)                          # jeux d'ÉVALUATION (graine distincte des jeux de réglage)
mocks = [(g, jl.simulate_obs(G, a_c, b_c, g, f["raghavan2010"][1], "raghavan2010", CC, rng, P["vtmax"], P["dvsky"], grid=GRID))
         for g in [0]*10 + [1]*10]
L = np.zeros((5, len(mocks))); infl = np.zeros(len(mocks))
for s in range(5):
    T = jl.templates(G, zs, b_c, CONT, CC, np.random.default_rng(9100 + s), R=R, vtmax=P["vtmax"], dvsky=P["dvsky"], grid=GRID)
    for i, (g, (lv, ps, k)) in enumerate(mocks):
        parts = jl.lam_parts(lv, ps, k, zs, T); L[s, i] = jl.lam(lv, ps, k, zs, T, parts)
        if s == 0: infl[i] = np.abs(jl.loo_influence(lv, ps, k, zs, T, parts)[1]).max()
    print(f"gabarits {s + 1}/5 ({time.time()-t0:.0f} s)", flush=True)
sd = L.std(0, ddof=1); crit_c = float(np.sqrt((sd**2).mean())); crit_d = float(np.quantile(infl, 0.9))
out = pd.DataFrame(dict(hyp=["newton" if g == 0 else "mond" for g, _ in mocks], lam_moyen=L.mean(0), ecart_type_gabarits=sd, influence_max=infl))
p_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"stability_{REL}.csv"); out.to_csv(p_out, index=False)
provenance(REL, "validation_joint/stability_study.py", dict(jeux=910, gabarits=[9100 + s for s in range(5)]), p_out, R=R, n_jeux=len(mocks))
print(out.round(2).to_string(index=False))
print(f"CRITÈRE (c) dispersion due aux gabarits < 1,0 : {crit_c:.2f} → {'OK' if crit_c < 1.0 else 'NON'} ; "
      f"(d) influence d'une paire (90e centile) < 1,5 : {crit_d:.2f} → {'OK' if crit_d < 1.5 else 'NON'}")
print("STABILITÉ :", "OK" if (crit_c < 1.0 and crit_d < 1.5) else "NON (le test conjoint reste en analyse secondaire)")
