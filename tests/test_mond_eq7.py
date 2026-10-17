"""Test de conformité : le boost MOND du pipeline doit reproduire l'équation 7 de Cookson et al. 2026 (arXiv:2602.24035),
    v_c/v_c,N = sqrt( ν (1 + K/3 [tanh(0.825 g_N,e / g_N,i)]^3.7) ),  ν et K évalués en g_N,t = (g_N,i² + g_N,e²)^½,
recodée ici indépendamment. Valeurs de référence en x = r_sky / r_M (r = (3/2)^½ r_sky), a0 = 1,2e-10, g_N,e = 1,184 a0.
Lancement : python tests/test_mond_eq7.py  (depuis la racine du dépôt)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

def eq7_reference(x_sky, gne=1.184):
    r = np.asarray(x_sky, float)*np.sqrt(1.5)
    gi = 1.0/r**2; gt = np.sqrt(gi**2 + gne**2)
    nu = lambda y: 1.0/(1.0 - np.exp(-np.sqrt(y)))
    eps = 1e-6
    K = (np.log(nu(gt*np.exp(eps))) - np.log(nu(gt*np.exp(-eps))))/(2*eps)
    return np.sqrt(nu(gt)*(1 + K/3*np.tanh(0.825*gne/gi)**3.7))

REF = {0.3: 1.0340, 0.5: 1.1044, 0.7: 1.1556, 1.0: 1.1691, 1.2: 1.1660, 1.5: 1.1664, 2.0: 1.1687, 5.0: 1.1700}

def main():
    from wbt_pipeline import mond_boost
    from calibrate import mond_boost3d
    xs = np.array(sorted(REF)); ref = np.array([REF[x] for x in xs])
    errs = {"référence recodée": np.abs(eq7_reference(xs) - ref).max(),
            "wbt_pipeline.mond_boost": np.abs(mond_boost(xs) - ref).max(),
            "calibrate.mond_boost3d": np.abs(mond_boost3d(xs*np.sqrt(1.5)) - ref).max()}
    for k, v in errs.items(): print(f"{k:28s} écart max {v:.5f}")
    bad = [k for k, v in errs.items() if v > 5e-4]
    if bad: sys.exit(f"ÉCHEC conformité équation 7 : {bad}")
    print("Conformité à l'équation 7 de Cookson et al. 2026 : OK")

if __name__ == "__main__":
    main()
