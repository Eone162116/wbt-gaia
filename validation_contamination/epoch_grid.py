"""Probabilité de détection par le crible par époques en fonction de (α photocentre [mas], P [an]),
indépendante de la distance. Échantillonnage et bruit réels de LSPM J1324+6124 (G=14,3, pré-publication DR4)."""
import os, numpy as np, pandas as pd, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
# Usage (depuis la racine, en 3 parties parallèles) : python validation_contamination/epoch_grid.py 0 3  (puis 1 3, 2 3)
# Nécessite validation_epoch/GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml (pré-publication ESA)
from astropy.io.votable import parse_single_table
from epoch_accel import transits_from_table, test_source, flag
rng = np.random.default_rng(7)
TR = transits_from_table(parse_single_table(os.path.join(os.path.dirname(HERE), "validation_epoch", "GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml")).to_table())
ref = TR[TR.source_id == 1663617687609809280].reset_index(drop=True)
uwe = float(pd.DataFrame([test_source(ref)]).uwe_5p.iloc[0])
LA = np.linspace(-2.5, 2.5, 21); LP = np.linspace(-2.5, 4.0, 21); NDRAW = 60
part, npart = int(sys.argv[1]), int(sys.argv[2])
def track(tau, alpha, P):
    e = rng.random()*0.9; w = 2*np.pi*rng.random(); W = 2*np.pi*rng.random(); ci = rng.uniform(-1, 1)
    M = 2*np.pi*(tau/P + rng.random()); E = M.copy()
    for _ in range(50): E = E - (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
    X, Y = np.cos(E) - e, np.sqrt(1 - e**2)*np.sin(E)
    A = alpha*(np.cos(w)*np.cos(W) - np.sin(w)*np.sin(W)*ci); B = alpha*(np.cos(w)*np.sin(W) + np.sin(w)*np.cos(W)*ci)
    F = alpha*(-np.sin(w)*np.cos(W) - np.cos(w)*np.sin(W)*ci); G = alpha*(-np.sin(w)*np.sin(W) + np.cos(w)*np.cos(W)*ci)
    return B*X + G*Y, A*X + F*Y
out = np.full((len(LA), len(LP)), np.nan)
for i, la in enumerate(LA):
    if i % npart != part: continue
    for j, lp in enumerate(LP):
        det = 0
        for _ in range(NDRAW):
            da, dd = track(ref.tau.values, 10**la, 10**lp)
            T = ref.copy(); T["w"] = da*T["sin"] + dd*T["cos"] + rng.normal(0, 1, len(T))*T.sig*uwe
            det += bool(flag(pd.DataFrame([test_source(T)]), 1e-3, 1.4).iloc[0])
        out[i, j] = det/NDRAW
np.save(os.path.join(HERE, f"pgrid_{part}.npy"), out)
# assemblage : np.nanmax des 3 parties -> pgrid.npy
