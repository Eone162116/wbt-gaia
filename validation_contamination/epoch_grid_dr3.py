"""Émulation du drapeau RUWE de DR3 : probabilité que RUWE > 1,25 en fonction de (α photocentre [mas], P [an]),
sur la fenêtre DR3 (34 mois, 25/07/2014 – 28/05/2017) de l'échantillonnage réel d'une source G≈14 (pré-publication DR4).
RUWE émulé = UWE du modèle à 5 paramètres / UWE médiane sans compagnon."""
import numpy as np, pandas as pd, sys
from astropy.io.votable import parse_single_table
from epoch_accel import transits_from_table, test_source
rng = np.random.default_rng(11)
TR = transits_from_table(parse_single_table("GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml").to_table())
ref = TR[TR.source_id == 1663617687609809280]
ref = ref[(ref.tau > -2.94) & (ref.tau < -0.09)].reset_index(drop=True)
uwe_ref = float(pd.DataFrame([test_source(ref)]).uwe_5p.iloc[0])
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
def uwe(T): return float(pd.DataFrame([test_source(T)]).uwe_5p.iloc[0])
# normalisation : UWE médiane sans compagnon
null = []
for _ in range(200):
    T = ref.copy(); T["w"] = rng.normal(0, 1, len(T))*T.sig*uwe_ref; null.append(uwe(T))
u0 = float(np.median(null))
if part == 0: print(f"{len(ref)} transits DR3 ; UWE médiane nulle {u0:.3f} ; faux positifs RUWE>1,25 : {np.mean(np.array(null)/u0 > 1.25):.3f}")
out = np.full((len(LA), len(LP)), np.nan)
for i, la in enumerate(LA):
    if i % npart != part: continue
    for j, lp in enumerate(LP):
        det = 0
        for _ in range(NDRAW):
            da, dd = track(ref.tau.values, 10**la, 10**lp)
            T = ref.copy(); T["w"] = da*T["sin"] + dd*T["cos"] + rng.normal(0, 1, len(T))*T.sig*uwe_ref
            det += uwe(T)/u0 > 1.25
        out[i, j] = det/NDRAW
np.save(f"pgrid_dr3_{part}.npy", out)
