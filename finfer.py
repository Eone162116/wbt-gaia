"""v0.6 — Mesure de la fraction de compagnons cachés f_in à partir des taux de drapeaux, sans aucune vitesse
(compatible avec l'aveuglement).
Observable : fraction de paires candidates (coupes indépendantes des vitesses) dont une composante est signalée
  - DR3 : RUWE > 1,25 ; détection modélisée par validation_contamination/pgrid_dr3.npy (RUWE émulé, fenêtre DR3) ;
  - DR4 : crible par époques (accélération ou UWE) ; détection modélisée par validation_contamination/pgrid.npy.
Modèle : P(paire signalée | f) = 1 − Π_c (1 − r0)(1 − f · p̄_c), p̄_c = probabilité moyenne qu'un compagnon tiré
de la population (forme donnée ; instable si a_in ≥ 0,2 × séparation de la binaire large) soit signalé.
Inversion de la moyenne sur les paires = taux observé ; bande ± nsig (binomial), élargie sur r0 et sur la forme."""
import numpy as np
from scipy.optimize import brentq
import contamination as ct

def pbar(m1, d, rsky, shape, grid, rng, ndraw=300, stability=0.2, q_range=(0.05, 1.0)):
    n = len(m1); q = rng.uniform(*q_range, (n, ndraw))
    a, Pyr = ct.draw_companions(np.repeat(m1, ndraw), q.ravel(), rng, shape)
    a, Pyr = a.reshape(n, ndraw), Pyr.reshape(n, ndraw)
    stable = a < stability*rsky[:, None]
    alpha = 1000*a*np.abs(q/(1 + q) - q**4/(1 + q**4))/d[:, None]
    return (ct.p_epoch(alpha, Pyr, grid)*stable).mean(1)

def rate_model(f, p1, p2, r0):
    return float(np.mean(1 - (1 - r0)**2*(1 - f*p1)*(1 - f*p2)))

def _solve(target, p1, p2, r0):
    lo, hi = rate_model(0, p1, p2, r0), rate_model(1, p1, p2, r0)
    if target <= lo: return 0.0
    if target >= hi: return 1.0
    return brentq(lambda f: rate_model(f, p1, p2, r0) - target, 0, 1)

def infer_band(flags, m1, m2, d, rsky, shapes, grid, r0_range, nsig=2.0, seed=0, stability=0.2, q_range=(0.05, 1.0)):
    """Renvoie {forme: (f_bas, f_central, f_haut)} : bande élargie sur r0 ∈ r0_range et ± nsig binomial."""
    rng = np.random.default_rng(seed)
    N = len(flags); r = float(np.mean(flags)); sr = np.sqrt(r*(1 - r)/N); out = {}
    for shape in shapes:
        p1 = pbar(m1, d, rsky, shape, grid, rng, stability=stability, q_range=q_range)
        p2 = pbar(m2, d, rsky, shape, grid, rng, stability=stability, q_range=q_range)
        lo = _solve(r - nsig*sr, p1, p2, max(r0_range)); hi = _solve(r + nsig*sr, p1, p2, min(r0_range))
        mid = _solve(r, p1, p2, float(np.mean(r0_range)))
        out[shape] = (lo, mid, hi)
    return out, r, sr
