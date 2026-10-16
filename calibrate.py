"""Calibration par simulations, utilisable AVANT la levée de l'aveuglement : elle n'utilise que la géométrie
(r/rM, masse, v_c) et les incertitudes de l'échantillon final, jamais les vitesses observées."""
import numpy as np, pandas as pd
from wbt_pipeline import nu, GNE, BASE

def step_stat(x, vt, x_low=0.3, x_high=1.0):
    lo, hi = vt[x < x_low], vt[x > x_high]
    if len(lo) < 10 or len(hi) < 10: return np.nan
    return np.median(hi) / np.median(lo) - 1

def mond_boost3d(x3):
    gi = 1.0/np.asarray(x3)**2; gt = np.hypot(gi, GNE); h = 1e-4
    Kd = (np.log(nu(gt*(1+h))) - np.log(nu(gt*(1-h)))) / (np.log(1+h) - np.log(1-h))
    return np.sqrt(nu(gt)*(1 + Kd/3*np.tanh((0.825*GNE/gi)**3.7)))

def kepler_geometry(n, rng, ecc="thermal"):
    e = np.sqrt(rng.random(n)) if ecc == "thermal" else rng.random(n)
    Mn = 2*np.pi*rng.random(n); E = Mn.copy()
    for _ in range(40): E = E - (E - e*np.sin(E) - Mn)/(1 - e*np.cos(E))
    r = np.stack([np.cos(E)-e, np.sqrt(1-e**2)*np.sin(E), np.zeros(n)], 1)
    v = np.stack([-np.sin(E), np.sqrt(1-e**2)*np.cos(E), np.zeros(n)], 1)/(1-e*np.cos(E))[:, None]
    u = rng.random((n, 3))
    a, b, c, d = (np.sqrt(1-u[:, 0])*np.sin(2*np.pi*u[:, 1]), np.sqrt(1-u[:, 0])*np.cos(2*np.pi*u[:, 1]),
                  np.sqrt(u[:, 0])*np.sin(2*np.pi*u[:, 2]), np.sqrt(u[:, 0])*np.cos(2*np.pi*u[:, 2]))
    R = np.stack([np.stack([1-2*(c*c+d*d), 2*(b*c-a*d), 2*(b*d+a*c)], 1),
                  np.stack([2*(b*c+a*d), 1-2*(b*b+d*d), 2*(c*d-a*b)], 1),
                  np.stack([2*(b*d-a*c), 2*(c*d+a*b), 1-2*(b*b+c*c)], 1)], 1)
    r = np.einsum("nij,nj->ni", R, r); v = np.einsum("nij,nj->ni", R, v)
    rsky = np.hypot(r[:, 0], r[:, 1])
    return v[:, :2]*np.sqrt(rsky)[:, None], np.linalg.norm(r, axis=1)/rsky

def simulate(x, vc, sv, truth, rng, vtmax=BASE["vtmax"], dvsky=BASE["dvsky"], ecc="thermal"):
    vt_vec, ratio = kepler_geometry(len(x), rng, ecc)
    if truth == "mond": vt_vec = vt_vec*mond_boost3d(x*ratio)[:, None]
    v = vt_vec*vc[:, None] + rng.normal(0, 1, (len(x), 2))*sv[:, None]
    va = np.linalg.norm(v, axis=1); vt = va/vc
    keep = (vt < vtmax) & (va < dvsky)
    return x[keep], vt[keep]

def calibrate(x, vc, sv, n_mocks=5000, seed=0, stat_kw=None, **kw):
    rng = np.random.default_rng(seed); stat_kw = stat_kw or {}
    out = {t: np.array([step_stat(*simulate(x, vc, sv, t, rng, **kw), **stat_kw) for _ in range(n_mocks)])
           for t in ("newton", "mond")}
    return pd.DataFrame(out)

def decide(S_obs, mocks, p_thr=0.00135):
    n, m = mocks.newton.dropna().values, mocks.mond.dropna().values
    p_newton = (np.sum(n >= S_obs) + 1)/(len(n) + 1)       # P(S ≥ S_obs | Newton)
    p_mond = (np.sum(m <= S_obs) + 1)/(len(m) + 1)         # P(S ≤ S_obs | MOND)
    # approximation gaussienne pour les queues au-delà de la résolution des simulations
    from scipy.stats import norm
    pg_newton = norm.sf(S_obs, n.mean(), n.std()); pg_mond = norm.cdf(S_obs, m.mean(), m.std())
    if p_mond < p_thr and pg_mond < p_thr and not (p_newton < p_thr): verdict = "MOND rejeté (Newton favorisé)"
    elif p_newton < p_thr and pg_newton < p_thr and not (p_mond < p_thr): verdict = "Newton rejeté (MOND favorisé)"
    elif p_newton < p_thr and p_mond < p_thr: verdict = "Les deux rejetés (ni Newton ni MOND : systématique ou autre physique)"
    else: verdict = "Non concluant"
    sep = (m.mean() - n.mean())/np.sqrt(0.5*(n.var() + m.var()))
    return dict(S_obs=S_obs, p_newton=p_newton, p_mond=p_mond, p_newton_gauss=pg_newton, p_mond_gauss=pg_mond,
                S_newton_mean=n.mean(), S_mond_mean=m.mean(), sigma_S=n.std(), separation_sigma=sep, verdict=verdict)
