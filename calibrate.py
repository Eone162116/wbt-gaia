"""Calibration par simulations (v0.7-B), utilisable AVANT la levée de l'aveuglement : n'utilise que la géométrie
(r/rM, séparation, masse, v_c) et les incertitudes de l'échantillon final, jamais les vitesses observées.
Nouveautés v0.7-B (audit externe du 17/10/2026) :
  - simulations par lots (vectorisées) : 50 000 par hypothèse pure, 20 000 par scénario de contamination ;
  - scénarios d'excentricité dépendant de la séparation, p(e) ∝ e^α avec α croissant de α_bas à 1 kau à α_haut à 30 kau
    (Hwang, Ting & Zakamska 2022 : superthermique au-delà de 10³ au) ; chaque hypothèse est jugée dans le scénario qui lui
    est le plus favorable ;
  - règle de rejet sur la borne haute de Clopper-Pearson à 95 % de la probabilité de queue, et sur l'approximation gaussienne."""
import numpy as np, pandas as pd
from scipy.stats import beta as _beta, norm as _norm
from wbt_pipeline import nu, GNE, BASE, observe

def step_stat(x, vt, x_low=0.3, x_high=1.0):
    lo, hi = vt[x < x_low], vt[x > x_high]
    if len(lo) < 10 or len(hi) < 10: return np.nan
    return np.median(hi) / np.median(lo) - 1

def mond_boost3d(x3):
    gi = 1.0/np.asarray(x3)**2; gt = np.hypot(gi, GNE); h = 1e-4
    Kd = (np.log(nu(gt*(1+h))) - np.log(nu(gt*(1-h)))) / (np.log(1+h) - np.log(1-h))
    return np.sqrt(nu(gt)*(1 + Kd/3*np.tanh(0.825*GNE/gi)**3.7))   # éq. 7 : [tanh(·)]^3,7 (corrigé v0.7)

def alpha_profile(rsky, a_lo, a_hi, s_lo=1e3, s_hi=3e4):
    t = np.clip((np.log10(rsky) - np.log10(s_lo))/(np.log10(s_hi) - np.log10(s_lo)), 0, 1)
    return a_lo + (a_hi - a_lo)*t

def alpha_from_x(x, a_lo, a_hi, x_lo=0.3, x_hi=1.0):
    """v0.8 : α = α_bas pour x ≤ 0,3, α_haut pour x ≥ 1, interpolé en log x entre les deux."""
    t = np.clip((np.log10(x) - np.log10(x_lo))/(np.log10(x_hi) - np.log10(x_lo)), 0, 1)
    return a_lo + (a_hi - a_lo)*t

def sample_ecc(n, rng, ecc="thermal", rsky=None, x=None):
    """'thermal' (α = 1), 'uniform' (α = 0), nombre α, (α_bas, α_haut) en séparation avec rsky,
    ou {"x": (α_bas, α_haut)} en r/rM avec x (v0.8, excentricité mesurée par les angles v–r) : p(e) ∝ e^α."""
    if isinstance(ecc, dict):
        a = alpha_from_x(x, *ecc["x"])
    elif isinstance(ecc, str):
        a = {"thermal": 1.0, "uniform": 0.0}[ecc]
    elif np.ndim(ecc) == 0:
        a = float(ecc)
    else:
        a = alpha_profile(rsky, ecc[0], ecc[1])
    return np.minimum(rng.random(n)**(1.0/(np.asarray(a) + 1.0)), 0.999)

def _orbits(e, rng):
    """Orbites képlériennes (a = 1), phase et orientation aléatoires : ṽ vectoriel projeté, r3D/rsky, rsky (en unités de a)."""
    n = len(e)
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
    rsky_ = np.hypot(r[:, 0], r[:, 1])
    return v[:, :2]*np.sqrt(rsky_)[:, None], np.linalg.norm(r, axis=1)/rsky_, rsky_

def kepler_geometry(n, rng, ecc="thermal", rsky=None, x=None):
    """Orbites conditionnées à la séparation projetée observée. Par défaut, dN/d ln a uniforme (β = 1, Öpik) : la phase
    et l'orientation sont uniformes. v0.8.1 : si ecc = {"x": ..., "beta": (β_bas, β_haut)}, la distribution des
    demi-grands axes dN/da ∝ a^-β (β interpolé en log x comme α) est prise en compte par rééchantillonnage pondéré
    ρ^(β-1), ρ = séparation projetée / a (4 candidats par paire)."""
    beta = ecc.get("beta") if isinstance(ecc, dict) else None
    if beta is None:
        vt, ratio, _ = _orbits(sample_ecc(n, rng, ecc, rsky, x), rng)
        return vt, ratio
    Kc = 4
    xr = np.repeat(x, Kc); rr = None if rsky is None else np.repeat(rsky, Kc)
    vt, ratio, rho = _orbits(sample_ecc(n*Kc, rng, ecc, rr, xr), rng)
    b = alpha_from_x(xr, *beta)
    w = (np.maximum(rho, 1e-6)**(b - 1.0)).reshape(n, Kc)
    cw = np.cumsum(w, 1); pick = (cw < rng.random(n)[:, None]*cw[:, -1:]).sum(1)
    idx = np.arange(n)*Kc + np.minimum(pick, Kc - 1)
    return vt[idx], ratio[idx]

def simulate(x, vc, sv, truth, rng, vtmax=BASE["vtmax"], dvsky=BASE["dvsky"], ecc="thermal", rsky=None):
    """Une réalisation (compatibilité : utilisée par les scripts de validation)."""
    vt_vec, ratio = kepler_geometry(len(x), rng, ecc, rsky, x)
    if truth == "mond": vt_vec = vt_vec*mond_boost3d(x*ratio)[:, None]
    v = observe(vt_vec*vc[:, None], sv, rng)
    va = np.linalg.norm(v, axis=1); vt = va/vc
    keep = (vt < vtmax) & (va < dvsky)
    return x[keep], vt[keep]

def simulate_batch(x, vc, sv, truth, rng, nb, vtmax=BASE["vtmax"], dvsky=BASE["dvsky"], ecc="thermal", rsky=None, stat_kw=None):
    """nb réalisations vectorisées ; renvoie les nb valeurs de S."""
    stat_kw = stat_kw or {}; N = len(x)
    X = np.tile(x, nb); VC = np.tile(vc, nb); SV = np.tile(np.asarray(sv), (nb, 1)) if np.ndim(sv) == 2 else np.tile(sv, nb)
    RS = None if rsky is None else np.tile(rsky, nb)
    vt_vec, ratio = kepler_geometry(N*nb, rng, ecc, RS, X)
    if truth == "mond": vt_vec = vt_vec*mond_boost3d(X*ratio)[:, None]
    v = observe(vt_vec*VC[:, None], SV, rng)
    va = np.linalg.norm(v, axis=1); vt = va/VC
    keep = (vt < vtmax) & (va < dvsky)
    out = np.empty(nb)
    for k in range(nb):
        s = slice(k*N, (k + 1)*N); kk = keep[s]
        out[k] = step_stat(X[s][kk], vt[s][kk], **stat_kw)
    return out

def calibrate(x, vc, sv, n_mocks=50000, seed=0, stat_kw=None, rsky=None, ecc="thermal", ecc_scenarios=None,
              vtmax=BASE["vtmax"], dvsky=BASE["dvsky"], batch=50):
    """Colonnes : newton, mond (excentricité de référence), puis newton_<scénario>, mond_<scénario>."""
    rng = np.random.default_rng(seed); out = {}
    runs = [("", ecc)] + [("_" + k, v if isinstance(v, dict) else tuple(v)) for k, v in (ecc_scenarios or {}).items()]
    for suffix, e in runs:
        for truth in ("newton", "mond"):
            vals = []
            for i in range(0, n_mocks, batch):
                vals.append(simulate_batch(x, vc, sv, truth, rng, min(batch, n_mocks - i), vtmax, dvsky, e, rsky, stat_kw))
            out[truth + suffix] = np.concatenate(vals)
    return pd.DataFrame(out)

def calibrate_envelope(geo, n_per_f=20000, seed=0, stat_kw=None, cfg=None, vtmax=BASE["vtmax"], dvsky=BASE["dvsky"], scenarios=None, ecc="thermal",
                       max_iter=200, missing_max=0.001):
    """Hypothèse nulle newtonienne contaminée (excentricité thermique, le cas le plus défavorable à Newton).
    scenarios : liste (forme, f_in) ; par défaut cfg['f_in'] (mode fixe, v0.5)."""
    import contamination as ct
    rng = np.random.default_rng(seed + 1); stat_kw = stat_kw or {}; rows = []; stats = {"paires": 0, "manquantes": 0}
    scenarios = scenarios or [(cfg.get("shape", "raghavan2010"), f) for f in cfg["f_in"]]
    kgg = lambda n, r, g: kepler_geometry(n, r, ecc, g.get("rsky"), g["x"])   # v0.8 ; v0.9.9 : sur le sous-ensemble tiré
    for shape, f in scenarios:
        c = dict(cfg, shape=shape)
        for _ in range(n_per_f):
            xx, vv = ct.contaminated_newton_cond(geo, f, rng, c, vtmax, dvsky, kgg, max_iter, stats)   # v0.9.9 : N simulé = N observé
            rows.append((shape, f, step_stat(xx, vv, **stat_kw)))
    frac = stats["manquantes"]/max(stats["paires"], 1)
    if frac > missing_max:                                   # v0.9.10 : règle fixée avant les données
        raise SystemExit(f"CALIBRATION REFUSÉE (enveloppe) : {stats['manquantes']} tirages de paires non reconstitués sur {stats['paires']} "
                         f"({frac:.3%} > {missing_max:.3%}).")
    out = pd.DataFrame(rows, columns=["shape", "f_in", "S"]); out.attrs["conditionnel"] = dict(stats, fraction=frac)
    return out

def _tail(s, S_obs, side):
    """Probabilité de queue : estimation ponctuelle k/N, borne haute Clopper-Pearson à 95 %, approximation gaussienne."""
    s = np.asarray(s, float); s = s[np.isfinite(s)]; N = len(s)
    k = int(np.sum(s >= S_obs)) if side == "upper" else int(np.sum(s <= S_obs))
    cp = 1.0 if k >= N else float(_beta.ppf(0.95, k + 1, N - k))
    g = float(_norm.sf(S_obs, s.mean(), s.std())) if side == "upper" else float(_norm.cdf(S_obs, s.mean(), s.std()))
    return dict(k=k, N=N, p=k/N, p_cp95=cp, p_gauss=g, S_mean=float(s.mean()), S_sd=float(s.std()))

def decide(S_obs, mocks, p_thr=0.00135, env=None):
    """Chaque hypothèse est jugée dans le scénario qui lui est le plus favorable :
    Newton : Newton pur thermique et chaque scénario de contamination (S le plus haut) ;
    MOND : MOND pur dans chaque scénario d'excentricité (S le plus bas).
    Rejet seulement si la borne haute Clopper-Pearson à 95 % ET l'approximation gaussienne sont sous le seuil."""
    rows = [dict(hyp="newton", scenario="pur_thermique", **_tail(mocks["newton"].values, S_obs, "upper"))]
    if env is not None:
        keys = ["shape", "f_in"] if "shape" in env.columns else ["f_in"]
        for key, g in env.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            lab = "_".join(str(k if isinstance(k, str) else round(float(k), 3)) for k in key)
            rows.append(dict(hyp="newton", scenario="contamination_" + lab, **_tail(g.S.values, S_obs, "upper")))
    for c in mocks.columns:
        if c.startswith("mond"):
            rows.append(dict(hyp="mond", scenario="thermique" if c == "mond" else c[5:], **_tail(mocks[c].values, S_obs, "lower")))
        elif c.startswith("newton_mes"):           # v0.8 : excentricité mesurée, partie de l'hypothèse nulle
            rows.append(dict(hyp="newton", scenario=c[7:], **_tail(mocks[c].values, S_obs, "upper")))
        elif c.startswith("newton_"):
            rows.append(dict(hyp="newton_info", scenario=c[7:], **_tail(mocks[c].values, S_obs, "upper")))
    T = pd.DataFrame(rows)
    nn, mm = T[T.hyp == "newton"], T[T.hyp == "mond"]
    pN_cp, pN_g, pM_cp, pM_g = nn.p_cp95.max(), nn.p_gauss.max(), mm.p_cp95.max(), mm.p_gauss.max()
    newton_rej = pN_cp < p_thr and pN_g < p_thr
    mond_rej = pM_cp < p_thr and pM_g < p_thr
    if mond_rej and not newton_rej: verdict = "MOND rejeté dans tous ses scénarios (Newton favorisé)"
    elif newton_rej and not mond_rej: verdict = "Newton rejeté dans tous ses scénarios (MOND favorisé)"
    elif newton_rej and mond_rej: verdict = "Les deux rejetés (ni Newton ni MOND : systématique ou autre physique)"
    else: verdict = "Non concluant"
    n0, m0 = T[(T.hyp == "newton") & (T.scenario == "pur_thermique")].iloc[0], T[(T.hyp == "mond") & (T.scenario == "thermique")].iloc[0]
    S_N_max, S_M_min = nn.S_mean.max(), mm.S_mean.min()
    sd = 0.5*(n0.S_sd + m0.S_sd)
    return dict(S_obs=S_obs, p_newton_cp95=pN_cp, p_newton_gauss=pN_g, p_newton_pur=n0.p, p_mond_cp95=pM_cp, p_mond_gauss=pM_g,
                p_mond_thermique=m0.p, S_newton_mean=n0.S_mean, S_newton_max=S_N_max, S_mond_mean=m0.S_mean, S_mond_min=S_M_min,
                sigma_S=n0.S_sd, separation_sigma=(m0.S_mean - n0.S_mean)/sd, separation_sigma_pire=(S_M_min - S_N_max)/sd,
                scenarios=T.to_dict("records"), verdict=verdict)
