"""Enveloppe de contamination de l'hypothèse nulle newtonienne (v0.5).
Une réalisation = vitesses newtoniennes sur la géométrie réelle de l'échantillon + compagnons internes cachés
(population de type Raghavan et al. 2010) + cribles DR4 (époques et image) + bruit réel + coupes.
N'utilise que la géométrie, les masses, les magnitudes et les incertitudes : jamais les vitesses observées
(utilisable avant la levée de l'aveuglement). Validation et justification : validation_contamination/."""
import os, numpy as np
from scipy.interpolate import RegularGridInterpolator
from wbt_pipeline import observe

_HERE = os.path.dirname(os.path.abspath(__file__))
_AX = (np.linspace(-2.5, 2.5, 21), np.linspace(-2.5, 4.0, 21))
_GRIDS = {}
def _grid(name):
    """'dr4' : crible par époques DR4 (pgrid.npy) ; 'dr3' : RUWE > 1,25 émulé sur la fenêtre DR3 (pgrid_dr3.npy)."""
    if name not in _GRIDS:
        f = {"dr4": "pgrid.npy", "dr3": "pgrid_dr3.npy"}[name]
        _GRIDS[name] = RegularGridInterpolator(_AX, np.load(os.path.join(_HERE, "validation_contamination", f)), bounds_error=False, fill_value=None)
    return _GRIDS[name]
_MTAB = np.array([0.05, 0.075, 0.08, 0.09, 0.10, 0.16, 0.23, 0.30, 0.37, 0.44, 0.50, 0.57, 0.60, 0.63, 0.70, 0.80, 1.0, 1.2])
_GTAB = np.array([20.0, 17.5, 17.0, 16.2, 15.1, 13.3, 12.0, 11.3, 10.6, 9.85, 9.25, 8.65, 8.4, 8.1, 7.5, 6.3, 4.7, 3.9])
_TAU = np.linspace(-2.75, 2.75, 40)

def p_epoch(alpha, Pyr, grid="dr4"):
    """Probabilité de détection astrométrique (mesurée sur l'échantillonnage réel de Gaia)."""
    la = np.log10(np.maximum(alpha, 1e-9))
    p = np.clip(_grid(grid)(np.stack([np.clip(la, -2.5, 2.5), np.clip(np.log10(Pyr), -2.5, 4.0)], -1)), 0, 1)
    return np.where(la < -2.5, 0.0, p)

SHAPES = {"raghavan2010": dict(kind="logP_days", mu=5.03, sig=2.28),          # type solaire (Raghavan et al. 2010)
          "mdwarf_pokemon": dict(kind="loga_au", mu=float(np.log10(7.91)), sig=1.1)}  # naines M (POKEMON 15 pc)

def draw_companions(m1, q, rng, shape):
    s = SHAPES[shape] if isinstance(shape, str) else shape
    n = len(m1)
    if s["kind"] == "logP_days":
        Pyr = 10**rng.normal(s["mu"], s["sig"], n)/365.25; a = (Pyr**2*m1*(1 + q))**(1/3)
    else:
        a = 10**rng.normal(s["mu"], s["sig"], n); Pyr = np.sqrt(a**3/(m1*(1 + q)))
    return a, Pyr

def mg_of_mass(m): return np.interp(m, _MTAB, _GTAB)

def theta50(dG): return np.interp(dG, [0, 4, 7], [0.6, 1.0, 2.0]) + np.where(dG > 7, (dG - 7)/3.0, 0.0)

def photocentre_pm(alpha, Pyr, rng):
    n = len(alpha)
    e = rng.random(n)*0.9; w = 2*np.pi*rng.random(n); W = 2*np.pi*rng.random(n); ci = rng.uniform(-1, 1, n)
    Mn = 2*np.pi*(_TAU[None, :]/Pyr[:, None] + rng.random(n)[:, None]); E = Mn.copy()
    for _ in range(40): E = E - (E - e[:, None]*np.sin(E) - Mn)/(1 - e[:, None]*np.cos(E))
    X, Y = np.cos(E) - e[:, None], np.sqrt(1 - e[:, None]**2)*np.sin(E)
    A = alpha*(np.cos(w)*np.cos(W) - np.sin(w)*np.sin(W)*ci); B = alpha*(np.cos(w)*np.sin(W) + np.sin(w)*np.cos(W)*ci)
    F = alpha*(-np.sin(w)*np.cos(W) - np.cos(w)*np.sin(W)*ci); G = alpha*(-np.sin(w)*np.sin(W) + np.cos(w)*np.cos(W)*ci)
    ra = B[:, None]*X + G[:, None]*Y; de = A[:, None]*X + F[:, None]*Y
    tc = _TAU - _TAU.mean()
    return np.stack([(ra*tc).sum(1), (de*tc).sum(1)], 1)/(tc**2).sum()

def contaminated_newton_cond(geo, f_in, rng, cfg, vtmax, dvsky, kepler_geometry_geo, max_iter=200, stats=None):
    """v0.9.9 — Réalisation newtonienne contaminée CONDITIONNÉE à la sélection : chaque paire est retirée jusqu'à passer le
    crible de contamination simulé et les coupes ṽ < vtmax et Δv < dvsky (et elles seules), comme la paire observée correspondante.
    kepler_geometry_geo(n, rng, geo_sous_ensemble). v0.9.10 : jusqu'à 200 essais, comptage dans stats.
    Renvoie (x, ṽ) des paires retenues (toutes, sauf de rares paires encore rejetées après max_iter essais)."""
    n = len(geo["x"]); vt = np.full(n, np.nan); kept = np.zeros(n, bool); todo = np.arange(n)
    for _ in range(max_iter):
        g = {k: v[todo] for k, v in geo.items()}
        v_, k_ = _contaminated_core(g, f_in, rng, cfg, vtmax, dvsky, lambda m, r: kepler_geometry_geo(m, r, g))
        ok = todo[k_]; vt[ok], kept[ok] = v_[k_], True; todo = todo[~k_]
        if not len(todo): break
    if stats is not None:                                    # v0.9.10-11 : comptes par zone
        import completude; completude.tally(stats, geo["x"], ~kept)
    return geo["x"][kept], vt[kept]

def contaminated_newton(geo, f_in, rng, cfg, vtmax, dvsky, kepler_geometry):
    """Une réalisation newtonienne contaminée. geo : dict d'arrays (x, vc, sv, d, rsky, M, m1, m2, G1, G2).
    Renvoie (x, ṽ) des paires retenues (version non conditionnée, conservée pour les scripts de validation)."""
    vt, keep = _contaminated_core(geo, f_in, rng, cfg, vtmax, dvsky, kepler_geometry)
    return geo["x"][keep], vt[keep]

def _contaminated_core(geo, f_in, rng, cfg, vtmax, dvsky, kepler_geometry):
    """Cœur commun : ṽ de toutes les paires et masque des paires retenues."""
    N = len(geo["x"])
    vt_vec, _ = kepler_geometry(N, rng)
    v = vt_vec*geo["vc"][:, None]
    M_hidden = np.zeros(N); kick = np.zeros((N, 2)); removed = np.zeros(N, bool)
    qlo, qhi = cfg["q_range"]; shape = cfg.get("shape", "raghavan2010")
    for mc, Gc in (("m1", "G1"), ("m2", "G2")):
        has = rng.random(N) < f_in
        q = rng.uniform(qlo, qhi, N); m1 = geo[mc]
        a_in, P_all = draw_companions(m1, q, rng, shape)
        has &= a_in < cfg["stability"]*geo["rsky"]
        i = np.where(has)[0]
        if len(i) == 0: continue
        Pyr, qq, aa, d = P_all[i], q[i], a_in[i], geo["d"][i]
        alpha = 1000*aa*np.abs(qq/(1 + qq) - qq**4/(1 + qq**4))/d
        kick[i] += 4.740470446*photocentre_pm(np.maximum(alpha, 1e-9), Pyr, rng)*d[:, None]/1000
        M_hidden[i] += qq*m1[i]
        scr = cfg["screen"]
        if scr == "aucun": continue
        det = rng.random(len(i)) < p_epoch(alpha, Pyr, "dr3" if scr.startswith("dr3_") else "dr4")
        if scr in ("epoques+image_pess", "epoques+image_opt", "dr3_ruwe+image_pess", "dr3_ruwe+image_opt"):
            dG = mg_of_mass(qq*m1[i]) - mg_of_mass(m1[i]); G2 = geo[Gc][i] + dG
            rho = aa*np.sqrt(1 - rng.uniform(-1, 1, len(i))**2)/d
            vis = G2 < 20.7
            if scr.endswith("image_pess"):
                p_img = vis/(1 + np.exp(-(np.log10(rho) - np.log10(theta50(dG)))/0.05))
            else:
                p_img = (vis & (rho > 0.15)).astype(float)
            det |= rng.random(len(i)) < p_img
        removed[i[det]] = True
    v = observe(v*np.sqrt(1 + M_hidden/geo["M"])[:, None] + kick, geo["sv"], rng)
    va = np.linalg.norm(v, axis=1); vt = va/geo["vc"]
    keep = (vt < vtmax) & (va < dvsky) & ~removed
    return vt, keep
