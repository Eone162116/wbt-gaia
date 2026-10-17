"""v0.8 — Excentricité mesurée sur l'échantillon par les angles v–r (Tokovinin 1998 ; Hwang, Ting & Zakamska 2022).
ψ = angle entre la séparation projetée et la vitesse relative projetée (corrigée de la perspective), replié sur [0°, 90°].
Dans le modèle opérationnel, le boost MOND est scalaire : ψ ne dépend pas de |ṽ|, la mesure se fait donc à l'étape aveugle.
α (p(e) ∝ e^α) est ajusté dans chaque zone de x (x < 0,3 et x > 1) par maximum de vraisemblance multinomiale sur
l'histogramme de ψ, avec des gabarits simulés sur la géométrie, le bruit (covariance complète) et les coupes de chaque paire.
Validation (17/10/2026) : injection-récupération sans biais pour α thermique, croissant ou décroissant ; un boost MOND
ne biaise pas α au-delà de la dispersion (validation_eccentricity/)."""
import numpy as np
from wbt_pipeline import observe

EDGES = np.linspace(0, 90, 10)
ALPHAS = np.round(np.arange(0.0, 3.01, 0.25), 2)

def _sky_rv(n, alpha, rng):
    e = np.minimum(rng.random(n)**(1/(alpha + 1)), 0.999)
    Mn = 2*np.pi*rng.random(n); E = Mn.copy()
    for _ in range(40): E = E - (E - e*np.sin(E) - Mn)/(1 - e*np.cos(E))
    r = np.stack([np.cos(E) - e, np.sqrt(1 - e**2)*np.sin(E), np.zeros(n)], 1)
    v = np.stack([-np.sin(E), np.sqrt(1 - e**2)*np.cos(E), np.zeros(n)], 1)/(1 - e*np.cos(E))[:, None]
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    w = rng.normal(size=(n, 3)); w -= (w*u).sum(1)[:, None]*u; w /= np.linalg.norm(w, axis=1)[:, None]
    Rm = np.stack([w, np.cross(u, w), u], 1)
    r3 = np.einsum("nij,nj->ni", Rm, r); v3 = np.einsum("nij,nj->ni", Rm, v)
    rs, vs = r3[:, :2], v3[:, :2]; rn = np.linalg.norm(rs, axis=1)
    return rs, vs*np.sqrt(rn)[:, None], np.linalg.norm(r3, axis=1)/rn

def templates(idx, x, vc, sv, rng, truth="newton", R=100, vtmax=2.5, dvsky=10.0, alphas=ALPHAS):
    """Histogrammes de ψ attendus (normalisés) pour chaque α, sur les paires idx."""
    from calibrate import mond_boost3d
    out = []
    X = np.repeat(idx, R)
    for a in alphas:
        rs, vt, ratio = _sky_rv(len(X), a, rng)
        if truth == "mond": vt = vt*mond_boost3d(x[X]*ratio)[:, None]
        v = observe(vt*vc[X][:, None], sv[X], rng)
        va = np.linalg.norm(v, axis=1); keep = (va/vc[X] < vtmax) & (va < dvsky)
        c = np.abs((rs*v).sum(1))/np.linalg.norm(rs, axis=1)/np.maximum(va, 1e-12)
        h, _ = np.histogram(np.degrees(np.arccos(np.clip(c, 0, 1)))[keep], EDGES)
        out.append((h + 0.5)/(h + 0.5).sum())
    return np.array(out)

def fit(psi_obs, tpl, alphas=ALPHAS):
    """α central et bande à ±2σ (Δ ln L = 2), bornée à la grille."""
    obs, _ = np.histogram(psi_obs, EDGES)
    ll = (obs[None, :]*np.log(tpl)).sum(1)
    fine = np.linspace(alphas[0], alphas[-1], 601); llf = np.interp(fine, alphas, ll); ok = fine[llf >= llf.max() - 2.0]
    return float(ok.min()), float(fine[np.argmax(llf)]), float(ok.max())

def measure(psi, x, vc, sv, rng, truth="newton", R=100, vtmax=2.5, dvsky=10.0, x_lo=0.3, x_hi=1.0):
    """{'bas': (α_min, α, α_max), 'haut': (...), 'n_bas': n, 'n_haut': n}."""
    out = {}
    for z, idx in (("bas", np.where(x < x_lo)[0]), ("haut", np.where(x > x_hi)[0])):
        tpl = templates(idx, x, vc, sv, rng, truth, R, vtmax, dvsky)
        out[z] = fit(psi[idx], tpl); out["n_" + z] = int(len(idx))
    return out

def scenarios(band_newton, band_mond):
    """Scénarios d'excentricité mesurés : central, le plus favorable à Newton (S maximal : α bas élevé, α haut faible),
    le plus favorable à MOND (S minimal : α bas faible, α haut élevé)."""
    bn, bm = band_newton, band_mond
    return {"mes_central": {"x": (bn["bas"][1], bn["haut"][1])},
            "mes_pire_newton": {"x": (bn["bas"][2], bn["haut"][0])},
            "mes_pire_mond": {"x": (bm["bas"][0], bm["haut"][2])}}
