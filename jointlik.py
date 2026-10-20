"""v0.9.1 — Test conjoint lissé et continu : Λ = 2 [max ln L(MOND) − max ln L(Newton)].
Observables par paire : log ṽ et ψ (angle séparation–vitesse relative), zones x < 0,3 et x > 1.
Changements par rapport à la v0.9 (instabilité constatée le 18/10/2026 : une paire pouvait déplacer Λ de 6,8) :
- gabarits = densités 2D fines (0,01 dex × 1°) lissées par un noyau gaussien (H_LV, H_PS), simulées SANS la coupe ṽ < vtmax ;
- vraisemblance évaluée par interpolation bilinéaire au point observé (plus de cases ; continue en les données) ;
- échelle globale η : la densité est décalée de η puis tronquée à vtmax et renormalisée exactement (Z(η) = F(c − η)) ;
- α interpolé linéairement entre les nœuds de gabarits (densités mélangées), profilé sur une grille fine (pas 0,1) ;
  η profilé sur une grille fine (pas 0,005 dex) : avec des densités lisses, Λ est une fonction continue des données ;
- nombres aléatoires communs : dans une zone, toutes les configurations (γ, contamination, α) partagent les mêmes tirages
  d'orbites, de compagnons et de bruit, si bien que le bruit Monte-Carlo des gabarits se compense dans Λ ;
- plancher : mélange (1 − EPS) × gabarit + EPS × densité uniforme sur le domaine observable (paires hors modèle).
Les hyperparamètres (H_LV, H_PS, EPS, grilles, R = 90 dans la configuration) ont été fixés le 18/10/2026 sur des jeux simulés
à la géométrie DR3 (graine de réglage distincte de celle de l'évaluation), avant tout calcul de Λ v0.9.1 sur des vitesses observées.
Physique des simulations (orbites, MOND, contamination, bruit, coupes) inchangée depuis la v0.9."""
import numpy as np
from scipy.ndimage import gaussian_filter

DLV, DPS = 0.01, 1.0
LVG = np.round(np.arange(-1.8, 0.6 + 1e-9, DLV), 4)    # centres en log ṽ
PSG = np.arange(0.0, 90.0 + 1e-9, DPS)                 # centres en ψ (degrés)
H_LV, H_PS = 0.06, 6.0                                  # largeurs du noyau (dex, degrés), fixées le 18/10/2026
EPS = 0.01                                              # poids du plancher uniforme
AL = np.round(np.arange(0.0, 3.01, 0.25), 2)            # nœuds de α (gabarits simulés)
AF = np.round(np.linspace(0.0, 3.0, 31), 3)             # α fin (profilage, pas 0,1)
ETA = np.round(np.linspace(-0.05, 0.05, 21), 5)         # η fin (profilage, pas 0,005 dex)
from wbt_pipeline import observe
import calibrate as cal
import contamination as ct

# v0.9.3 : l'ancien bloc de constantes v0.9 (LV, PS, AL, ETA à 11 points) qui écrasait ETA ci-dessus a été retiré
# (audit externe du 19/10/2026, point L). Les études v0.9.1 et v0.9.2 avaient tourné avec un pas de η de 0,01 dex.

def _orbits(e, rng):
    n = len(e)
    Mn = 2*np.pi*rng.random(n); E = Mn.copy()
    for _ in range(40): E = E - (E - e*np.sin(E) - Mn)/(1 - e*np.cos(E))
    r = np.stack([np.cos(E) - e, np.sqrt(1 - e**2)*np.sin(E), np.zeros(n)], 1)
    v = np.stack([-np.sin(E), np.sqrt(1 - e**2)*np.cos(E), np.zeros(n)], 1)/(1 - e*np.cos(E))[:, None]
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    w = rng.normal(size=(n, 3)); w -= (w*u).sum(1)[:, None]*u; w /= np.linalg.norm(w, axis=1)[:, None]
    Rm = np.stack([w, np.cross(u, w), u], 1)
    r3 = np.einsum("nij,nj->ni", Rm, r); v3 = np.einsum("nij,nj->ni", Rm, v)
    rs = r3[:, :2]; rn = np.linalg.norm(rs, axis=1)
    return rs, v3[:, :2]*np.sqrt(rn)[:, None], np.linalg.norm(r3, axis=1)/rn, rn

def simulate_obs(G, a_lohi, b_lohi, gamma, f_in, shape, ccfg, rng, vtmax=2.5, dvsky=10.0, kc=4, mass_bias=1.0, grid="dr3"):
    """Une réalisation de l'échantillon G (dict d'arrays : x, vc, sv (N,4), d, rsky, M, m1, m2, G1, G2).
    Renvoie (log ṽ, ψ, retenu) par paire."""
    r_orb, r_con, r_obs = rng if isinstance(rng, tuple) else (rng, rng, rng)   # v0.9.1 : flux séparés possibles
    x = G["x"]; n = len(x)
    a = cal.alpha_from_x(x, *a_lohi); b = cal.alpha_from_x(x, *b_lohi)
    X = np.repeat(np.arange(n), kc)
    e = np.minimum(r_orb.random(n*kc)**(1.0/(a[X] + 1.0)), 0.999)
    rs, vt, ratio, rho = _orbits(e, r_orb)
    w = (np.maximum(rho, 1e-6)**(b[X] - 1.0)).reshape(n, kc); cw = np.cumsum(w, 1)
    pick = np.minimum((cw < r_orb.random(n)[:, None]*cw[:, -1:]).sum(1), kc - 1); j = np.arange(n)*kc + pick
    rs, vt, ratio = rs[j], vt[j], ratio[j]
    if gamma: vt = vt*cal.mond_boost3d(x*ratio)[:, None]
    v = vt*G["vc"][:, None]*np.sqrt(mass_bias)
    removed = np.zeros(n, bool); Mh = np.zeros(n)
    if f_in > 0:
        for mc, Gc in (("m1", "G1"), ("m2", "G2")):
            has = r_con.random(n) < f_in; q = r_con.uniform(*ccfg["q_range"], n); m1 = G[mc]
            a_in, Pyr = ct.draw_companions(m1, q, r_con, shape)
            has &= a_in < ccfg["stability"]*G["rsky"]; i = np.where(has)[0]
            if len(i) == 0: continue
            qq, aa, dd = q[i], a_in[i], G["d"][i]
            al = 1000*aa*np.abs(qq/(1 + qq) - qq**4/(1 + qq**4))/dd
            v[i] += 4.740470446*ct.photocentre_pm(np.maximum(al, 1e-9), Pyr[i], r_con)*dd[:, None]/1000
            Mh[i] += qq*m1[i]
            det = r_con.random(len(i)) < ct.p_epoch(al, Pyr[i], grid)
            dG = ct.mg_of_mass(qq*m1[i]) - ct.mg_of_mass(m1[i]); G2 = G[Gc][i] + dG
            rh = aa*np.sqrt(1 - r_con.uniform(-1, 1, len(i))**2)/dd
            det |= r_con.random(len(i)) < (G2 < 20.7)/(1 + np.exp(-(np.log10(rh) - np.log10(ct.theta50(dG)))/0.05))
            removed[i[det]] = True
    v = observe(v*np.sqrt(1 + Mh/G["M"])[:, None], G["sv"], r_obs)
    va = np.linalg.norm(v, axis=1); keep = (va/G["vc"] < vtmax) & (va < dvsky) & ~removed
    c = np.abs((rs*v).sum(1))/np.linalg.norm(rs, axis=1)/np.maximum(va, 1e-12)
    return np.log10(np.maximum(va/G["vc"], 1e-6)), np.degrees(np.arccos(np.clip(c, 0, 1))), keep

def sub(G, idx, rep=1):
    k = np.tile(idx, rep); return {key: (val[k] if np.ndim(val) == 1 else val[k]) for key, val in G.items()}


def simulate_obs_cond(G, *args, max_iter=200, stats=None, **kw):
    """v0.9.9 — Jeu simulé CONDITIONNÉ à la sélection : chaque paire de la géométrie observée est retirée jusqu'à passer les
    mêmes coupes que les données (crible de contamination, ṽ < vtmax, Δv sur le ciel). Raison : la géométrie observée a déjà
    passé ces coupes ; en retirant les paires simulées détectées, les jeux simulés gardaient 65 à 76 % des paires (DR3).
    Coupes simulées, et elles seules : crible de contamination simulé, ṽ < vtmax, Δv sur le ciel < dvsky. Les autres étapes de
    sélection (filtre HR, RUWE, dégroupage, présence et écart des vitesses radiales) agissent par la géométrie observée, fixe.
    v0.9.10-11 : jusqu'à 200 essais ; stats (dict) reçoit les comptes par zone (completude.tally). Quasi-complétude contrôlée
    avec tolérance fixée : la fonction peut rendre moins de paires que N, et la règle par scénario et par zone en décide.
    Échantillonnage par rejet, paire par paire (les paires sont indépendantes à géométrie donnée) : la loi de chaque paire est
    exactement sa loi conditionnelle. Après max_iter essais, les rares paires restantes sont marquées non retenues.
    Mêmes arguments que simulate_obs ; renvoie (log ṽ, ψ, retenue)."""
    n = len(G["x"]); lv = np.full(n, np.nan); ps = np.full(n, np.nan); kept = np.zeros(n, bool); todo = np.arange(n)
    for _ in range(max_iter):
        a, b, k = simulate_obs(sub(G, todo), *args, **kw)
        ok = todo[k]; lv[ok], ps[ok], kept[ok] = a[k], b[k], True; todo = todo[~k]
        if not len(todo): break
    if stats is not None:                                     # v0.9.11 : comptes par zone (completude.tally)
        import completude; completude.tally(stats, G["x"], ~kept)
    return lv, ps, kept

def check_completeness(stats, max_frac, label, diag_path=None, section=None):
    """v0.9.10-11 : relais vers completude.check (règle par scénario et par zone, diagnostic écrit avant tout refus)."""
    import completude
    return completude.check(stats, max_frac, label, diag_path, section)

def _density(lv, ps):
    """Densité lissée non tronquée sur la grille fine (par dex et par degré) et sa fonction de répartition en log ṽ."""
    e_lv = np.r_[LVG - DLV/2, LVG[-1] + DLV/2]; e_ps = np.r_[PSG - DPS/2, PSG[-1] + DPS/2]
    h, _, _ = np.histogram2d(np.clip(lv, LVG[0], LVG[-1]), np.clip(ps, 0, 90), [e_lv, e_ps])
    h = gaussian_filter(h, (H_LV/DLV, H_PS/DPS), mode=("nearest", "reflect"))
    D = h/(h.sum()*DLV*DPS)
    cdf = np.r_[0.0, np.cumsum(D.sum(1)*DPS*DLV)]          # aux bords e_lv
    return D, cdf

def templates(G, zones, beta, cont_scen, ccfg, rng, R=30, vtmax=2.5, dvsky=10.0, grid="dr3"):
    """Gabarits lissés : dict(D[z, γ, c, α, lv, ψ] (densité par dex et par degré), cdf[z, γ, c, α, bord], c=log10 vtmax)."""
    nz, ng, nc, na = 2, 2, len(cont_scen), len(AL)
    Dz = np.zeros((nz, ng, nc, na, len(LVG), len(PSG)), np.float32); cdf = np.zeros((nz, ng, nc, na, len(LVG) + 1))   # float32 : identique en calibration et à la levée
    for zi, z in enumerate(("bas", "haut")):
        Gz = sub(G, zones[z], R); kids = np.random.SeedSequence(int(rng.integers(2**62))).spawn(3)
        for g in (0, 1):
            for ci, (shape, f) in enumerate(cont_scen):
                for ai, a in enumerate(AL):
                    crn = tuple(np.random.default_rng(kd) for kd in kids)      # mêmes tirages pour γ, c et α
                    lv, ps, k = simulate_obs(Gz, (a, a), beta, g, f, shape, ccfg, crn, np.inf, dvsky, grid=grid)
                    D, F = _density(lv[k], ps[k])
                    Dz[zi, g, ci, ai] = D; cdf[zi, g, ci, ai] = F
    return dict(D=Dz, cdf=cdf, c=float(np.log10(vtmax)))

def _lnp_zone(lv, ps, T, zi):
    """ln p par paire pour une zone : tableau [γ, c, α_fin, η, paire] (float32). Interpolation bilinéaire vectorisée
    sur tous les gabarits à la fois ; densité décalée de η, tronquée à c = log10 vtmax et renormalisée."""
    c = T["c"]; e_lv = np.r_[LVG - DLV/2, LVG[-1] + DLV/2]; U = 1.0/((c - LVG[0])*90.0)
    lv = np.clip(lv, LVG[0], c); n = len(lv); ne = len(ETA)
    D = T["D"][zi]; ng, nc, na, nl, npsi = D.shape; Df = D.reshape(ng*nc*na, nl*npsi)
    u = np.clip((lv[None, :] - ETA[:, None] - LVG[0])/DLV, 0, nl - 1 - 1e-9); v = np.clip(ps/DPS, 0, npsi - 1 - 1e-9)
    i0 = np.floor(u).astype(int); fu = u - i0; j0 = np.floor(v).astype(int)[None, :]; fv = (v - np.floor(v))[None, :]
    k00 = (i0*npsi + j0).ravel(); fu = fu.ravel(); fv = np.broadcast_to(fv, (ne, n)).ravel()
    d = ((1 - fu)*(1 - fv)*Df[:, k00] + (1 - fu)*fv*Df[:, k00 + 1] + fu*(1 - fv)*Df[:, k00 + npsi] + fu*fv*Df[:, k00 + npsi + 1])
    Z = np.stack([np.interp(c - ETA, e_lv, F) for F in T["cdf"][zi].reshape(-1, nl + 1)])      # P(log ṽ_sim + η < c)
    P = ((1 - EPS)*d.reshape(-1, ne, n)/np.maximum(Z, 1e-12)[:, :, None] + EPS*U).reshape(ng, nc, na, ne, n)
    W = np.clip(1 - np.abs((AF[:, None] - AL[None, :])/(AL[1] - AL[0])), 0, None)     # (α_fin, nœud) : interpolation linéaire
    return np.log(np.einsum("fa,gcaen->gcfen", W, P, optimize=True)).astype(np.float32)

def lam_parts(lv, ps, keep, zones, T):
    """ln p par paire et par zone (pour Λ et l'influence par exclusion une à une)."""
    return [(_lnp_zone(lv[zones[z]][keep[zones[z]]], ps[zones[z]][keep[zones[z]]], T, zi), np.asarray(zones[z])[keep[zones[z]]])
            for zi, z in enumerate(("bas", "haut"))]

def _lam_from(sums):
    ll = sum(s.max(2) for s in sums)                     # (γ, c, η) : α profilé par zone, zones sommées
    best = ll.reshape(ll.shape[0], -1).max(1)            # contamination et η communs aux zones
    return float(2*(best[1] - best[0]))

def lam(lv, ps, keep, zones, T, parts=None):
    """Λ = 2 [max ln L(MOND) − max ln L(Newton)], profilé sur α (par zone), contamination et η (communs)."""
    parts = parts or lam_parts(lv, ps, keep, zones, T)
    return _lam_from([p.sum(-1, dtype=np.float64) for p, _ in parts])

def loo_influence(lv, ps, keep, zones, T, parts=None):
    """ΔΛ = Λ(sans la paire i) − Λ pour chaque paire retenue des deux zones. Renvoie (indices, ΔΛ, Λ)."""
    parts = parts or lam_parts(lv, ps, keep, zones, T)
    sums = [p.sum(-1, dtype=np.float64) for p, _ in parts]; L0 = _lam_from(sums)
    idx, dl = [], []
    for zi, (p, ii) in enumerate(parts):
        for j in range(p.shape[-1]):
            s2 = list(sums); s2[zi] = sums[zi] - p[..., j]
            idx.append(ii[j]); dl.append(_lam_from(s2) - L0)
    return np.array(idx), np.array(dl), L0
