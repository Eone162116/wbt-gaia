"""v0.9 — Test conjoint (norme et direction des vitesses) : rapport de vraisemblance profilé Λ = 2 [ln L(MOND) − ln L(Newton)].
Observables par paire : log ṽ et ψ (angle séparation–vitesse relative), dans deux zones de x (x < 0,3 ; x > 1).
Gabarits 2D simulés sur la géométrie, le bruit et les coupes de l'échantillon, pour chaque hypothèse, chaque α (par zone)
et chaque scénario de contamination (forme × f_in mesurés), avec la pente β mesurée.
Nuisances profilées : α dans chaque zone, scénario de contamination (commun aux zones), échelle globale η de ṽ (±0,05 dex,
commune aux zones : absorbe les erreurs de masse uniformes). Λ n'utilise donc ni la normalisation absolue de ṽ, ni une
excentricité supposée. Simulation seulement avant la levée ; Λ observé calculé une seule fois à la levée."""
import numpy as np
from wbt_pipeline import observe
import calibrate as cal
import contamination as ct

LV = np.linspace(-1.6, np.log10(2.5), 11)          # bornes en log ṽ
PS = np.linspace(0, 90, 7)                          # bornes en ψ (degrés)
AL = np.round(np.arange(0.0, 3.01, 0.25), 2)       # grille de α
ETA = np.linspace(-0.05, 0.05, 11)                 # décalages globaux en log ṽ (dex)

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
    x = G["x"]; n = len(x)
    a = cal.alpha_from_x(x, *a_lohi); b = cal.alpha_from_x(x, *b_lohi)
    X = np.repeat(np.arange(n), kc)
    e = np.minimum(rng.random(n*kc)**(1.0/(a[X] + 1.0)), 0.999)
    rs, vt, ratio, rho = _orbits(e, rng)
    w = (np.maximum(rho, 1e-6)**(b[X] - 1.0)).reshape(n, kc); cw = np.cumsum(w, 1)
    pick = np.minimum((cw < rng.random(n)[:, None]*cw[:, -1:]).sum(1), kc - 1); j = np.arange(n)*kc + pick
    rs, vt, ratio = rs[j], vt[j], ratio[j]
    if gamma: vt = vt*cal.mond_boost3d(x*ratio)[:, None]
    v = vt*G["vc"][:, None]*np.sqrt(mass_bias)
    removed = np.zeros(n, bool); Mh = np.zeros(n)
    if f_in > 0:
        for mc, Gc in (("m1", "G1"), ("m2", "G2")):
            has = rng.random(n) < f_in; q = rng.uniform(*ccfg["q_range"], n); m1 = G[mc]
            a_in, Pyr = ct.draw_companions(m1, q, rng, shape)
            has &= a_in < ccfg["stability"]*G["rsky"]; i = np.where(has)[0]
            if len(i) == 0: continue
            qq, aa, dd = q[i], a_in[i], G["d"][i]
            al = 1000*aa*np.abs(qq/(1 + qq) - qq**4/(1 + qq**4))/dd
            v[i] += 4.740470446*ct.photocentre_pm(np.maximum(al, 1e-9), Pyr[i], rng)*dd[:, None]/1000
            Mh[i] += qq*m1[i]
            det = rng.random(len(i)) < ct.p_epoch(al, Pyr[i], grid)
            dG = ct.mg_of_mass(qq*m1[i]) - ct.mg_of_mass(m1[i]); G2 = G[Gc][i] + dG
            rh = aa*np.sqrt(1 - rng.uniform(-1, 1, len(i))**2)/dd
            det |= rng.random(len(i)) < (G2 < 20.7)/(1 + np.exp(-(np.log10(rh) - np.log10(ct.theta50(dG)))/0.05))
            removed[i[det]] = True
    v = observe(v*np.sqrt(1 + Mh/G["M"])[:, None], G["sv"], rng)
    va = np.linalg.norm(v, axis=1); keep = (va/G["vc"] < vtmax) & (va < dvsky) & ~removed
    c = np.abs((rs*v).sum(1))/np.linalg.norm(rs, axis=1)/np.maximum(va, 1e-12)
    return np.log10(np.maximum(va/G["vc"], 1e-6)), np.degrees(np.arccos(np.clip(c, 0, 1))), keep

def hist2(lv, ps):
    h, _, _ = np.histogram2d(np.clip(lv, LV[0], LV[-1] - 1e-9), ps, [LV, PS]); return h.ravel()

def sub(G, idx, rep=1):
    k = np.tile(idx, rep); return {key: (val[k] if np.ndim(val) == 1 else val[k]) for key, val in G.items()}

def templates(G, zones, beta, cont_scen, ccfg, rng, R=30, vtmax=2.5, dvsky=10.0, grid="dr3"):
    """T[z, γ, c, α, bin] (log-probabilités lissées). cont_scen : liste de (forme, f_in)."""
    T = np.zeros((2, 2, len(cont_scen), len(AL), (len(LV) - 1)*(len(PS) - 1)))
    for zi, z in enumerate(("bas", "haut")):
        Gz = sub(G, zones[z], R)
        for g in (0, 1):
            for ci, (shape, f) in enumerate(cont_scen):
                for ai, a in enumerate(AL):
                    lv, ps, k = simulate_obs(Gz, (a, a), beta, g, f, shape, ccfg, rng, vtmax, dvsky, grid=grid)
                    h = hist2(lv[k], ps[k]) + 0.3; T[zi, g, ci, ai] = np.log(h/h.sum())
    return T

def lam(lv, ps, keep, zones, T):
    """Λ = 2 [max ln L(MOND) − max ln L(Newton)], profilé sur α (par zone), contamination et η."""
    best = np.full(2, -np.inf)
    for e in ETA:
        H = np.stack([hist2(lv[zones[z]][keep[zones[z]]] - e, ps[zones[z]][keep[zones[z]]]) for z in ("bas", "haut")])
        ll = np.einsum("zb,zgcab->zgca", H, T).max(3).sum(0)          # (γ, c) : max sur α par zone, somme des zones
        best = np.maximum(best, ll.max(1))
    return float(2*(best[1] - best[0]))
