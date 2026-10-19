"""HISTORIQUE (étude exploratoire antérieure à la v0.9, conservée pour la traçabilité ; audit externe du 19/10/2026, points E, F, G, I).
Ses chiffres ne se transfèrent pas au test conjoint lissé v0.9.1 : vraisemblance binnée, sans renormalisation exacte de la
troncature ni contamination ; seuils à 3σ extrapolés d'une loi normale estimée sur 150 jeux ; seuils recalculés dans chaque
scénario perturbé (les taux de rejet de ces scénarios ne sont donc pas ceux de la règle nominale figée) ; « biais de masse
+20 % » = masse gravitationnelle vraie 20 % plus grande que la masse photométrique (vitesses × √1,2), sans recalcul des
masses, de x ni de la sélection.
Vraisemblance conjointe (ṽ, ψ) vs marche S : étude de puissance par simulation seulement (aucune vitesse observée dans la
statistique simulée ; la géométrie provient d'une sélection qui utilise des coupes cinématiques).
Géométrie, bruit et coupes de l'échantillon principal DR3 ; β fixé aux valeurs mesurées (séparations) ; contamination non
incluse (comparaison à armes égales des deux tests). Vérités : Newton ou MOND avec α (bas, haut) = (2,0 ; 0,25), le
scénario central mesuré sur DR3. Test conjoint : rapport de vraisemblance profilé sur α_bas et α_haut, à partir de gabarits
2D (log ṽ, ψ) par zone. Test S : rejet au-delà des coins de la bande d'α mesurée par ψ (règle v0.8), approximation gaussienne."""
import os, sys, json, time
sys.path.insert(0, os.getcwd())   # à lancer depuis la racine du dépôt après « build » : python validation_joint/joint_power.py
import numpy as np, pandas as pd
from wbt_pipeline import select, hr_band, observe
import calibrate as cal, eccfit as ef
from scipy.stats import norm

t0 = time.time()
CFG = json.load(open("config_prereg.json", encoding="utf-8")); P = CFG["primary"]
K = pd.read_pickle("K_dr3.pkl"); hb, hs, _ = hr_band(K)
kw = {k: P[k] for k in ["dmax", "smin", "smax", "vtmax", "svt", "ruwe", "drv", "dvsky", "sph", "hr", "ipd", "degroup", "screen", "epoch"]}
m, _ = select(K, (hb, hs, P["hr_height"]), **kw); S0 = K[m].reset_index(drop=True)
BETA = (1.43, 2.14); A_TRUE = (2.0, 0.25); rng = np.random.default_rng(31)
LV = np.linspace(-1.6, np.log10(2.5), 11); PS = np.linspace(0, 90, 7)
AL = np.round(np.arange(0.0, 3.01, 0.25), 2)

def draw(x, vc, sv, alpha, gamma, beta, kc=4, f_kick=0.0, mass_bias=1.0):
    """Une orbite par paire (α et β constants dans la zone), avec ψ ; gamma=1 : boost MOND."""
    n = len(x); X = np.repeat(np.arange(n), kc)
    rs, vt, ratio = ef._sky_rv(n*kc, alpha, rng); rho = np.linalg.norm(rs, axis=1)
    w = (rho**(beta - 1)).reshape(n, kc); cw = np.cumsum(w, 1)
    pick = np.minimum((cw < rng.random(n)[:, None]*cw[:, -1:]).sum(1), kc - 1); j = np.arange(n)*kc + pick
    rs, vt, ratio = rs[j], vt[j], ratio[j]
    if gamma: vt = vt*cal.mond_boost3d(x*ratio)[:, None]
    vv = vt*vc[:, None]*np.sqrt(mass_bias)
    if f_kick > 0:                                # coups de vitesse log-uniformes 0,03–1 km/s, direction aléatoire
        k = rng.random(n) < f_kick; amp = 10**rng.uniform(np.log10(0.03), 0.0, k.sum()); ph = rng.uniform(0, 2*np.pi, k.sum())
        vv[k] += np.stack([amp*np.cos(ph), amp*np.sin(ph)], 1)
    v = observe(vv, sv, rng); va = np.linalg.norm(v, axis=1)
    keep = (va/vc < P["vtmax"]) & (va < P["dvsky"])
    c = np.abs((rs*v).sum(1))/np.linalg.norm(rs, axis=1)/np.maximum(va, 1e-12)
    return np.log10(va/vc)[keep], np.degrees(np.arccos(np.clip(c, 0, 1)))[keep]

def hist2(lv, ps):
    h, _, _ = np.histogram2d(np.clip(lv, LV[0], LV[-1] - 1e-9), ps, [LV, PS]); return h

def zone_arrays(S, scale):
    out = {}
    for z, mk in (("bas", S.x.values < 0.3), ("haut", S.x.values > 1)):
        Sz = S[mk]; rep = np.tile(np.arange(len(Sz)), scale)
        out[z] = (Sz.x.values[rep], Sz.vc.values[rep], np.stack([Sz.cv_aa, Sz.cv_ad, Sz.cv_dd, Sz.fd], 1)[rep], BETA[0] if z == "bas" else BETA[1])
    return out

def templates(Z, R=40):
    T = {}
    for z, (x, vc, sv, b) in Z.items():
        xr, vcr, svr = np.tile(x, R), np.tile(vc, R), np.tile(sv, (R, 1))
        for g in (0, 1):
            T[(z, g)] = np.array([(lambda h: (h + 0.3)/(h + 0.3).sum())(hist2(*draw(xr, vcr, svr, a, g, b))) for a in AL])
    return T

def fit_psi(p, tpl):
    obs, _ = np.histogram(p, PS); ll = (obs[None, :]*np.log(tpl)).sum(1)
    fine = np.linspace(AL[0], AL[-1], 601); llf = np.interp(fine, AL, ll); ok = fine[llf >= llf.max() - 2.0]
    return float(ok.min()), float(fine[np.argmax(llf)]), float(ok.max())

ETA = np.linspace(-0.05, 0.05, 11)          # décalage global en log ṽ (dex)
def lnL_prof(H, T, z, g):
    return float(np.max((H[None]*np.log(T[(z, g)])).sum((1, 2))))
def lam_eta(lvs, pss, T, Z):
    best = {0: -np.inf, 1: -np.inf}
    for e in ETA:
        H = {z: hist2(lvs[z] - e, pss[z]) for z in Z}
        for g in (0, 1): best[g] = max(best[g], sum(lnL_prof(H[z], T, z, g) for z in Z))
    return 2*(best[1] - best[0])

def run(scale, nsets=150, f_kick=0.0, mass_bias=1.0, tag=""):
    Z = zone_arrays(S0, scale); T = templates(Z)
    # gabarits ψ (1D) pour la bande d'α du test S, et grille de moyennes de S (Newton, MOND) en fonction de (α_bas, α_haut)
    Tpsi = {z: T[(z, 0)].sum(1) for z in Z}
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]; mu = {}
    xa = np.concatenate([Z["bas"][0], Z["haut"][0]]); vca = np.concatenate([Z["bas"][1], Z["haut"][1]]); sva = np.concatenate([Z["bas"][2], Z["haut"][2]])
    for g in (0, 1):
        for al in grid:
            for ah in grid:
                vals = []
                for _ in range(30):
                    lvb, _ = draw(Z["bas"][0], Z["bas"][1], Z["bas"][2], al, g, BETA[0]); lvh, _ = draw(Z["haut"][0], Z["haut"][1], Z["haut"][2], ah, g, BETA[1])
                    vals.append(np.median(10**lvh)/np.median(10**lvb) - 1)
                mu[(g, al, ah)] = (np.mean(vals), np.std(vals))
    interp = lambda g, al, ah, k: float(np.mean([mu[(g, min(grid, key=lambda v: abs(v - al)), min(grid, key=lambda v: abs(v - ah)))][k]]))
    res = []
    for truth in (0, 1):
        for _ in range(nsets):
            H = {}; S_lv = {}; ps = {}
            for z, (x, vc, sv, b) in Z.items():
                lv, p = draw(x, vc, sv, A_TRUE[0] if z == "bas" else A_TRUE[1], truth, b, f_kick=f_kick, mass_bias=mass_bias)
                S_lv[z] = lv; ps[z] = p
            lam = lam_eta(S_lv, ps, T, Z)
            S = np.median(10**S_lv["haut"])/np.median(10**S_lv["bas"]) - 1
            bnd = {z: fit_psi(ps[z], Tpsi[z]/Tpsi[z].sum(1, keepdims=True)) for z in Z}
            mN = interp(0, bnd["bas"][2], bnd["haut"][0], 0); sN = interp(0, bnd["bas"][2], bnd["haut"][0], 1)
            mM = interp(1, bnd["bas"][0], bnd["haut"][2], 0); sM = interp(1, bnd["bas"][0], bnd["haut"][2], 1)
            res.append(dict(truth=truth, lam=lam, S=S, S_newton_rej=S > mN + 3*sN, S_mond_rej=S < mM - 3*sM))
    R = pd.DataFrame(res)
    lN, lM = R[R.truth == 0].lam, R[R.truth == 1].lam
    thrN = lN.mean() + norm.isf(0.00135)*lN.std()      # Newton rejeté si Λ au-dessus
    thrM = lM.mean() - norm.isf(0.00135)*lM.std()      # MOND rejeté si Λ en dessous
    out = dict(cas=tag, echelle=scale, lam_newton_moy=float(lN.mean()), lam_mond_moy=float(lM.mean()), N=int(len(Z["bas"][0]) + len(Z["haut"][0])),
               conjoint_MOND_rejete_si_Newton=float(np.mean(lN < thrM)), conjoint_Newton_rejete_si_MOND=float(np.mean(lM > thrN)),
               conjoint_separation_sigma=float((lM.mean() - lN.mean())/(0.5*(lN.std() + lM.std()))),
               S_MOND_rejete_si_Newton=float(R[R.truth == 0].S_mond_rej.mean()), S_Newton_rejete_si_MOND=float(R[R.truth == 1].S_newton_rej.mean()))
    print(json.dumps(out, ensure_ascii=False), f"({time.time()-t0:.0f} s)", flush=True)
    return out

TPL_CACHE = {}
rows = [run(1, tag="référence (η profilé)"), run(1, tag="biais de masse uniforme +20 %", mass_bias=1.2),
        run(1, tag="compagnons non modélisés 30 %", f_kick=0.3), run(2, tag="référence ×2")]
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "joint_power_dr3.csv"), index=False)
print("FIN", flush=True)
