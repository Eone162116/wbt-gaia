"""Modèle de contamination réaliste par compagnons internes, filtré par les cribles DR4, sur la géométrie réelle
de l'échantillon principal (DR3, spécification pré-enregistrée).
Population (par composante de binaire large) :
  - présence d'un compagnon interne avec probabilité f_in ;
  - période log-normale, log10 P[j] ~ N(5,03 ; 2,28) (Raghavan et al. 2010) ; q uniforme sur [0,05 ; 1] ; e uniforme [0 ; 0,9] ;
  - stabilité hiérarchique : a_in < 0,2 × séparation de la binaire large, sinon pas de compagnon.
Effets : biais de masse (masse cachée non comptée dans v_c) et « coup » de mouvement propre = vitesse moyenne du
photocentre sur les 5,5 ans de DR4 (régression linéaire de la trajectoire), bruit réel, coupes ṽ < 2,5 et |v| < 10 km/s.
Cribles :
  - époques : p_det(α, P) mesurée par injection sur l'échantillonnage réel DR4 (pgrid.npy, indépendant de d) ;
  - image pessimiste : résolution catalogue DR3 (50 % à 0,6″/1″/2″ pour ΔG = 0/4/7), G₂ < 20,7 ;
  - image optimiste : toute séparation > 0,15″ (diffraction), G₂ < 20,7.
Statistique : biais de S par rapport à la même géométrie sans contamination, global et par distance (< ou > 100 pc)."""
import os, sys, numpy as np, pandas as pd, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))   # racine du dépôt
# À lancer depuis la racine du dépôt (K_dr3.pkl et config_prereg.json) : python validation_contamination/contamination_model.py
from scipy.interpolate import RegularGridInterpolator
from wbt_pipeline import select, hr_band, MG_SUN, VC_1AU, BASE
from calibrate import kepler_geometry, step_stat

rng = np.random.default_rng(20261017)
CFG = json.load(open("config_prereg.json", encoding="utf-8")); P = CFG["primary"]
K = pd.read_pickle("K_dr3.pkl"); b, s, _ = hr_band(K)
kw = {k: P[k] for k in ["dmax", "smin", "smax", "vtmax", "svt", "ruwe", "drv", "dvsky", "sph", "hr", "ipd", "degroup", "screen", "epoch"]}
m, _ = select(K, (b, s, P["hr_height"]), **kw)
S = K[m].reset_index(drop=True); N = len(S)
x, vc, d, rsky, Mtot = S.x.values, S.vc.values, S.d.values, S.rsky.values, S.M.values
sv = (S.svt*S.vc).values
mass = lambda MG: (lambda mm: np.where(mm < 0.7, mm + 0.05, mm))(10**(0.0725*(MG_SUN - MG)))
m_comp = np.stack([mass(S.MG1.values), mass(S.MG2.values)], 1)
G_comp = np.stack([S.MG1.values, S.MG2.values], 1) + 5*np.log10(d/10)[:, None]
near = d < 100
print(f"échantillon : N = {N} ({near.sum()} à moins de 100 pc)")

pgrid = np.load(os.path.join(HERE, "pgrid.npy"))
interp = RegularGridInterpolator((np.linspace(-2.5, 2.5, 21), np.linspace(-2.5, 4.0, 21)), pgrid, bounds_error=False, fill_value=None)
def p_epoch(alpha, Pyr):
    la = np.clip(np.log10(alpha), -2.5, 2.5); lp = np.clip(np.log10(Pyr), -2.5, 4.0)
    p = np.clip(interp(np.stack([la, lp], -1)), 0, 1)
    return np.where(np.log10(alpha) < -2.5, 0.0, p)

mtab = np.array([0.05, 0.075, 0.08, 0.09, 0.10, 0.16, 0.23, 0.30, 0.37, 0.44, 0.50, 0.57, 0.60, 0.63, 0.70, 0.80, 1.0, 1.2])
gtab = np.array([20.0, 17.5, 17.0, 16.2, 15.1, 13.3, 12.0, 11.3, 10.6, 9.85, 9.25, 8.65, 8.4, 8.1, 7.5, 6.3, 4.7, 3.9])
MGm = lambda mm: np.interp(mm, mtab, gtab)
def theta50(dG): return np.interp(dG, [0, 4, 7], [0.6, 1.0, 2.0]) + np.where(dG > 7, (dG - 7)/3.0, 0.0)

TAU = np.linspace(-2.75, 2.75, 40)
def photocentre_pm(alpha, Pyr, n):
    """Vitesse moyenne du photocentre sur la fenêtre DR4 (mas/an), orientation/phase/excentricité aléatoires."""
    e = rng.random(n)*0.9; w = 2*np.pi*rng.random(n); W = 2*np.pi*rng.random(n); ci = rng.uniform(-1, 1, n)
    Mn = 2*np.pi*(TAU[None, :]/Pyr[:, None] + rng.random(n)[:, None]); E = Mn.copy()
    for _ in range(40): E = E - (E - e[:, None]*np.sin(E) - Mn)/(1 - e[:, None]*np.cos(E))
    X, Y = np.cos(E) - e[:, None], np.sqrt(1 - e[:, None]**2)*np.sin(E)
    A = alpha*(np.cos(w)*np.cos(W) - np.sin(w)*np.sin(W)*ci); B = alpha*(np.cos(w)*np.sin(W) + np.sin(w)*np.cos(W)*ci)
    F = alpha*(-np.sin(w)*np.cos(W) - np.cos(w)*np.sin(W)*ci); G = alpha*(-np.sin(w)*np.sin(W) + np.cos(w)*np.cos(W)*ci)
    ra = B[:, None]*X + G[:, None]*Y; de = A[:, None]*X + F[:, None]*Y
    tc = TAU - TAU.mean()
    return np.stack([(ra*tc).sum(1)/(tc**2).sum(), (de*tc).sum(1)/(tc**2).sum()], 1)

def realise(f_in, screen):
    vt_vec, ratio = kepler_geometry(N, rng)                     # vérité newtonienne, ṽ vectoriel
    v = vt_vec*vc[:, None]                                     # km/s (masse photométrique)
    M_hidden = np.zeros(N); kick = np.zeros((N, 2)); removed = np.zeros(N, bool)
    for c in (0, 1):
        has = rng.random(N) < f_in
        Pd = 10**rng.normal(5.03, 2.28, N); q = rng.uniform(0.05, 1.0, N); m1 = m_comp[:, c]
        a_in = ((Pd/365.25)**2*m1*(1 + q))**(1/3)
        has &= a_in < 0.2*rsky
        if not has.any(): continue
        i = np.where(has)[0]; Pyr = Pd[i]/365.25; qq = q[i]; aa = a_in[i]
        Bph = qq/(1 + qq) - qq**4/(1 + qq**4)
        alpha = 1000*aa*np.abs(Bph)/d[i]                        # mas
        pm = photocentre_pm(np.maximum(alpha, 1e-6), Pyr, len(i))
        kick[i] += 4.740470446*pm*d[i][:, None]/1000            # km/s
        M_hidden[i] += qq*m1[i]
        if screen == "aucun": continue
        det = rng.random(len(i)) < p_epoch(np.maximum(alpha, 1e-6), Pyr)
        if screen in ("epoques+image_pess", "epoques+image_opt"):
            dG = MGm(qq*m1[i]) - MGm(m1[i]); G2 = G_comp[i, c] + dG
            rho = aa*np.sqrt(1 - rng.uniform(-1, 1, len(i))**2)/d[i]   # arcsec
            vis = G2 < 20.7
            if screen == "epoques+image_pess":
                p_img = vis/(1 + np.exp(-(np.log10(rho) - np.log10(theta50(dG)))/0.05))
            else:
                p_img = (vis & (rho > 0.15)).astype(float)
            det |= rng.random(len(i)) < p_img
        removed[i[det]] = True
    v = v*np.sqrt(1 + M_hidden/Mtot)[:, None] + kick + rng.normal(0, 1, (N, 2))*sv[:, None]
    va = np.linalg.norm(v, axis=1); vt = va/vc
    keep = (vt < BASE["vtmax"]) & (va < BASE["dvsky"]) & ~removed
    return dict(n=keep.sum(), S=step_stat(x[keep], vt[keep]),
                S_near=step_stat(x[keep & near], vt[keep & near]), S_far=step_stat(x[keep & ~near], vt[keep & ~near]))

R = 300
rows = []
for f_in in (0.0, 0.3, 0.5):
    for screen in (("aucun",) if f_in == 0 else ("aucun", "epoques", "epoques+image_pess", "epoques+image_opt")):
        res = pd.DataFrame([realise(f_in, screen) for _ in range(R)])
        rows.append(dict(f_in=f_in, crible=screen, N_med=res.n.median(), S=res.S.mean(), S_err=res.S.std()/np.sqrt(R),
                         S_proche=res.S_near.mean(), S_loin=res.S_far.mean(), diff_loin_moins_proche=(res.S_far - res.S_near).mean(),
                         diff_err=(res.S_far - res.S_near).std()/np.sqrt(R), diff_sd=(res.S_far - res.S_near).std()))
T = pd.DataFrame(rows)
ref = T[T.f_in == 0].iloc[0]
T["biais_S"] = T.S - ref.S; T["biais_diff"] = T.diff_loin_moins_proche - ref.diff_loin_moins_proche
T.to_csv(os.path.join(HERE, "contamination_results.csv"), index=False)
pd.set_option("display.width", 250)
print(T[["f_in", "crible", "N_med", "S", "biais_S", "S_proche", "S_loin", "biais_diff", "diff_sd"]].round(4).to_string(index=False))
