"""Test d'injection-récupération du WBT.
Principe : on garde pour chaque paire réelle de l'échantillon de base (V0) sa masse, sa distance, sa séparation
projetée et son incertitude de vitesse, et on remplace sa vitesse par une vitesse simulée :
  - géométrie képlérienne (excentricité thermique f(e)=2e, orientation et phase aléatoires) ; ṽ newtonien
    ne dépend que de la géométrie (invariance d'échelle) ;
  - MOND : vitesse multipliée par le boost de Cookson éq. 7 évalué au rayon 3D tiré ;
  - compagnons cachés (fraction f_cb) : biais de masse sqrt(1+q/2) et, pour a_in > 10 au, recul du photocentre ;
    0,3-10 au exclus (supposés éliminés par la coupe RUWE) ;
  - bruit gaussien 2D à l'incertitude réelle de la paire, puis mêmes coupes cinématiques.
Statistiques : préférence Newton/MOND (χ² sur médianes binées, comme Cookson) et amplitude de marche
  S = médiane(ṽ | x > 1) / médiane(ṽ | x < 0,3) - 1, plus robuste et sans ajustement.
Limites : rayon 3D tiré sans pondération par la distribution des demi-grands axes ; boost MOND appliqué à la
vitesse képlérienne (pas d'intégration d'orbites MOND) ; modèle de contamination simplifié."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wbt_pipeline import *

rng = np.random.default_rng(20261016)
K = pd.read_pickle("K.pkl"); hr = hr_band(K)
m0, vt0 = select(K, hr)                       # échantillon de base V0
S = K[m0].reset_index(drop=True); S["vt"] = vt0[m0]
N = len(S)
x, M, vc = S.x.values, S.M.values, S.vc.values
sv = (S.svt*S.vc).values                      # incertitude sur v (km/s)

def mond_boost3d(x3):
    gi = 1.0/np.asarray(x3)**2; gt = np.hypot(gi, GNE); h = 1e-4
    Kd = (np.log(nu(gt*(1+h)))-np.log(nu(gt*(1-h))))/(np.log(1+h)-np.log(1-h))
    return np.sqrt(nu(gt)*(1+Kd/3*np.tanh((0.825*GNE/gi)**3.7)))

def kepler_geometry(n):
    e = np.sqrt(rng.random(n))                               # thermique
    Mn = 2*np.pi*rng.random(n); E = Mn.copy()
    for _ in range(40): E = E-(E-e*np.sin(E)-Mn)/(1-e*np.cos(E))
    r = np.stack([np.cos(E)-e, np.sqrt(1-e**2)*np.sin(E), np.zeros(n)], 1)
    v = np.stack([-np.sin(E), np.sqrt(1-e**2)*np.cos(E), np.zeros(n)], 1)/(1-e*np.cos(E))[:, None]
    # rotation aléatoire uniforme (quaternions)
    u = rng.random((n, 3)); q = np.stack([np.sqrt(1-u[:, 0])*np.sin(2*np.pi*u[:, 1]), np.sqrt(1-u[:, 0])*np.cos(2*np.pi*u[:, 1]),
                                          np.sqrt(u[:, 0])*np.sin(2*np.pi*u[:, 2]), np.sqrt(u[:, 0])*np.cos(2*np.pi*u[:, 2])], 1)
    a, b, c, d = q.T
    R = np.stack([np.stack([1-2*(c*c+d*d), 2*(b*c-a*d), 2*(b*d+a*c)], 1),
                  np.stack([2*(b*c+a*d), 1-2*(b*b+d*d), 2*(c*d-a*b)], 1),
                  np.stack([2*(b*d-a*c), 2*(c*d+a*b), 1-2*(b*b+c*c)], 1)], 1)
    r = np.einsum("nij,nj->ni", R, r); v = np.einsum("nij,nj->ni", R, v)
    rsky = np.hypot(r[:, 0], r[:, 1]); vsky = v[:, :2]
    return vsky*np.sqrt(rsky)[:, None], np.linalg.norm(r, axis=1)/rsky   # ṽ vectoriel (Newton), r3D/rsky

def simulate(truth="newton", f_cb=0.0):
    vt_vec, ratio = kepler_geometry(N)
    if truth == "mond": vt_vec *= mond_boost3d(x*ratio)[:, None]
    v = vt_vec*vc[:, None]                                  # km/s, vrai si masse photométrique juste
    cb = rng.random(N) < f_cb
    if cb.any():
        k = cb.sum(); qq = rng.uniform(0.1, 1.0, k); m1 = M[cb]/2
        v[cb] *= np.sqrt(1+qq*m1/M[cb])[:, None]            # masse cachée non comptée
        la = np.where(rng.random(k) < 0.5, rng.uniform(np.log10(0.01), np.log10(0.3), k),
                      rng.uniform(np.log10(10), np.log10(300), k))
        ain = 10**la; far = ain > 1
        B = qq/(1+qq)-qq**4/(1+qq**4)                       # décalage photocentre-barycentre
        vin = B*VC_1AU*np.sqrt(m1*(1+qq)/ain)
        uz = rng.uniform(-1, 1, k); phi = rng.uniform(0, 2*np.pi, k); proj = np.sqrt(1-uz**2)
        kick = (vin*proj*far)[:, None]*np.stack([np.cos(phi), np.sin(phi)], 1)
        v[cb] += kick
    v += rng.normal(0, 1, (N, 2))*sv[:, None]
    vabs = np.linalg.norm(v, axis=1); vt = vabs/vc
    keep = (vt < BASE["vtmax"]) & (vabs < BASE["dvsky"])
    return x[keep], vt[keep]

def step_stat(xx, vv):
    lo, hi = vv[xx < 0.3], vv[xx > 1.0]
    return np.median(hi)/np.median(lo)-1

def run(truth, f_cb, R=300):
    out = []
    for _ in range(R):
        xx, vv = simulate(truth, f_cb)
        B, s = binned_test(xx, vv)
        out.append(dict(truth=truth, f_cb=f_cb, n=len(xx), log10_pref=np.log10(s["pref_newton"]), step=step_stat(xx, vv)))
    return pd.DataFrame(out)

if __name__ == "__main__":
    scen = [("newton", 0.0), ("mond", 0.0), ("newton", 0.15), ("newton", 0.3), ("newton", 0.5), ("mond", 0.3)]
    D = pd.concat([run(t, f) for t, f in scen], ignore_index=True)
    D.to_csv("injection_results.csv", index=False)
    summ = D.groupby(["truth", "f_cb"]).agg(n=("n", "median"), step_med=("step", "median"),
                                           step_lo=("step", lambda s: s.quantile(.025)), step_hi=("step", lambda s: s.quantile(.975)),
                                           frac_mond_pref=("log10_pref", lambda s: (s < 0).mean()),
                                           frac_mond_pref_x10=("log10_pref", lambda s: (s < -1).mean()),
                                           med_log10=("log10_pref", "median")).reset_index()
    summ.to_csv("injection_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))

    # bootstrap des données réelles
    bs = []
    for _ in range(2000):
        i = rng.integers(0, N, N)
        B, s = binned_test(x[i], S.vt.values[i])
        bs.append(dict(log10_pref=np.log10(s["pref_newton"]), step=step_stat(x[i], S.vt.values[i])))
    BS = pd.DataFrame(bs); BS.to_csv("bootstrap_real.csv", index=False)
    obs_step = step_stat(x, S.vt.values)
    Bobs, sobs = binned_test(x, S.vt.values)
    from scipy.stats import chi2
    print(f"\nDonnées réelles V0 : N={N}, marche S = {obs_step:+.3f} "
          f"[IC95 bootstrap {BS.step.quantile(.025):+.3f}, {BS.step.quantile(.975):+.3f}]")
    print(f"log10(Newton/MOND) = {np.log10(sobs['pref_newton']):.2f} "
          f"[IC95 {BS.log10_pref.quantile(.025):.2f}, {BS.log10_pref.quantile(.975):.2f}] ; "
          f"fraction bootstrap pro-MOND = {(BS.log10_pref<0).mean():.3f}")
    print(f"Adéquation Newton : χ²={sobs['chi2_N']:.2f}/{sobs['dof']} ddl, p={chi2.sf(sobs['chi2_N'], sobs['dof']):.3f} ; "
          f"MOND : χ²={sobs['chi2_M']:.2f}, p={chi2.sf(sobs['chi2_M'], sobs['dof']):.3f}")

    # figure
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    order = [("newton", 0.0, "Newton, propre", "#4C72B0"), ("mond", 0.0, "MOND, propre", "#DD8452"),
             ("newton", 0.15, "Newton, 15 % compagnons", "#8172B3"), ("newton", 0.3, "Newton, 30 % compagnons", "#937860"),
             ("newton", 0.5, "Newton, 50 % compagnons", "#C44E52")]
    bins = np.linspace(-0.25, 0.45, 50)
    for t, f, lab, c in order:
        ax.hist(D[(D.truth == t) & (D.f_cb == f)].step, bins=bins, histtype="step", lw=2, color=c, label=lab, density=True)
    ax.hist(BS.step, bins=bins, color="#999999", alpha=0.35, label="Données réelles (bootstrap)", density=True)
    ax.axvline(obs_step, color="k", lw=1.5, label=f"Données réelles : S = {obs_step:+.3f}")
    ax.set_xlabel("Amplitude de marche S = médiane(ṽ | r/rM > 1) / médiane(ṽ | r/rM < 0,3) − 1")
    ax.set_ylabel("Densité")
    ax.set_title("Marche MOND injectée bien retrouvée ; données DR3 côté Newton, à ~2σ seulement",
                 loc="left", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); plt.savefig("fig_injection_recovery.png", dpi=150)
    print("figure écrite")
