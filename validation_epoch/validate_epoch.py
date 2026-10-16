"""Validation du test d'accélération sur la pré-publication DR4 (12 sources, juin 2026).
1. Classification des 12 sources : 9 simples/QSO doivent passer, 3 binaires orbitales doivent être signalées.
2. Injection de compagnons cachés képlériens sur l'échantillonnage réel d'une source G≈14 :
   probabilité de détection en fonction du demi-grand axe interne a_in et du rapport de masse q,
   pour une primaire de 0,6 M☉ à 130 pc (composante typique de binaire large)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io.votable import parse_single_table
from epoch_accel import transits_from_table, test_all, test_source, flag

P_THR, UWE_THR = 1e-3, 1.4
rng = np.random.default_rng(20261016)
t = parse_single_table("GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml").to_table()
TR = transits_from_table(t)

# ---------------------------------------------------------------- 1. classification
meta = pd.DataFrame([
 (4181040337841125632, "HD 183633", "simple"), (3926186255616949504, "G=19, ϖ=1", "simple"),
 (435469040545191680, "G=14, ϖ=1", "simple"), (1663617687609809280, "LSPM J1324+6124", "simple"),
 (2309425390592896, "G=14, ϖ=0,9", "simple"), (20694084440761600, "G=14, ϖ=4,9", "simple"),
 (2237987199365376, "QSO 1", "qso"), (60730287810150016, "QSO 2", "qso"), (10973744521070720, "QSO 3", "qso"),
 (3937211745905473024, "HD 114762 (P≈84 j)", "orbite"), (1457486023639239296, "Gaia-4 (P≈570 j)", "orbite"),
 (4318465066420528000, "Gaia BH3 (P≈11,6 ans)", "orbite")], columns=["source_id", "nom", "groupe"])
S = meta.merge(test_all(TR), on="source_id")
S["signale"] = flag(S, P_THR, UWE_THR)
S["attendu"] = S.groupe == "orbite"
S.to_csv("prerelease_classification.csv", index=False)
print(S[["nom", "groupe", "n_transits", "parallax_5p", "uwe_5p", "g_tot", "g_sig", "p_acc7", "signale", "attendu"]].round(4).to_string(index=False))
print(f"classification correcte : {(S.signale == S.attendu).sum()}/12")

# ---------------------------------------------------------------- 2. injection de compagnons cachés
ref = TR[TR.source_id == 1663617687609809280].reset_index(drop=True)      # G=14,3 : composante typique
uwe_ref = float(S.loc[S.source_id == 1663617687609809280, "uwe_5p"].iloc[0])
D_PC, M1 = 130.0, 0.6

def photocentre_track(tau, a_au, q, rng):
    m2 = q*M1; P = np.sqrt(a_au**3/(M1 + m2))                          # années
    B = q/(1 + q) - q**4/(1 + q**4)                                     # décalage photocentre (L ∝ m⁴)
    alpha = 1000*a_au*B/D_PC                                            # mas
    e = np.sqrt(rng.random()); w = 2*np.pi*rng.random(); W = 2*np.pi*rng.random(); ci = rng.uniform(-1, 1)
    M = 2*np.pi*(tau/P + rng.random()); E = M.copy()
    for _ in range(50): E = E - (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
    X, Y = np.cos(E) - e, np.sqrt(1 - e**2)*np.sin(E)
    A_ = alpha*(np.cos(w)*np.cos(W) - np.sin(w)*np.sin(W)*ci); B_ = alpha*(np.cos(w)*np.sin(W) + np.sin(w)*np.cos(W)*ci)
    F_ = alpha*(-np.sin(w)*np.cos(W) - np.cos(w)*np.sin(W)*ci); G_ = alpha*(-np.sin(w)*np.sin(W) + np.cos(w)*np.cos(W)*ci)
    return B_*X + G_*Y, A_*X + F_*Y, P, alpha                           # Δα*, Δδ (mas)

def detected(a_au, q):
    da, dd, P, alpha = photocentre_track(ref.tau.values, a_au, q, rng)
    T = ref.copy()
    T["w"] = da*T["sin"] + dd*T["cos"] + rng.normal(0, 1, len(T))*T.sig*uwe_ref
    s = pd.DataFrame([test_source(T)])
    return bool(flag(s, P_THR, UWE_THR).iloc[0]), P, alpha

grid_a = np.logspace(-1.5, 3, 28); grid_q = [0.1, 0.2, 0.4, 0.7]
rows = []
for q in grid_q:
    for a in grid_a:
        res = [detected(a, q) for _ in range(150)]
        rows.append(dict(q=q, a_au=a, P_yr=res[0][1], alpha_mas=res[0][2], p_det=np.mean([r[0] for r in res])))
D = pd.DataFrame(rows); D.to_csv("injection_companions.csv", index=False)
null = np.mean([detected(1e-6, 0.1)[0] for _ in range(400)])
print(f"\nfaux positifs (aucun compagnon) : {null:.3f}")
for q in grid_q:
    d = D[D.q == q]; ok = d[d.p_det >= 0.5]
    rng_txt = f"{ok.a_au.min():.2f}–{ok.a_au.max():.0f} au" if len(ok) else "aucune"
    print(f"q = {q} : détection ≥ 50 % pour a_in = {rng_txt}")

fig, ax = plt.subplots(figsize=(9.5, 5.4))
cols = ["#C44E52", "#DD8452", "#4C72B0", "#55A868"]
for q, c in zip(grid_q, cols):
    d = D[D.q == q]; ax.plot(d.a_au, d.p_det, "-o", ms=3, color=c, label=f"q = {q}")
ax.axvspan(0.3, 10, color="#999999", alpha=0.12)
ax.text(1.7, 0.03, "zone supposée\nfiltrée par RUWE\n(modèle jouet DR3)", ha="center", fontsize=8, color="#555555")
ax.axhline(null, color="k", lw=0.8, ls=":"); ax.text(0.035, null + 0.02, f"faux positifs {null:.2%}", fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("Demi-grand axe du compagnon caché a_in (au)")
ax.set_ylabel("Probabilité de détection")
ax.set_title("Crible DR4 par époques : quels compagnons cachés sont détectés ?", loc="left", fontweight="bold")
fig.text(0.01, 0.01, f"Primaire 0,6 M☉ à {D_PC:.0f} pc, échantillonnage réel DR4 d'une source G=14,3 (pré-publication) ; détection = accélération (p<{P_THR:g}) ou UWE>{UWE_THR} ; 150 orbites par point.",
         fontsize=8, color="#555555")
ax.legend(frameon=False, fontsize=9, loc="center right"); ax.set_ylim(-0.02, 1.02)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig("fig_epoch_detection.png", dpi=150)
print("figure écrite")
