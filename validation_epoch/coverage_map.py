"""Carte de couverture des compagnons cachés pour une composante de binaire large (0,6 M☉ à 130 pc) :
crible par époques (injection sur la pré-publication DR4) vs résolution image de Gaia.
Hypothèses (à réviser avec les données DR4) :
- relation masse–M_G de séquence principale approchée (type Pecaut & Mamajek), incertitude ~0,3–0,5 mag ;
- résolution catalogue à 50 % : 0,6″ (ΔG≈0), 1″ (ΔG≈4), 2″ (ΔG≈7) d'après El-Badry et al. 2021/2024 (DR3), interpolée ;
- limite de diffraction ~0,15″ pour ipd_frac_multi_peak / SEAPipe (borne optimiste, dépendance au contraste non publiée) ;
- limite de détection Gaia G ≈ 20,7 ; projection moyenne ρ ≈ 0,8 a/d."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D_PC, M1, PROJ, GLIM = 130.0, 0.6, 0.8, 20.7
mtab = np.array([0.075, 0.08, 0.09, 0.10, 0.16, 0.23, 0.30, 0.37, 0.44, 0.50, 0.57, 0.60, 0.63])
gtab = np.array([17.5, 17.0, 16.2, 15.1, 13.3, 12.0, 11.3, 10.6, 9.85, 9.25, 8.65, 8.4, 8.1])
MG = lambda m: np.interp(m, mtab, gtab)
dmod = 5*np.log10(D_PC/10); G1 = MG(M1) + dmod

def theta50(dG):           # résolution catalogue (50 %), arcsec
    return np.interp(dG, [0, 4, 7], [0.6, 1.0, 2.0]) + np.where(dG > 7, (dG - 7)/3.0, 0.0)

inj = pd.read_csv("injection_companions.csv")
rows = []
for q in [0.1, 0.2, 0.4, 0.7]:
    d = inj[inj.q == q]; ok = d[d.p_det >= 0.5]
    a_lo, a_hi = ok.a_au.min(), ok.a_au.max()
    m2 = q*M1; dG = MG(m2) - MG(M1); G2 = G1 + dG
    visible = G2 < GLIM
    a_res = theta50(dG)*D_PC/PROJ if visible else np.inf
    a_diff = 0.15*D_PC/PROJ if visible else np.inf
    rows.append(dict(q=q, m2=m2, dG=round(float(dG), 1), G2=round(float(G2), 1), epoch_lo_au=round(a_lo, 2), epoch_hi_au=round(a_hi, 1),
                     image_res50_au=round(float(a_res), 0) if np.isfinite(a_res) else None,
                     image_diffraction_au=round(float(a_diff), 0) if np.isfinite(a_diff) else None,
                     gap_pessimiste=f"{a_hi:.0f}–{a_res:.0f} au" if np.isfinite(a_res) else f"> {a_hi:.0f} au (compagnon trop faible : jamais imagé)",
                     gap_optimiste=("aucun" if a_diff <= a_hi else f"{a_hi:.0f}–{a_diff:.0f} au") if np.isfinite(a_diff) else f"> {a_hi:.0f} au"))
T = pd.DataFrame(rows); T.to_csv("coverage_gap.csv", index=False)
pd.set_option("display.width", 220); print(T.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5.4))
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
for i, r in T.iterrows():
    y = i
    ax.barh(y, r.epoch_hi_au - r.epoch_lo_au, left=r.epoch_lo_au, height=0.55, color="#4C72B0", alpha=0.85)
    if pd.notna(r.image_res50_au):
        ax.barh(y, r.image_res50_au - r.epoch_hi_au, left=r.epoch_hi_au, height=0.55, color="#C44E52", alpha=0.35)
        ax.barh(y, 5000 - r.image_res50_au, left=r.image_res50_au, height=0.55, color="#55A868", alpha=0.8)
        ax.plot([r.image_diffraction_au]*2, [y - 0.32, y + 0.32], color="k", lw=1.5)
    else:
        ax.barh(y, 5000 - r.epoch_hi_au, left=r.epoch_hi_au, height=0.55, color="#C44E52", alpha=0.6, hatch="//")
handles = [Patch(color="#4C72B0", alpha=0.85, label="Crible par époques DR4 (détection ≥ 50 %)"),
           Patch(color="#C44E52", alpha=0.35, label="Angle mort si seul le catalogue résout"),
           Patch(color="#55A868", alpha=0.8, label="Résolu dans le catalogue Gaia (≥ 50 %)"),
           Patch(facecolor="#C44E52", alpha=0.6, hatch="//", label="Compagnon trop faible pour l'imagerie (G > 20,7)"),
           Line2D([0], [0], color="k", lw=1.5, label="Limite de diffraction ~0,15″ (borne optimiste)")]
ax.set_yticks(range(len(T))); ax.set_yticklabels([f"q = {q} (ΔG ≈ {dg})" for q, dg in zip(T.q, T.dG)])
ax.set_xscale("log"); ax.set_xlim(0.03, 5000); ax.set_xlabel("Demi-grand axe du compagnon caché a_in (au)")
sec = ax.secondary_xaxis("top", functions=(lambda a: a*PROJ/D_PC, lambda t: t*D_PC/PROJ)); sec.set_xlabel("Séparation projetée typique à 130 pc (″)")
ax.set_title("Couverture des compagnons cachés : époques DR4 + imagerie Gaia", loc="left", fontweight="bold", pad=34)
ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
ax.spines["right"].set_visible(False)
fig.text(0.01, 0.01, "Primaire 0,6 M☉ à 130 pc. Résolution catalogue : El-Badry et al. 2021/2024 (DR3). Relation masse–M_G approchée. Sensibilité de SEAPipe au contraste non publiée.",
         fontsize=7.5, color="#555555")
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig("fig_coverage_gap.png", dpi=150)
print("figure écrite")
