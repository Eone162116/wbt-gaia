"""Test d'accélération sur l'astrométrie par époque de Gaia DR4 (table epoch_astrometry).

Modèle le long du balayage (coordonnée de plan local w, en mas, au point de référence ra0/dec0) :
    w = Δα*(τ) sin θ + Δδ(τ) cos θ + ϖ f_w
    Δα*(τ) = Δα*0 + μα* τ + ½ gα τ² + ⅙ jα τ³      (idem pour δ)
τ : temps barycentrique TCB en années juliennes depuis J2017.5 ; θ : angle de position du balayage ; f_w : facteur de parallaxe AL.
Modèles emboîtés : 5 paramètres (simple), 7 (accélération), 9 (accélération + variation).

Les mesures CCD d'un même transit partagent l'attitude : elles sont moyennées par transit (pondération AGIS
1/(σ² + bruit_source²)), puis les tests F comparent les modèles en normalisant par le résidu observé,
ce qui rend le test robuste à une sous-estimation globale des erreurs.

Une binaire large de quelques kau produit une accélération astrométrique d'environ 1e-4 mas/an² à 130 pc,
indétectable : toute accélération détectée signale un compagnon proche, non résolu."""
import numpy as np, pandas as pd
from scipy.stats import f as fdist

NS_PER_YR = 1e9 * 86400 * 365.25
T_REF = 7.5                                   # J2017.5 en années depuis J2010.0

def transits_from_table(t):
    """Table epoch_astrometry (astropy Table ou DataFrame, colonnes tableaux par CCD) -> DataFrame par transit."""
    rows = []
    for r in (t.iterrows() if isinstance(t, pd.DataFrame) else enumerate(t)):
        r = r[1]
        used = np.asarray(r["used_by_agis_al"], bool)
        w = np.asarray(r["centroid_pos_al"], float); s = np.asarray(r["centroid_pos_error_al"], float)
        ok = used & np.isfinite(w) & np.isfinite(s) & (np.abs(w) < 1e4)
        if ok.sum() == 0: continue
        eps = float(r["agis_source_excess_noise"]) if np.isfinite(r["agis_source_excess_noise"]) else 0.0
        om = 1.0 / (s[ok]**2 + eps**2)
        tt = (np.asarray(r["obs_time_tcb"], float)[ok] + float(r["obs_time_bary_corr"])) / NS_PER_YR - T_REF
        th = np.radians(np.asarray(r["scan_pos_angle"], float)[ok])
        rows.append(dict(source_id=int(r["source_id"]), tau=np.sum(om*tt)/om.sum(),
                         sin=np.sum(om*np.sin(th))/om.sum(), cos=np.sum(om*np.cos(th))/om.sum(),
                         fw=float(r["parallax_factor_al"]), w=np.sum(om*w[ok])/om.sum(),
                         sig=om.sum()**-0.5, nccd=int(ok.sum()), g_mag=float(r["g_mag"])))
    return pd.DataFrame(rows)

def design(T, npar):
    s, c, tau = T["sin"].values, T["cos"].values, T["tau"].values
    cols = [s, c, T["fw"].values, tau*s, tau*c]
    if npar >= 7: cols += [0.5*tau**2*s, 0.5*tau**2*c]
    if npar >= 9: cols += [tau**3/6*s, tau**3/6*c]
    return np.stack(cols, 1)

def wls(T, npar):
    A = design(T, npar); wt = 1.0/T["sig"].values
    Aw, bw = A*wt[:, None], T["w"].values*wt
    x, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    chi2 = float(np.sum((bw - Aw@x)**2))
    cov = np.linalg.inv(Aw.T@Aw)
    return x, cov, chi2

def test_source(T):
    """Statistiques d'un source à partir de ses transits."""
    n = len(T)
    if n < 15: return dict(n_transits=n)
    x5, c5, q5 = wls(T, 5); x7, c7, q7 = wls(T, 7); x9, c9, q9 = wls(T, 9)
    F7 = ((q5 - q7)/2) / (q7/(n - 7)); p7 = float(fdist.sf(F7, 2, n - 7))
    F9 = ((q7 - q9)/2) / (q9/(n - 9)); p9 = float(fdist.sf(F9, 2, n - 9))
    s7 = q7/(n - 7)                                   # facteur d'échelle : erreurs sous-estimées éventuelles
    g = x7[5:7]; cg = c7[5:7, 5:7]*s7
    sig_g = float(np.sqrt(g @ np.linalg.solve(cg, g)))  # significativité de l'accélération (σ, 2 ddl)
    return dict(n_transits=n, parallax_5p=x5[2], parallax_err_5p=float(np.sqrt(c5[2, 2]*q5/(n-5))),
                pmra_5p=x5[3], pmdec_5p=x5[4], uwe_5p=float(np.sqrt(q5/(n - 5))),
                g_ra=g[0], g_dec=g[1], g_tot=float(np.hypot(*g)), g_sig=sig_g, p_acc7=p7, p_acc9=p9,
                g_mag=float(T.g_mag.median()))

def test_all(transits):
    return pd.DataFrame([dict(source_id=sid, **test_source(T)) for sid, T in transits.groupby("source_id")])

def flag(stats, p_thr=1e-3, uwe_thr=None):
    """Drapeau « compagnon proche probable ». uwe_thr : seuil d'UWE (fixé avant la levée, à partir de la
    distribution des composantes candidates par tranche de magnitude, cf. pré-enregistrement)."""
    f = (stats.p_acc7 < p_thr) | (stats.p_acc9 < p_thr)
    if uwe_thr is not None: f |= stats.uwe_5p > uwe_thr
    return f

def load_epoch_file(path):
    """Astrométrie par époque déjà téléchargée (VOTable de l'archive ou de la pré-publication)."""
    from astropy.io.votable import parse
    from astropy.table import vstack
    return vstack([t.to_table() for t in parse(path).iter_tables()])

def fetch_epoch_dr4(source_ids, chunk=500):
    """Astrométrie par époque via DataLink (astroquery), renvoyée en table astropy (colonnes-tableaux par CCD).
    Non testé : DR4 indisponible au 16/10/2026. Le libellé de release (« Gaia DR4 ») est à vérifier le jour J ;
    à défaut, télécharger depuis l'interface de l'archive et passer le fichier via WBT_EPOCH_FILE."""
    from astroquery.gaia import Gaia
    from astropy.table import vstack
    parts = []
    ids = list(map(int, np.unique(source_ids)))
    for i in range(0, len(ids), chunk):
        res = Gaia.load_data(ids=ids[i:i+chunk], data_release="Gaia DR4", retrieval_type="EPOCH_ASTROMETRY",
                             data_structure="RAW", format="votable")
        for key, tabs in res.items():
            for tab in tabs: parts.append(tab.to_table())
    return vstack(parts)
