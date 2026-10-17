"""
Wide Binary Test (WBT) - pipeline headless, reproduction de Cookson et al. 2026 (MNRAS, arXiv:2602.24035)
Entrée : catalogue Gaia DR3 (parallax > 6 mas, parallax_over_error > 10) en tranches CSV.
Chaque choix d'analyse est un paramètre -> multivers de spécifications.
"""
import glob, numpy as np, pandas as pd
from scipy.spatial import cKDTree

KAPPA = 4.740470446          # km/s par (mas/an * kpc)
ARCSEC_PER_RAD = 206264.806
GM_SUN = 1.32712440018e20    # m^3 s^-2
AU = 1.495978707e11          # m
A0 = 1.2e-10                 # m s^-2
VC_1AU = np.sqrt(GM_SUN / AU) / 1e3   # 29.78 km/s
MG_SUN = 4.67                # Willmer 2018, bande G
GNE = 1.184                  # champ externe newtonien-équivalent / a0 (Cookson eq. 4)

# --------------------------------------------------------------------------- catalogue
def _bool(col):
    return col.astype(str).str.strip().str.lower().isin(["true", "1", "t"])

def star_screen_dr4(df, rv_gof_max=4.0):
    """Crible étoile à étoile des compagnons cachés à partir des colonnes DR4 de gaia_source.
    Colonne absente (DR3) => condition ignorée (vraie)."""
    ok = pd.Series(True, index=df.index)
    if "astrometry_origin" in df:   ok &= df.astrometry_origin.fillna("all_source_astrometry").astype(str).str.strip() == "all_source_astrometry"
    if "radial_velocity_origin" in df:
        o = df.radial_velocity_origin.astype(str).str.strip(); ok &= df.radial_velocity_origin.isna() | (o == "all_source_rvs")
    if "unreliable_astrometry" in df: ok &= ~_bool(df.unreliable_astrometry)
    if "blended" in df:              ok &= df.blended.fillna(0) == 0
    if "rv_assumed_sb2" in df:       ok &= ~_bool(df.rv_assumed_sb2)
    if "fraction_double_lined" in df: ok &= df.fraction_double_lined.fillna(0) == 0
    if "rv_renormalised_gof" in df:  ok &= df.rv_renormalised_gof.fillna(0) < rv_gof_max
    return ok

def load_catalog(pattern="s40_*.csv"):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pattern))], ignore_index=True)
    df = df.drop_duplicates("source_id")   # garde-fou : tranches qui se chevauchent
    df = df.dropna(subset=["pmra", "pmdec"]).reset_index(drop=True)
    df["snr"] = df.parallax / df.parallax_error
    df["scr4_star"] = star_screen_dr4(df)
    ra, de = np.radians(df.ra.values), np.radians(df.dec.values)
    df["x"], df["y"], df["z"] = np.cos(de)*np.cos(ra), np.cos(de)*np.sin(ra), np.sin(de)
    return df

# --------------------------------------------------------------------------- liens compagnons
def find_links(df, snr_main=40.0, plx_main=6.4, rmax_au=1e5, chunk=40000):
    """Pour chaque étoile 'principale' (S/N>snr_main), liste les compagnons co-mouvants
    dans le catalogue large (rayon projeté < rmax_au). Critères El-Badry & Rix 2018 / El-Badry+2021 :
    parallaxes compatibles à 3 sigma ; |dmu| < 0.44 plx^1.5 theta^-0.5 + 2 sigma_dmu (M=5 Msun)."""
    xyz = df[["x", "y", "z"]].values
    tree = cKDTree(xyz)
    main = np.where((df.snr.values > snr_main) & (df.parallax.values > plx_main))[0]
    plx, eplx = df.parallax.values, df.parallax_error.values
    pmra, pmde = df.pmra.values, df.pmdec.values
    epa, epd = df.pmra_error.values, df.pmdec_error.values
    I, J = [], []
    for c0 in range(0, len(main), chunk):
        idx = main[c0:c0+chunk]
        theta = rmax_au / (1000.0 / plx[idx]) / ARCSEC_PER_RAD          # rad
        r = 2*np.sin(theta/2)
        lists = tree.query_ball_point(xyz[idx], r=r)
        n = np.fromiter((len(l) for l in lists), int, len(lists))
        ii = np.repeat(idx, n); jj = np.concatenate([np.asarray(l, int) for l in lists])
        keep = ii != jj
        ii, jj = ii[keep], jj[keep]
        # séparation angulaire
        dot = np.clip((xyz[ii]*xyz[jj]).sum(1), -1, 1)
        th_as = np.arccos(dot) * ARCSEC_PER_RAD
        s_au = th_as * 1000.0 / plx[ii]
        ok = s_au < rmax_au
        ok &= np.abs(plx[ii]-plx[jj]) < 3*np.hypot(eplx[ii], eplx[jj])
        dmu = np.hypot(pmra[ii]-pmra[jj], pmde[ii]-pmde[jj])
        sdmu = np.sqrt(epa[ii]**2+epa[jj]**2+epd[ii]**2+epd[jj]**2)
        ok &= dmu < 0.44*plx[ii]**1.5*np.maximum(th_as, 1e-3)**-0.5 + 2*sdmu
        I.append(ii[ok]); J.append(jj[ok])
    return np.concatenate(I), np.concatenate(J)

# --------------------------------------------------------------------------- paires + cinématique
def build_pairs(df, I, J, snr_main=40.0):
    snr = df.snr.values
    links = pd.DataFrame({"i": I, "j": J})
    # nombre de compagnons par étoile principale, selon le catalogue de dégroupage
    n10 = links.groupby("i").size()
    n40 = links[snr[links.j.values] > snr_main].groupby("i").size()
    # paires mutuelles entre étoiles principales
    p = links[(snr[links.j.values] > snr_main) & (links.i < links.j)]
    rev = set(zip(links.j.values, links.i.values))
    p = p[[ (a, b) in rev for a, b in zip(p.i.values, p.j.values) ]].reset_index(drop=True)
    a, b = p.i.values, p.j.values
    out = pd.DataFrame({"i": a, "j": b,
                        "nc10_i": n10.reindex(a).fillna(0).values, "nc10_j": n10.reindex(b).fillna(0).values,
                        "nc40_i": n40.reindex(a).fillna(0).values, "nc40_j": n40.reindex(b).fillna(0).values})
    return out

def kinematics(df, P):
    g = lambda c: (df[c].values[P.i.values], df[c].values[P.j.values])
    (p1, p2), (e1, e2) = g("parallax"), g("parallax_error")
    w1, w2 = e1**-2, e2**-2
    pw = (w1*p1 + w2*p2)/(w1+w2); epw = (w1+w2)**-0.5
    d = 1000.0/pw
    (ra1, ra2), (de1, de2) = g("ra"), g("dec")
    (x1, x2), (y1, y2), (z1, z2) = g("x"), g("y"), g("z")
    th = np.arccos(np.clip(x1*x2+y1*y2+z1*z2, -1, 1))*ARCSEC_PER_RAD
    rsky = th*d                                                     # AU
    (G1, G2), (c1, c2) = g("phot_g_mean_mag"), g("bp_rp")
    MG1, MG2 = G1-5*np.log10(d/10), G2-5*np.log10(d/10)
    def mass(MG):
        m = 10**(0.0725*(MG_SUN-MG)); return np.where(m < 0.7, m+0.05, m)
    M = mass(MG1)+mass(MG2)
    (pa1, pa2), (pd1, pd2) = g("pmra"), g("pmdec")
    (sa1, sa2), (sd1, sd2) = g("pmra_error"), g("pmdec_error")
    (rv1, rv2), (srv1, srv2) = g("radial_velocity"), g("radial_velocity_error")
    # vitesse projetée sans correction sphérique
    dma, dmd = pa2-pa1, pd2-pd1
    dmu = np.hypot(dma, dmd)
    v_raw = KAPPA*d/1000*dmu
    # correction sphérique : vecteurs 3D avec RV systémique commune
    wr1, wr2 = srv1**-2, srv2**-2
    rvsys = np.where(np.isfinite(rv1) & np.isfinite(rv2), (wr1*rv1+wr2*rv2)/(wr1+wr2),
                     np.where(np.isfinite(rv1), rv1, rv2))
    def basis(ra, de):
        a, dd = np.radians(ra), np.radians(de)
        er = np.stack([np.cos(dd)*np.cos(a), np.cos(dd)*np.sin(a), np.sin(dd)], 1)
        ea = np.stack([-np.sin(a), np.cos(a), np.zeros_like(a)], 1)
        ed = np.stack([-np.sin(dd)*np.cos(a), -np.sin(dd)*np.sin(a), np.cos(dd)], 1)
        return er, ea, ed
    er1, ea1, ed1 = basis(ra1, de1); er2, ea2, ed2 = basis(ra2, de2)
    k = (KAPPA*d/1000)[:, None]
    rvs = np.nan_to_num(rvsys)[:, None]
    V1 = k*(pa1[:, None]*ea1+pd1[:, None]*ed1)+rvs*er1
    V2 = k*(pa2[:, None]*ea2+pd2[:, None]*ed2)+rvs*er2
    n = er1+er2; n /= np.linalg.norm(n, axis=1)[:, None]
    dV = V2-V1
    v_sph = np.linalg.norm(dV-(dV*n).sum(1)[:, None]*n, axis=1)
    # incertitude sur v (mouvements propres + distance commune)
    sa, sd = np.hypot(sa1, sa2), np.hypot(sd1, sd2)
    sdmu = np.sqrt((dma*sa)**2+(dmd*sd)**2)/np.maximum(dmu, 1e-9)
    sv = np.hypot(KAPPA*d/1000*sdmu, v_raw*epw/pw)
    vc = VC_1AU*np.sqrt(M/rsky)
    rM = np.sqrt(GM_SUN*M/A0)/AU
    return pd.DataFrame(dict(
        d=d, rsky=rsky, M=M, x=rsky/rM, vc=vc,
        vt_sph=v_sph/vc, vt_raw=v_raw/vc, svt=sv/vc, v_sph=v_sph, v_raw=v_raw,
        drv=np.abs(rv1-rv2), both_rv=np.isfinite(rv1) & np.isfinite(rv2),
        ruwe_max=np.maximum(*g("ruwe")), ipd_max=np.maximum(*g("ipd_frac_multi_peak")),
        c1=c1, c2=c2, MG1=MG1, MG2=MG2,
        nc10=np.maximum(P.nc10_i.values, P.nc10_j.values), nc40=np.maximum(P.nc40_i.values, P.nc40_j.values),
        sid1=df.source_id.values[P.i.values], sid2=df.source_id.values[P.j.values],
        scr4=np.logical_and(*g("scr4_star")), flags_ok=np.ones(len(P), bool)))

# --------------------------------------------------------------------------- filtre HR
def hr_band(K_all, slope=3.3, height=0.8):
    """Parallélogramme le long de la séquence principale (Cookson §3.2.4).
    Hypothèse : bande centrée sur la médiane de MG - slope*(BP-RP) des étoiles propres
    (l'intercept exact n'est pas publié)."""
    c = np.concatenate([K_all.c1, K_all.c2]); m = np.concatenate([K_all.MG1, K_all.MG2])
    ok = (c > 0.5) & (c < 3.5) & (m > 4) & (m < 14)
    b = np.nanmedian(m[ok]-slope*c[ok])
    return b, slope, height

def in_hr(c, MG, b, slope, height):
    r = MG-slope*c
    return (c > 0.5) & (c < 3.5) & (MG > 4) & (MG < 14) & (r > b-height/2) & (r < b+height/2)

# --------------------------------------------------------------------------- coupes paramétrées
BASE = dict(dmax=130, smin=1000, smax=30000, vtmax=2.5, svt=0.1, ruwe=1.25, drv=10, dvsky=10,
            sph=True, hr=True, ipd=False, degroup="snr10", screen="dr3", epoch=False)

def select(K, hrpar, **kw):
    o = {**BASE, **kw}
    vt = K.vt_sph if o["sph"] else K.vt_raw
    v = K.v_sph if o["sph"] else K.v_raw
    m = (K.d < o["dmax"]) & (K.rsky > o["smin"]) & (K.rsky < o["smax"])
    m &= (K.ruwe_max < o["ruwe"]) & K.both_rv & (K.drv < o["drv"]) & (v < o["dvsky"])
    m &= (vt < o["vtmax"]) & (K.svt < o["svt"])
    if o["hr"]:
        m &= in_hr(K.c1, K.MG1, *hrpar) & in_hr(K.c2, K.MG2, *hrpar)
    if o["ipd"]:
        m &= K.ipd_max == 0
    if o["screen"] == "dr4":
        m &= K.scr4 & K.flags_ok
    if o["epoch"] and "epoch_ok" in K.columns:      # absent sur DR3 : ignoré (signalé dans le journal)
        m &= K.epoch_ok
    if o["degroup"] == "snr10": m &= K.nc10 == 1
    elif o["degroup"] == "snr40": m &= K.nc40 == 1
    return m.values, vt.values

# --------------------------------------------------------------------------- prédictions
def nu(y):  return 1.0/(1.0-np.exp(-np.sqrt(y)))
def mond_boost(x):
    """v_c/v_c,N, Cookson eq. 7, avec r = r_sky*sqrt(3/2)."""
    x3 = np.asarray(x)*np.sqrt(1.5)
    gi = 1.0/x3**2
    gt = np.hypot(gi, GNE)
    h = 1e-4
    K = (np.log(nu(gt*(1+h)))-np.log(nu(gt*(1-h))))/(np.log(1+h)-np.log(1-h))
    return np.sqrt(nu(gt)*(1+K/3*np.tanh(0.825*GNE/gi)**3.7))   # éq. 7 : [tanh(·)]^3,7 (corrigé v0.7)

EDGES = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 6.0])

def binned_test(x, vt, edges=EDGES):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        n = int(s.sum())
        if n < 5: continue
        med = np.median(vt[s]); mad = np.median(np.abs(vt[s]-med))
        rows.append(dict(lo=lo, hi=hi, n=n, xmed=float(np.median(x[s])), med=med, err=1.4826*mad/np.sqrt(n)))
    B = pd.DataFrame(rows)
    if len(B) < 3: return B, dict(chi2_N=np.nan, chi2_M=np.nan, dof=0, pref_newton=np.nan, n=int(len(x)))
    w = 1/B.err**2
    A_N = (w*B.med).sum()/w.sum(); chiN = (w*(B.med-A_N)**2).sum()
    f = mond_boost(B.xmed.values)
    A_M = (w*B.med*f).sum()/(w*f*f).sum(); chiM = (w*(B.med-A_M*f)**2).sum()
    B["newton"], B["mond"] = A_N, A_M*f
    return B, dict(chi2_N=chiN, chi2_M=chiM, dof=len(B)-1, A_N=A_N,
                   pref_newton=float(np.exp((chiM-chiN)/2)), n=int(len(x)))
