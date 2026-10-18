"""Orchestrateur du WBT pré-enregistré.
  python run_wbt.py <dr3|dr4> fetch       téléchargement + vérification contre COUNT(*) serveur
  python run_wbt.py <dr3|dr4> build       paires, cinématique, crible DR4 (flags des candidats)
  python run_wbt.py <dr3|dr4> blind       échantillon principal, diagnostics AVEUGLES, calibration par simulations
  python run_wbt.py <dr3|dr4> freeze      manifeste SHA-256 (code, config, échantillon, simulations)
  python run_wbt.py <dr3|dr4> unblind     vérifie le gel, lève l'aveuglement, applique la règle de décision
  python run_wbt.py <dr3|dr4> multiverse  analyses secondaires (seulement après la levée)
  python run_wbt.py <dr3|dr4> all         enchaîne fetch, build, blind (s'arrête avant le gel)"""
import glob, itertools, json, os, sys, time, numpy as np, pandas as pd
from wbt_pipeline import (load_catalog, find_links, build_pairs, kinematics, hr_band, select, binned_test, BASE)
import calibrate as cal, blind as bl

CFG = json.load(open("config_prereg.json", encoding="utf-8"))
REL = sys.argv[1]; STAGE = sys.argv[2]
P = CFG["primary"]; ST = CFG["statistic"]
CODE = ["wbt_pipeline.py", "calibrate.py", "blind.py", "run_wbt.py", "fetch_gaia.py", "epoch_accel.py", "contamination.py",
        "finfer.py", "validation_contamination/pgrid.npy", "validation_contamination/pgrid_dr3.npy", "config_prereg.json"]
def frozen_files():
    mes = CFG["calibration"]["contamination"].get("f_in_mode") == "mesure"
    return (CODE + [f"mocks_{REL}.csv", f"envelope_{REL}.csv"] + ([f"finfer_{REL}.csv"] if mes else [])
            + [f"K_{REL}.pkl", f"sample_{REL}.csv", f"CONTROLS_{REL}_aveugle.json"]      # v0.7-D : données analysées et contrôles
            + ([f"eccfit_{REL}.csv", "eccfit.py"] if CFG["calibration"].get("ecc_mode") == "mesure" else [])   # v0.8
            + ([f"joint_{REL}.npz", f"joint_mocks_{REL}.csv", "jointlik.py"] if CFG["calibration"].get("joint") else [])   # v0.9
            + ([f"epoch_stats_{REL}.csv"] if REL == "dr4" else []))
pattern = f"{REL}_*.csv" if glob.glob(f"{REL}_*.csv") else "s40_*.csv"      # s40_* = tranches DR3 de développement

def log(*a):
    msg = " ".join(str(x) for x in a); print(msg, flush=True)
    open(f"log_{REL}.txt", "a", encoding="utf-8").write(time.strftime("%H:%M:%S ") + msg + "\n")

def primary_selection(K):
    b, slope, _ = hr_band(K)
    if P["hr_intercept"] is not None: b = P["hr_intercept"]
    kw = {k: P[k] for k in ["dmax", "smin", "smax", "vtmax", "svt", "ruwe", "drv", "dvsky", "sph", "hr", "ipd", "degroup", "screen", "epoch"]}
    m, vt = select(K, (b, slope, P["hr_height"]), **kw)
    return m, vt, b

if STAGE in ("fetch", "all"):
    import fetch_gaia as fg
    fg.fetch_catalog(REL); log("catalogue complet :", fg.check_complete(REL))

if STAGE in ("build", "all"):
    df = load_catalog(pattern); I, J = find_links(df, snr_main=P["snr_main"], plx_main=1000/(1.05*max(CFG["multiverse"]["dmax"])))
    Pp = build_pairs(df, I, J, snr_main=P["snr_main"]); K = kinematics(df, Pp)
    log(f"{len(df)} étoiles, {len(I)} liens, {len(Pp)} paires mutuelles ; crible DR4 étoile : {df.scr4_star.mean():.3f}")
    if REL == "dr4":                      # flags de all_source_flags pour les seuls candidats plausibles
        import fetch_gaia as fg
        cand = (K.rsky < 30000) & (K.d < max(CFG["multiverse"]["dmax"])) & K.both_rv
        ids = np.concatenate([K.sid1[cand], K.sid2[cand]])
        F = fg.fetch_extra(REL, ids, CFG["flags_table"]["table"], CFG["flags_table"]["columns"]).set_index("source_id")
        ok = (F.non_single_star.fillna(0) == 0) & (F.sea_env_flag.fillna(0) == 0)
        K["flags_ok"] = ok.reindex(K.sid1).fillna(False).values & ok.reindex(K.sid2).fillna(False).values
        log(f"flags récupérés pour {len(F)} sources ; paires candidates propres : {K.flags_ok[cand].mean():.3f}")
    K.to_pickle(f"K_{REL}.pkl")

if STAGE == "epoch" or (STAGE == "all" and REL == "dr4"):
    if REL != "dr4": raise SystemExit("astrométrie par époque disponible seulement pour DR4")
    import epoch_accel as ea
    K = pd.read_pickle(f"K_{REL}.pkl"); E = CFG["epoch"]; G = CFG["multiverse"]
    # v0.7-C : candidats = coupes indépendantes des vitesses seulement, sans RUWE ni dégroupage ni HR :
    # sur-ensemble des candidats de f_in et de toutes les spécifications du multivers
    hb, hs, _ = hr_band(K)
    cand, _ = select(K, (hb, hs, 0.8), hr=False, sph=True, dmax=max(G["dmax"]), ruwe=1e9, degroup=None, screen="dr3",
                     ipd=False, epoch=False, vtmax=1e9, dvsky=1e9, drv=1e9, svt=P["svt"])
    ids = np.unique(np.concatenate([K.sid1[cand], K.sid2[cand]]))
    log(f"crible par époques : téléchargement pour {len(ids)} composantes (≈ {len(ids)*0.15/1000:.1f} Go en RAW)")
    raw = ea.load_epoch_file(os.environ["WBT_EPOCH_FILE"]) if os.environ.get("WBT_EPOCH_FILE") else ea.fetch_epoch_dr4(ids)
    St = ea.test_all(ea.transits_from_table(raw)); St.to_csv(f"epoch_stats_{REL}.csv", index=False)
    bad = set(St.source_id[ea.flag(St, E["p_acc"], E["uwe_max"]) | (St.n_transits < E["min_transits"])])
    have = set(St.source_id)
    def _state(a, b):                       # v0.7-C : trois états distincts
        if a in bad or b in bad: return "signalé"
        if a not in have or b not in have: return "manquant"
        return "propre"
    K["epoch_state"] = [_state(a, b) for a, b in zip(K.sid1, K.sid2)]
    K["epoch_ok"] = K.epoch_state == "propre"
    cs = K.epoch_state[cand].value_counts(normalize=True)
    log(f"crible par époques : {len(ids)} composantes candidates, {len(have)} avec données, {len(bad)} signalées ; paires candidates : "
        + ", ".join(f"{k} {cs.get(k, 0):.3f}" for k in ("propre", "signalé", "manquant")))
    K.to_pickle(f"K_{REL}.pkl")

if STAGE in ("blind", "all"):
    K = pd.read_pickle(f"K_{REL}.pkl"); m, vt, b = primary_selection(K)
    if P.get("epoch") and "epoch_ok" not in K.columns:      # v0.7-D : bloquant sur DR4
        if REL == "dr4": raise SystemExit("ARRÊT : crible par époques obligatoire sur DR4 (primary.epoch). Lance d'abord : python run_wbt.py dr4 epoch")
        log("crible par époques absent (DR3, répétition) : ignoré")
    S = K[m]; x = S.x.values; vc = S.vc.values; N = len(S)
    sv = np.stack([S.cv_aa, S.cv_ad, S.cv_dd, S.fd], 1) if "cv_aa" in S.columns else (S.svt*S.vc).values   # v0.7-C
    vt_blind = bl.blind(vt[m])
    log(f"[AVEUGLE] spécification principale : N = {N} (cible {CFG['decision']['n_target']}, minimum {CFG['decision']['n_min_powered']})"
        f" ; HR intercept {b:.3f} ; r/rM : {np.sum(x < ST['x_low'])} bas, {np.sum(x > ST['x_high'])} hauts")
    B, _ = binned_test(x, vt_blind)
    log("[AVEUGLE] médianes binées (permutées, sans information physique) :", B[["lo", "hi", "n", "med"]].round(3).to_dict("records"))
    log(f"[AVEUGLE] S permuté = {cal.step_stat(x, vt_blind, ST['x_low'], ST['x_high']):+.3f} (attendu ≈ 0)")
    pd.DataFrame(dict(sid1=S.sid1, sid2=S.sid2, x=x, M=S.M.values, vc=vc, sv=(S.svt*S.vc).values)).to_csv(f"sample_{REL}.csv", index=False)
    C = CFG["calibration"]; t = time.time()
    quick = os.environ.get("WBT_QUICK") == "1"          # v0.7-B : mode rapide, répétitions seulement
    if quick and REL == "dr4": raise SystemExit("WBT_QUICK interdit sur DR4 : la calibration complète est obligatoire.")
    if quick: log("MODE RAPIDE (WBT_QUICK=1) : simulations divisées par 10, répétition seulement")
    nm = C["n_mocks"]//10 if quick else C["n_mocks"]
    log(f"effectifs par zone : x < {ST['x_low']} : {int((x < ST['x_low']).sum())} ; x > {ST['x_high']} : {int((x > ST['x_high']).sum())}")
    ecc_sc, ecc_env = C.get("ecc_scenarios"), "thermal"
    if C.get("ecc_mode") == "mesure":                 # v0.8 : excentricité mesurée par les angles v–r, à l'aveugle
        import eccfit as ef
        if "psi" not in S.columns: raise SystemExit("K ne contient pas l'angle ψ : relance d'abord « build » (v0.8).")
        rng_e = np.random.default_rng(C["seed"] + 7); Re = C["ecc_fit"]["R"]//(2 if quick else 1)
        bN = ef.measure(S.psi.values, x, vc, sv, rng_e, "newton", Re, P["vtmax"], P["dvsky"])
        bM = ef.measure(S.psi.values, x, vc, sv, rng_e, "mond", Re, P["vtmax"], P["dvsky"])
        # v0.8.1 : pente des demi-grands axes, mesurée sur les séparations projetées des paires candidates (sans vitesses)
        mcb, _ = select(K, (b, hr_band(K)[1], P["hr_height"]), dmax=P["dmax"], smin=500, smax=60000, svt=P["svt"], sph=True, hr=P["hr"],
                        ipd=P["ipd"], ruwe=P["ruwe"], degroup=P["degroup"], screen=P["screen"], epoch=P.get("epoch", False),
                        vtmax=1e9, dvsky=1e9, drv=1e9)
        s_all = K.rsky.values[mcb]; rs_ = S.rsky.values
        zr = {"bas": np.quantile(rs_[x < ST["x_low"]], [0.05, 0.95]), "haut": np.quantile(rs_[x > ST["x_high"]], [0.05, 0.95])}
        bB = {z: ef.beta_fit(s_all, *zr[z]) for z in zr}
        ecc_sc = ef.scenarios(bN, bM, bB); ecc_env = ecc_sc["mes_pire_newton"]
        rows_ = [dict(gabarits=h, zone=z, alpha_min=bb[z][0], alpha=bb[z][1], alpha_max=bb[z][2], n=bb["n_" + z])
                 for h, bb in (("newton", bN), ("mond", bM)) for z in ("bas", "haut")]
        rows_ += [dict(gabarits="beta_separations", zone=z, alpha_min=bB[z][0], alpha=bB[z][1], alpha_max=bB[z][2],
                       n=int(((s_all >= zr[z][0]) & (s_all < zr[z][1])).sum())) for z in zr]
        pd.DataFrame(rows_).to_csv(f"eccfit_{REL}.csv", index=False)
        log(f"pente des demi-grands axes (séparations projetées, sans vitesses) : β(x < {ST['x_low']}) = {bB['bas'][1]:.2f} "
            f"[{bB['bas'][0]:.2f} ; {bB['bas'][2]:.2f}] ; β(x > {ST['x_high']}) = {bB['haut'][1]:.2f} [{bB['haut'][0]:.2f} ; {bB['haut'][2]:.2f}]")
        log(f"excentricité mesurée (angles v–r, sans |ṽ|) : α(x < {ST['x_low']}) = {bN['bas'][1]:.2f} [{bN['bas'][0]:.2f} ; {bN['bas'][2]:.2f}] ; "
            f"α(x > {ST['x_high']}) = {bN['haut'][1]:.2f} [{bN['haut'][0]:.2f} ; {bN['haut'][2]:.2f}] ; scénarios : "
            + ", ".join(f"{k} {tuple(round(a, 2) for a in v['x'])}" for k, v in ecc_sc.items()))
    mocks = cal.calibrate(x, vc, sv, n_mocks=nm, seed=C["seed"], ecc=C["eccentricity"], ecc_scenarios=ecc_sc,
                          rsky=S.rsky.values, stat_kw=dict(x_low=ST["x_low"], x_high=ST["x_high"]), vtmax=P["vtmax"], dvsky=P["dvsky"])
    mocks.to_csv(f"mocks_{REL}.csv", index=False)
    log(f"calibration : {nm} simulations par colonne ({time.time()-t:.0f} s) ; "
        + ", ".join(f"{c} {mocks[c].mean():+.3f} ± {mocks[c].std():.3f}" for c in mocks.columns))
    from wbt_pipeline import MG_SUN
    _mass = lambda MG: (lambda mm: np.where(mm < 0.7, mm + 0.05, mm))(10**(0.0725*(MG_SUN - MG)))
    dmod = 5*np.log10(S.d.values/10)
    geo = dict(x=x, vc=vc, sv=sv, d=S.d.values, rsky=S.rsky.values, M=S.M.values, m1=_mass(S.MG1.values), m2=_mass(S.MG2.values),
               G1=S.MG1.values + dmod, G2=S.MG2.values + dmod)
    CC = dict(C["contamination"]); t2 = time.time()
    CC["screen"] = CC.get("screen_by_release", {}).get(REL, CC["screen"])
    scenarios = None
    if CC.get("f_in_mode") == "mesure":            # v0.6 : f_in mesuré sur les drapeaux des paires candidates, sans vitesses
        import finfer as fi
        hb, hs, _ = hr_band(K)
        if P["hr_intercept"] is not None: hb = P["hr_intercept"]
        mc, _ = select(K, (hb, hs, P["hr_height"]), dmax=P["dmax"], smin=P["smin"], smax=P["smax"], svt=P["svt"], sph=P["sph"],
                       hr=P["hr"], degroup=P["degroup"], screen="dr3", epoch=False, ipd=False, ruwe=1e9, vtmax=1e9, dvsky=1e9, drv=1e9)
        Cd = K[mc]
        if REL == "dr4" and "epoch_state" in Cd.columns:     # v0.7-C : seules les paires évaluées entrent dans le taux
            n_man = int((Cd.epoch_state == "manquant").sum()); Cd = Cd[Cd.epoch_state != "manquant"]
            flags, grid = (Cd.epoch_state == "signalé").values, "dr4"
            log(f"f_in : {n_man} paires candidates sans données d'époque exclues du dénominateur")
        else: flags, grid = (Cd.ruwe_max > 1.25).values, "dr3"
        band, r, sr = fi.infer_band(flags, _mass(Cd.MG1.values), _mass(Cd.MG2.values), Cd.d.values, Cd.rsky.values, CC["shapes"],
                                    grid, CC["r0"][grid], CC["nsig"], seed=C["seed"], stability=CC["stability"], q_range=CC["q_range"])
        scenarios = [(sh, f) for sh, fs in band.items() for f in fs]
        pd.DataFrame([dict(shape=sh, f_bas=fs[0], f_central=fs[1], f_haut=fs[2], taux=r, taux_err=sr, n=len(flags), drapeau=grid)
                      for sh, fs in band.items()]).to_csv(f"finfer_{REL}.csv", index=False)
        log(f"f_in mesuré (drapeau {grid}, sans vitesses) : taux signalé {r:.3f} ± {sr:.3f} sur {len(flags)} paires candidates ; "
            + " ; ".join(f"{sh} {fs[0]:.2f}–{fs[2]:.2f}" for sh, fs in band.items()))
    env = cal.calibrate_envelope(geo, n_per_f=(CC["n_mocks_per_f"]//10 if quick else CC["n_mocks_per_f"]), seed=C["seed"], stat_kw=dict(x_low=ST["x_low"], x_high=ST["x_high"]),
                                 cfg=CC, vtmax=P["vtmax"], dvsky=P["dvsky"], scenarios=scenarios, ecc=ecc_env)
    env.to_csv(f"envelope_{REL}.csv", index=False)
    g = env.groupby(["shape", "f_in"]).S
    log(f"enveloppe de contamination ({CC['screen']}, {time.time()-t2:.0f} s) : S newtonien moyen : "
        + ", ".join(f"{sh[:8]} {f:.2f} → {v:+.3f}" for (sh, f), v in g.mean().items())
        + f" ; séparation pire MOND vs pire Newton : {(min(mocks[c].mean() for c in mocks.columns if c.startswith('mond')) - max(g.mean().max(), mocks.newton.mean()))/mocks.newton.std():.1f} σ")
    # v0.7-D : contrôles avant levée (section 14), archivés et gelés
    rng_c = np.random.default_rng(12345); vt_m = vt[m]
    perm = np.array([cal.step_stat(x, rng_c.permutation(vt_m), ST["x_low"], ST["x_high"]) for _ in range(500)])
    s_bl = cal.step_stat(x, vt_blind, ST["x_low"], ST["x_high"]); z_perm = (s_bl - np.nanmean(perm))/np.nanstd(perm)
    nlo_c, nhi_c = int((x < ST["x_low"]).sum()), int((x > ST["x_high"]).sum())
    ctrl = dict(
        permutation=dict(S_aveugle=s_bl, moyenne_permutations=float(np.nanmean(perm)), ecart_type=float(np.nanstd(perm)), z=float(z_perm), ok=bool(abs(z_perm) < 3)),
        injection=dict(S_newton_moyen=float(mocks.newton.mean()), S_mond_moyen=float(mocks.mond.mean()),
                       ok=bool(abs(mocks.newton.mean()) < 0.02 and mocks.mond.mean() > 0.10)),
        effectifs=dict(n_bas=nlo_c, n_haut=nhi_c, ok=bool(min(nlo_c, nhi_c) >= CFG["decision"].get("n_min_per_zone", 0))))
    if CC.get("f_in_mode") == "mesure" and REL == "dr4":
        band_r, rr, _ = fi.infer_band((Cd.ruwe_max > 1.25).values, _mass(Cd.MG1.values), _mass(Cd.MG2.values), Cd.d.values, Cd.rsky.values,
                                      CC["shapes"], "dr3", CC["r0"]["dr3"], CC["nsig"], seed=C["seed"], stability=CC["stability"], q_range=CC["q_range"])
        ov = all(min(band[sh][2], band_r[sh][2]) >= max(band[sh][0], band_r[sh][0]) for sh in band)
        ctrl["f_in_comparaison"] = dict(epoques=band, ruwe_grille_dr3_approchee=band_r, recouvrement=bool(ov))
    else:
        ctrl["f_in_comparaison"] = "non applicable (un seul drapeau disponible)"
    json.dump(ctrl, open(f"CONTROLS_{REL}_aveugle.json", "w", encoding="utf-8", newline="\n"), indent=2, default=float, ensure_ascii=False)
    if C.get("joint"):                      # v0.9 : test conjoint, gabarits et calibration de Λ (simulation seulement)
        import jointlik as jl
        JC = C["joint"]; t3 = time.time(); rng_j = np.random.default_rng(C["seed"] + 9)
        grid_j = "dr3" if CC["screen"].startswith("dr3") else "dr4"
        zs = {"bas": np.where(x < ST["x_low"])[0], "haut": np.where(x > ST["x_high"])[0]}
        if C.get("ecc_mode") == "mesure":
            a_c, b_c = ecc_sc["mes_central"]["x"], ecc_sc["mes_central"].get("beta", (1.0, 1.0))
            a_pn, b_pn = ecc_sc["mes_pire_newton"]["x"], ecc_sc["mes_pire_newton"].get("beta", (1.0, 1.0))
            a_pm, b_pm = ecc_sc["mes_pire_mond"]["x"], ecc_sc["mes_pire_mond"].get("beta", (1.0, 1.0))
        else:
            a_c = a_pn = a_pm = b_c = b_pn = b_pm = (1.0, 1.0)
        fb = band if CC.get("f_in_mode") == "mesure" else {"raghavan2010": (0.0, 0.3, 0.5)}
        sh0 = "raghavan2010" if "raghavan2010" in fb else list(fb)[0]
        cont = [(sh0, 0.0)] + [(sh, fb[sh][1]) for sh in fb] + [(sh0, fb[sh0][2])]
        Tj = jl.templates(geo, zs, b_c, cont, CC, rng_j, JC["R_templates"]//(2 if quick else 1), P["vtmax"], P["dvsky"], grid_j)
        nJ = JC["n_par_scenario"]//(10 if quick else 1)
        scen_j = [("newton", "central", a_c, b_c, fb[sh0][1]), ("newton", "pire_newton_f_haut", a_pn, b_pn, fb[sh0][2]),
                  ("mond", "central", a_c, b_c, fb[sh0][1]), ("mond", "pire_mond_f0", a_pm, b_pm, 0.0), ("mond", "pire_mond_f_haut", a_pm, b_pm, fb[sh0][2])]
        rows_j = []
        for hyp, lab, a_, b_, f_ in scen_j:
            for _ in range(nJ):
                lv_, ps_, k_ = jl.simulate_obs(geo, a_, b_, int(hyp == "mond"), f_, sh0, CC, rng_j, P["vtmax"], P["dvsky"], grid=grid_j)
                rows_j.append((hyp, lab, jl.lam(lv_, ps_, k_, zs, Tj)))
        Jm = pd.DataFrame(rows_j, columns=["hyp", "scenario", "lam"]); Jm.to_csv(f"joint_mocks_{REL}.csv", index=False)
        np.savez(f"joint_{REL}.npz", T=Tj, cont=np.array([f"{s_}:{v_}" for s_, v_ in cont]))
        gj = Jm.groupby(["hyp", "scenario"]).lam
        log(f"test conjoint ({time.time()-t3:.0f} s, {nJ} simulations par scénario) : Λ moyen ± σ "
            + ", ".join(f"{h}/{s_} {gj.mean()[(h, s_)]:+.1f} ± {gj.std()[(h, s_)]:.1f}" for h, s_ in gj.mean().index))
    log("contrôles avant levée : " + " ; ".join(f"{k} {'OK' if isinstance(v, dict) and v.get('ok', v.get('recouvrement', True)) else ('—' if isinstance(v, str) else 'ÉCHEC')}" for k, v in ctrl.items()))

def frozen_arrays():
    s = pd.read_csv(f"sample_{REL}.csv")
    return dict(sample_ids=np.sort(np.concatenate([s.sid1.values, s.sid2.values]).astype(np.int64)))

if STAGE == "freeze":
    files = frozen_files()
    man = bl.freeze(f"FREEZE_{REL}.json", files, frozen_arrays())
    log("gel enregistré :", man["frozen_at_utc"], "manifeste", man["manifest_sha256"][:16])

if STAGE == "unblind":
    man = bl.verify(f"FREEZE_{REL}.json", frozen_files(), frozen_arrays())
    K = pd.read_pickle(f"K_{REL}.pkl"); m, vt, b = primary_selection(K)
    x = K.x.values[m]; v = vt[m]
    S_obs = cal.step_stat(x, v, ST["x_low"], ST["x_high"])
    res = cal.decide(S_obs, pd.read_csv(f"mocks_{REL}.csv"), CFG["decision"]["p_threshold"], env=pd.read_csv(f"envelope_{REL}.csv"))
    B, s = binned_test(x, v)
    res.update(N=int(m.sum()), binned_log10_newton_over_mond=float(np.log10(s["pref_newton"])), unblinded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               freeze_manifest=man["manifest_sha256"])
    if res["N"] < CFG["decision"]["n_min_powered"]: res["verdict"] += " — étude déclarée sous-dimensionnée"
    nlo, nhi = int((x < ST["x_low"]).sum()), int((x > ST["x_high"]).sum()); res.update(n_bas=nlo, n_haut=nhi)
    if min(nlo, nhi) < CFG["decision"].get("n_min_per_zone", 0): res["verdict"] += " — zone de x sous-dimensionnée"
    if CFG["calibration"].get("joint"):      # v0.9 : test conjoint, Λ observé calculé une seule fois
        import jointlik as jl
        Tj = np.load(f"joint_{REL}.npz")["T"]; Jm = pd.read_csv(f"joint_mocks_{REL}.csv")
        zs = {"bas": np.where(x < ST["x_low"])[0], "haut": np.where(x > ST["x_high"])[0]}
        lam_obs = jl.lam(np.log10(v), K.psi.values[m], np.ones(len(x), bool), zs, Tj)
        J = pd.DataFrame([dict(hyp=h, scenario=s_, **cal._tail(g_.lam.values, lam_obs, "upper" if h == "newton" else "lower"))
                          for (h, s_), g_ in Jm.groupby(["hyp", "scenario"])])
        jn, jm = J[J.hyp == "newton"], J[J.hyp == "mond"]; thr = CFG["decision"]["p_threshold"]
        nrej = jn.p_cp95.max() < thr and jn.p_gauss.max() < thr; mrej = jm.p_cp95.max() < thr and jm.p_gauss.max() < thr
        vj = ("MOND rejeté dans tous ses scénarios (Newton favorisé)" if mrej and not nrej else
              "Newton rejeté dans tous ses scénarios (MOND favorisé)" if nrej and not mrej else
              "Les deux rejetés (ni Newton ni MOND : systématique ou autre physique)" if nrej and mrej else "Non concluant")
        res["conjoint"] = dict(lam_obs=lam_obs, p_newton_cp95=float(jn.p_cp95.max()), p_newton_gauss=float(jn.p_gauss.max()),
                               p_mond_cp95=float(jm.p_cp95.max()), p_mond_gauss=float(jm.p_gauss.max()), verdict=vj, scenarios=J.to_dict("records"))
        if CFG["decision"].get("primary_test") == "conjoint":
            suffix = res["verdict"][len(res["verdict"].split(" — ")[0]):]
            res["verdict_S"] = res["verdict"]; res["verdict"] = vj + suffix; res["test_principal"] = "conjoint"
    # v0.7-D : contrôle en régime newtonien (forte accélération, x < 0,3), attendu proche de zéro sous les deux hypothèses
    def _s_newt(xx, vv):
        lo, hi = vv[xx < 0.15], vv[(xx >= 0.15) & (xx < 0.3)]
        return np.median(hi)/np.median(lo) - 1 if len(lo) >= 10 and len(hi) >= 10 else np.nan
    Ssub = K[m]; sv_c = np.stack([Ssub.cv_aa, Ssub.cv_ad, Ssub.cv_dd, Ssub.fd], 1); rng_c = np.random.default_rng(54321)
    simN = np.array([_s_newt(*cal.simulate(Ssub.x.values, Ssub.vc.values, sv_c, "newton", rng_c, P["vtmax"], P["dvsky"], "thermal", Ssub.rsky.values)) for _ in range(1000)])
    simM = np.array([_s_newt(*cal.simulate(Ssub.x.values, Ssub.vc.values, sv_c, "mond", rng_c, P["vtmax"], P["dvsky"], "thermal", Ssub.rsky.values)) for _ in range(1000)])
    s_n = _s_newt(x, v); z_n = (s_n - np.nanmean(simN))/np.nanstd(simN)
    ctrl_l = dict(regime_newtonien=dict(S_obs=s_n, attendu_newton=float(np.nanmean(simN)), attendu_mond=float(np.nanmean(simM)),
                                        ecart_type=float(np.nanstd(simN)), z=float(z_n), ok=bool(abs(z_n) < 3)))
    res["controles"] = dict(avant_levee=json.load(open(f"CONTROLS_{REL}_aveugle.json", encoding="utf-8")), a_la_levee=ctrl_l)
    json.dump(res, open(f"RESULT_{REL}.json", "w", encoding="utf-8"), indent=2, default=float)
    log("[LEVÉE]", json.dumps(res, indent=1, default=float, ensure_ascii=False))

if STAGE == "multiverse":
    if not (os.path.exists(f"RESULT_{REL}.json") and os.path.exists(f"FREEZE_{REL}.json")):
        raise SystemExit("multivers interdit avant la levée de l'aveuglement")
    man = bl.verify(f"FREEZE_{REL}.json", frozen_files(), frozen_arrays())
    if json.load(open(f"RESULT_{REL}.json", encoding="utf-8")).get("freeze_manifest") != man["manifest_sha256"]:
        raise SystemExit("multivers interdit : RESULT ne correspond pas au gel en cours (fichier étranger ou périmé)")
    K = pd.read_pickle(f"K_{REL}.pkl"); b, slope, _ = hr_band(K)
    if P["hr_intercept"] is not None: b = P["hr_intercept"]
    G = CFG["multiverse"]; rows = []
    for combo in itertools.product(*G.values()):
        spec = dict(zip(G.keys(), combo)); hrh = spec.pop("hr")
        if REL == "dr3" and (spec["screen"] == "dr4" or spec.get("epoch")): continue
        m, vt = select(K, (b, slope, hrh or 0.8), hr=hrh is not None, ipd=P["ipd"], **spec)
        x, v = K.x.values[m], vt[m]
        rows.append(dict(**spec, hr=hrh, n=int(m.sum()), S=cal.step_stat(x, v, ST["x_low"], ST["x_high"])))
    R = pd.DataFrame(rows); R.to_csv(f"multiverse_{REL}.csv", index=False)
    log(f"multivers : {len(R)} spécifications ; S médian {R.S.median():+.3f} ; part S > +0,075 : {(R.S > 0.075).mean():.2f}")
