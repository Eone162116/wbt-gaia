"""Télécharge le catalogue Gaia DR3 (parallaxe > 6 mas, S/N parallaxe > 10) : 1 426 271 étoiles.
Contraintes du TAP ESA en anonyme (constatées oct. 2026) : les jobs asynchrones sont tués à 60 s,
les requêtes synchrones tronquent SILENCIEUSEMENT à 50 000 lignes -> 40 tranches de parallaxe à volume égal.
Parallélisable : M=3 R=0 python fetch_gaia.py & M=3 R=1 ... & M=3 R=2 ...  (relançable, saute les tranches faites)"""
import requests, os, time
SYNC = "https://gea.esac.esa.int/tap-server/tap/sync"
COLS = ("source_id, ra, dec, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, pmra_pmdec_corr, "
        "phot_g_mean_mag, bp_rp, radial_velocity, radial_velocity_error, ruwe, ipd_frac_multi_peak")
DMAX, N = 1000/6.0, 40
edges = [DMAX*(k/N)**(1/3) for k in range(N+1)]
plx = [1e9]+[1000/d for d in edges[1:]]
M, R = int(os.environ.get("M", "1")), int(os.environ.get("R", "0"))
for attempt in range(4):
    for k in [k for k in range(N) if k % M == R]:
        out = f"s40_{k:02d}.csv"
        if os.path.exists(out) and os.path.getsize(out) > 1000: continue
        hi, lo = plx[k], plx[k+1]
        q = f"SELECT {COLS} FROM gaiadr3.gaia_source WHERE parallax >= {lo:.6f} AND parallax < {hi:.6f} AND parallax_over_error > 10"
        t = time.time()
        try:
            r = requests.post(SYNC, data=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q), timeout=600)
            if r.status_code == 200 and r.content[:9] == b"source_id":
                nrows = r.content.count(b"\n")-1
                if nrows >= 50000: print(k, "TRONQUÉ à 50 000 lignes : augmenter N"); continue
                open(out, "wb").write(r.content)
            print(attempt, k, f"{lo:.3f}-{hi:.3f} mas", r.status_code, round(time.time()-t), "s", flush=True)
        except Exception as e:
            print(attempt, k, "exception", e, flush=True)
