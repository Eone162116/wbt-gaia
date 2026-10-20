"""Téléchargement Gaia (DR3 ou DR4) robuste aux limites du TAP ESA anonyme.
- Requêtes synchrones (les jobs asynchrones anonymes sont tués à 60 s).
- Tranches de parallaxe à volume égal ; toute tranche qui atteint la limite de lignes (troncature silencieuse)
  est automatiquement coupée en deux, récursivement.
- Relançable : les tranches déjà présentes sont sautées. Parallélisable : M=3 R=0|1|2.
- fetch_extra() : récupère une table annexe (ex. all_source_flags) pour une liste de source_id seulement.
Usage : python fetch_gaia.py [dr3|dr4]"""
import json, os, sys, time, requests, numpy as np, pandas as pd

CFG = json.load(open(os.path.join(os.path.dirname(__file__) or ".", "config_prereg.json"), encoding="utf-8"))
TAP_URL = os.environ.get("WBT_TAP_URL", CFG["tap"]["url"])   # v0.9.6 : plan B, miroir officiel si l'archive ESA est saturée

def _sync(query, timeout=600):
    r = requests.post(TAP_URL, data=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=query), timeout=timeout)
    if r.status_code != 200 or not r.content.startswith(b"source_id"):
        raise RuntimeError(f"TAP {r.status_code}: {r.content[:300]!r}")
    return r.content

def columns(release):
    cols = list(CFG["columns_base"])
    if release == "dr4": cols += CFG["columns_dr4_extra"]
    return cols

def server_count(release):
    sch = CFG["tap"]["schema"][release]; t = CFG["tap"]
    q = (f"SELECT COUNT(*) AS n FROM {sch}.gaia_source WHERE parallax > {t['plx_min_catalog']} "
         f"AND parallax_over_error > {t['snr_catalog']}")
    r = requests.post(TAP_URL, data=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q), timeout=300)
    return int(r.text.strip().splitlines()[-1])

def slice_count(release, lo, hi):
    """v0.9.1 : nombre de lignes attendu par le serveur pour une tranche [lo, hi[ de parallaxe."""
    sch = CFG["tap"]["schema"][release]
    q = (f"SELECT COUNT(*) AS n FROM {sch}.gaia_source WHERE parallax >= {lo:.6f} AND parallax < {hi:.6f} "
         f"AND parallax_over_error > {CFG['tap']['snr_catalog']}")
    r = requests.post(TAP_URL, data=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q), timeout=300)
    return int(r.text.strip().splitlines()[-1])

def fetch_range(release, lo, hi, outdir, depth=0):
    out = os.path.join(outdir, f"{release}_{lo:.6f}_{hi:.6f}.csv")
    if os.path.exists(out) and os.path.getsize(out) > 100: return
    sch = CFG["tap"]["schema"][release]
    q = (f"SELECT {', '.join(columns(release))} FROM {sch}.gaia_source WHERE parallax >= {lo:.6f} AND parallax < {hi:.6f} "
         f"AND parallax_over_error > {CFG['tap']['snr_catalog']}")
    t = time.time(); content = _sync(q)
    n = content.count(b"\n") - 1
    if n >= CFG["tap"]["row_limit"]:                     # troncature silencieuse : on coupe en deux (volume égal)
        mid = 1000 / (((1000/lo)**3 + (1000/min(hi, 1e4))**3) / 2) ** (1/3)
        print(f"{'  '*depth}{lo:.3f}-{hi:.3f} : {n} lignes = limite, découpe en {lo:.3f}-{mid:.3f}-{hi:.3f}", flush=True)
        fetch_range(release, lo, mid, outdir, depth+1); fetch_range(release, mid, hi, outdir, depth+1); return
    ref = slice_count(release, lo, hi)                    # v0.9.1 : troncature silencieuse SOUS la limite de lignes
    if n != ref: raise RuntimeError(f"tranche {lo:.6f}-{hi:.6f} incomplète : {n} lignes reçues, {ref} attendues")
    open(out, "wb").write(content)
    print(f"{'  '*depth}{lo:.3f}-{hi:.3f} mas : {n} lignes, {time.time()-t:.0f} s", flush=True)

def fetch_catalog(release, outdir=".", n_slices=None):
    t = CFG["tap"]; dmax = 1000 / t["plx_min_catalog"]
    if TAP_URL != CFG["tap"]["url"]: print(f"serveur TAP de repli : {TAP_URL} (complétude vérifiée contre ce même serveur)", flush=True)
    n_slices = n_slices or int(np.ceil(1.3 * server_count(release) / t["row_limit"]))   # identique pour tous les terminaux
    edges = [dmax*(k/n_slices)**(1/3) for k in range(n_slices+1)]
    plx = [1e6] + [1000/d for d in edges[1:]]
    M, R = int(os.environ.get("M", "1")), int(os.environ.get("R", "0"))
    for k in range(n_slices):
        if k % M != R: continue
        for attempt in range(3):
            try: fetch_range(release, plx[k+1], plx[k], outdir); break
            except Exception as e: print("erreur", k, e, flush=True); time.sleep(10)

def check_complete(release, outdir="."):
    import glob
    ids = pd.concat([pd.read_csv(f, usecols=["source_id"]) for f in glob.glob(os.path.join(outdir, f"{release}_*.csv"))]).source_id
    n, nu, ref = len(ids), ids.nunique(), server_count(release)
    print(f"local {n} lignes, {nu} uniques / serveur {ref}")
    return nu == ref and n == nu

def fetch_extra(release, source_ids, table, cols, chunk=400):
    """Table annexe pour une liste de source_id (crible final, quelques milliers de sources)."""
    sch = CFG["tap"]["schema"][release]; parts = []
    ids = np.unique(np.asarray(source_ids, dtype=np.int64))
    for i in range(0, len(ids), chunk):
        lst = ",".join(map(str, ids[i:i+chunk]))
        q = f"SELECT source_id, {', '.join(cols)} FROM {sch}.{table} WHERE source_id IN ({lst})"
        from io import BytesIO
        parts.append(pd.read_csv(BytesIO(_sync(q))))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["source_id"]+cols)

def repair(release, outdir="."):
    """v0.9.1 : compare chaque tranche locale au compte du serveur et supprime les tranches incomplètes,
    qu'une nouvelle passe de fetch_catalog retéléchargera. Renvoie la liste des tranches supprimées."""
    import glob, re
    bad = []
    for f in sorted(glob.glob(os.path.join(outdir, f"{release}_*.csv"))):
        lo, hi = map(float, re.findall(rf"{release}_([\d.]+)_([\d.]+)\.csv", os.path.basename(f))[0])
        n = sum(1 for _ in open(f, "rb")) - 1; ref = slice_count(release, lo, hi)
        if n != ref: bad.append(f); print(f"incomplète : {os.path.basename(f)} ({n} lignes sur {ref}) — supprimée", flush=True); os.remove(f)
    print(f"{len(bad)} tranche(s) incomplète(s) supprimée(s)" + (" : relance « python run_wbt.py " + release + " fetch »." if bad else "."))
    return bad

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "repair": repair(sys.argv[2]); sys.exit()
    rel = sys.argv[1] if len(sys.argv) > 1 else CFG["release"]
    fetch_catalog(rel)
