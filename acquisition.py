"""v0.9.7 — Acquisition traçable des données (audit externe du 20/10/2026, points F3, F8, F9).
- read_flags_file(path, ids) : lecture stricte d'un fichier de drapeaux all_source_flags téléchargé à part (plan B) :
  colonnes présentes, source_id uniques, tous les identifiants demandés présents, valeurs numériques ; sinon ARRÊT.
- flags_ok(F) : propre seulement si les deux drapeaux valent 0 ; un drapeau vide ou NaN est inconnu, donc NON propre.
- record(rel, section, **info) : ACQUISITION_<rel>.json, journal durable des sources de données (serveur interrogé,
  fichiers de repli et leurs empreintes, effectifs, inventaire des tranches), avec historique des réécritures (v0.9.8).
  Obligatoire en DR4 (require), et gelé avec le reste (run_wbt.frozen_files)."""
import json, os, time
import numpy as np, pandas as pd

FLAG_COLS = ("non_single_star", "sea_env_flag")

def read_flags_file(path, ids):
    F = pd.read_csv(path)
    miss = [c for c in ("source_id",) + FLAG_COLS if c not in F.columns]
    if miss: raise SystemExit(f"ARRÊT : colonnes absentes de {path} : {miss}")
    if F.source_id.duplicated().any():
        raise SystemExit(f"ARRÊT : {int(F.source_id.duplicated().sum())} source_id en double dans {path}")
    need = pd.Index(np.unique(np.asarray(ids, dtype=np.int64)))
    F = F[F.source_id.isin(need)].copy()
    absent = need.difference(pd.Index(F.source_id))
    if len(absent):
        raise SystemExit(f"ARRÊT : {len(absent)} source_id demandés absents de {path} (sur {len(need)}). Retélécharge le fichier complet.")
    for c in FLAG_COLS:
        v = pd.to_numeric(F[c], errors="coerce")
        if (v.isna() & F[c].notna()).any(): raise SystemExit(f"ARRÊT : valeurs non numériques dans la colonne {c} de {path}")
        F[c] = v
    return F.set_index("source_id")

def flags_ok(F):
    """Propre seulement si les deux drapeaux valent 0 ; vide ou NaN = inconnu = non propre (v0.9.7)."""
    return (F[FLAG_COLS[0]] == 0) & (F[FLAG_COLS[1]] == 0)

REQUIRED_DR4 = ("catalogue", "drapeaux", "epoques")     # v0.9.8 : sections exigées avant « blind » et « freeze » en DR4

def record(rel, section, **info):
    """Écrit une section ; une version antérieure différente de la même section est conservée dans « historique » (v0.9.8)."""
    p = f"ACQUISITION_{rel}.json"
    d = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    if section in d:
        prev = dict(d[section]); prev.pop("historique", None); hist = d[section].get("historique", [])
        info["historique"] = hist + [prev]
    d[section] = dict(date_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **info)
    json.dump(d, open(p, "w", encoding="utf-8", newline="\n"), indent=2, ensure_ascii=False, default=str)
    return d

def require(rel, sections=REQUIRED_DR4):
    """v0.9.8 : arrêt si le journal d'acquisition manque ou s'il lui manque une section exigée."""
    p = f"ACQUISITION_{rel}.json"
    if not os.path.exists(p): raise SystemExit(f"ARRÊT : {p} absent ; la provenance des données est obligatoire en {rel.upper()}.")
    d = json.load(open(p, encoding="utf-8")); miss = [s for s in sections if s not in d]
    if miss: raise SystemExit(f"ARRÊT : sections absentes de {p} : {miss} (relancer l'étape correspondante : fetch, build ou epoch).")
    return d

def inventory(pattern):
    """v0.9.8 : inventaire des tranches du catalogue (fichier, octets, SHA-256 avec fins de ligne normalisées)."""
    import glob
    from blind import sha256_file
    return [dict(fichier=os.path.basename(f), octets=os.path.getsize(f), sha256=sha256_file(f)) for f in sorted(glob.glob(pattern))]
