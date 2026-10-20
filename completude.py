"""v0.9.11 — Contrôle de complétude des simulations de calibration (audit externe du 20/10/2026 sur la v0.9.10, réserves A, B, C, D).
Règle unique, fixée avant les données DR4 : dans CHAQUE scénario de chaque calibration (simulations pures de S, enveloppe de
contamination de S, test conjoint), la fraction de tirages de paires perdus ou non reconstitués doit rester ≤ manquantes_max,
globalement ET dans chaque zone (x < x_low, x > x_high). Sinon la calibration est refusée. Les comptes par scénario et par zone
sont écrits dans COMPLETUDE_<rel>.json AVANT tout refus, pour diagnostic. Il s'agit d'une quasi-complétude contrôlée avec
tolérance fixée, pas d'une égalité stricte N simulé = N observé."""
import json, os
import numpy as np

def tally(stats, x, lost, x_low=0.3, x_high=1.0):
    """Ajoute à stats (dict) les comptes d'une réalisation : x des paires simulées, lost = masque booléen des paires perdues."""
    x = np.asarray(x); lost = np.asarray(lost, bool)
    for z, m in (("tout", np.ones(len(x), bool)), ("bas", x < x_low), ("haut", x > x_high)):
        stats[f"paires_{z}"] = stats.get(f"paires_{z}", 0) + int(m.sum())
        stats[f"perdues_{z}"] = stats.get(f"perdues_{z}", 0) + int((m & lost).sum())
    return stats

def fractions(stats):
    return {z: stats.get(f"perdues_{z}", 0)/max(stats.get(f"paires_{z}", 0), 1) for z in ("tout", "bas", "haut")}

def check(stats, max_frac, label, diag_path=None, section=None):
    """Vérifie un scénario ; écrit d'abord le diagnostic (si diag_path), puis refuse si une fraction dépasse max_frac."""
    fr = fractions(stats)
    if diag_path:
        d = json.load(open(diag_path, encoding="utf-8")) if os.path.exists(diag_path) else {}
        d.setdefault(section or "calibration", {})[label] = dict(stats, fractions=fr, max_frac=max_frac)
        json.dump(d, open(diag_path, "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
    bad = {z: f for z, f in fr.items() if f > max_frac}
    if bad:
        raise SystemExit(f"CALIBRATION REFUSÉE ({label}) : fraction de tirages de paires perdus au-delà de {max_frac:.3%} -> "
                         + ", ".join(f"{z} {f:.3%}" for z, f in bad.items()) + (f" (détail : {diag_path})" if diag_path else ""))
    return fr
