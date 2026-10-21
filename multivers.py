"""v0.9.13 — Multivers de spécifications : énumération testable, garde sur la valeur de Cookson, résumé par axe
(audit externe du 21/10/2026 sur la v0.9.12). Le multivers est DESCRIPTIF : la part de spécifications au-delà d'un seuil n'est
pas une probabilité, elle dépend du nombre de modalités de chaque axe. Les variantes d'ordonnée HR ont été ajoutées le 21/10/2026,
après le diagnostic de sensibilité sur DR3 ; elles sont secondaires et ne changent pas le verdict de S."""
import itertools
import pandas as pd

def check_cookson(h):
    """Renvoie l'ordonnée de Cookson (centre de la bande) si elle est utilisable, sinon None. Garde (v0.9.13) : date de réception,
    source et nature obligatoires ; une valeur reçue après la date limite est refusée, au lieu d'être ajoutée après coup."""
    v = (h or {}).get("valeur")
    if v is None: return None
    if not h.get("recue_le") or not h.get("source"):
        raise SystemExit("valeur de Cookson sans date de réception (recue_le) ou sans source archivée : non utilisable")
    if str(h["recue_le"]) > str(h["date_limite"]):
        raise SystemExit(f"valeur de Cookson reçue le {h['recue_le']}, après la date limite {h['date_limite']} : non utilisable")
    if h.get("nature") != "centre":
        raise SystemExit("préciser nature = « centre » : l'ordonnée doit être celle du CENTRE de la bande (convertir une borne avant saisie)")
    return float(v)

def ordonnee(o, b, hc=None):
    """Ordonnée du centre de la bande pour une modalité : « mediane », « mediane±d » ou « cookson »."""
    if o == "cookson": return hc
    return b + (float(o[len("mediane"):]) if o != "mediane" else 0.0)

def specs(G, rel, hc=None):
    """Liste des spécifications (dict des options de select, hauteur HR, modalité d'ordonnée), dans l'ordre du produit cartésien."""
    G = dict(G)
    if hc is not None: G["hr_ordonnee"] = list(G["hr_ordonnee"]) + ["cookson"]
    out = []
    for combo in itertools.product(*G.values()):
        spec = dict(zip(G.keys(), combo)); hrh = spec.pop("hr"); ho = spec.pop("hr_ordonnee", "mediane")
        if rel == "dr3" and (spec["screen"] == "dr4" or spec.get("epoch")): continue
        if hrh is None and ho != "mediane": continue       # sans filtre HR, l'ordonnée ne joue pas
        out.append((spec, hrh, ho))
    return out

def by_axis(R, axes):
    """Résumé descriptif de S par modalité de chaque axe (n spécifications, médiane, min, max)."""
    rows = []
    for a in axes:
        for lev, g in R.groupby(R[a].astype(str), dropna=False):
            rows.append(dict(axe=a, modalite=lev, n_specs=len(g), S_mediane=g.S.median(), S_min=g.S.min(), S_max=g.S.max()))
    return pd.DataFrame(rows)
