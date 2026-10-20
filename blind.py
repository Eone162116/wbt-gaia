"""Aveuglement du WBT.
- Clé secrète stockée HORS du dépôt (variable WBT_BLIND_KEY ou ~/.wbt_blind_key), jamais affichée.
- En mode aveugle, les ṽ sont permutés entre paires : la relation r/rM - ṽ est détruite, S ≈ 0 par construction,
  mais toutes les vérifications de sélection restent possibles.
- freeze() écrit un manifeste SHA-256 (code, configuration, identifiants de l'échantillon, simulations de calibration).
- La levée exige que chaque empreinte corresponde encore : toute modification après le gel bloque la levée."""
import hashlib, json, os, secrets, time, numpy as np

KEY_PATH = os.environ.get("WBT_BLIND_KEY", os.path.expanduser("~/.wbt_blind_key"))

def _key():
    if not os.path.exists(KEY_PATH):
        with open(KEY_PATH, "w", encoding="utf-8") as f: f.write(str(secrets.randbits(63)))
        os.chmod(KEY_PATH, 0o600)
    return int(open(KEY_PATH, encoding="utf-8").read().strip())

def blind(vt):
    return np.random.default_rng(_key()).permutation(np.asarray(vt))

TEXT_EXT = (".py", ".json", ".csv", ".txt", ".md")

def sha256_file(p):
    """Fichiers texte : CRLF ramené à LF avant hachage (v0.7-D2), pour que l'empreinte survive aux conversions
    de fins de ligne (Windows, Git). Fichiers binaires (.pkl, .npy) : octet par octet."""
    h = hashlib.sha256()
    if str(p).lower().endswith(TEXT_EXT):
        h.update(open(p, "rb").read().replace(b"\r\n", b"\n"))
        return h.hexdigest()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def sha256_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def freeze(path, files, arrays):
    man = dict(frozen_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               files={f: sha256_file(f) for f in files},
               arrays={k: sha256_array(v) for k, v in arrays.items()})
    man["manifest_sha256"] = hashlib.sha256(json.dumps(man, sort_keys=True).encode()).hexdigest()
    json.dump(man, open(path, "w", encoding="utf-8"), indent=2)
    return man

def verify(path, files, arrays):
    man = json.load(open(path, encoding="utf-8"))
    if set(man["files"]) != set(files) or set(man["arrays"]) != set(arrays):     # v0.9.8 : ensembles exacts (audit v0.9.7, P0)
        raise SystemExit("LEVÉE REFUSÉE : la liste des fichiers gelés ne correspond pas -> en trop au gel "
                         f"{sorted(set(man['files']) - set(files))}, absents du gel {sorted(set(files) - set(man['files']))}")
    bad = [f for f in files if man["files"].get(f) != sha256_file(f)]
    bad += [k for k, v in arrays.items() if man["arrays"].get(k) != sha256_array(v)]
    if bad:
        raise SystemExit(f"LEVÉE REFUSÉE : modifié depuis le gel -> {bad}")
    return man

def write_completion(path, files):
    """v0.9.4 : écrit, à la toute fin de « blind », les empreintes des fichiers qui seront gelés."""
    rec = dict(completed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               files={f: (sha256_file(f) if os.path.exists(f) else None) for f in files})
    json.dump(rec, open(path, "w", encoding="utf-8"), indent=2)
    return rec

def check_completion(path, files):
    """v0.9.4 : refuse le gel si « blind » n'est pas allé au bout, ou si un fichier a changé depuis."""
    if not os.path.exists(path):
        raise SystemExit("GEL REFUSÉ : « blind » n'a pas été mené à son terme (marqueur " + path + " absent). Relance « blind » jusqu'au bout.")
    rec = json.load(open(path, encoding="utf-8"))
    if set(rec["files"]) != set(files):                                          # v0.9.8 : ensembles exacts
        raise SystemExit("GEL REFUSÉ : la liste des fichiers à geler a changé depuis la fin de « blind » -> "
                         f"{sorted(set(rec['files']) ^ set(files))}. Relance « blind ».")
    bad = [f for f in files if rec["files"].get(f) != (sha256_file(f) if os.path.exists(f) else None)]
    if bad:
        raise SystemExit("GEL REFUSÉ : modifié depuis la fin de « blind » -> " + str(bad) + ". Relance « blind » avec le code actuel.")
    return rec
