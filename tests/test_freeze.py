"""v0.9.8 — Tests du gel (audit v0.9.7, P0) : la levée et le gel comparent les ENSEMBLES exacts de fichiers, pas seulement
les empreintes des fichiers attendus. Lancer depuis la racine : python tests/test_freeze.py"""
import os, sys, tempfile, numpy as np
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # v0.9.11 : console Windows (cp1252)
import blind as bl
d = tempfile.mkdtemp(); cwd = os.getcwd(); os.chdir(d)
def refused(fn, *a):
    try: fn(*a)
    except SystemExit: return True
    return False
try:
    for f, c in (("a.py", "x = 1\n"), ("b.csv", "1,2\n"), ("ACQUISITION_drX.json", "{}\n")): open(f, "w").write(c)
    files, arr = ["a.py", "b.csv", "ACQUISITION_drX.json"], {"ids": np.arange(5)}
    bl.freeze("F.json", files, arr); bl.verify("F.json", files, arr)
    os.remove("ACQUISITION_drX.json")
    assert refused(bl.verify, "F.json", files[:2], arr), "fichier gelé disparu de la liste : levée non refusée"
    assert refused(bl.verify, "F.json", files[:2] + ["c.py"], arr), "fichier ajouté à la liste : levée non refusée"
    open("ACQUISITION_drX.json", "w").write("{}\n"); open("b.csv", "a").write("3,4\n")
    assert refused(bl.verify, "F.json", files, arr), "fichier modifié : levée non refusée"
    bl.write_completion("M.json", files[:2])
    assert refused(bl.check_completion, "M.json", files), "liste changée depuis blind : gel non refusé"
    print("tests du gel : OK (4 cas)")
finally: os.chdir(cwd)
