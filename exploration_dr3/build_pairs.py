"""Catalogue -> liens co-mouvants (100 kau) -> paires mutuelles -> cinématique (K.pkl). ~30 s."""
from wbt_pipeline import *
df = load_catalog()
I, J = find_links(df)
P = build_pairs(df, I, J)
K = kinematics(df, P)
K.to_pickle("K.pkl")
print(len(df), "étoiles ;", len(I), "liens ;", len(P), "paires mutuelles")
