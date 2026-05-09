"""
Optimização Heurística 2025/26 - Trabalho de Grupo
Data Exploration
"""

import pandas as pd

# Carregar o dataset
df = pd.read_csv('dataset_playlist.csv')

# The dataset has duplicate track_ids (same song in multiple genres).
# We deduplicate by track_id to get a clean pool of unique songs.
df_unique = df.drop_duplicates(subset='track_id').copy()
df_unique['duration_min'] = df_unique['duration_ms'] / 60000

print("Informações sobre o Dataset")
print(f"Total de linhas (com duplicados): {len(df)}")
print(f"Musicas unicas (por track_id):   {len(df_unique)}")
print(f"Colunas: {df.columns.tolist()}")
print(f"\nValores ausentes:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

print("\nEstatísticas descritivas (músicas únicas):")
cols = ['popularity', 'duration_min', 'instrumentalness',
        'danceability', 'tempo', 'acousticness', 'liveness', 'valence']
print(df_unique[cols].describe().round(3))

print("\n" + "=" * 60)
print("PL1 — instrumentalness >= 0.66")
print("=" * 60)
pl1_pool = df_unique[df_unique['instrumentalness'] >= 0.66]
print(f"Candidate songs : {len(pl1_pool)}")
print(f"Avg popularity  : {pl1_pool['popularity'].mean():.1f}")
print(f"Avg duration    : {pl1_pool['duration_min'].mean():.2f} min")
print(f"Total duration  : {pl1_pool['duration_min'].sum():.1f} min")

print("\n" + "=" * 60)
print("PL2 — avg danceability = 0.5, tempo >= 120 BPM per song")
print("=" * 60)
pl2_pool = df_unique[df_unique['tempo'] >= 120]
print(f"Candidate songs (tempo >= 120): {len(pl2_pool)}")
print(f"Danceability stats:")
print(pl2_pool['danceability'].describe().round(3))
# Songs close to danceability 0.5 (good anchors)
near_05 = pl2_pool[(pl2_pool['danceability'] >= 0.45) & (pl2_pool['danceability'] <= 0.55)]
print(f"Songs with danceability in [0.45, 0.55]: {len(near_05)}")

print("\n" + "=" * 60)
print("PL3 — >= 15 min acoustic genre + >= 4 live songs (liveness > 0.8)")
print("=" * 60)
acoustic_pool = df_unique[df_unique['track_genre'] == 'acoustic']
live_pool = df_unique[df_unique['liveness'] > 0.8]
overlap = df_unique[(df_unique['track_genre'] == 'acoustic') & (df_unique['liveness'] > 0.8)]
print(f"Acoustic genre songs : {len(acoustic_pool)}")
print(f"Live songs           : {len(live_pool)}")
print(f"Both acoustic + live : {len(overlap)}")
print(f"Total acoustic duration available: {acoustic_pool['duration_min'].sum():.1f} min")
print(f"Avg acoustic song duration: {acoustic_pool['duration_min'].mean():.2f} min")
print(f"Songs needed for 15 min acoustic: ~{15 / acoustic_pool['duration_min'].mean():.0f} songs")

print("\n" + "=" * 60)
print("PL4 — total valence >= 7.0")
print("=" * 60)
print(f"Valence range: {df_unique['valence'].min():.3f} – {df_unique['valence'].max():.3f}")
print(f"Avg valence  : {df_unique['valence'].mean():.3f}")
print(f"Songs with valence > 0.7 : {len(df_unique[df_unique['valence'] > 0.7])}")
print(f"Songs with valence > 0.8 : {len(df_unique[df_unique['valence'] > 0.8])}")
print(f"\nIf playlist has 9 songs → need avg valence >= {7/9:.3f}")
print(f"If playlist has 10 songs → need avg valence >= {7/10:.3f}")

print("\n" + "=" * 60)
print("DURATION FEASIBILITY (32–35 min per playlist)")
print("=" * 60)
print(f"Avg song duration (all):      {df_unique['duration_min'].mean():.2f} min")
print(f"Estimated songs per playlist: {32/df_unique['duration_min'].mean():.0f}–{35/df_unique['duration_min'].mean():.0f} songs")

print("\n" + "=" * 60)
print("GENRES AVAILABLE")
print("=" * 60)
print(df_unique['track_genre'].value_counts().head(15))