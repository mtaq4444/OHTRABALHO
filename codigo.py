"""
Optimização Heurística 2025/26 - Trabalho de Grupo
Data Exploration
"""

import pandas as pd
# ============================================================
# 1. LEITURA E PREPARAÇÃO DO DATASET
# ============================================================

df = pd.read_csv("dataset_playlist.csv")
df.columns = df.columns.str.strip()


if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])


# ============================================================
# 2. ANÁLISE INICIAL DO DATASET
# ============================================================
# Remover músicas duplicadas pelo track_id
df_unique = df.drop_duplicates(subset="track_id").copy()

# Converter duração de milissegundos para minutos
df_unique["duration_min"] = df_unique["duration_ms"] / 60000

print("\n===== INFORMAÇÕES SOBRE O DATASET =====")
print(f"Total de linhas com duplicados: {len(df)}")
print(f"Músicas únicas por track_id: {len(df_unique)}")
print(f"Número de duplicados removidos: {len(df) - len(df_unique)}")

print("\nColunas do dataset:")
print(df.columns.tolist())

print("\nValores ausentes:")
missing = df_unique.isnull().sum()
print(missing[missing > 0])

print("\n===== ESTATÍSTICAS DESCRITIVAS =====")
cols = [
    "popularity",
    "duration_min",
    "instrumentalness",
    "danceability",
    "tempo",
    "acousticness",
    "liveness",
    "valence"
]
print(df_unique[cols].describe().round(3))


print("\n===== PL1 — INSTRUMENTALNESS >= 0.66 =====")
pl1_pool = df_unique[df_unique["instrumentalness"] >= 0.66]

print("Número de músicas candidatas:", len(pl1_pool))
print("Popularidade média:", round(pl1_pool["popularity"].mean(), 2))
print("Duração média:", round(pl1_pool["duration_min"].mean(), 2), "min")
print("Duração total disponível:", round(pl1_pool["duration_min"].sum(), 2), "min")


print("\n===== PL2 — TEMPO >= 120 BPM E DANCEABILITY MÉDIA 0.5 =====")
pl2_pool = df_unique[df_unique["tempo"] >= 120]

print("Número de músicas candidatas:", len(pl2_pool))
print("Danceability mínima:", round(pl2_pool["danceability"].min(), 3))
print("Danceability média:", round(pl2_pool["danceability"].mean(), 3))
print("Danceability máxima:", round(pl2_pool["danceability"].max(), 3))

near_05 = pl2_pool[
    (pl2_pool["danceability"] >= 0.45) &
    (pl2_pool["danceability"] <= 0.55)
]

print("Músicas com danceability entre 0.45 e 0.55:", len(near_05))


print("\n===== PL3 — GÉNERO ACÚSTICO E MÚSICAS LIVE =====")
acoustic_pool = df_unique[df_unique["track_genre"] == "acoustic"]
live_pool = df_unique[df_unique["liveness"] > 0.8]
overlap = df_unique[
    (df_unique["track_genre"] == "acoustic") &
    (df_unique["liveness"] > 0.8)
]

print("Músicas do género acoustic:", len(acoustic_pool))
print("Músicas live com liveness > 0.8:", len(live_pool))
print("Músicas acoustic e live:", len(overlap))
print("Duração total acoustic disponível:", round(acoustic_pool["duration_min"].sum(), 2), "min")
print("Duração média acoustic:", round(acoustic_pool["duration_min"].mean(), 2), "min")

if acoustic_pool["duration_min"].mean() > 0:
    print(
        "Nº aproximado de músicas acoustic necessário para 15 min:",
        round(15 / acoustic_pool["duration_min"].mean())
    )


print("\n===== PL4 — VALENCE TOTAL >= 7.0 =====")
print("Valence mínima:", round(df_unique["valence"].min(), 3))
print("Valence média:", round(df_unique["valence"].mean(), 3))
print("Valence máxima:", round(df_unique["valence"].max(), 3))
print("Músicas com valence > 0.7:", len(df_unique[df_unique["valence"] > 0.7]))
print("Músicas com valence > 0.8:", len(df_unique[df_unique["valence"] > 0.8]))

print("Se a playlist tiver 9 músicas, precisa de valence média >=", round(7 / 9, 3))
print("Se a playlist tiver 10 músicas, precisa de valence média >=", round(7 / 10, 3))


print("\n===== VIABILIDADE DA DURAÇÃO DAS PLAYLISTS =====")
duracao_media = df_unique["duration_min"].mean()

print("Duração média das músicas:", round(duracao_media, 2), "min")
print(
    "Estimativa de músicas por playlist:",
    round(32 / duracao_media),
    "a",
    round(35 / duracao_media),
    "músicas"
)


print("\n===== GÉNEROS MAIS FREQUENTES =====")
print(df_unique["track_genre"].value_counts().head(15))


# ============================================================
# 3. FUNÇÕES AUXILIARES
# ============================================================

def duracao_total(playlist):
    return sum(m["duration_min"] for m in playlist)


def popularidade_total(playlist):
    return sum(m["popularity"] for m in playlist)


def imprimir_playlist(nome, playlist):
    print("\n" + "=" * 60)
    print(nome)
    print("=" * 60)

    print("Nº músicas:", len(playlist))
    print("Duração:", round(duracao_total(playlist), 2), "min")
    print("Popularidade:", popularidade_total(playlist))

    for i, m in enumerate(playlist, 1):
        print(
            f"{i}. {m['track_name']} - {m['artists']} "
            f"| pop={m['popularity']} "
            f"| dur={m['duration_min']:.2f}"
        )


def analisar_popularidade(nome, playlist):
    mais_popular = max(playlist, key=lambda m: m["popularity"])
    menos_popular = min(playlist, key=lambda m: m["popularity"])

    print("\n" + nome)
    print("Mais popular:")
    print(f"{mais_popular['track_name']} - {mais_popular['artists']}")
    print("Popularidade:", mais_popular["popularity"])

    print("Menos popular:")
    print(f"{menos_popular['track_name']} - {menos_popular['artists']}")
    print("Popularidade:", menos_popular["popularity"])

    print(
        "Diferença:",
        mais_popular["popularity"] - menos_popular["popularity"]
    )



# ============================================================
# Alínea b) Desenvolvimento de uma heurística construtiva  
# ============================================================
# 1. INICIALIZAÇÃO
M =  M = df_unique.copy()

PL1, PL2, PL3, PL4 = [], [], [], []

D1 = D2 = D3 = D4 = 0
P1 = P2 = P3 = P4 = 0

# 2. PLAYLISTS
# PL1 — INSTRUMENTALNESS >= 0.66

while D1 < 32:
    admissiveis = M[
        (M["instrumentalness"] >= 0.66) &
        (M["duration_min"] + D1 <= 35)
    ]

    if admissiveis.empty:
        break

    musica = admissiveis.sort_values("popularity", ascending=False).iloc[0]

    PL1.append(musica)
    D1 += musica["duration_min"]
    P1 += musica["popularity"]

    M = M[M["track_id"] != musica["track_id"]]

# PL2 – Dance  TEMPO >= 120 E MÉDIA DANCEABILITY ≈ 0.5

SOMA_DANCE2 = 0
DM2 = 0

while D2 < 32:
    admissiveis = M[
        (M["tempo"] >= 120) &
        (M["duration_min"] + D2 <= 35)
    ].copy()

    if admissiveis.empty:
        break

    admissiveis["desvio"] = abs(
        ((SOMA_DANCE2 + admissiveis["danceability"]) / (len(PL2) + 1)) - 0.5
    )

    musica = admissiveis.sort_values(
        by=["desvio", "popularity"],
        ascending=[True, False]
    ).iloc[0]

    PL2.append(musica)
    D2 += musica["duration_min"]
    P2 += musica["popularity"]
    SOMA_DANCE2 += musica["danceability"]
    DM2 = SOMA_DANCE2 / len(PL2)

    M = M[M["track_id"] != musica["track_id"]]

# PL3 – Acústica / Live PELO MENOS 15 MIN ACÚSTICO + 4 LIVE

A3 = 0
L3 = 0

while D3 < 35 and (D3 < 32 or A3 < 15 or L3 < 4):
    admissiveis = M[
        M["duration_min"] + D3 <= 35
    ].copy()

    if admissiveis.empty:
        break

    admissiveis["score"] = admissiveis["popularity"]

    if A3 < 15:
        admissiveis.loc[
            admissiveis["acousticness"] >= 0.5, "score"
        ] += 30

    if L3 < 4:
        admissiveis.loc[
            admissiveis["liveness"] >= 0.8, "score"
        ] += 50

    musica = admissiveis.sort_values(
        "score",
        ascending=False
    ).iloc[0]

    PL3.append(musica)
    D3 += musica["duration_min"]
    P3 += musica["popularity"]

    if musica["acousticness"] >= 0.5:
        A3 += musica["duration_min"]

    if musica["liveness"] >= 0.8:
        L3 += 1

    M = M[M["track_id"] != musica["track_id"]]


# PL4 – Forte componente positiva VALENCE TOTAL >= 7


V4 = 0

while D4 < 35 and (D4 < 32 or V4 < 7):
    admissiveis = M[
        M["duration_min"] + D4 <= 35
    ].copy()

    if admissiveis.empty:
        break

    admissiveis["score"] = admissiveis["popularity"]
    admissiveis["score"] += admissiveis["valence"] * 20

    musica = admissiveis.sort_values(
        "score",
        ascending=False
    ).iloc[0]

    PL4.append(musica)
    D4 += musica["duration_min"]
    P4 += musica["popularity"]
    V4 += musica["valence"]

    M = M[M["track_id"] != musica["track_id"]]

# 3. VERIFICAÇÃO FINAL

ids_usados = (
    [m["track_id"] for m in PL1] +
    [m["track_id"] for m in PL2] +
    [m["track_id"] for m in PL3] +
    [m["track_id"] for m in PL4]
)
sem_repetidas = len(ids_usados) == len(set(ids_usados))

instrumental_PL1_valido = min([m["instrumentalness"] for m in PL1]) >= 0.66
tempo_PL2_valido = min([m["tempo"] for m in PL2]) >= 120

dance_PL2_valido = abs(DM2 - 0.5) <= 0.001

solucao_admissivel = (
    32 <= D1 <= 35 and
    32 <= D2 <= 35 and
    32 <= D3 <= 35 and
    32 <= D4 <= 35 and
    instrumental_PL1_valido and
    tempo_PL2_valido and
    A3 >= 15 and
    L3 >= 4 and
    V4 >= 7 and
    0.45 <= DM2 <= 0.55
)

# 4. Resultados finais

print("\n===== RESULTADOS FINAIS =====\n")

print("PL1 – Instrumental")
print("Nº músicas:", len(PL1))
print("Duração:", round(D1, 2), "min")
print("Popularidade:", P1)
for i, m in enumerate(PL1, 1):
    print(f"  {i}. {m['track_name']} - {m['artists']}")
print()

print("PL2 – Dance")
print("Nº músicas:", len(PL2))
print("Duração:", round(D2, 2), "min")
print("Popularidade:", P2)
print("Danceability média (DM2):", round(DM2, 3))
for i, m in enumerate(PL2, 1):
    print(f"  {i}. {m['track_name']} - {m['artists']}")
print()

print("PL3 – Acústica / Live")
print("Nº músicas:", len(PL3))
print("Duração total:", round(D3, 2), "min")
print("Popularidade:", P3)
print("Minutos acústicos (A3):", round(A3, 2))
print("Número de músicas live (L3):", L3)
print("Requisito acústico:", "OK" if A3 >= 15 else "NÃO OK")
print("Requisito live:", "OK" if L3 >= 4 else "NÃO OK")
for i, m in enumerate(PL3, 1):
    print(f"  {i}. {m['track_name']} - {m['artists']}")
print()

print("PL4 – Forte componente positiva")
print("Nº músicas:", len(PL4))
print("Duração total:", round(D4, 2), "min")
print("Popularidade:", P4)
print("Valence total (V4):", round(V4, 2))
print("Requisito valence:", "OK" if V4 >= 7 else "NÃO OK")
for i, m in enumerate(PL4, 1):
    print(f"  {i}. {m['track_name']} - {m['artists']}")

print("\n===== VERIFICAÇÃO FINAL =====")

print("PL1 duração válida:", 32 <= D1 <= 35)
print("PL1 instrumentalness válida:", instrumental_PL1_valido)

print("PL2 duração válida:", 32 <= D2 <= 35)
print("PL2 tempo válido:", tempo_PL2_valido)
print("PL2 danceability média válida:", 0.45 <= DM2 <= 0.55)

print("PL3 duração válida:", 32 <= D3 <= 35)
print("PL3 requisito acústico válido:", A3 >= 15)
print("PL3 requisito live válido:", L3 >= 4)

print("PL4 duração válida:", 32 <= D4 <= 35)
print("PL4 valence válida:", V4 >= 7)

print("\n===== POPULARIDADE DAS PLAYLISTS =====")
print("Popularidade PL1:", P1)
print("Popularidade PL2:", P2)
print("Popularidade PL3:", P3)
print("Popularidade PL4:", P4)

print("\n===== MÚSICA MAIS E MENOS POPULAR POR PLAYLIST =====")

analisar_popularidade("PL1 – Instrumental", PL1)
analisar_popularidade("PL2 – Dance", PL2)
analisar_popularidade("PL3 – Acústica / Live", PL3)
analisar_popularidade("PL4 – Forte componente positiva", PL4)

print("\n===== POPULARIDADE TOTAL =====")
P_total = P1 + P2 + P3 + P4
print("Popularidade total da solução:", P_total)

if solucao_admissivel:
    print("\nA solução obtida é admissível.")
else:
    print("\nA solução obtida NÃO é admissível.")



# ============================================================
# Alínea c) — Solução vizinha por substituição na PL4
# ============================================================

print("\n===== ALÍNEA C — SOLUÇÃO VIZINHA =====")

# Música removida da PL4
musica_removida = PL4[-1]

# IDs já usados na solução original
ids_usados = set(
    [m["track_id"] for m in PL1] +
    [m["track_id"] for m in PL2] +
    [m["track_id"] for m in PL3] +
    [m["track_id"] for m in PL4]
)

# Procurar músicas ainda não usadas
candidatas = df_unique[
    ~df_unique["track_id"].isin(ids_usados)
].copy()

found = False

for _, musica_nova in candidatas.sort_values("popularity", ascending=False).iterrows():

    PL4_viz = PL4.copy()
    PL4_viz[-1] = musica_nova

    D4_viz = sum(m["duration_min"] for m in PL4_viz)
    V4_viz = sum(m["valence"] for m in PL4_viz)
    P4_viz = sum(m["popularity"] for m in PL4_viz)

    if 32 <= D4_viz <= 35 and V4_viz >= 7:
        found = True
        break

if found:
    print("Estrutura de vizinhança: substituição de uma música por outra música não usada.")
    print("\nPlaylist alterada: PL4")

    print("\nMúsica removida:")
    print(musica_removida["track_name"], "-", musica_removida["artists"])
    print("Duração:", round(musica_removida["duration_min"], 2))
    print("Valence:", round(musica_removida["valence"], 3))
    print("Popularidade:", musica_removida["popularity"])

    print("\nMúsica adicionada:")
    print(musica_nova["track_name"], "-", musica_nova["artists"])
    print("Duração:", round(musica_nova["duration_min"], 2))
    print("Valence:", round(musica_nova["valence"], 3))
    print("Popularidade:", musica_nova["popularity"])

    print("\nPL4 original:")
    print("Duração:", round(D4, 2))
    print("Valence:", round(V4, 2))
    print("Popularidade:", P4)

    print("\nPL4 vizinha:")
    print("Duração:", round(D4_viz, 2))
    print("Valence:", round(V4_viz, 2))
    print("Popularidade:", P4_viz)

    print("\nPopularidade total original:", P1 + P2 + P3 + P4)
    print("Popularidade total vizinha:", P1 + P2 + P3 + P4_viz)

else:
    print("Nenhuma solução vizinha admissível encontrada para PL4.")