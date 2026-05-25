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

# ============================================================
# Alínea D) — Pesquisa Tabu
# ============================================================

print("\n===== ALÍNEA D — PESQUISA TABU =====")

# ------------------------------------------------------------
# Funções auxiliares para a Pesquisa Tabu
# ------------------------------------------------------------

def copiar_solucao(solucao):
    """Cria uma cópia das playlists da solução."""
    return {
        "PL1": solucao["PL1"].copy(),
        "PL2": solucao["PL2"].copy(),
        "PL3": solucao["PL3"].copy(),
        "PL4": solucao["PL4"].copy()
    }


def popularidade_solucao(solucao):
    """Calcula a função objetivo: popularidade total das quatro playlists."""
    return (
        popularidade_total(solucao["PL1"]) +
        popularidade_total(solucao["PL2"]) +
        popularidade_total(solucao["PL3"]) +
        popularidade_total(solucao["PL4"])
    )


def ids_solucao(solucao):
    """Devolve o conjunto de track_id usados na solução."""
    return set(
        [m["track_id"] for m in solucao["PL1"]] +
        [m["track_id"] for m in solucao["PL2"]] +
        [m["track_id"] for m in solucao["PL3"]] +
        [m["track_id"] for m in solucao["PL4"]]
    )


def verificar_playlist(nome_playlist, playlist):
    """Verifica se uma playlist respeita as suas restrições específicas."""
    D = duracao_total(playlist)

    # Restrição comum: duração entre 32 e 35 minutos
    if not (32 <= D <= 35):
        return False

    # PL1: músicas com instrumentalness >= 0.66
    if nome_playlist == "PL1":
        return min([m["instrumentalness"] for m in playlist]) >= 0.66

    # PL2: tempo >= 120 BPM e danceability média entre 0.45 e 0.55
    if nome_playlist == "PL2":
        tempo_valido = min([m["tempo"] for m in playlist]) >= 120
        dance_media = sum(m["danceability"] for m in playlist) / len(playlist)
        return tempo_valido and (0.45 <= dance_media <= 0.55)

    # PL3: pelo menos 15 min acústicos e pelo menos 4 músicas live
    if nome_playlist == "PL3":
        A3_temp = sum(m["duration_min"] for m in playlist if m["acousticness"] >= 0.5)
        L3_temp = sum(1 for m in playlist if m["liveness"] >= 0.8)
        return A3_temp >= 15 and L3_temp >= 4

    # PL4: valence total >= 7
    if nome_playlist == "PL4":
        V4_temp = sum(m["valence"] for m in playlist)
        return V4_temp >= 7

    return False


def solucao_admissivel_tabu(solucao):
    """Verifica se a solução completa é admissível."""
    ids = (
        [m["track_id"] for m in solucao["PL1"]] +
        [m["track_id"] for m in solucao["PL2"]] +
        [m["track_id"] for m in solucao["PL3"]] +
        [m["track_id"] for m in solucao["PL4"]]
    )

    sem_repetidas = len(ids) == len(set(ids))

    return (
        sem_repetidas and
        verificar_playlist("PL1", solucao["PL1"]) and
        verificar_playlist("PL2", solucao["PL2"]) and
        verificar_playlist("PL3", solucao["PL3"]) and
        verificar_playlist("PL4", solucao["PL4"])
    )


# ------------------------------------------------------------
# Inicialização da Pesquisa Tabu
# ------------------------------------------------------------

# A solução inicial é a solução obtida pela heurística construtiva greedy
solucao_atual = {
    "PL1": PL1.copy(),
    "PL2": PL2.copy(),
    "PL3": PL3.copy(),
    "PL4": PL4.copy()
}

# A melhor solução começa por ser a solução inicial
melhor_solucao = copiar_solucao(solucao_atual)

valor_atual = popularidade_solucao(solucao_atual)
melhor_valor = valor_atual

print("Popularidade da solução inicial:", valor_atual)

# Dimensão da lista tabu
# Cada movimento fica proibido temporariamente durante as últimas 5 atualizações
dimensao_tabu = 5
lista_tabu = []

# Critérios de paragem
max_iteracoes = 20          # limite máximo de iterações
max_sem_melhoria = 5        # termina se não melhorar durante 5 iterações consecutivas
sem_melhoria = 0

playlists = ["PL1", "PL2", "PL3", "PL4"]

# Lista onde será guardada a informação de cada iteração para exportar para CSV
output_tabu = []


# ------------------------------------------------------------
# Ciclo principal da Pesquisa Tabu
# ------------------------------------------------------------

for iteracao in range(1, max_iteracoes + 1):

    melhor_vizinha = None
    melhor_valor_vizinha = -1
    melhor_movimento = None

    ids_atuais = ids_solucao(solucao_atual)

    # Gerar soluções vizinhas por substituição de uma música
    for nome_pl in playlists:

        playlist_atual = solucao_atual[nome_pl]

        for pos_remover, musica_removida in enumerate(playlist_atual):

            # Músicas candidatas ainda não usadas na solução atual
            candidatas = df_unique[
                ~df_unique["track_id"].isin(ids_atuais)
            ].copy()

            # Para reduzir o tempo computacional, analisam-se apenas as 100 mais populares
            candidatas = candidatas.sort_values("popularity", ascending=False).head(100)

            for _, musica_nova in candidatas.iterrows():

                # Movimento: playlist alterada, música removida e música adicionada
                movimento = (
                    nome_pl,
                    musica_removida["track_id"],
                    musica_nova["track_id"]
                )

                # Criar solução vizinha
                solucao_vizinha = copiar_solucao(solucao_atual)
                solucao_vizinha[nome_pl][pos_remover] = musica_nova

                # Verificar se a solução vizinha continua admissível
                if not solucao_admissivel_tabu(solucao_vizinha):
                    continue

                valor_vizinha = popularidade_solucao(solucao_vizinha)

                # Verificar se o movimento está na lista tabu
                movimento_tabu = movimento in lista_tabu

                # Critério de aspiração:
                # um movimento tabu pode ser aceite se melhorar a melhor solução global
                if movimento_tabu and valor_vizinha <= melhor_valor:
                    continue

                # Escolher a melhor solução vizinha admissível
                if valor_vizinha > melhor_valor_vizinha:
                    melhor_vizinha = solucao_vizinha
                    melhor_valor_vizinha = valor_vizinha
                    melhor_movimento = movimento

    # Se não existirem soluções vizinhas admissíveis, o algoritmo termina
    if melhor_vizinha is None:
        print("Não foram encontradas soluções vizinhas admissíveis.")
        break

    # Atualizar solução atual
    solucao_atual = melhor_vizinha
    valor_atual = melhor_valor_vizinha

    # Atualizar lista tabu
    lista_tabu.insert(0, melhor_movimento)

    if len(lista_tabu) > dimensao_tabu:
        lista_tabu.pop()

    # Atualizar melhor solução global
    if valor_atual > melhor_valor:
        melhor_solucao = copiar_solucao(solucao_atual)
        melhor_valor = valor_atual
        sem_melhoria = 0
        houve_melhoria = "Sim"
    else:
        sem_melhoria += 1
        houve_melhoria = "Não"

    print(f"\nIteração {iteracao}")
    print("Movimento escolhido:", melhor_movimento)
    print("Popularidade da solução atual:", valor_atual)
    print("Melhor popularidade encontrada:", melhor_valor)
    print("Houve melhoria global:", houve_melhoria)
    print("Lista Tabu:", lista_tabu)

    # Guardar informação da iteração para ficheiro output
    output_tabu.append({
        "iteracao": iteracao,
        "playlist_alterada": melhor_movimento[0],
        "track_id_removido": melhor_movimento[1],
        "track_id_adicionado": melhor_movimento[2],
        "movimento": str(melhor_movimento),
        "popularidade_atual": valor_atual,
        "melhor_popularidade": melhor_valor,
        "houve_melhoria_global": houve_melhoria,
        "lista_tabu": str(lista_tabu)
    })

    # Critério de paragem por estagnação
    if sem_melhoria >= max_sem_melhoria:
        print("\nCritério de paragem atingido: sem melhoria durante", max_sem_melhoria, "iterações consecutivas.")
        break


# ------------------------------------------------------------
# Resultados finais da Pesquisa Tabu
# ------------------------------------------------------------

print("\n===== MELHOR SOLUÇÃO APÓS PESQUISA TABU =====")
print("Popularidade inicial:", P_total)
print("Melhor popularidade obtida:", melhor_valor)
print("Melhoria obtida:", melhor_valor - P_total)

for nome_pl in playlists:
    playlist = melhor_solucao[nome_pl]

    print("\n" + nome_pl)
    print("Nº músicas:", len(playlist))
    print("Duração:", round(duracao_total(playlist), 2), "min")
    print("Popularidade:", popularidade_total(playlist))

    for i, m in enumerate(playlist, 1):
        print(
            f"  {i}. {m['track_name']} - {m['artists']} "
            f"| pop={m['popularity']} "
            f"| dur={m['duration_min']:.2f}"
        )

print("\nSolução final admissível:", solucao_admissivel_tabu(melhor_solucao))


# ------------------------------------------------------------
# Exportar ficheiros de output da Pesquisa Tabu
# ------------------------------------------------------------

# 1) Ficheiro com a evolução da Pesquisa Tabu por iteração
_df_output_tabu = pd.DataFrame(output_tabu)
_df_output_tabu.to_csv(
    "output_pesquisa_tabu.csv",
    index=False,
    encoding="utf-8-sig"
)

# 2) Ficheiro com a melhor solução final encontrada
linhas_solucao_final = []

for nome_pl in playlists:
    for i, m in enumerate(melhor_solucao[nome_pl], 1):
        linhas_solucao_final.append({
            "playlist": nome_pl,
            "ordem": i,
            "track_id": m["track_id"],
            "track_name": m["track_name"],
            "artists": m["artists"],
            "duration_min": m["duration_min"],
            "popularity": m["popularity"],
            "instrumentalness": m["instrumentalness"],
            "danceability": m["danceability"],
            "tempo": m["tempo"],
            "acousticness": m["acousticness"],
            "liveness": m["liveness"],
            "valence": m["valence"]
        })

_df_solucao_final = pd.DataFrame(linhas_solucao_final)
_df_solucao_final.to_csv(
    "melhor_solucao_pesquisa_tabu.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nFicheiro de output criado: output_pesquisa_tabu.csv")
print("Ficheiro com a melhor solução criado: melhor_solucao_pesquisa_tabu.csv")