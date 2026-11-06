#%%
"""
Inter 166 Analytics v2 - Simulação de Monte Carlo do Brasileirão 2026

Em vez de olhar só os pontos acumulados, este script:
  1. Raspa da Wikipedia a classificação E a matriz de resultados (placar de cada jogo).
  2. Ajusta um modelo de Poisson por time (força de ataque, força de defesa, mando de campo).
  3. Simula os jogos restantes N vezes, jogo a jogo, aplicando os critérios de desempate
     (pontos > vitórias > saldo > gols pró).
  4. Conta em quantas simulações cada time terminou campeão, no G4/G6, no Z4 etc.

Por construção as probabilidades são coerentes: título soma 100% e rebaixamento soma 400%.
"""
import io
import re
import sys
from datetime import date

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import PoissonRegressor

N_SIMULACOES = 100_000
SEED = 166
URL = "https://pt.wikipedia.org/wiki/Campeonato_Brasileiro_de_Futebol_de_2026_-_S%C3%A9rie_A"
HEADERS = {"User-Agent": "Inter166-Analytics/2.0 (projeto de estudo; github.com/henriquereolonpain-sys)"}
PLACAR = re.compile(r"^(\d+)\s*[–-]\s*(\d+)$")


def limpar_nome(nome: str) -> str:
    """Remove marcadores da Wikipedia como '(Q)', '(R)', '(X)'."""
    return re.sub(r"\s*\([A-Z]\)\s*$", "", str(nome)).strip()


#%%
# Passo 1: Coleta (classificação + matriz de confrontos)
def coletar(url: str = URL):
    html = requests.get(url, headers=HEADERS, timeout=30).text
    tabelas = pd.read_html(io.StringIO(html))

    classif = next(t for t in tabelas if {"Pts", "J", "V", "GP", "GC"} <= set(map(str, t.columns)))
    classif = classif.iloc[:, :10].copy()
    classif.columns = ["pos", "time", "pts", "j", "v", "e", "d", "gp", "gc", "sg"]
    classif["time"] = classif["time"].map(limpar_nome)
    classif = classif[["time", "pts", "j", "v", "gp", "gc"]].astype(
        {"pts": int, "j": int, "v": int, "gp": int, "gc": int}
    )

    matriz = next(t for t in tabelas if str(t.columns[0]).startswith("Mandante"))
    matriz = matriz.set_index(matriz.columns[0])
    nomes = [limpar_nome(n) for n in matriz.index]

    jogados, restantes = [], []
    for i, mandante in enumerate(nomes):
        for j, visitante in enumerate(nomes):
            if i == j:
                continue
            m = PLACAR.match(str(matriz.iloc[i, j]))
            if m:
                jogados.append((mandante, visitante, int(m.group(1)), int(m.group(2))))
            else:
                restantes.append((mandante, visitante))

    jogados = pd.DataFrame(jogados, columns=["mandante", "visitante", "gm", "gv"])
    restantes = pd.DataFrame(restantes, columns=["mandante", "visitante"])

    # Checagem de sanidade: jogos restantes por time precisam bater com 38 - J
    faltam = pd.concat([restantes["mandante"], restantes["visitante"]]).value_counts()
    esperado = (38 - classif.set_index("time")["j"])
    divergentes = esperado[esperado != faltam.reindex(esperado.index).fillna(0)]
    if len(divergentes):
        print("ATENÇÃO: jogos restantes não batem com a classificação:", divergentes.to_dict())

    return classif, jogados, restantes


#%%
# Passo 2: Modelo de Poisson (ataque, defesa, mando)
def ajustar_poisson(jogados: pd.DataFrame, alpha: float = 0.01):
    """
    log(gols) = intercepto + ataque[time] - defesa[adversário] + mando
    Cada jogo vira duas linhas (gols do mandante e gols do visitante).
    O alpha (L2) puxa as forças para a média e evita overfitting com poucos jogos.
    """
    times = sorted(set(jogados["mandante"]) | set(jogados["visitante"]))
    k = len(times)
    pos = {t: n for n, t in enumerate(times)}

    linhas, gols = [], []
    for r in jogados.itertuples():
        for atk, dfs, g, casa in ((r.mandante, r.visitante, r.gm, 1), (r.visitante, r.mandante, r.gv, 0)):
            x = np.zeros(2 * k + 1)
            x[pos[atk]] = 1          # força de ataque de quem marca
            x[k + pos[dfs]] = 1      # fragilidade defensiva de quem sofre
            x[-1] = casa             # mando de campo
            linhas.append(x)
            gols.append(g)

    modelo = PoissonRegressor(alpha=alpha, max_iter=1000).fit(np.array(linhas), np.array(gols))
    coef = modelo.coef_
    forcas = pd.DataFrame({
        "time": times,
        "ataque": np.exp(coef[:k]),
        "defesa_fragil": np.exp(coef[k:2 * k]),
    })
    return modelo, forcas, pos, k


def lambdas(modelo, pos, k, mandante, visitante):
    def linha(atk, dfs, casa):
        x = np.zeros(2 * k + 1)
        x[pos[atk]], x[k + pos[dfs]], x[-1] = 1, 1, casa
        return x
    X = np.vstack([linha(mandante, visitante, 1), linha(visitante, mandante, 0)])
    lm, lv = modelo.predict(X)
    return lm, lv


#%%
# Passo 3: Monte Carlo dos jogos restantes
def simular(classif, restantes, modelo, pos, k, n=N_SIMULACOES, seed=SEED):
    rng = np.random.default_rng(seed)
    times = list(classif["time"])
    idx = {t: i for i, t in enumerate(times)}
    T = len(times)

    pts = np.tile(classif["pts"].to_numpy(), (n, 1)).astype(np.int32)
    vit = np.tile(classif["v"].to_numpy(), (n, 1)).astype(np.int32)
    gp = np.tile(classif["gp"].to_numpy(), (n, 1)).astype(np.int32)
    gc = np.tile(classif["gc"].to_numpy(), (n, 1)).astype(np.int32)

    for r in restantes.itertuples():
        lm, lv = lambdas(modelo, pos, k, r.mandante, r.visitante)
        h, a = idx[r.mandante], idx[r.visitante]
        gm, gv = rng.poisson(lm, n), rng.poisson(lv, n)
        pts[:, h] += np.where(gm > gv, 3, np.where(gm == gv, 1, 0))
        pts[:, a] += np.where(gv > gm, 3, np.where(gm == gv, 1, 0))
        vit[:, h] += gm > gv
        vit[:, a] += gv > gm
        gp[:, h] += gm; gc[:, h] += gv
        gp[:, a] += gv; gc[:, a] += gm

    # Critérios: pontos > vitórias > saldo > gols pró > sorteio
    chave = (pts.astype(np.int64) * 10**9 + vit * 10**6 + (gp - gc + 500) * 10**3 + gp
             ).astype(np.float64) + rng.random((n, T)) * 0.5
    ordem = np.argsort(-chave, axis=1)
    posicao = np.empty_like(ordem)
    posicao[np.arange(n)[:, None], ordem] = np.arange(1, T + 1)

    res = pd.DataFrame({"time": times, "pts_atual": classif["pts"], "jogos": classif["j"]})
    res["pts_medio_final"] = pts.mean(axis=0).round(1)
    res["campeao_%"] = ((posicao == 1).mean(axis=0) * 100)
    res["g4_%"] = ((posicao <= 4).mean(axis=0) * 100)
    res["g6_%"] = ((posicao <= 6).mean(axis=0) * 100)
    res["sula_7a12_%"] = (((posicao >= 7) & (posicao <= 12)).mean(axis=0) * 100)
    res["rebaixamento_%"] = ((posicao >= 17).mean(axis=0) * 100)
    res["posicao_media"] = posicao.mean(axis=0).round(1)
    dist = pd.DataFrame((posicao[:, :, None] == np.arange(1, T + 1)).mean(axis=0) * 100,
                        index=times, columns=range(1, T + 1))
    return res.round(2), dist.round(2), pts


#%%
# Passo 4: Execução
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    classif, jogados, restantes = coletar()
    print(f"Jogos disputados: {len(jogados)} | Jogos restantes: {len(restantes)}")

    modelo, forcas, pos, k = ajustar_poisson(jogados)
    print(f"Vantagem de mando (multiplicador de gols): {np.exp(modelo.coef_[-1]):.2f}x")
    print(forcas.assign(saldo_esperado=lambda d: d.ataque - d.defesa_fragil)
          .sort_values("saldo_esperado", ascending=False).round(2).to_string(index=False))

    res, dist, pts = simular(classif, restantes, modelo, pos, k)
    print(res.to_string(index=False))
    print("\nChecagem -> soma título:", res["campeao_%"].sum().round(1),
          "| soma rebaixamento:", res["rebaixamento_%"].sum().round(1))

    # Probabilidade de cada placar nos jogos que faltam (útil pro post/BI)
    jogos_prev = []
    for r in restantes.itertuples():
        lm, lv = lambdas(modelo, pos, k, r.mandante, r.visitante)
        g = np.arange(11)
        from scipy.stats import poisson
        pm, pv = poisson.pmf(g, lm), poisson.pmf(g, lv)
        M = np.outer(pm, pv)
        jogos_prev.append({
            "mandante": r.mandante, "visitante": r.visitante,
            "gols_esp_mandante": round(lm, 2), "gols_esp_visitante": round(lv, 2),
            "vitoria_mandante_%": round(np.tril(M, -1).sum() * 100, 1),
            "empate_%": round(np.trace(M) * 100, 1),
            "vitoria_visitante_%": round(np.triu(M, 1).sum() * 100, 1),
        })
    jogos_prev = pd.DataFrame(jogos_prev)

    hoje = date.today().isoformat()
    res.insert(0, "data_execucao", hoje)
    res.to_csv("previsoes_brasileirao_2026_v2.csv", index=False, encoding="utf-8-sig")
    dist.to_csv("distribuicao_posicoes_2026.csv", encoding="utf-8-sig")
    jogos_prev.to_csv("jogos_restantes_probabilidades.csv", index=False, encoding="utf-8-sig")
    print("\nArquivos salvos: previsoes_brasileirao_2026_v2.csv, distribuicao_posicoes_2026.csv, "
          "jogos_restantes_probabilidades.csv")
