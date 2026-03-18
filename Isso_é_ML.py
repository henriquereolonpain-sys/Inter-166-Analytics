#%%
# Passo 1: Importação da Base de Dados
import io
import requests
import pandas as pd
import matplotlib.pyplot as plt 
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.tree import plot_tree

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "adaoduque/campeonato-brasileiro-de-futebol",
    "campeonato-brasileiro-full.csv"
)
print(df.head())

# %%
# Passo 2: Filtro de Temporadas a partir de 2006 e Cálculo de Pontos por Partida
df['data'] = pd.to_datetime(df['data'], format='mixed')
df['ano'] = df['data'].dt.year
df = df[df['ano'] >= 2006].copy()

def calcular_pontos_mandante(row):
    if row['vencedor'] == row['mandante']:
        return 3
    elif row['vencedor'] == '-':
        return 1
    else:
        return 0

def calcular_pontos_visitante(row):
    if row['vencedor'] == row['visitante']:
        return 3
    elif row['vencedor'] == '-':
        return 1
    else:
        return 0

df['pts_mandante'] = df.apply(calcular_pontos_mandante, axis=1)
df['pts_visitante'] = df.apply(calcular_pontos_visitante, axis=1)

print(df[['ano', 'rodata', 'mandante', 'visitante', 'vencedor', 'pts_mandante', 'pts_visitante']].head())


# %%
# Passo 3: Agrupamento da Pontuação Acumulada na 6ª Rodada
df.rename(columns={'rodata': 'rodada'}, inplace=True)

df_recorte = df[df['rodada'] <= 6].copy()

df_mandantes = df_recorte[['ano', 'mandante', 'pts_mandante']].rename(columns={'mandante': 'time', 'pts_mandante': 'pontos'})
df_visitantes = df_recorte[['ano', 'visitante', 'pts_visitante']].rename(columns={'visitante': 'time', 'pts_visitante': 'pontos'})

df_empilhado = pd.concat([df_mandantes, df_visitantes])

tabela_rodada_6 = df_empilhado.groupby(['ano', 'time'])['pontos'].sum().reset_index()

print(tabela_rodada_6.head(10))


# %%
# Passo 4: Criação da Variável Alvo (Rebaixados) e Consolidação do Dataset

df_mandantes_all = df[['ano', 'mandante', 'pts_mandante']].rename(columns={'mandante': 'time', 'pts_mandante': 'pontos'})
df_visitantes_all = df[['ano', 'visitante', 'pts_visitante']].rename(columns={'visitante': 'time', 'pts_visitante': 'pontos'})

df_empilhado_all = pd.concat([df_mandantes_all, df_visitantes_all])
tabela_final = df_empilhado_all.groupby(['ano', 'time'])['pontos'].sum().reset_index()

tabela_final['posicao'] = tabela_final.groupby('ano')['pontos'].rank(method='first', ascending=False)
tabela_final['rebaixado'] = tabela_final['posicao'].apply(lambda x: 1 if x >= 17 else 0)

df_modelo = pd.merge(tabela_rodada_6, tabela_final[['ano', 'time', 'rebaixado']], on=['ano', 'time'], how='left')

print(df_modelo.head(10))
print("\nTotal de rebaixados na base:", df_modelo['rebaixado'].sum())


#%%
# Passo 5: Benchmarking de Modelos

X = df_modelo[['pontos']]
y = df_modelo['rebaixado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

modelo_nb = GaussianNB()
modelo_nb.fit(X_res, y_res)
y_pred_nb = modelo_nb.predict(X_test)

modelo_lr = LogisticRegression(random_state=42)
modelo_lr.fit(X_res, y_res)
y_pred_lr = modelo_lr.predict(X_test)

modelo_rf = RandomForestClassifier(random_state=42)
modelo_rf.fit(X_res, y_res)
y_pred_rf = modelo_rf.predict(X_test)

df_comparacao = pd.DataFrame({
    'Modelo': ['Naive Bayes', 'Regressao Logistica', 'Random Forest'],
    'Acuracia': [accuracy_score(y_test, y_pred_nb), accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf)],
    'F1-Score (Rebaixados)': [f1_score(y_test, y_pred_nb), f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_rf)]
})

print("--- Comparacao de Modelos (Rebaixamento) ---")
print(df_comparacao)
print("\n")

tabela_final['campeao'] = tabela_final['posicao'].apply(lambda x: 1 if x == 1 else 0)
tabela_final['libertadores'] = tabela_final['posicao'].apply(lambda x: 1 if x <= 5 else 0)

colunas_merge = ['ano', 'time', 'campeao', 'libertadores']
df_modelo_cl = pd.merge(df_modelo, tabela_final[colunas_merge], on=['ano', 'time'], how='left', suffixes=('', '_drop'))
df_modelo_cl = df_modelo_cl.loc[:, ~df_modelo_cl.columns.str.endswith('_drop')]

y_camp = df_modelo_cl['campeao']
y_lib = df_modelo_cl['libertadores']

X_camp_res, y_camp_res = sm.fit_resample(X, y_camp)
modelo_rf_campeao = RandomForestClassifier(random_state=42)
modelo_rf_campeao.fit(X_camp_res, y_camp_res)

X_lib_res, y_lib_res = sm.fit_resample(X, y_lib)
modelo_rf_libertadores = RandomForestClassifier(random_state=42)
modelo_rf_libertadores.fit(X_lib_res, y_lib_res)

plt.figure(figsize=(20,10))
plot_tree(modelo_rf.estimators_[0], 
          feature_names=['pontos'], 
          class_names=['Nao Rebaixado', 'Rebaixado'], 
          filled=True, 
          rounded=True, 
          max_depth=3)
plt.title("Visualizacao de 1 Arvore da Random Forest (Logica de Rebaixamento)")
plt.show()

plt.figure(figsize=(20,10))
plot_tree(modelo_rf.estimators_[0], 
          feature_names=['pontos'], 
          class_names=['Nao Rebaixado', 'Rebaixado'], 
          filled=True, 
          rounded=True, 
          max_depth=3)
plt.title("Visualizacao de 1 Arvore da Random Forest (Logica de Rebaixamento)")
plt.show()


# %%
# Passo 6: Web scraping e Previsão 2026

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

tabela_final['campeao'] = tabela_final['posicao'].apply(lambda x: 1 if x == 1 else 0)
tabela_final['libertadores'] = tabela_final['posicao'].apply(lambda x: 1 if x <= 5 else 0)

colunas_merge = ['ano', 'time', 'campeao', 'libertadores']
df_modelo_cl = pd.merge(df_modelo, tabela_final[colunas_merge], on=['ano', 'time'], how='left', suffixes=('', '_drop'))
df_modelo_cl = df_modelo_cl.loc[:, ~df_modelo_cl.columns.str.endswith('_drop')]

X_geral = df_modelo_cl[['pontos']]
y_camp = df_modelo_cl['campeao']
y_lib = df_modelo_cl['libertadores']

sm = SMOTE(random_state=42)

X_camp_res, y_camp_res = sm.fit_resample(X_geral, y_camp)
modelo_rf_campeao = RandomForestClassifier(random_state=42)
modelo_rf_campeao.fit(X_camp_res, y_camp_res)

X_lib_res, y_lib_res = sm.fit_resample(X_geral, y_lib)
modelo_rf_libertadores = RandomForestClassifier(random_state=42)
modelo_rf_libertadores.fit(X_lib_res, y_lib_res)

url = "https://pt.wikipedia.org/wiki/Campeonato_Brasileiro_de_Futebol_de_2026_-_S%C3%A9rie_A"
headers = {'User-Agent': 'Mozilla/5.0'}
resposta = requests.get(url, headers=headers)

tabelas = pd.read_html(io.StringIO(resposta.text), match="Pts")
df_2026 = tabelas[0]

df_2026 = df_2026.iloc[:, [1, 2]].copy()
df_2026.columns = ['time', 'pontos']

df_2026['previsao_rebaixamento'] = modelo_rf.predict(df_2026[['pontos']])

prob_queda = modelo_rf.predict_proba(df_2026[['pontos']])
df_2026['risco_queda'] = (prob_queda[:, 1] * 100).round(2).astype(str) + '%'

prob_camp = modelo_rf_campeao.predict_proba(df_2026[['pontos']])
df_2026['chance_campeao'] = (prob_camp[:, 1] * 100).round(2).astype(str) + '%'

prob_lib = modelo_rf_libertadores.predict_proba(df_2026[['pontos']])
df_2026['chance_libertadores'] = (prob_lib[:, 1] * 100).round(2).astype(str) + '%'

df_2026.index = df_2026.index + 1

print(df_2026)


# %%
# Passo 7: Envio dos Dados para o csv
df_2026.to_csv('previsoes_brasileirao_2026.csv', index=False, encoding='utf-8')
print("Previsoes salvas em 'previsoes_brasileirao_2026.csv' com sucesso!")


