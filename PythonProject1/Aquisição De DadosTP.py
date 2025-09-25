# ==========================================
# Integração com Google Drive (para Colab)
# ==========================================
from google.colab import drive
drive.mount('/content/drive')

# ==========================================
# Caminhos de entrada e saída
# ==========================================
book_local = 'shallue_local_curves_'
book_global = 'shallue_global_curves_'

# Caminho do dataset original (Kepler KOI dataset)
path_input = "/content/drive/MyDrive/Colab Notebooks/ML Dados/lighkurve_KOI_dataset.csv"

# Caminhos de saída para os resultados pré-processados
path_local = '/content/drive/MyDrive/Colab Notebooks/ML Dados/Resultados Testes' + book_local + '.xlsx'
path_global = '/content/drive/MyDrive/Colab Notebooks/ML Dados/Resultados Testes' + book_global + '.xlsx'
# ==========================================
# Importações
# ==========================================
import pandas as pd
import numpy as np
import time
from lightkurve import search_lightcurve
import sys
import warnings
warnings.simplefilter("ignore")

# ==========================================
# 1. Leitura e filtragem do dataset inicial
# ==========================================
# Carregar arquivo CSV original
lc = pd.read_csv(path_input, sep=",")

# Selecionar apenas as colunas relevantes
lc = lc[['kepid','koi_disposition','koi_period','koi_time0bk','koi_duration','koi_quarters']]

# Mostrar quantidade inicial de curvas
print('total inicial de curvas: %d\n' % (lc.shape[0]))

# Remover valores ausentes e candidatos (ficar só com confirmados e falsos positivos)
lc = lc.dropna()
lc = lc[lc.koi_disposition != 'CANDIDATE']
lc = lc.reset_index(drop=True)

# Estatísticas iniciais
print('falsos positivos: %d, confirmados: %d\n\ntotal atualizado: %d\n' % (
    (lc.koi_disposition == 'FALSE POSITIVE').sum(),
    (lc.koi_disposition == 'CONFIRMED').sum(),
    lc.shape[0]
))

# Percentual por classe
perc_class = ((lc.koi_disposition == 'FALSE POSITIVE').sum()*100)/lc.shape[0]
print('falsos positivos: %.2f %% confirmados: %.2f %% \n' % (perc_class, 100-perc_class))

# ==========================================
# 2. Pré-processamento das curvas de luz
# ==========================================
curvas_locais = []   # armazena curvas locais (recorte no trânsito)
labels_locais = []   # rótulos das curvas locais
curvas_globais = []  # armazena curvas globais (longa duração)
labels_globais = []  # rótulos das curvas globais

start_time = time.time()

# Iterar sobre um subconjunto do dataset (aqui de 5000 a 6000 como exemplo)
for index, row in lc[5000:6000].iterrows():
    period, t0, duration_hours = row[2], row[3], row[4]

    try:
        # Buscar curvas no Kepler (Lightkurve)
        lcs = search_lightcurve(str(row[0]), author='Kepler', cadence='long').download_all()

        if (lcs is not None):
            # Juntar segmentos
            lc_raw = lcs.stitch()

            # Remover outliers
            lc_clean = lc_raw.remove_outliers(sigma=3)

            # Dobragem no período orbital
            temp_fold = lc_clean.fold(period, epoch_time=t0)

            # Criar máscara para marcar pontos de trânsito
            fractional_duration = (duration_hours / 24.0) / period
            phase_mask = np.abs(temp_fold.phase.value) < (fractional_duration * 1.5)
            transit_mask = np.in1d(lc_clean.time.value, temp_fold.time_original.value[phase_mask])

            # Flatten da curva (separando tendência do trânsito)
            lc_flat, trend_lc = lc_clean.flatten(return_trend=True, mask=transit_mask)

            # Curva dobrada final
            lc_fold = lc_flat.fold(period, epoch_time=t0)

            # ------------------------------------------
            # GLOBAL preprocessing
            # ------------------------------------------
            lc_global = lc_fold.bin(bins=2001).normalize() - 1
            lc_global = (lc_global / np.abs(np.nanmin(lc_global.flux))) * 2.0 + 1

            # ------------------------------------------
            # LOCAL preprocessing
            # ------------------------------------------
            # Seleciona região ao redor do trânsito
            phase_mask = (lc_fold.phase > -4*fractional_duration) & (lc_fold.phase < 4.0*fractional_duration)
            lc_zoom = lc_fold[phase_mask]

            lc_local = lc_zoom.bin(bins=201).normalize() - 1
            lc_local = (lc_local / np.abs(np.nanmin(lc_local.flux))) * 2.0 + 1

            # ------------------------------------------
            # Armazenar curvas e rótulos
            # ------------------------------------------
            labels_locais.append(row[1])
            curvas_locais.append(lc_local.flux.value)

            labels_globais.append(row[1])
            curvas_globais.append(lc_global.flux.value)

            print(index, 'OK')
        else:
            print(index, 'not downloaded')

    except Exception as e:
        print(index, e)

# Tempo de execução
t = time.time() - start_time
print('Tempo para importar curvas de luz: %f seconds\n' % t)

# ==========================================
# 3. Criar datasets estruturados
# ==========================================
dataset_global = pd.DataFrame(curvas_globais)
dataset_local = pd.DataFrame(curvas_locais)

# Checagem de consistência (tamanho das curvas)
for i in range(1, len(curvas_globais)):
    if len(curvas_globais[i]) != len(curvas_globais[i-1]):
        print("A curva %d possui tamanho diferente. Tamanho: %d" % (i, len(curvas_globais[i])))
print("Caso nenhuma diferença apareça, todas as curvas GLOBAIS têm %d pontos.\n" % len(curvas_globais[0]))

for i in range(1, len(curvas_locais)):
    if len(curvas_locais[i]) != len(curvas_locais[i-1]):
        print("A curva %d possui tamanho diferente. Tamanho: %d" % (i, len(curvas_locais[i])))
print("Caso nenhuma diferença apareça, todas as curvas LOCAIS têm %d pontos.\n" % len(curvas_locais[0]))

# Verificação de NaNs
print("Quantidade de NaN na base GLOBAL: %s" % dataset_global.isna().sum(axis=1).sum())
print("Quantidade de NaN na base LOCAL: %s\n" % dataset_local.isna().sum(axis=1).sum())

# Percentual de NaNs antes da interpolação
perc_nan_glob = (dataset_global.isna().sum(axis=1).sum()*100) / dataset_global.count(axis=1).sum()
perc_nan_loc = (dataset_local.isna().sum(axis=1).sum()*100) / dataset_local.count(axis=1).sum()
print("Porcentagem de valores GLOBAL substituídos: %.2f %%" % perc_nan_glob)
print("Porcentagem de valores LOCAL substituídos: %.2f %%\n" % perc_nan_loc)

# Interpolação para preencher valores faltantes
dataset_global = dataset_global.interpolate(axis=1)
dataset_local = dataset_local.interpolate(axis=1)

# Conferir se ficou zerado
print("NaNs GLOBAL após interpolação: %s" % dataset_global.isna().sum(axis=1).sum())
print("NaNs LOCAL após interpolação: %s\n" % dataset_local.isna().sum(axis=1).sum())

# ==========================================
# 4. Adicionar rótulos e salvar resultados
# ==========================================
labels_glob = pd.Series(labels_globais)
labels_loc = pd.Series(labels_locais)
dataset_global['label'] = labels_glob
dataset_local['label'] = labels_loc

# Exportar datasets finais
dataset_global.to_csv(path_global, index=False)
dataset_local.to_csv(path_local, index=False)
