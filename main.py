import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns


# Carregar os dados
url = "archive/Country-data.csv"
data = pd.read_csv(url)

# Verificando as primeiras linhas e informações gerais do dataset
print(data.head())
print(data.info())
print(f"Total de países: {data['country'].nunique()}")

# Remover a coluna "country" se ela não for usada na clusterização
data = data.drop(columns=['country'])

# 1. Análise Exploratória com Gráficos
# Histograma para cada variável
data.hist(bins=20, figsize=(15, 10))
plt.suptitle("Distribuição das Variáveis")
plt.show()

# Boxplots para observar a faixa dinâmica e outliers de cada variável
plt.figure(figsize=(15, 8))
sns.boxplot(data=data)
plt.title("Boxplot das Variáveis")
plt.xticks(rotation=45)
plt.show()

# Mapa de calor para ver a correlação entre as variáveis
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Calor das Correlações")
plt.show()

# 2. Pré-processamento dos Dados
# Normalizando os dados para a mesma escala usando MinMaxScaler
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Verificando os dados escalados
print(data_scaled.describe())

# O dataset está pronto para a clusterização
# 1. Clusterização com K-Means
# Definindo o número de clusters como 3
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(data_scaled)

# Adicionando os rótulos dos clusters ao DataFrame
data_scaled['KMeans_Cluster'] = kmeans_labels

# Análise de cada cluster
print("Distribuição dos clusters K-Means:")
print(data_scaled['KMeans_Cluster'].value_counts())

# Calcular a média das dimensões em cada grupo para interpretação
kmeans_clusters_summary = data_scaled.groupby('KMeans_Cluster').mean()
print("\nMédia das variáveis em cada cluster K-Means:")
print(kmeans_clusters_summary)

# Identificar o país que melhor representa cada cluster (mais próximo do centróide)
# Como removemos a coluna 'country' no pré-processamento, vamos recarregar para análise
original_data = pd.read_csv("archive/Country-data.csv")
data_scaled['country'] = original_data['country']

# Corrigindo a seleção do país representativo
centroids = kmeans.cluster_centers_

for i in range(3):
    # Selecionar o cluster atual
    cluster_data = data_scaled[data_scaled['KMeans_Cluster'] == i]
    
    # Calcular a distância de cada país ao centróide do cluster
    distances = ((cluster_data.iloc[:, :-2] - centroids[i]) ** 2).sum(axis=1)
    
    # Encontrar o país com a menor distância ao centróide
    representative_country = cluster_data.loc[distances.idxmin(), 'country']
    print(f"País que melhor representa o Cluster {i}: {representative_country}")

# Visualizando a distribuição das dimensões para cada cluster
plt.figure(figsize=(10, 6))
kmeans_clusters_summary.plot(kind='bar', figsize=(15, 6), title="Distribuição média das variáveis em cada cluster K-Means")
plt.ylabel("Média Normalizada")
plt.xlabel("Cluster")
plt.legend(loc='upper right')
plt.show()

# 2. Clusterização Hierárquica
# Criando o linkage para a Clusterização Hierárquica
linkage_matrix = linkage(data_scaled.iloc[:, :-2], method='ward')

# Dendograma
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, labels=data_scaled['country'].values, leaf_rotation=90, leaf_font_size=10)
plt.title("Dendrograma da Clusterização Hierárquica")
plt.xlabel("Países")
plt.ylabel("Distância")
plt.show()

# Aplicando o modelo de Clusterização Hierárquica com 3 clusters para comparação
hierarchical = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
hierarchical_labels = hierarchical.fit_predict(data_scaled.iloc[:, :-2])

# Adicionando os rótulos da clusterização hierárquica ao DataFrame
data_scaled['Hierarchical_Cluster'] = hierarchical_labels

# Comparação das distribuições de clusters K-Means vs Hierárquico
print("\nDistribuição dos clusters Hierárquicos:")
print(data_scaled['Hierarchical_Cluster'].value_counts())

# 3. Comparação dos Resultados de Clusterização
# Comparando K-Means e Clusterização Hierárquica
comparison = data_scaled[['country', 'KMeans_Cluster', 'Hierarchical_Cluster']]
print("\nComparação dos resultados de clusterização entre K-Means e Clusterização Hierárquica:")
print(comparison)

# Análise dos clusters
sns.countplot(x="KMeans_Cluster", hue="Hierarchical_Cluster", data=data_scaled)
plt.title("Comparação de Clusterização: K-Means vs Hierárquica")
plt.show()