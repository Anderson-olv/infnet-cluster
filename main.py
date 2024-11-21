import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from scipy.stats import zscore


class CountryClustering:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        self.scaled_data = None
        self.kmeans = None
        self.hierarchical = None
        self.clusters_kmeans = None
        self.clusters_hierarchical = None
        self.centroids = None

    def count_countries(self):
        """Conta e retorna o número de países únicos no dataset."""
        if "country" in self.data.columns:
            num_countries = self.data["country"].nunique()
            print(f"Número de países únicos no dataset: {num_countries}")
            return num_countries
        else:
            raise KeyError("A coluna 'country' não está disponível no dataset.")

    def plot_variable_ranges(self, output_dir="plots_original"):
        """Mostra gráficos da faixa dinâmica das variáveis antes do pré-processamento."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        numeric_columns = self.data.select_dtypes(include=np.number).columns

        # Boxplot
        plt.figure(figsize=(10, 6))
        self.data[numeric_columns].boxplot()
        plt.title("Faixa Dinâmica das Variáveis (Dados Originais)")
        plt.ylabel("Valores")
        plt.xlabel("Variáveis")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/boxplot_original.png")
        plt.show()
        plt.close()

        # Histogramas
        for column in numeric_columns:
            plt.figure(figsize=(6, 4))
            plt.hist(self.data[column], bins=15, alpha=0.7, edgecolor='black')
            plt.title(f"Distribuição da Variável (Original): {column}")
            plt.xlabel(column)
            plt.ylabel("Frequência")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/hist_{column}_original.png")
            plt.show()
            plt.close()

    def preprocess_data(self):
        """Realiza o pré-processamento dos dados."""
        self.data.dropna(inplace=True)
        z_scores = np.abs(zscore(self.data.iloc[:, 1:]))
        self.data = self.data[(z_scores < 3).all(axis=1)]
        self.data.reset_index(drop=True, inplace=True)

        data_values = self.data.drop(columns=['country'])
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(data_values)
        print("Pré-processamento concluído!")
        return self.scaled_data

    def apply_kmeans(self, n_clusters=3):
        """Aplica o algoritmo K-Médias."""
        print("\nAplicando K-Médias...")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.clusters_kmeans = self.kmeans.fit_predict(self.scaled_data)
        self.centroids = self.kmeans.cluster_centers_
        self.data["Cluster_KMeans"] = self.clusters_kmeans
        print("K-Médias concluído! Clusters atribuídos.")

    def apply_hierarchical(self, n_clusters=3):
        """Aplica a Clusterização Hierárquica."""
        print("\nAplicando Clusterização Hierárquica...")
        self.hierarchical = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
        self.clusters_hierarchical = self.hierarchical.fit_predict(self.scaled_data)
        self.data["Cluster_Hierarchical"] = self.clusters_hierarchical
        print("Clusterização Hierárquica concluída! Clusters atribuídos.")

    def analyze_clusters(self):
        """Gera análises detalhadas dos clusters obtidos."""
        numeric_columns = self.data.select_dtypes(include=np.number).columns
        analyses = {}

        if "Cluster_KMeans" in self.data.columns:
            analyses["Cluster_KMeans"] = self.data.groupby("Cluster_KMeans")[numeric_columns].agg(['mean', 'median', 'std'])
            print("\nAnálise dos Clusters (K-Médias):")
            for cluster, stats in analyses["Cluster_KMeans"].iterrows():
                print(f"\nCluster {cluster + 1}:")
                print(stats)

        if "Cluster_Hierarchical" in self.data.columns:
            analyses["Cluster_Hierarchical"] = self.data.groupby("Cluster_Hierarchical")[numeric_columns].agg(['mean', 'median', 'std'])
            print("\nAnálise dos Clusters (Hierárquica):")
            for cluster, stats in analyses["Cluster_Hierarchical"].iterrows():
                print(f"\nCluster {cluster + 1}:")
                print(stats)

        return analyses

    def represent_clusters(self):
        """Identifica o país que melhor representa cada cluster no K-Médias."""
        representatives = []
        print("\nPaíses Representativos por Cluster (K-Médias):")
        for cluster in range(self.kmeans.n_clusters):
            cluster_data = self.data[self.data["Cluster_KMeans"] == cluster]
            indices = cluster_data.index
            distances = np.linalg.norm(self.scaled_data[indices] - self.centroids[cluster], axis=1)
            representative_country = cluster_data.iloc[np.argmin(distances)]["country"]
            representatives.append((cluster, representative_country))
            print(f"Cluster {cluster + 1}: País Representativo - {representative_country}")
            print(f"Total de países no cluster: {len(cluster_data)}\n")
        return representatives

    def plot_dendrogram(self, output_dir="plots"):
        """Gera e salva o dendrograma da Clusterização Hierárquica."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.figure(figsize=(10, 7))
        linked = linkage(self.scaled_data, method="ward")
        dendrogram(linked, labels=self.data["country"].values, leaf_rotation=90, leaf_font_size=8)
        plt.title("Dendrograma - Clusterização Hierárquica")
        plt.xlabel("Países")
        plt.ylabel("Distância Euclidiana")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dendrograma.png")
        plt.show()

    def compare_methods(self):
        """Compara os clusters gerados por K-Médias e Clusterização Hierárquica."""
        if "Cluster_KMeans" in self.data.columns and "Cluster_Hierarchical" in self.data.columns:
            comparison = pd.crosstab(self.data["Cluster_KMeans"], self.data["Cluster_Hierarchical"])
            print("\nComparação entre K-Médias e Clusterização Hierárquica:")
            print(comparison)
            return comparison
        else:
            raise KeyError("Os clusters K-Médias e/ou Hierárquicos não estão disponíveis no dataset.")

    def plot_clusters(self, output_dir="plots"):
        """
        Gera e salva gráficos de dispersão para visualização dos clusters.
        
        Os clusters são plotados com base nas duas primeiras dimensões escaladas.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.clusters_kmeans is None:
            raise ValueError("Os clusters K-Médias ainda não foram gerados. Execute apply_kmeans primeiro.")

        # Gráfico de dispersão com as duas primeiras dimensões escaladas
        plt.figure(figsize=(12, 8))
        for cluster in range(self.kmeans.n_clusters):
            # Selecionar dados do cluster atual
            cluster_data = self.scaled_data[self.data["Cluster_KMeans"] == cluster]
            plt.scatter(
                cluster_data[:, 0],  # Primeira dimensão
                cluster_data[:, 1],  # Segunda dimensão
                s=70, label=f"Cluster {cluster + 1}"
            )
        
        # Adicionar os centróides ao gráfico
        plt.scatter(
            self.centroids[:, 0], self.centroids[:, 1],
            s=300, c="yellow", label="Centróides", marker="o", edgecolor="black", alpha=0.8
        )

        # Adicionar linhas conectando centróides aos pontos representativos
        for cluster in range(self.kmeans.n_clusters):
            cluster_data = self.scaled_data[self.data["Cluster_KMeans"] == cluster]
            distances = np.linalg.norm(cluster_data - self.centroids[cluster], axis=1)
            closest_point = cluster_data[np.argmin(distances)]
            plt.plot(
                [self.centroids[cluster, 0], closest_point[0]],
                [self.centroids[cluster, 1], closest_point[1]],
                color="black", linestyle="--", linewidth=1.5, alpha=0.6
            )
            # Destacar o ponto representativo
            plt.scatter(closest_point[0], closest_point[1], color="red", s=100, edgecolor="black", label=f"Representativo Cluster {cluster + 1}")

        # Personalizar o gráfico
        plt.title("Clusters (K-Médias)", fontsize=14)
        plt.xlabel("Dimensão 1 (Escalada)", fontsize=12)
        plt.ylabel("Dimensão 2 (Escalada)", fontsize=12)
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Salvar e mostrar o gráfico
        plt.savefig(f"{output_dir}/clusters_scatter_improved.png")
        plt.show()
        plt.close()

if __name__ == "__main__":
    clustering = CountryClustering(filepath="archive/Country-data.csv")
    
    # Contar e exibir o número de países
    clustering.count_countries()
    
    # Mostrar gráficos da faixa dinâmica das variáveis antes do pré-processamento
    clustering.plot_variable_ranges()
    
    # Realizar o pré-processamento dos dados
    clustering.preprocess_data()
    
    # Aplicar K-Médias e gerar gráficos
    clustering.apply_kmeans(n_clusters=3)
    clustering.plot_clusters(output_dir="plots")
    
    # Identificar e exibir os países representativos por cluster (K-Médias)
    representatives = clustering.represent_clusters()
    print("\nPaíses Representativos por Cluster (K-Médias):", representatives)
    
    # Aplicar Clusterização Hierárquica e gerar dendrograma
    clustering.apply_hierarchical(n_clusters=3)
    clustering.plot_dendrogram()
    
    # Analisar os clusters e salvar os resultados
    analyses = clustering.analyze_clusters()
    
    # Comparar métodos de clusterização
    comparison = clustering.compare_methods()
    print("\nComparação entre Métodos:\n", comparison)
