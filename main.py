import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from scipy.stats import zscore


class CountryClustering:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        self.scaled_data = None
        self.kmeans = None
        self.kmedoids = None
        self.hierarchical = None
        self.clusters_kmeans = None
        self.clusters_hierarchical = None
        self.clusters_kmedoids = None
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

    def apply_kmedoids(self, n_clusters=3):
        """Aplica o algoritmo K-Medoides."""
        print("\nAplicando K-Medoides...")
        self.kmedoids = KMedoids(n_clusters=n_clusters, random_state=42, method="pam")
        self.clusters_kmedoids = self.kmedoids.fit_predict(self.scaled_data)
        self.data["Cluster_KMedoids"] = self.clusters_kmedoids
        print("K-Medoides concluído! Clusters atribuídos.")

    def apply_hierarchical(self, n_clusters=3):
        """Aplica a Clusterização Hierárquica."""
        print("\nAplicando Clusterização Hierárquica...")
        self.hierarchical = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
        self.clusters_hierarchical = self.hierarchical.fit_predict(self.scaled_data)
        self.data["Cluster_Hierarchical"] = self.clusters_hierarchical
        print("Clusterização Hierárquica concluída! Clusters atribuídos.")

    def analyze_clusters(self):
        """Gera análises dos clusters obtidos."""
        numeric_columns = self.data.select_dtypes(include=np.number).columns
        analyses = {}

        if "Cluster_KMeans" in self.data.columns:
            analyses["Cluster_KMeans"] = self.data.groupby("Cluster_KMeans")[numeric_columns].agg(['mean', 'median', 'std'])
            print("\nAnálise dos Clusters (K-Médias):")
            print(analyses["Cluster_KMeans"])

        if "Cluster_Hierarchical" in self.data.columns:
            analyses["Cluster_Hierarchical"] = self.data.groupby("Cluster_Hierarchical")[numeric_columns].agg(['mean', 'median', 'std'])
            print("\nAnálise dos Clusters (Hierárquica):")
            print(analyses["Cluster_Hierarchical"])

        if "Cluster_KMedoids" in self.data.columns:
            analyses["Cluster_KMedoids"] = self.data.groupby("Cluster_KMedoids")[numeric_columns].agg(['mean', 'median', 'std'])
            print("\nAnálise dos Clusters (K-Medoides):")
            print(analyses["Cluster_KMedoids"])

        return analyses

    def represent_clusters(self):
        """Identifica o país que melhor representa cada cluster no K-Médias."""
        representatives = []
        for cluster in range(self.kmeans.n_clusters):
            cluster_data = self.data[self.data["Cluster_KMeans"] == cluster]
            indices = cluster_data.index
            distances = np.linalg.norm(self.scaled_data[indices] - self.centroids[cluster], axis=1)
            representative_country = cluster_data.iloc[np.argmin(distances)]["country"]
            representatives.append((cluster, representative_country))
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
        """Gera e salva gráficos de dispersão para visualização dos clusters."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.clusters_kmeans is None:
            raise ValueError("Os clusters K-Médias ainda não foram gerados. Execute apply_kmeans primeiro.")

        plt.figure(figsize=(10, 6))
        for cluster in range(self.kmeans.n_clusters):
            cluster_data = self.data[self.data["Cluster_KMeans"] == cluster]
            plt.scatter(
                cluster_data.iloc[:, 1],  # Exemplo: Variável X
                cluster_data.iloc[:, 2],  # Exemplo: Variável Y
                s=50, label=f"Cluster {cluster + 1}"
            )

        plt.scatter(
            self.centroids[:, 0], self.centroids[:, 1],
            s=200, c="yellow", label="Centróides", marker="o", edgecolor="black"
        )

        plt.title("Clusters (K-Médias)")
        plt.xlabel("Variável X")
        plt.ylabel("Variável Y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/clusters_scatter.png")
        plt.show()


if __name__ == "__main__":
    clustering = CountryClustering(filepath="Country-data.csv")
    clustering.count_countries()
    clustering.plot_variable_ranges()
    clustering.preprocess_data()
    clustering.apply_kmeans(n_clusters=3)
    clustering.plot_clusters(output_dir="plots")
    clustering.apply_kmedoids(n_clusters=3)
    clustering.apply_hierarchical(n_clusters=3)
    clustering.plot_dendrogram()
    clustering.analyze_clusters()
    comparison = clustering.compare_methods()
    print("\nComparação entre Métodos:\n", comparison)
