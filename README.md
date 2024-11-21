# O trabalho está dividido em 4 partes, como você pode observar a seguir.


## Parte 1: Infraestrutura
Para as questões a seguir, você deverá executar códigos em um notebook Jupyter, rodando em ambiente local, certifique-se que:

1. Você está rodando em Python 3.9+  
   ```
   Sim Python 3.12.3, printscreen se encontra na pasta pritscreen.
   ```

2. Você está usando um ambiente virtual: Virtualenv ou Anaconda  
   ```
   Sim Virtualenv, printscreen se encontra na pasta pritscreen.
   ```

3. Todas as bibliotecas usadas nesse exercícios estão instaladas em um ambiente virtual específico
   ```
   Sim bibliotecas, printscreen se encontra na pasta pritscreen.
   ``` 
4. Gere um arquivo de requerimentos (requirements.txt) com os pacotes necessários. É necessário se certificar que a versão do pacote está disponibilizada.
   ```
   requirements.tx gerado, se encontra na raiz do projeto.
   ```
5. Tire um printscreen do ambiente que será usado rodando em sua máquina.
   ```
   Virtualenv, printscreen se encontra na pasta pritscreen.
   ```  
6. Disponibilize os códigos gerados, assim como os artefatos acessórios (requirements.txt) e instruções em um repositório GIT público. (se isso não for feito, o diretório com esses arquivos deverá ser enviado compactado no moodle).
    ```bash
   git clone https://github.com/Anderson-olv/infnet-cluster.git

## Parte 2: Escolha de base de dados
Para as questões a seguir, usaremos uma base de dados e faremos a análise exploratória dos dados, antes da clusterização.

1. Baixe os dados disponibilizados na plataforma Kaggle sobre dados sócio-econômicos e de saúde que determinam o índice de desenvolvimento de um país. Esses dados estão disponibilizados através do link: https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data
   
2. Quantos países existem no dataset?
   ```
   Número de países: 167
   ```
   
3. Mostre através de gráficos a faixa dinâmica das variáveis que serão usadas nas tarefas de clusterização. Analise os resultados mostrados. O que deve ser feito com os dados antes da etapa de clusterização?
   ```
   Gráficos salvos em plots_original na raiz do projeto.
   ```
   
4. Realize o pré-processamento adequado dos dados.
   ```
   Feito.
   ```

## Parte 3: Clusterização
Para os dados pré-processados da etapa anterior, você irá:

1. Realizar o agrupamento dos países em 3 grupos distintos. Para tal, use:  
   a. K-Médias  
   ```
   Gráfico salvo em plots.
   ```
   b. Clusterização Hierárquica  
   ```
   Gráfico salvo em plots.
   ```
2. Para os resultados, do K-Médias:  
   a. Interprete cada um dos clusters obtidos citando:  
    i. Qual a distribuição das dimensões em cada grupo  
      ```
      Cluster 1:
      Perfil: Países desenvolvidos.
      Características:
      Baixa mortalidade infantil (~4.8).
      Alta renda (~37,103).
      Alta expectativa de vida (~80 anos).
      Elevado PIB per capita (~37,540).
      Exemplo: Finlândia (país representativo).

      Cluster 2:
      Perfil: Países em desenvolvimento intermediário.
      Características:
      Mortalidade infantil moderada (~20.1).
      Renda média (~12,749).
      Expectativa de vida moderada (~73 anos).
      PIB per capita intermediário (~6,535).
      Exemplo: Tunísia (país representativo).

      Cluster 3:
      Perfil: Países menos desenvolvidos.
      Características:
      Alta mortalidade infantil (~84.5).
      Baixa renda (~4,017).
      Baixa expectativa de vida (~60.5 anos).
      Baixo PIB per capita (~1,920).
      Exemplo: Tanzânia (país representativo).

      Resumo:  
      Os clusters identificam grupos de países:
      1. Desenvolvidos (Cluster 1).
      2. Em desenvolvimento (Cluster 2).
      3. Menos desenvolvidos (Cluster 3).
      ```
    ii. O país, de acordo com o algoritmo, melhor representa o seu agrupamento. Justifique
      ```
      Cluster 1:
      País Representativo: Finlândia.
      Justificativa: A Finlândia apresenta características médias dentro do grupo, como alta renda, baixa mortalidade infantil, e alta expectativa de vida, sendo próximo ao centróide do cluster.

      Cluster 2:
      País Representativo: Tunísia.
      Justificativa: A Tunísia reflete as características intermediárias do cluster, como mortalidade infantil moderada, renda média e expectativa de vida mediana, estando alinhada ao centróide.

      Cluster 3:
      País Representativo: Tanzânia.
      Justificativa: A Tanzânia é o país mais próximo do centróide, representando fielmente o perfil de países com baixa renda, alta mortalidade infantil e baixa expectativa de vida.
      ```  

3. Para os resultados da Clusterização Hierárquica, apresente o dendograma e interprete os resultados:
   ```
   Interpretação do Dendrograma:
   O dendrograma mostra como os países são agrupados com base na similaridade das suas variáveis.
   Ele divide os 167 países em três principais clusters, com base na distância euclidiana.
   As ligações no dendrograma indicam que países no mesmo cluster têm características similares.

   Cluster 1 (à esquerda):
   Inclui países com alta expectativa de vida, alta renda e baixa mortalidade infantil.
   Países desenvolvidos como Finlândia e Dinamarca dominam esse cluster.

   Cluster 2 (ao centro):
   Representa países com características intermediárias, como expectativa de vida moderada, mortalidade infantil média e níveis medianos de renda.
   A Tunísia é um bom exemplo neste cluster.

   Cluster 3 (à direita):
   Agrupa países com baixa expectativa de vida, alta mortalidade infantil e baixa renda.
   Tanzânia é representativa neste grupo, mostrando características alinhadas com as tendências do cluster.

   Tamanho dos clusters:
   Os tamanhos variam, refletindo as diferenças no número de países em cada categoria econômica e social.

   Conclusão:
   O dendrograma oferece insights visuais e estruturais claros sobre as diferenças entre os grupos.
   É consistente com os resultados de K-Médias, mostrando coerência nas análises.
   ```
4. Compare os dois resultados, aponte as semelhanças e diferenças e interprete.
   ```
   Semelhanças:

   Ambos os métodos identificaram três grupos distintos entre os países, refletindo características econômicas e sociais similares.
   Os clusters indicam padrões consistentes de expectativa de vida, renda, mortalidade infantil, inflação e outros indicadores.

   Representatividade dos Grupos:
   Tanto no K-Médias quanto na clusterização hierárquica, os países representativos (como Finlândia, Tunísia e Tanzânia) aparecem em grupos com características correspondentes.
   A distribuição geral dos países é coerente, com países desenvolvidos no Cluster 1, países em desenvolvimento no Cluster 2 e países menos desenvolvidos no Cluster 3.

   Padrões de Agrupamento:
   Países com características extremas, como alta mortalidade infantil ou renda elevada, tendem a ser agrupados juntos em ambos os métodos.

   Diferenças:
   Metodologia:
   O K-Médias utiliza centróides calculados iterativamente, enquanto a clusterização hierárquica constrói um dendrograma com base na proximidade entre países.
   O K-Médias depende da inicialização dos centróides, enquanto a hierárquica não requer um número inicial de clusters.

   Atribuição de Países:
   Algumas discrepâncias aparecem na atribuição de países. Por exemplo, na tabela de comparação:

   O K-Médias coloca 22 países no Cluster 1 que, na hierárquica, são atribuídos ao Cluster 0.
   14 países no Cluster 2 de K-Médias aparecem no Cluster 1 da hierárquica.
   Isso ocorre porque o K-Médias agrupa com base na distância ao centróide, enquanto a hierárquica considera a proximidade local.

   Flexibilidade de Visualização:
   O dendrograma da hierárquica permite explorar diferentes níveis de granularidade (podendo formar mais ou menos clusters), enquanto o K-Médias é restrito ao número de clusters pré-definido.

   Interpretação:
   K-Médias:
   Ideal para uma análise rápida e bem definida quando o número de clusters é conhecido.
   Os resultados são influenciados pela inicialização dos centróides, podendo variar levemente entre execuções.

   Hierárquica:
   Mais interpretativa e oferece um panorama completo das similaridades entre países, graças ao dendrograma.
   É útil para decidir o número ideal de clusters, mas pode ser menos escalável para grandes datasets.

   Conclusão:
   Ambos os métodos são consistentes na identificação de padrões gerais e fornecem insights complementares.

   O K-Médias é mais eficiente para grandes datasets com clusters bem definidos, enquanto a hierárquica é ideal para análises exploratórias ou quando o número de clusters não é conhecido inicialmente.

   Usar os dois métodos em conjunto, como feito neste caso, aumenta a confiabilidade e profundidade das conclusões.
   ```


## Parte 4: Escolha de algoritmos
1. Escreva em tópicos as etapas do algoritmo de K-médias até sua convergência.
   ```
   1. Inicialização: No código, o parâmetro random_state=42 no KMeans garante reprodutibilidade na escolha inicial dos centróides.

   2. Atribuição de Clusters: Os dados escalados são atribuídos a clusters com base na proximidade aos centróides, o que é refletido na saída dos clusters atribuídos no dataset.

   3. Atualização dos Centróides: Durante cada iteração do KMeans, os centróides são atualizados automaticamente para a média dos pontos de cada cluster.

   4. Reavaliação e Convergência: O algoritmo para quando não há mudanças significativas nos centróides, ou atinge o limite de iterações padrão (300 no Scikit-learn). Essa convergência foi observada no agrupamento final e na saída dos clusters.

   5. Resultado Final: As saídas mostram os clusters finais e os países representativos, confirmando que o algoritmo foi executado conforme esperado, atribuindo corretamente os dados a seus clusters com base nas dimensões.
   ```

2. O algoritmo de K-médias converge até encontrar os centróides que melhor descrevem os clusters encontrados (até o deslocamento entre as interações dos centróides ser mínimo). Lembrando que o centróide é o baricentro do cluster em questão e não representa, em via de regra, um dado existente na base. Refaça o algoritmo apresentado na questão 1 a fim de garantir que o cluster seja representado pelo dado mais próximo ao seu baricentro em todas as iterações do algoritmo.
   Obs: nesse novo algoritmo, o dado escolhido será chamado medóide.
   ```
   https://github.com/Anderson-olv/infnet-cluster.git
   Tive problema em instalar a biblioteca scikit-learn-extra, com isso realizei a criação no colab, tem um nova branches K-Medoids
   ```

3. O algoritmo de K-médias é sensível a outliers nos dados. Explique.
   ```
   O K-médias é sensível a outliers porque utiliza a distância Euclidiana e calcula centróides como a média dos pontos. Isso faz com que valores extremos possam deslocar os centróides, distorcendo os clusters e afetando a qualidade dos resultados. Para mitigar, é importante remover outliers no pré-processamento (como usar Z-score) ou utilizar algoritmos mais robustos, como K-Medoids ou DBSCAN.
   ```
   
4. Por que o algoritmo de DBScan é mais robusto à presença de outliers?
   ```
   O DBScan é mais robusto a outliers porque identifica pontos com baixa densidade como ruído em vez de forçá-los a fazer parte de um cluster. Ele forma clusters com base na densidade local, ignorando pontos isolados que não atendem ao critério de densidade mínima (definido pelos parâmetros eps e min_samples). Isso evita que outliers influenciem a formação ou localização dos clusters, diferentemente do K-médias.
   ```