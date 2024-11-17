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
   
3. Mostre através de gráficos a faixa dinâmica das variáveis que serão usadas nas tarefas de clusterização. Analise os resultados mostrados. O que deve ser feito com os dados antes da etapa de clusterização?
   
4. Realize o pré-processamento adequado dos dados.


## Parte 3: Clusterização
Para os dados pré-processados da etapa anterior, você irá:

1. Realizar o agrupamento dos países em 3 grupos distintos. Para tal, use:  
   a. K-Médias  
   b. Clusterização Hierárquica  

2. Para os resultados, do K-Médias:  
   a. Interprete cada um dos clusters obtidos citando:  
    i. Qual a distribuição das dimensões em cada grupo  

    ii. O país, de acordo com o algoritmo, melhor representa o seu agrupamento. Justifique  

3. Para os resultados da Clusterização Hierárquica, apresente o dendograma e interprete os resultados:

4. Compare os dois resultados, aponte as semelhanças e diferenças e interprete.


## Parte 4: Escolha de algoritmos
1. Escreva em tópicos as etapas do algoritmo de K-médias até sua convergência.
   
2. O algoritmo de K-médias converge até encontrar os centróides que melhor descrevem os clusters encontrados (até o deslocamento entre as interações dos centróides ser mínimo). Lembrando que o centróide é o baricentro do cluster em questão e não representa, em via de regra, um dado existente na base. Refaça o algoritmo apresentado na questão 1 a fim de garantir que o cluster seja representado pelo dado mais próximo ao seu baricentro em todas as iterações do algoritmo.
   Obs: nesse novo algoritmo, o dado escolhido será chamado medóide.

3. O algoritmo de K-médias é sensível a outliers nos dados. Explique.
   
4. Por que o algoritmo de DBScan é mais robusto à presença de outliers?