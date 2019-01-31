from collections import Counter
import pandas as pd

df = pd.read_csv('situacao_do_cliente.csv')

#para pegar varias colunas, é preciso passar um array com os nomes das colunas
X_df = df[['recencia', 'frequencia', 'semanas_de_inscricao']]
Y_df = df['situacao']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values


porcentagem_de_treino = 0.8
porcentagem_de_teste = 0.1

tamanho_de_treino = porcentagem_de_treino * len(Y)
tamanho_de_teste = porcentagem_de_teste * len(Y)
tamanho_de_validacao = len(Y) - tamanho_de_teste - tamanho_de_treino

#0 até 799
treino_dados = X[0:int(tamanho_de_treino)]
treino_marcacoes = Y[0:int(tamanho_de_treino)]

#800 ate 899
fim_de_teste = tamanho_de_teste + tamanho_de_treino
teste_dados = X[int(tamanho_de_treino):int(fim_de_teste)]
teste_marcacoes = Y[int(tamanho_de_treino):int(fim_de_teste)]

#900 ate 999
validacao_dados = X[int(fim_de_teste):]
validacao_marcacoes = Y[int(fim_de_teste):]


def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)
    resultado = modelo.predict(teste_dados)
    acertos = (resultado == teste_marcacoes)

    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_dados)

    taxa_de_acerto = 100 * total_de_acertos/total_de_elementos
    msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
    print(msg)
    return taxa_de_acerto
    
resultados = {}

# OneVersusRest
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0))
resultadoOneVsRest = fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados,treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultadoOneVsRest] = modeloOneVsRest

#MultinomialNB
from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial

#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost

maximo = max(resultados)
vencedor = resultados[maximo]

resultado = vencedor.predict(validacao_dados)
acertos = (resultado == validacao_marcacoes)

total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_marcacoes)

taxa_de_acerto = 100 * total_de_acertos/total_de_elementos
msg = "Taxa de acerdo do vencedor no mundo real: {0}".format(taxa_de_acerto)
print(msg)
#a eficacia do algoritomo que chuta tudo 0 ou 1
acerto_base = max(Counter(validacao_marcacoes).values())#elemento mais comum de todos
taxa_de_acerto_base = 100.0 * acerto_base/len(validacao_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

print("Total de testes: %d " % len(validacao_dados))