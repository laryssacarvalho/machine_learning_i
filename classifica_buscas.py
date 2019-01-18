from collections import Counter
import pandas as pd

#data frame
df = pd.read_csv('busca.csv')

#para pegar varias colunas, é preciso passar um array com os nomes das colunas
X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

#90% iniciais
porcentagem_de_treino = 0.9
tamanho_de_treino = porcentagem_de_treino * len(Y)

treino_dados = X[:int(tamanho_de_treino)]
treino_marcacoes = Y[:int(tamanho_de_treino)]

#10% finais
tamanho_de_teste = len(Y) - tamanho_de_treino

teste_dados = X[-int(tamanho_de_teste):]
teste_marcacoes = Y[-int(tamanho_de_teste):]



from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)
acertos = (resultado == teste_marcacoes)

total_de_acertos = sum(acertos)
total_de_elementos = len(teste_dados)

taxa_de_acerto = 100 * total_de_acertos/total_de_elementos
print("Taxa de acerto do algoritmo: %f:" % taxa_de_acerto)
print(total_de_elementos)

#a eficacia do algoritomo que chuta tudo 0 ou 1
acerto_base = max(Counter(teste_marcacoes).values())#elemento mais comum de todos
taxa_de_acerto_base = 100.0 * acerto_base/len(teste_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)