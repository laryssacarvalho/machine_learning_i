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
diferencas = resultado - teste_marcacoes

acertos = [d for d in diferencas if d == 0]

total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)

taxa_de_acerto = 100 * total_de_acertos/total_de_elementos
print(taxa_de_acerto)
print(total_de_elementos)

