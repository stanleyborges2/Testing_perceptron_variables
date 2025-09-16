import numpy as np

class Perceptron:
    def __init__(self):
        pass

    # Função de ativação - Sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Função de ativação - Degrau
    def step(self, x):
        return 1 if x >= 0 else 0

    # Treinamento
    def train(self, inputs, outputs, learning_rate=0.1, epochs=100, activation="sigmoid"):
        self.inputs = inputs
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation

        # Inicializa pesos e bias aleatórios
        w1, w2, bias = np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)

        for i in range(epochs):
            for j in range(len(inputs)):
                soma = w1 * inputs[j][0] + w2 * inputs[j][1] + bias

                #  função de ativação escolhida
                if activation == "sigmoid":
                    saida = self.sigmoid(soma)
                else:
                    saida = self.step(soma)

                # Atualização dos pesos e bias
                erro = outputs[j][0] - saida
                w1 = w1 + learning_rate * erro * inputs[j][0]
                w2 = w2 + learning_rate * erro * inputs[j][1]
                bias = bias + (learning_rate * erro)

        return w1, w2, bias

    # Predição
    def predict(self, weights, x1, x2, activation="sigmoid"):
        soma = (x1 * weights[0]) + (x2 * weights[1]) + weights[2]

        if activation == "sigmoid":
            return 1 if self.sigmoid(soma) > 0.5 else 0
        else:
            return self.step(soma)


if __name__ == '__main__':
    # Entradas possíveis
    inputs = [[0,0], [0,1], [1,0], [1,1]]

    #
    outputs = [[0], [0], [0], [1]]   
 
 
    perceptron = Perceptron()

    # Treinando (você pode variar learning_rate, epochs e activation)
    pesos = perceptron.train(inputs=inputs,
                             outputs=outputs,
                             learning_rate=0.1,
                             epochs=100,
                             activation="sigmoid")  # Troque para "step" se quiser

    # Testando
    print("\nResultados:")
    for entrada in inputs:
        pred = perceptron.predict(pesos, entrada[0], entrada[1], activation="sigmoid")
        print(f"{entrada} -> {pred}")
