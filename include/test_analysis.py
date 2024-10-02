import matplotlib.pyplot as plt
import numpy as np


class TestAnalysis:

    def __init__(self, test_points):
        self.test_X, self.test_Y = self.BreakTupleList(test_points)

        
        self.maximums_X, self.maximums_Y, self.minimums_X, self.minimums_Y = self.CriticalPoints(self.test_X, self.test_Y)
        self.curve_mean_X, self.curve_mean_Y = self.CurveMean(self.maximums_X, self.maximums_Y, self.minimums_X, self.minimums_Y)

#        self.adjusted_X, self.adjusted_Y = self.Adjust(self.test_X, self.test_Y, self.curve_mean_Y)
#        self.ad_maximums_X, self.ad_maximums_Y, self.ad_minimums_X, self.ad_minimums_Y = self.CriticalPoints(self.adjusted_X, self.adjusted_Y)

        self.amplitude_X, self.amplitude_Y = self.Amplitude(self.maximums_X, self.maximums_Y, self.minimums_X, self.minimums_Y)
                
        self.frequency = len(self.maximums_X)

        self.all_analysis = {
            'test_X': self.test_X,
            'test_Y': self.test_Y,
            'curve_mean_X': self.curve_mean_X,
            'curve_mean_Y': self.curve_mean_Y,
            'maximums_X': self.maximums_X,
            'maximums_Y': self.maximums_Y,
            'minimums_X': self.minimums_X,
            'minimums_Y': self.minimums_Y,
            #'adjusted_X': self.adjusted_X,
            #'adjusted_Y': self.adjusted_Y,
            #'ad_maximums_X': self.ad_maximums_X,
            #'ad_maximums_Y': self.ad_maximums_Y,
            #'ad_minimums_X': self.ad_minimums_X,
            #'ad_minimums_Y': self.ad_minimums_Y,
            'amplitude_X': self.amplitude_X,
            'amplitude_Y': self.amplitude_Y,
            'frequency': self.frequency
            }

    def BreakTupleList(self, test_points):
        test_X = []
        test_Y = []
        for point in test_points:
            test_X.append(point[0])
            test_Y.append(point[1])
        return test_X, test_Y 

    def CurveMean(self, maximums_X, maximums_Y, minimums_X, minimums_Y):
        meanX = []
        meanY = []

        i = 0
        if maximums_X[0] < minimums_X[0]:
            while i < (len(maximums_Y)):
                meanY.append((maximums_Y[i] + minimums_Y[i])/2)
                meanX.append((maximums_X[i] + minimums_X[i])/2)
                i +=1
        elif maximums_X[0] > minimums_X[0]:
            while i < (len(maximums_Y)):
                meanY.append((maximums_Y[i] + minimums_Y[i])/2)
                meanX.append((maximums_X[i] + minimums_X[i])/2)
                i +=1
        return meanX, meanY
        
    def Adjust(self, X, Y, mean_Y):
        adjusted_X = []
        adjusted_Y = []

        i = 0
        while i < (len(Y)):
            adjusted_X.append(X[i])

            adjusted_Y.append(Y[i]-mean_Y[i])
            i += 1

        return adjusted_X, adjusted_Y

    def CriticalPoints(self, X, Y):
        maximums_X = []
        maximums_Y = []

        minimums_X = []
        minimums_Y = []
        if (Y[1] < Y[0]):
            maximums_X.append(X[0])
            maximums_Y.append(Y[0])
        elif (Y[1] > Y[0]):
            minimums_X.append(X[0])
            minimums_Y.append(Y[0])

        i = 1
        while i < (len(Y)-1):
            if (Y[i-1] < Y[i] and Y[i+1] < Y[i]):
                maximums_X.append(X[i])
                maximums_Y.append(Y[i])
            elif (Y[i-1] > Y[i] and Y[i+1] > Y[i]):
                minimums_X.append(X[i])
                minimums_Y.append(Y[i])
            i += 1

        if (Y[i-1] < Y[i]):
            maximums_X.append(X[i])
            maximums_Y.append(Y[i])
        elif (Y[i-1] > Y[i]):
            minimums_X.append(X[i])
            minimums_Y.append(Y[i])

        if(len(maximums_Y) == len(minimums_Y)):
            print('quantidade igual')
        elif(len(maximums_Y) > len(minimums_Y)):
            print('max maior')
        elif(len(maximums_Y) < len(minimums_Y)):
            print('min maior')
        print(len(maximums_Y), len(minimums_Y))

        return maximums_X, maximums_Y, minimums_X, minimums_Y

    def Amplitude(self, maximums_X, maximums_Y, minimums_X, minimums_Y):
        amplitude_X = []
        amplitude_Y = []

        i = 0
        if maximums_X[0] < minimums_X[0]:
            while i < (len(maximums_Y)-1):
                amplitude_Y.append(abs(maximums_Y[i]-minimums_Y[i]))
                amplitude_X.append((maximums_X[i] + minimums_X[i])/2)

                amplitude_Y.append(abs(maximums_Y[i+1]-minimums_Y[i]))
                amplitude_X.append((maximums_X[i+1] + minimums_X[i])/2)
                i +=1
        elif maximums_X[0] > minimums_X[0]:
            while i < (len(maximums_Y)-1):
                amplitude_Y.append(abs(maximums_Y[i]-minimums_Y[i]))
                amplitude_X.append((maximums_X[i] + minimums_X[i])/2)

                amplitude_Y.append(abs(maximums_Y[i]-minimums_Y[i+1]))
                amplitude_X.append((maximums_X[i] + minimums_X[i+1])/2)
                i +=1

        return amplitude_X, amplitude_Y



'''
def gerarPontos(a, b, c, d):
        X = np.linspace(0, 100, 100)  # 100 pontos entre 0 e 10
        Y = a * X + np.sin(b * X) + d * np.cos(c * X)
        return X, Y

    # função interessante
    # test_X, test_Y = gerarPontos(1, 5, 0.1, 7)

    test_X, test_Y = gerarPontos(0.1, 1, 0.1, 7)
'''

'''
    plt.figure()
    plt.title('Exame')
    plt.plot(test_X, test_Y, linestyle='-', color='blue')
    plt.plot(curve_mean_X, curve_mean_Y, linestyle='-', color='red')
    plt.plot(maximums_X, maximums_Y, 'bo')
    plt.plot(minimums_X, minimums_Y, 'bo')

    plt.figure()
    plt.title('Exame com Curva Ajustada')
    plt.plot(adjusted_X, adjusted_Y, linestyle='-', color='blue')
    plt.plot(ad_maximums_X, ad_maximums_Y, 'bo')
    plt.plot(ad_minimums_X, ad_minimums_Y, 'bo')

    plt.figure()
    plt.title('Amplitude')
    plt.xlim(0, max(amplitude_X)+1)  # Definir o intervalo do eixo X
    plt.ylim(0, max(amplitude_Y)+1)  # Definir o intervalo do eixo Y
    plt.plot(amplitude_X, amplitude_Y, linestyle='-', color='green')

    plt.show()
'''


'''
def FourierAnalysis():
    # Geração de dados de exemplo
    x_data = np.linspace(0, 10, 100)
    y_data = 2.0 + 3.0 * np.sin(1.5 * x_data) + 1.0 * np.cos(2.0 * x_data) + np.random.normal(0, 0.5, x_data.size)

    # Número de termos para a série
    N = 2
    T = x_data[-1] - x_data[0]

    # Cálculo de a0
    a0 = np.mean(y_data)

    # Inicializando os coeficientes
    a_n = np.zeros(N)
    b_n = np.zeros(N)

    # Cálculo dos coeficientes a_n e b_n
    for n in range(1, N + 1):
        a_n[n - 1] = (2 / T) * np.sum(y_data * np.cos(n * x_data))
        b_n[n - 1] = (2 / T) * np.sum(y_data * np.sin(n * x_data))

    # Definindo a função ajustada
    def fitted_func(t):
        return a0 + sum(a_n[n - 1] * np.cos(n * t) + b_n[n - 1] * np.sin(n * t) for n in range(1, N + 1))

    # Calculando os valores ajustados
    y_fit = fitted_func(x_data)

    # Plotando os dados e a função ajustada
    plt.scatter(x_data, y_data, label='Dados', color='blue', s=10)
    plt.plot(x_data, y_fit, label='Ajuste', color='red')
    plt.xlabel('Eixo X (Tempo)')
    plt.ylabel('Eixo Y (Amplitude)')
    plt.title('Ajuste de Dados com Senos e Cossenos (sem SciPy)')
    plt.legend()
    plt.grid()
    plt.show()

    # Exibindo os coeficientes ajustados
    print(f'Coeficiente constante (a0): {a0}')
    print(f'Coeficientes a_n: {a_n}')
    print(f'Coeficientes b_n: {b_n}')
'''