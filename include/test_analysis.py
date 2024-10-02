
import matplotlib.pyplot as plt
import numpy as np


class TestAnalysis:

    def __init__(self, test_points, type):
        self.test_X, self.test_Y = self.BreakTupleList(test_points)
        if type == "V":
            copy = self.test_X
            self.test_X = self.test_Y
            self.test_Y = copy

        self.maximums_X, self.maximums_Y, self.minimums_X, self.minimums_Y = self.CriticalPoints(self.test_X, self.test_Y)
        self.curve_mean_X, self.curve_mean_Y = self.CurveMean(self.test_X, self.test_Y)

        self.amplitude_X, self.amplitude_Y = self.Amplitude(self.test_X, self.test_Y)

        self.frequency = len(self.maximums_X)

        if type == "H":
            print('hori')
            self.all_analysis = {
                'test_X': self.test_X,
                'test_Y': self.test_Y,
                'curve_mean_X': self.curve_mean_X,
                'curve_mean_Y': self.curve_mean_Y,
                'maximums_X': self.maximums_X,
                'maximums_Y': self.maximums_Y,
                'minimums_X': self.minimums_X,
                'minimums_Y': self.minimums_Y,
                'amplitude_X': self.amplitude_X,
                'amplitude_Y': self.amplitude_Y,
                'frequency': self.frequency
            }
        elif type == "V":
            print('vertical')
            self.all_analysis = {
                'test_X': self.test_Y,
                'test_Y': self.test_X,
                'curve_mean_X': self.curve_mean_Y,
                'curve_mean_Y': self.curve_mean_X,
                'maximums_X': self.maximums_Y,
                'maximums_Y': self.maximums_X,
                'minimums_X': self.minimums_Y,
                'minimums_Y': self.minimums_X,
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

    def CurveMean(self, test_X, test_Y):
        meanX = []
        meanY = []

        i = 0
        while i < (len(test_X)-1):
            meanY.append((test_Y[i] + test_Y[i+1])/2)
            meanX.append((test_X[i] + test_X[i+1])/2)
            i +=1

        return meanX, meanY

    def Amplitude(self, test_X, test_Y):
        amplitude_X = []
        amplitude_Y = []

        i = 0
        while i < (len(test_X)-2):
            amplitude_Y.append(abs(test_Y[i]-test_Y[i+1]))
            amplitude_X.append((test_X[i] + test_X[i+1])/2)
            i +=1

        return amplitude_X, amplitude_Y



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





'''
def gerarPontos(a, b, c, d):
        X = np.linspace(0, 100, 100)  # 100 pontos entre 0 e 10
        Y = a * X + np.sin(b * X) + d * np.cos(c * X)
        return X, Y

    # função interessante
    # test_X, test_Y = gerarPontos(1, 5, 0.1, 7)

    test_X, test_Y = gerarPontos(0.1, 1, 0.1, 7)
'''
