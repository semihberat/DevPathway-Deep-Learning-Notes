import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
""" Girilen her değer 0 ile 1 arasında bir sayı döndürecektir. Bu da sayılar arasında bir korelasyon oluşmasına 
ve istatistiksel bir çıkarım yapmaya elverişli hale getirir. """
