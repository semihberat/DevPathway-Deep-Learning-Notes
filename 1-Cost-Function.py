import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Ağırlıkların ve girdilerin basitçe çarpımı
def predict(X, weights):
    z = np.dot(X, weights)
    return sigmoid(z)

# Maliyet Fonksiyonun implementasyonu 
def compute_cost(X, y, weights):
    m = len(y)
    predictions = predict(X, weights) ## ℎ𝜃(𝑥) yani tahmini deger hesaplanıyor.
    cost = -1/m * (np.dot(y, np.log(predictions)) + np.dot((1 - y), np.log(1 - predictions))) ## ve blog üzerinde de anlattığımız gibi maliyet hesaplanıyor.
    return cost

# Değerler array olarak belirtilerek test edilebilir.
# Örnek veri seti
X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])  # Bias terimi için sabit bir 1'lik sütun eklendi
y = np.array([0, 0, 1, 1])  # Gerçek etiketler

# Başlangıçta ağırlıklar sıfır olarak ayarlanır
weights = np.zeros(X.shape[1])

# Maliyet fonksiyonunu hesapla
cost = compute_cost(X, y, weights)
print("Maliyet:", cost)
