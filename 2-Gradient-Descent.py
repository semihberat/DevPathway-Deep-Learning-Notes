# Gradyanları hesaplama
def compute_gradients(X, y, weights):
    m = len(y)
    predictions = predict(X, weights)  # Tahminler
    error = predictions - y
    gradients = np.dot(X.T, error) / m  # Gradyanlar
    return gradients

# Gradient Descent algoritması
def gradient_descent(X, y, weights, learning_rate, iterations):
    for i in range(iterations):
        gradients = compute_gradients(X, y, weights)  # Gradyanları hesapla
        weights -= learning_rate * gradients  # Ağırlıkları güncelle
        if i % 100 == 0:  # Her 100 iterasyonda maliyeti yazdır
            cost = compute_cost(X, y, weights)
            print(f"Iteration {i}: Maliyet = {cost}")
    return weights
