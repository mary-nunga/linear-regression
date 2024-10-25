import pandas as pd
import matplotlib.pyplot as plt


# Function to calculate Mean Squared Error
def mean_squared_error(actual, predicted):
    n = len(actual)
    mse = sum((actual[i] - predicted[i]) ** 2 for i in range(n)) / n
    return mse


# Function to perform Gradient Descent
def gradient_descent(x, y, m, c, learning_rate):
    n = len(x)
    y_predicted = [(m * x[i] + c) for i in range(n)]

    m_gradient = -(2 / n) * sum(x[i] * (y[i] - y_predicted[i]) for i in range(n))
    c_gradient = -(2 / n) * sum((y[i] - y_predicted[i]) for i in range(n))

    m_new = m - learning_rate * m_gradient
    c_new = c - learning_rate * c_gradient

    return m_new, c_new


# Function to train the model
def train_model(x, y, learning_rate, epochs):
    m = 0  # Initial slope
    c = 0  # Initial intercept

    for epoch in range(epochs):
        m, c = gradient_descent(x, y, m, c, learning_rate)
        y_predicted = [(m * x[i] + c) for i in range(len(x))]
        error = mean_squared_error(y, y_predicted)
        print(f'Epoch {epoch + 1}: Mean Squared Error = {error}')

    return m, c


# Function to predict office price
def predict_office_price(size, m, c):
    return m * size + c


# Load the Excel file
file_path = r'C:\Users\user\Desktop\Nairobi_Office_Price.csv'  # Update this path
data = pd.read_csv(file_path)

# Extract the columns from the dataset
x = data['SIZE'].tolist()
y = data['PRICE'].tolist()

# Training parameters
learning_rate = 0.0001
epochs = 10

# Training the model
m, c = train_model(x, y, learning_rate, epochs)

# Plotting the line of best fit
plt.scatter(x, y, color='blue')
y_predicted = [m * i + c for i in x]
plt.plot(x, y_predicted, color='red')
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Office Price')
plt.title('Office Size vs Price')
plt.show()

# Predict the price
predicted_price = predict_office_price(100, m, c)
print(f'Predicted price for a 100 sq. ft. office: {predicted_price}')
