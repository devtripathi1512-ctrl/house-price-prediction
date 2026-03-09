import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample data
data = {'Size':[1000,1500,2000,2500], 'Price':[5000,7000,9000,11000]}
df = pd.DataFrame(data)

X = df[['Size']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

print("Predicted price for 3000 sq ft:", model.predict([[3000]]))
print("Model accuracy:", model.score(X_test, y_test))