import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

customers = pd.read_csv('StudentsPerformance.csv')

display(customers.head())
customers.head()
customers.info()
display(customers.describe())

sns.jointplot('reading score', 'writing score', data=customers)
sns.pairplot(customers)
sns.lmplot('reading score', 'writing score', data=customers)

X = customers[['writing score', 'reading score', 'math score']] 
y = customers[['math score']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm = LinearRegression()

lm.fit(X_train, y_train)

print(lm.coef_)

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

print(mae, mse, rmse)

coeffs = pd.DataFrame(data=lm.coef_.transpose(), index=X.columns, columns=['Coefficient'])
coeffs.plot()
display(coeffs)
plt.show()