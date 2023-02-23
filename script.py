
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
f = pd.read_csv('tennis_stats.csv')

df = pd.DataFrame(f)

print(df.head(5))


# perform exploratory analysis here:

plt.scatter(df['FirstServeReturnPointsWon'], df['Wins'])
plt.title('First Serve Return Points Won vs Wins')
plt.xlabel('First Serve Return Points Won')
plt.ylabel('Wins')
plt.show()
plt.clf()

plt.scatter(df['BreakPointsOpportunities'], df['Wins'])
plt.title('Break Points Opportunities vs Wins')
plt.xlabel('Break Points Opportunities')
plt.ylabel('Wins')
plt.show()
plt.clf()

# perform single feature linear regressions here:

X = df[['FirstServePointsWon']]
y = df['Wins']

X_train, X_test, y_train, y_test = train_test_split(X, y)

lm = LinearRegression()
model = lm.fit(X_train, y_train)

print("Train score:")
print(lm.score(X_train, y_train))

print("Test score:")
print(lm.score(X_test, y_test))

y_predict = lm.predict(X_test)
plt.scatter(X_test, y_test)
plt.plot(X_test, y_predict, color='red',)
plt.xlabel('First Serve')
plt.ylabel('Wins')
plt.show()
plt.clf()


# perform two feature linear regressions here:

features = ['FirstServe', 'ServiceGamesPlayed']
X = df[features]
y = df['Wins']

X_train, X_test, y_train, y_test = train_test_split(X, y)

lm = LinearRegression()
model = lm.fit(X_train, y_train)

print("Train score:")
print(lm.score(X_train, y_train))

print("Test score:")
print(lm.score(X_test, y_test))

y_predict = lm.predict(X_test)

plt.scatter(y_test, y_predict)
plt.plot(X_test, y_predict, color='red',)


# perform multiple feature linear regressions here:

x = df[['Aces', 'DoubleFaults', 'FirstServe', 'FirstServePointsWon', 'SecondServePointsWon',
        'BreakPointsFaced', 'BreakPointsSaved', 'ServiceGamesPlayed', 'ServiceGamesWon', 'TotalServicePointsWon']]

y = df[['Wins']]

x_train, x_test, y_train, y_test = train_test_split(x, y)

lm = LinearRegression()

model = lm.fit(x_train, y_train)

y_predict = lm.predict(x_test)

print("Train score:")
print(lm.score(x_train, y_train))

print("Test score:")
print(lm.score(x_test, y_test))

plt.scatter(y_test, y_predict)
plt.title('Predicted Wins vs. Actual Wins')
plt.xlabel('Actual Wins')
plt.ylabel('Predicted Wins')
plt.plot(X_test, y_predict, alpha=0.4)
