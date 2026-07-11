import pandas as pd
from sklearn.linear_model import LinearRegression
data={
    'hours':[1,2,3,4,5,6],
    'marks':[35,45,55,65,75,85]
}
df =pd.DataFrame(data)
X=df[['hours']]
y=df['marks']
model=LinearRegression()
model.fit(X,y)
pred=model.predict([[7]])
print("prediction: ",pred[0])