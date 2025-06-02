from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

class Modelo:
    def __init__(self, df, X_cols, y_col):
        self.dataFrame = df
        self.X_cols = X_cols
        self.y_col = y_col

        #Calculated DF's for further use
        X=self.dataFrame[X_cols].values #Independent variables dataframe
        y=self.dataFrame[y_col].values #Output dataframe

        #Calculated and separated DF's for further trainig and testing
        X_train, X_test, y_train, y_test = train_test_split(X,y)

        #Standard Scaler / Normalization    
        self.sc_x = StandardScaler().fit(X) 
        self.sc_y = StandardScaler().fit(y)

        #Standardize dataframes
        #Transformar/Normalizar nuestros datos
        self.X_train = self.sc_x.transform(X_train)
        self.X_test = self.sc_x.transform(X_test)
        self.y_train = self.sc_y.transform(y_train)
        self.y_test = self.sc_y.transform(y_test)

        self.TrainModel()
        self.PredictTest()

    def TrainModel(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
    
    def GetModel(self):
        return self.model if self.model is not None else "Execute TrainModel() First"
    
    def Predict(self, X_predict):
        return self.model.predict(X_predict)
    
    def PredictTest(self):
        self.y_pred = self.Predict(self.X_test)
    
    def GetMetrics(self):
        mse = metrics.mean_squared_error(self.y_test,self.y_pred)
        r2 = metrics.r2_score(self.y_test,self.y_pred)
        print("R2: ", r2)
        print("MSE: ", mse)

    def GetSummary(self, intercept=True):
        #Create a copy DF to pass to statsmodel
        X_train_sm = pd.DataFrame(self.X_train, columns=self.X_cols)

        # Add and intercept constant
        if intercept:
            X_train_sm = sm.add_constant(X_train_sm)

        # Adjust model
        ols_model = sm.OLS(self.y_train, X_train_sm).fit()

        # Ver el R2
        print("R2 statsmodels:", ols_model.rsquared)

        # Shows the summary
        print(ols_model.summary())
        pass

    def GetResidualsPlot(self):
        residuals = np.subtract(self.y_test, self.y_pred)
        plt.scatter(self.y_pred, residuals)
        plt.show()