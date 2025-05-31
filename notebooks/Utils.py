from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def EntrenarModelo(X_cols, y_col, df):
    X=df[X_cols].values #Data Frame sin la columna a predecir
    y=df[y_col].values #Data Frame con la columna a predecir

    X_train, X_test, y_train, y_test = train_test_split(X, y) #Dividimos los DF en conjuntos de entrenamiento y de test

    #Crear el objeto que nos ayudara a normalizar los datos
    sc_x = StandardScaler().fit(X) 
    sc_y = StandardScaler().fit(y)

    #Transformar/Normalizar nuestros datos
    X_train = sc_x.transform(X_train)
    X_test = sc_x.transform(X_test)
    y_train = sc_y.transform(y_train)
    y_test = sc_y.transform(y_test)

    ##Creacion del modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test