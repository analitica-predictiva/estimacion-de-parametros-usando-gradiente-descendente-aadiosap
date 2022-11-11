"""
Optimización usando gradiente descendente - Regresión polinomial
-----------------------------------------------------------------------------------------

En este laboratio se estimarán los parámetros óptimos de un modelo de regresión 
polinomial de grado `n`.

"""


def pregunta_01():
    """
    Complete el código presentado a continuación.
    """
    # Importe pandas
    import pandas as pd

    # Importe PolynomialFeatures
    from sklearn.preprocessing import PolynomialFeatures

    # Cargue el dataset `data.csv`
    data = pd.read_csv("data.csv")

    # Cree un objeto de tipo `PolynomialFeatures` con grado `2`
    poly = PolynomialFeatures(degree=2)

    # Transforme la columna `x` del dataset `data` usando el objeto `poly`
    x_poly = poly.fit_transform(data[["x"]])

    # Retorne x y y
    return x_poly, data.y


def pregunta_02():

    # Importe numpy
    import numpy as np

    x_poly, y = pregunta_01()

    # Fije la tasa de aprendizaje en 0.0001 y el número de iteraciones en 1000
    learning_rate = 0.001
    n_iterations = 1000

    # Defina el parámetro inicial `params` como un arreglo de tamaño 3 con ceros
    params = np.zeros(3)
    for i in range(n_iterations):

        # Compute el pronóstico con los parámetros actuales
        y_pred = np.sum(np.multiply(params, x_poly),axis=1)

        # Calcule el error
        error = [y - y_pred for y, y_pred in zip(y, y_pred)]

        # Calcule el gradiente
        x_1 = [item[1] for item in x_poly.tolist()]
        x_2 = [item[2] for item in x_poly.tolist()]
        gradient_w0 = -2 * sum(error)
        gradient_w1 = -2 * sum([error * x_1 for error, x_1 in zip(error, x_1)])
        gradient_w2 = -4 * sum([error * x_2 for error, x_2 in zip(error, x_2)])
        gradient = np.array([gradient_w0,gradient_w1,gradient_w2])

        # Actualice los parámetros
        params = params - learning_rate * gradient

    return params
