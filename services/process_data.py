import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def process_data():
    snapshots = {}

    # Cargar datos
    csv = 'data/study_performance.csv'
    df = pd.read_csv(csv)
    snapshots['original'] = df.head(10).copy()

    # Seleccionar columnas de puntajes
    score_columns = [column for column in df.columns if column.endswith('score')]

    # Calcular puntaje promedio y eliminar columnas individuales
    df['score'] = round(df[score_columns].sum(axis = 1) / 30)
    snapshots['score'] = df.head(10).copy()
    df.drop(columns = score_columns, inplace = True)

    # One Hot Encoding para variables categóricas
    encoder = OneHotEncoder(sparse_output = False)

    # Genero
    encoded_data = encoder.fit_transform(df[['gender']])
    encoded_df = pd.DataFrame(encoded_data, columns = encoder.get_feature_names_out(['gender']))
    df = pd.concat([df, encoded_df], axis = 1)
    snapshots['gender'] = df.head(10).copy()
    df.drop(columns = ['gender'], inplace = True)

    # Curso preparación para las pruebas
    encoded_data = encoder.fit_transform(df[['test_preparation_course']])
    encoded_df = pd.DataFrame(encoded_data, columns = encoder.get_feature_names_out(['test_preparation_course']))
    df = pd.concat([df, encoded_df], axis = 1)
    snapshots['test_preparation_course'] = df.head(10).copy()
    df.drop(columns = ['test_preparation_course'], inplace = True)

    # Alimentación
    encoded_data = encoder.fit_transform(df[['lunch']])
    encoded_df = pd.DataFrame(encoded_data, columns = encoder.get_feature_names_out(['lunch']))
    df = pd.concat([df, encoded_df], axis = 1)
    snapshots['lunch'] = df.head(10).copy()
    df.drop(columns = ['lunch'], inplace = True)

    # Raza/etnia
    encoded_data = encoder.fit_transform(df[['race_ethnicity']])
    encoded_df = pd.DataFrame(encoded_data, columns = encoder.get_feature_names_out(['race_ethnicity']))
    df = pd.concat([df, encoded_df], axis = 1)
    snapshots['race_ethnicity'] = df.head(10).copy()
    df.drop(columns = ['race_ethnicity'], inplace = True)

    # Ordinal encoding para variables ordinales
    ordinal_categories = ['some high school', 'high school', 'some college', 'associate\'s degree', 'bachelor\'s degree', 'master\'s degree']
    encoder = OrdinalEncoder(categories = [ordinal_categories])
    encoded_data = encoder.fit_transform(df[['parental_level_of_education']])
    encoded_df = pd.DataFrame(encoded_data, columns = ['parental_level_of_education_encoded'])
    df = pd.concat([df, encoded_df], axis = 1)
    snapshots['parental_level_of_education'] = df.head(10).copy()
    df.drop(columns = ['parental_level_of_education'], inplace = True)

    return df, snapshots


def train_models(df):
    x = df.drop(columns = ['score'], axis = 1)
    y = df['score']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 21)

    random_forest_model = RandomForestRegressor(n_estimators = 100, random_state = 42)
    random_forest_model.fit(x_train, y_train)

    linear_regression_model = LinearRegression()
    linear_regression_model.fit(x_train, y_train)

    random_forest_predictions = random_forest_model.predict(x_test).round()
    linear_regression_predictions = linear_regression_model.predict(x_test).round()

    random_forest_mse = mean_squared_error(y_test, random_forest_predictions)
    linear_regression_mse = mean_squared_error(y_test, linear_regression_predictions)

    return random_forest_mse, linear_regression_mse
