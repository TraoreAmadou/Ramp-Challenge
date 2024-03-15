from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


class Regressor(BaseEstimator):
    def __init__(self):
        # Specify the categorical columns
        categorical_cols = ['Type', 'Heating_Type',
                            'Insulation_Level', 'Windows_Type', 'Roof_Type',
                            'Wall_Material', 'Elevator_Presence',
                            'Basement_Presence', 'Energy_Rating',
                            'LED_Lighting', 'Energy_Management_System']
        
        num_columns = ['Surface', 'Age', 'Outside_Temperature',
                       'Number_of_Rooms',
                       'Occupancy_Rate', 'Parking_Spaces', 'Residents_Count',
                       'Distance_From_City_Center']

        # Pipeline pour les variables numériques
        self.numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),  # Imputation des valeurs manquantes par la médiane
                ("scaler", StandardScaler())  # Standardisation des variables numériques
            ]
        )

        # Pipeline pour les variables catégoriques
        self.categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),  # Imputation des valeurs manquantes par une constante
                ("onehot", OneHotEncoder(handle_unknown="ignore"))  # Encodage OneHot pour les variables catégoriques
            ]
        )

        # Combine les transformateurs
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", self.numeric_transformer, num_columns),  # Applique numeric_transformer aux variables numériques
                ("cat", self.categorical_transformer, categorical_cols)  # Applique categorical_transformer aux variables catégoriques
            ]
        )
                
        self.model = LinearRegression()
        self.pipe = make_pipeline(self.preprocessor, self.model)

    def fit(self, X, y):
        self.pipe.fit(X, y)

    def predict(self, X):
        return self.pipe.predict(X)

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)

