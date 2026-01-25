#Place in util folder

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC, LinearSVR, SVR, SVC
from sklearn.linear_model import LassoCV, ElasticNetCV, Ridge, SGDRegressor, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn import metrics
import random, logging

# df = pd.read_csv(r"C:\Users\reyde\Desktop\Formulations.csv")
# x = df.drop(["Viscosity"], axis=1)
# y = df["Viscosity"]

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# newmodel = LassoCV()
# newmodel.fit(x_train, y_train)
# newmodel.get_params()


model_dict = {
    "Linear SVC": LinearSVC,
    "Lasso": LassoCV,
    "ElasticNet": ElasticNetCV,
    "Ridge Regression": Ridge,
    "Linear SVR": LinearSVR,
    "rbf SVR": SVR,
    "SGD Regressor": SGDRegressor,
    "Naive Bayes": GaussianNB,
    "KNeighbors Classifier": KNeighborsClassifier,
    "SVC": SVC,
    "SGD Classifier": SGDClassifier
}

LinearSVC_params = {
    "penalty": ["l1", "l2"],
    "loss": ["hinge", "squared_hinge"],
    "class_weight": ["balance", None],
}

RidgeRegression_params = {}

LinearSVR_params = {}

rbfSVR_params = {}

SGDRegressor_params= {}

GaussianNB_params = {}

KNeighborsClassifier_params = {}

SVC_params = {}

SGDClassifier_params = {}

cv_params = {
    "Linear SVC": LinearSVC_params,
    "Lasso": False,
    "ElasticNet": False,
    "Ridge Regression": RidgeRegression_params,
    "Linear SVR": LinearSVR_params,
    "rbf SVR": rbfSVR_params,
    "SGD Regressor": SGDRegressor_params,
    "Naive Bayes": GaussianNB_params,
    "KNeighbors Classifier": KNeighborsClassifier_params,
    "SVC": SVC_params,
    "SGD Classifier": SGDClassifier_params,
}


class model_predicting:
    def __init__(self, y_class=None) -> None:
        self.cv = "random"
        self.supported_models = {
            "SGD Regressor",
            "Lasso",
            "ElasticNet",
            "Ridge Regression",
            "Linear SVR",
            "rbf SVR",
            "SGD Regressor",
            "Naive Bayes",
            "KNeighbors Classifier",
            "SVC",
            "Linear SVC",
            "SGD Classifier"
        }
        self.scalers = {
            "MinMax": MinMaxScaler,
            "Standard": StandardScaler,
            "Robust": RobustScaler,
        }
        self.y_class = y_class

    def descrete_check(self, y_column: pd.Series):
        unique = y_column.nunique()
        total_values = len(y_column)
        value_counts = y_column.value_counts()

        greater_than_1_count = 0

        if unique >= total_values / 2:
            return True

        for index in value_counts.index:
            if value_counts[index] > 1:
                greater_than_1_count = greater_than_1_count + 1

        if greater_than_1_count < unique / 2:
            return True
        else:
            return False

    def y_classifier(self, y_column: pd.Series):

        if self.descrete_check(y_column):
            y_class = "regression"
            return y_class
        else:
            y_class = "classification"
            return y_class

    def model_predict(self, y_column: pd.Series, column_num: int, y_class: str):
        model_recommendations = {}
        if y_class == "classification":
            if len(y_column) < 100000:
                model_recommendations["simple"] = "Linear SVC"

                # if y_column.dtype == str:
                if isinstance(y_column[0], str) :
                    model_recommendations["recommended1"] = "Naive Bayes"

                else:
                    model_recommendations["recommended1"] = "KNeighbors Classifier"
                    model_recommendations["recommended2"] = "SVC"
            else:
                model_recommendations["recommended1"] = "SGD Classifier"

        elif y_class == "regression":
            if len(y_column) < 100000:
                if column_num > 10:
                    model_recommendations["recommended1"] = "Lasso"
                    model_recommendations["recommended2"] = "ElasticNet"

                else:
                    model_recommendations["recommended1"] = "Ridge Regression"
                    model_recommendations["recommended2"] = "Linear SVR"
                    model_recommendations["recommended3"] = "rbf SVR"

            else:
                model_recommendations["recommended1"] = "SGD Regressor"

        else:
            model_recommendations["none"] = "No Recommended Models"

        return model_recommendations

    def scaling_check(self, y: pd.Series):
        q3 = y.quantile(0.75)
        q1 = y.quantile(0.25)
        iqr = q3 - q1
        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)
        outliers = (y < lower) | (y > upper)

        scaler_recommendations = {}

        if len(outliers) > 1:
            scaler_recommendations["recommended1"] = "Robust"
            scaler_recommendations["recommended2"] = "Remove Outliers then Standard"

        else:
            scaler_recommendations["recommended1"] = "Standard"
            scaler_recommendations["recommended2"] = "MinMax"

        return scaler_recommendations

    def outlier_removal(self, y: pd.Series):
        q3 = y.quantile(0.75)
        q1 = y.quantile(0.25)
        iqr = q3 - q1
        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)

        y_no_outliers = y[((y >= lower) & (y <= upper))]


        return y_no_outliers

    def model_train(
        self, model_name, y_variable: str, df: pd.DataFrame, y_class, scaler_name, remove_outliers
    ):
        x = df.drop([y_variable], axis=1)
        y = df[y_variable]
        
        if remove_outliers == "yes":
            y = self.outlier_removal(y)
            outliers = True
        
        else:
            outliers = False
        
        state = random.randint(1, 1000000)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=state
        )

        if scaler_name in self.scalers:
            scaler = self.scalers[f"{scaler_name}"]()
            x_train, x_test = self.data_scaling(x_train, x_test, scaler)

        if y_class == "classification":
            trained_model, accuracy, y_pred = self.train_classifier(
                model_name, x_train, x_test, y_train, y_test
            )
            return trained_model, state, accuracy, y_pred, outliers

        elif y_class == "regression":
            trained_model, accuracy, y_pred = self.train_regressor(
                model_name, x_train, x_test, y_train, y_test
            )
            return trained_model, state, accuracy, y_pred, outliers

        else:
            return state, state, state, state, state

    def data_scaling(self, x_train, x_test, scaler):

        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        return x_train_scaled, x_test_scaled

    def train_classifier(self, model_name, x_train, x_test, y_train, y_test):
        logging.info(f"Training {str(model_name)} model with data")

        model = model_dict[model_name]()
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        logging.info(f"{str(model_name)}model trained successfully")

        return model, accuracy, y_pred

    def train_regressor(self, model_name, x_train, x_test, y_train, y_test):
        logging.info(f"Training {str(model_name)} model with data")

        model = model_dict[model_name]()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        residuals = np.subtract(y_test, y_pred)
        accuracy = model.score(x_test, y_test)

        logging.info(f"{str(model_name)}model trained successfully")
        return model, accuracy, y_pred
    
    def confusion_matrix(self, y, y_pred, state):
        _, y_test = train_test_split(y, test_size=0.25, random_state=state)
        cm = metrics.confusion_matrix
        cm_report = metrics.classification_report(y_test, y_pred)

        return cm, cm_report

    # def generate_param_lists(model_class=Lasso):

    #     """
    #     Generate parameter lists for a scikit-learn model class.

    #     For bool parameters: [True, False]
    #     For float/int parameters: 5 evenly spaced values with default in the middle
    #     """
    #     # Get the signature of the class __init__ method
    #     sig = inspect.signature(model_class.__init__)

    #     param_lists = {}

    #     for param_name, param in sig.parameters.items():
    #         if param_name == 'self':
    #             continue

    #         default_value = param.default
    #         annotation = param.annotation

    #         # Skip if no default value
    #         if default_value == inspect.Parameter.empty:
    #             continue

    #         # Handle boolean parameters
    #         if isinstance(default_value, bool):
    #             param_lists[param_name] = [True, False]

    #         # Handle numeric parameters (int or float)
    #         elif isinstance(default_value, (int, float)) and default_value is not None:
    #             # Create 5 evenly spaced values with default in the middle
    #             if default_value == 0:
    #                 # Special case: if default is 0, create a range around it
    #                 if isinstance(default_value, int):
    #                     param_lists[param_name] = [-2, -1, 0, 1, 2]
    #                 else:
    #                     param_lists[param_name] = [-0.2, -0.1, 0.0, 0.1, 0.2]
    #             elif default_value > 0:
    #                 # Create range from half to double the default value
    #                 lower = default_value * 0.5
    #                 upper = default_value * 2.0
    #                 values = np.linspace(lower, upper, 5)

    #                 if isinstance(default_value, int):
    #                     param_lists[param_name] = [int(v) for v in values]
    #                 else:
    #                     param_lists[param_name] = values.tolist()
    #             else:
    #                 # For negative default values
    #                 lower = default_value * 2.0
    #                 upper = default_value * 0.5
    #                 values = np.linspace(lower, upper, 5)

    #                 if isinstance(default_value, int):
    #                     param_lists[param_name] = [int(v) for v in values]
    #                 else:
    #                     param_lists[param_name] = values.tolist()

    #         # # Handle string parameters with specific options
    #         # elif isinstance(default_value, str):
    #         #     # For Lasso, we know some specific parameter options
    #         #     if param_name == 'selection':
    #         #         param_lists[param_name] = ['cyclic', 'random']
    #         #     elif param_name == 'solver':
    #         #         param_lists[param_name] = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
    #         #     else:
    #         #         param_lists[param_name] = [default_value]

    #         # Handle None default (could be various types)
    #         elif default_value is None:
    #             param_lists[param_name] = [None]

    #     return param_lists
