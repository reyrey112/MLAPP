import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Docstring for Model
    abstract class for all models

    """

    @abstractmethod
    def train(self, x_train, y_train):
        """
        Docstring for train
        abstract method for training the model

        :param x_train: training data features
        :param y_train: training data labels
        """
        pass

class LinearRegressionModel(Model):
    """
    Docstring for LinearRegression
    concrete class for linear regression model

    """

    def train(self, x_train, y_train, **kwargs):
        """
        Docstring for train
        method for training the linear regression model

        :param x_train: training data features
        :param y_train: training data labels
        """
        # implementation of linear regression training algorithm
        logging.info("Training linear regression model with data")
        # ...

        try:
            reg = LinearRegression(**kwargs)
            reg.fit(x_train, y_train)
            logging.info("Linear regression model trained successfully")
            return reg  # return the trained model
        
        except Exception as e:
            logging.error(f"Error training linear regression model: {e}")
            raise e # re-raise the exception after logging it  
        
