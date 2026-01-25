import logging
from abc import ABC, abstractmethod
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Docstring for Evaluation
    Abstract class for evaluating our models
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the evaluation score based on the true and predicted labels.
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        """
        pass

class MSE(Evaluation):
    """
    Docstring for MSE
    Evaluation Strategy for calculating the Mean Squared Error (MSE) score.
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error (MSE) score.
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        Returns:
            float: The MSE score.
        """

        try:
            logging.info("Calculating MSE score...")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE Score: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error calculating MSE score: {e}")
            raise e

class R2(Evaluation):
    """
    Docstring for R2
    Evaluation Strategy for calculating the R-squared (R2) score.
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the R-squared (R2) score.
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        Returns:
            float: The R2 score.
        """
        try:
            logging.info("Calculating R2 score...")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error calculating R2 score: {e}")
            raise e
        
class RMSE(Evaluation):
    """
    Docstring for RMSE
    Evaluation Strategy for calculating the Root Mean Squared Error (RMSE) score.
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the Root Mean Squared Error (RMSE) score.
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        Returns:
            float: The RMSE score.
        """
        try:
            logging.info("Calculating RMSE score...")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"RMSE Score: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error calculating RMSE score: {e}")
            raise e