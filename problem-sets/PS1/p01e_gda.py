import numpy as np
import util
import matplotlib.pyplot as plt
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    # *** START CODE HERE ***
    gda = GDA()
    gda.fit(x_train, y_train)
    print(gda.theta)
    y_pred = gda.predict(x_eval)
    m, _ = x_eval.shape
    accuracy = np.sum(y_pred == y_eval) / m
    print(f'Accuracy: {accuracy}')
    util.plot(x_eval, y_eval, theta = gda.theta, save_path = "problem-sets/PS1/data/gda_test.png")
    # *** END CODE HERE ***




class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        phi = np.sum(y)/m
        mu_0 = np.dot(x.T, 1-y)/np.sum(1-y)
        mu_1 = np.dot(x.T, y)/np.sum(y)
        y_reshaped = np.reshape(y, (m,-1))
        mu_x = y_reshaped * mu_1 + (1-y_reshaped) * mu_0
        x_centered = x - mu_x
        sigma = 1/m * np.dot(x_centered.T, x_centered)
        sigma_inv = np.linalg.inv(sigma)
        theta = sigma_inv @ (mu_1-mu_0)
        theta_0 = 1 / 2 * mu_0 @ sigma_inv @ mu_0 - 1 / 2 * mu_1 @ sigma_inv @ mu_1 - np.log((1 - phi) / phi)
    
        self.theta = np.insert(theta, 0 , theta_0)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return (util.add_intercept(x) @ self.theta) > 0
        # *** END CODE HERE





if __name__ == '__main__':
    main(train_path='problem-sets/PS1/data/ds2_train.csv',
         eval_path='problem-sets/PS1/data/ds2_valid.csv',
         pred_path='data/pred_logreg.csv')