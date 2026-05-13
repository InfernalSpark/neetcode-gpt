import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        z1 = np.dot(x, W1.T) + b1
        y1 = np.maximum(0, z1)
        z2 = np.dot(y1, W2.T) + b2

        loss = np.mean(np.power(z2 - y_true, 2))


        op = 2*(z2 - y_true)/len(z2)
        dW2 = np.outer(op, y1)
        db2 = op
        relu_grad = np.dot(op, W2) * (z1 > 0)
        dW1 = np.outer(relu_grad, x)
        db1 = relu_grad

        return {
            'loss': np.round(loss, 4),
            'dW1': np.round(dW1 + 0.0, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist(),
        }

