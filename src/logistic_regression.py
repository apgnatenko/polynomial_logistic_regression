import numpy as np
from utils import sigmoid
import math


class LogisticRegression():
    
    def __init__(self, is_reg):
        """
        Args:
          is_reg : (bool) Flag to compute regularized logistic regression or not
        """
        if is_reg:
            self._cost_function = self._compute_cost_reg
            self._gradient_function = self._compute_gradient_reg
        else:
            self._cost_function = self._compute_cost
            self._gradient_function = self._compute_gradient
        
    def _compute_cost(self, X, y, w, b, *argv):
        """
        Computes the cost over all examples
        Args:
          X : (ndarray Shape (m,n)) data, m examples by n features
          y : (ndarray Shape (m,))  target value 
          w : (ndarray Shape (n,))  values of parameters of the model      
          b : (scalar)              value of bias parameter of the model
          *argv : unused, for compatibility with regularized version below
        Returns:
          total_cost : (scalar) cost 
        """
        m, n = X.shape
        
        loss_sum = 0
        # Loop over each training example
        for i in range(m): 
            # First calculate z_wb = w[0]*X[i][0]+...+w[n-1]*X[i][n-1]+b
            z_wb = 0 
            for j in range(n): 
              # Add the corresponding term to z_wb
                z_wb_ij = w[j] * X[i][j]
                z_wb += z_wb_ij
            # Add the bias term to z_wb
            z_wb += b
            # Calculate the prediction for this example
            f_wb = sigmoid(z_wb)
            # Calculate loss
            loss =  -y[i]*np.log(f_wb)-(1-y[i])*np.log(1-f_wb)

            loss_sum += loss

        total_cost = (1 / m) * loss_sum       
        return total_cost

    def _compute_gradient(self, X, y, w, b, *argv): 
        """
        Computes the gradient for logistic regression 
    
        Args:
          X : (ndarray Shape (m,n)) data, m examples by n features
          y : (ndarray Shape (m,))  target value 
          w : (ndarray Shape (n,))  values of parameters of the model      
          b : (scalar)              value of bias parameter of the model
          *argv : unused, for compatibility with regularized version below
        Returns
          dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
          dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
        """
        m, n = X.shape
        dj_dw = np.zeros(w.shape)
        dj_db = 0.

        for i in range(m):
            # First calculate z_wb = w[0]*X[i][0]+...+w[n-1]*X[i][n-1]+b
            z_wb = 0 
            # Loop over each feature
            for j in range(n): 
              # Add the corresponding term to z_wb
                z_wb_ij = w[j] * X[i][j]
                z_wb += z_wb_ij
            # Add the bias term to z_wb
            z_wb += b
            # Calculate the prediction for this example
            f_wb = sigmoid(z_wb)

            # Calculate the error
            dj_db_i = f_wb - y[i]

            # add that to dj_db
            dj_db += dj_db_i

            # get dj_dw for each attribute
            for j in range(n):
                # Calculate the gradient from the i-th example for j-th attribute
                dj_dw_ij =  (f_wb - y[i])* X[i][j]
                dj_dw[j] += dj_dw_ij

        # divide dj_db and dj_dw by total number of examples
        dj_dw = dj_dw / m
        dj_db = dj_db / m
        return dj_db, dj_dw

    def _compute_cost_reg(self, X, y, w, b, lambda_ = 1):
        """
        Computes the cost over all examples
        Args:
          X : (ndarray Shape (m,n)) data, m examples by n features
          y : (ndarray Shape (m,))  target value 
          w : (ndarray Shape (n,))  values of parameters of the model      
          b : (scalar)              value of bias parameter of the model
          lambda_ : (scalar, float) Controls amount of regularization
        Returns:
          total_cost : (scalar)     cost 
        """

        m, n = X.shape
        
        # Calls the compute_cost function
        cost_without_reg = self._compute_cost(X, y, w, b) 
        reg_cost = 0.

        for j in range(n):
            reg_cost_j = w[j]**2
            reg_cost += reg_cost_j
            
        reg_cost = ((lambda_)/(2*m))*reg_cost

        # Add the regularization cost to get the total cost
        total_cost = cost_without_reg + reg_cost
        return total_cost

    def _compute_gradient_reg(self, X, y, w, b, lambda_ = 1): 
        """
        Computes the gradient for logistic regression with regularization
    
        Args:
          X : (ndarray Shape (m,n)) data, m examples by n features
          y : (ndarray Shape (m,))  target value 
          w : (ndarray Shape (n,))  values of parameters of the model      
          b : (scalar)              value of bias parameter of the model
          lambda_ : (scalar,float)  regularization constant
        Returns
          dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
          dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

        """
        m, n = X.shape
        
        dj_db, dj_dw = self._compute_gradient(X, y, w, b)

        for j in range(n):
            dj_dw_reg = (lambda_/m)*w[j]
            dj_dw[j] = dj_dw[j] + dj_dw_reg
        return dj_db, dj_dw
    
    def gradient_descent(self, X, y, w_in, b_in, alpha, num_iters, lambda_): 
        """
        Performs batch gradient descent to learn theta. Updates theta by taking 
        num_iters gradient steps with learning rate alpha
        
        Args:
          X :    (ndarray Shape (m, n) data, m examples by n features
          y :    (ndarray Shape (m,))  target value 
          w_in : (ndarray Shape (n,))  Initial values of parameters of the model
          b_in : (scalar)              Initial value of parameter of the model
          alpha : (float)              Learning rate
          num_iters : (int)            Number of iterations to run gradient descent
          lambda_ : (scalar, float)    Regularization constant
          
        Returns:
          w : (ndarray Shape (n,)) Updated values of parameters of the model after
              running gradient descent
          b : (scalar)                Updated value of parameter of the model after
              running gradient descent
        """

        # An array to store cost J and w's at each iteration primarily for graphing later
        J_history = []
        w_history = []
        
        for i in range(num_iters):

            # Calculate the gradient and update the parameters
            dj_db, dj_dw = self._gradient_function(X, y, w_in, b_in, lambda_)   

            # Update Parameters using w, b, alpha and gradient
            w_in = w_in - alpha * dj_dw               
            b_in = b_in - alpha * dj_db              
          
            # Save cost J at each iteration
            if i<100000:      # prevent resource exhaustion 
                cost =  self._cost_function(X, y, w_in, b_in, lambda_)
                J_history.append(cost)

            # Print cost every at intervals 10 times or as many iterations if < 10
            if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
                w_history.append(w_in)
                print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
            
        return w_in, b_in, J_history, w_history # return w and J, w history for graphing

    def predict(self, X, w, b): 
        """
        Predict whether the label is 0 or 1 using learned logistic
        regression parameters w
        
        Args:
          X : (ndarray Shape (m,n)) data, m examples by n features
          w : (ndarray Shape (n,))  values of parameters of the model      
          b : (scalar)              value of bias parameter of the model

        Returns:
          p : (ndarray (m,)) The predictions for X using a threshold at 0.5
        """
        m, n = X.shape   
        p = np.zeros(m)

        # Loop over each example
        for i in range(m):   
            z_wb = 0
            # Loop over each feature
            for j in range(n): 
                # Add the corresponding term to z_wb
                z_wb_ij = w[j]*X[i][j]
                z_wb += z_wb_ij
            
            # Add bias term 
            z_wb += b
            
            # Calculate the prediction for this example
            f_wb = sigmoid(z_wb)

            # Apply the threshold
            p[i] = 1 if f_wb>=0.5 else 0
        return p
