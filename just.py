import pandas as pd
import numpy as np
import warnings

# Printing first few records
sales_df = pd.read_csv(‘Advertising.csv’)
sales_df.head()
X = sales_df[[‘TV’, ‘Radio’, ‘Newspaper’]]
Y = sales_df[‘Sales’]
             
Y = np.array( (Y - Y.mean() ) / Y.std() )
X = X.apply( lambda rec: ( rec - rec.mean() ) / rec.std(),axis = 0 )

def initialize( dim ):

        np.random.seed(seed=42)
        random.seed(42)
        #Initialize the bias.
        b = random.random()
        #Initialize the weights.
        w = np.random.rand( dim )
        return b, w

def predict_Y( b, w, X ):
    return b + np.matmul( X, w )

b, w = initialize( 3 )
Y_hat = predict_Y( b, w, X)
Y_hat[0:10]

def get_cost( Y, Y_hat ):
    Y_resid = Y - Y_hat
    return np.sum( np.matmul( Y_resid.T, Y_resid ) ) / len( Y_resid )

get_cost( Y, Y_hat )


def update_beta( x, y, y_hat, b_0, w_0, learning_rate ):

        db = (np.sum( y_hat - y ) * 2) / len(y)

        dw = (np.dot( ( y_hat - y ), x ) * 2 ) / len(y)
        b_1 = b_0 - learning_rate * db
        #update beta
        w_1 = w_0 - learning_rate * dw
        return b_1, w_1


b, w = update_beta( X, Y, Y_hat, b, w, 0.01 )
print( “After first update - Bias: ”, b, “ Weights: ”, w )

def run_gradient_descent( X,Y,alpha = 0.01,num_iterations = 100):

    b, w = initialize( X.shape[1] )
    iter_num = 0
    # gd_iterations_df keeps track of the cost every 10 iterations
    gd_iterations_df = pd.DataFrame(columns = [‘iteration’, ‘cost’])
    result_idx = 0


    for each_iter in range(num_iterations):
    Y_hat = predict_Y( b, w, X )
    # Calculate the cost

    this_cost = get_cost( Y, Y_hat )
    # Save the previous bias and weights

    prev_b = b
    prev_w = w

    # Update and calculate the new values of bias and weights
    b, w = update_beta( X, Y, Y_hat, prev_b, prev_w, alpha)
    # For every 10 iterations, store the cost i.e. MSE
    if( iter_num % 10 == 0 ):
        gd_iterations_df.loc[result_idx] = [iter_num, this_cost]
        result_idx = result_idx + 1

    iter_num += 1
    print( “Final estimate of b and w: ”, b, w )
    #return the final bias, weights and the cost at the end
    return gd_iterations_df, b, w

gd_iterations_df, b, w = run_gradient_descent( X, Y, alpha =0.001, num_iterations = 200 )

gd_iterations_df[0:10]