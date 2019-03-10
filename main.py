#!/usr/bin/env python3
import os
import numpy
import scipy
import matplotlib
import mnist
import pickle
import scipy.special
matplotlib.use('agg')
from matplotlib import pyplot as plt
import time

from matplotlib import pyplot

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables


def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = numpy.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        # shuffle the training data
        numpy.random.seed(8675309)
        perm = numpy.random.permutation(60000)
        Xs_tr = numpy.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = numpy.ascontiguousarray(Ys_tr[:,perm])
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = numpy.ascontiguousarray(Xs_te)
        Ys_te = numpy.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset
#calculate the prediction vector given W and X, namely softmax(WX)
#
# X        training examples (d * n)
# W         parameters       (c * d)
#
#returns preds, a (c * n) matrix of predictions
def mult_logreg_pred(W,X):
    WX= numpy.matmul(W,X) #apply the linear parameters
    preds=scipy.special.softmax(WX, axis=0) # apply softmax with respect to each observation
    
    return preds

# compute the gradient of the multinomial logistic regression objective, with regularization
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the gradient of the model parameters
def multinomial_logreg_grad(Xs, Ys, gamma, W):
    # TODO students should implement this
    c, d = numpy.shape(W)
    dummy, n=numpy.shape(Xs)
    grad = numpy.zeros((c, d)) # gradient has the same size as W
    H = mult_logreg_pred(W, Xs)
    signed_error=(H-Ys)
    grad = (numpy.matmul(signed_error, Xs.T))/n+gamma*W
    return grad

# compute the gradient of the multinomial logistic regression objective, with regularization (SAME AS PROGRAMMING ASSIGNMENT 2)
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    # TODO students should use their implementation from programming assignment 2
    X_sub = Xs[:, ii]
    Y_sub = Ys[:, ii]
    return multinomial_logreg_grad( X_sub, Y_sub, gamma, W)


# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    # TODO students should use their implementation from programming assignment 1
    preds_real=mult_logreg_pred(W,Xs)
    c, n = numpy.shape(preds_real)
    pred_lables=numpy.zeros((c,n))
    label_position=numpy.argmax(preds_real,axis=0)
    for j,i in enumerate(label_position):
        pred_lables[i,j]=1
    accuracy=numpy.trace(numpy.matmul(Ys,pred_lables.T))/n
    error_percent=1-accuracy
    return error_percent

# compute the cross-entropy loss of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss(Xs, Ys, gamma, W):
    # TODO students should implement this
    c, n = numpy.shape(Ys)
    preds = mult_logreg_pred(W,Xs)
    log_y_hat = -numpy.log(preds)
    y_log_y_hat = Ys*log_y_hat
    empirical_risk = numpy.sum(Ys*log_y_hat)/n + gamma/2*numpy.linalg.norm(W)**2
    return empirical_risk

# gradient descent (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set, or equivalently iterations of gradient descent) to run
# monitor_period  how frequently, in terms of epochs/iterations to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" epochs
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):
    # TODO students should use their implementation from programming assignment 1
    param_list=[]
    W_prev=W0;
    j=0;
    for i in range(num_epochs):
        j+=1
        grad_f = multinomial_logreg_grad(Xs, Ys, gamma, W_prev);
        W_next = W_prev - alpha*grad_f
        if j == monitor_period:
            param_list.append(W_next)
            j=0
            print("difference in W's",numpy.linalg.norm(W_prev-W_next))
        W_prev=W_next


    return param_list


# gradient descent with nesterov momentum
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# num_epochs      number of epochs (passes through the training set, or equivalently iterations of gradient descent) to run
# monitor_period  how frequently, in terms of epochs/iterations to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" epochs
def gd_nesterov(Xs, Ys, gamma, W0, alpha, beta, num_epochs, monitor_period):
    # TODO students should implement this
    param_list=[]
    W_prev = W0
    V_prev = W0
    j = 0
    for i in range(num_epochs):
        j+=1
        grad_f = multinomial_logreg_grad(Xs, Ys, gamma, W_prev)
        V_next = W_prev - alpha*grad_f
        W_next= (1+beta)*V_next - beta*V_prev
        if j == monitor_period:
            param_list.append(W_next)
            j=0
            print("difference in W's",numpy.linalg.norm(W_prev-W_next))
        W_prev = W_next
        V_prev = V_next
    return param_list


# SGD: run stochastic gradient descent with minibatching and sequential sampling order (SAME AS PROGRAMMING ASSIGNMENT 2)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_minibatch_sequential_scan(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should use their implementation from programming assignment 2
    c, n = Ys.shape
    d, n = Xs.shape
    W_prev = W0
    Ws_output = []
    if (n % B != 0):
        raise ValueError("B must divide n evenly")
    q = 0
    for j in range(0, num_epochs):
        for i in range(n//B):
            sample = range(i*B,(i+1)*B)
            
            W_next = W_prev - alpha * multinomial_logreg_grad_i(Xs, Ys, numpy.array(sample), gamma, W_prev)
            if ((q +1)% monitor_period == 0):
                Ws_output.append(W_next)
            q=q+1
            W_prev = W_next
    return Ws_output


# SGD + Momentum: add momentum to the previous algorithm
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, monitor_period):
    # TODO students should implement this
    c, n = Ys.shape
    d, n = Xs.shape
    W_prev = W0
    V = 0
    Ws_output = []
    if (n % B != 0):
        raise ValueError("B must divide n evenly")
    q = 0
    for j in range(0, num_epochs):
        for i in range(n//B):
            sample = range(i*B,(i+1)*B)
            V = beta*V - alpha * multinomial_logreg_grad_i(Xs, Ys, numpy.array(sample), gamma, W_prev)
            W_next = W_prev + V
            if ((q + 1) % monitor_period == 0):
                Ws_output.append(W_next)
            q=q+1
            W_prev = W_next
    return Ws_output




# Adam Optimizer
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# rho1            first moment decay rate ρ1
# rho2            second moment decay rate ρ2
# B               minibatch size
# eps             small factor used to prevent division by zero in update step
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def adam(Xs, Ys, gamma, W0, alpha, rho1, rho2, B, eps, num_epochs, monitor_period):
    # TODO students should implement this
    c, n = Ys.shape
    d, n = Xs.shape
    W_prev = W0
    r = numpy.zeros((c,d))
    s = numpy.zeros((c,d))
    Ws_output = []
    if (n % B != 0):
        raise ValueError("B must divide n evenly")
    t = 0
    for j in range(0, num_epochs):
        for i in range(n//B):
            sample = range(i*B,(i+1)*B)
            grad = multinomial_logreg_grad_i(Xs, Ys, numpy.array(sample), gamma, W_prev)
            s = rho1*s + (1 - rho1)*grad
            r = rho2*r + (1 - rho2)*numpy.power(grad,2)
            s_hat = s/(1-rho1**(t+1))
            r_hat = r/(1-rho2**(t+1))
            W_next= W_prev - alpha/numpy.sqrt(r_hat+eps)*s_hat
            if ((t + 1)% monitor_period == 0):
                Ws_output.append(W_next)
            t=t+1
            W_prev = W_next
    return Ws_output

# code for generating graphs
def problem_1_error_driver():
    Xs_tr, Ys_tr, Xs_te, Ys_te = load_MNIST_dataset()
    
    d, n = numpy.shape(Xs_tr)
    c, n = numpy.shape(Ys_tr)
    
    W0 = numpy.random.rand(c,d)
    
    gamma = 0.0001
    alpha = 1
    beta_1 = 0.9
    beta_2 = 0.99
    monitor_period = 1
    num_epochs = 100
    
    index = numpy.linspace(1,100,100)
    W_GD = gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, num_epochs, monitor_period)
    W_GDM_B1 = gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha, beta_1, num_epochs, monitor_period)
    W_GDM_B2 = gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha, beta_2, num_epochs, monitor_period)

    W_GD_tr_err = []
    W_GDM_B1_tr_err = []
    W_GDM_B2_tr_err = []
    
    index= numpy.linspace(1,100,100)
    
    if len(W_GD) != len(W_GDM_B1) or len(W_GD) != len(W_GDM_B2) or len(W_GDM_B1) != len(W_GDM_B2):
        print("length of arrays not equal")

    for i in range(len(W_GD)):
        W_GD_tr_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, W_GD[i]))
        W_GDM_B1_tr_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, W_GDM_B1[i]))
        W_GDM_B2_tr_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, W_GDM_B2[i]))

    W_GD_te_err = []
    W_GDM_B1_te_err = []
    W_GDM_B2_te_err = []

    for i in range(len(W_GD)):
        W_GD_te_err.append(multinomial_logreg_error(Xs_te, Ys_te, W_GD[i]))
        W_GDM_B1_te_err.append(multinomial_logreg_error(Xs_te, Ys_te, W_GDM_B1[i]))
        W_GDM_B2_te_err.append(multinomial_logreg_error(Xs_te, Ys_te, W_GDM_B2[i]))

    W_GD_loss = []
    W_GDM_B1_loss = []
    W_GDM_B2_loss = []

    for i in range(len(W_GD)):
        W_GD_loss.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, W_GD[i]))
        W_GDM_B1_loss.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, W_GDM_B1[i]))
        W_GDM_B2_loss.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, W_GDM_B2[i]))
    
    
    plt.figure()
    plt.plot(index,  W_GD_tr_err, 'r-')
    plt.plot(index,  W_GDM_B1_tr_err, 'b-')
    plt.plot(index,  W_GDM_B2_tr_err, 'g-')
    plt.ylabel('Percent error')
    plt.xlabel('Epoch number')
    plt.title('GD vs. GD with momentum, training error')
    plt.savefig('train_err_GD.pdf')

    plt.figure()
    plt.plot(index,  W_GD_te_err, 'r-')
    plt.plot(index,  W_GDM_B1_te_err, 'b-')
    plt.plot(index,  W_GDM_B2_te_err, 'g-')
    plt.ylabel('Percent error')
    plt.xlabel('Epoch number')
    plt.title('GD vs. GD with momentum, testing error')
    plt.savefig('test_err_GD.pdf')

    plt.figure()
    plt.plot(index,  W_GD_loss, 'r-')
    plt.plot(index,  W_GDM_B1_loss, 'b-')
    plt.plot(index,  W_GDM_B2_loss, 'g-')
    plt.ylabel('Percent error')
    plt.xlabel('Epoch number')
    plt.title('GD vs. GD with momentum, training loss')
    plt.savefig('loss_GD.pdf')

def problem_1_time_driver():
    Xs_tr, Ys_tr, Xs_te, Ys_te = load_MNIST_dataset()
    
    d, n = numpy.shape(Xs_tr)
    c, n = numpy.shape(Ys_tr)
    
    W0 = numpy.random.rand(c,d)
    
    gamma = 0.0001
    alpha = 1
    beta_1 = 0.9
    beta_2 = 0.99
    monitor_period = 1
    num_epochs = 100
    
    time_alg1 = 0
    time_alg2 = 0
    time_alg3 = 0
    
    
    for i in range(5):
        start = time.clock()
        gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, num_epochs, monitor_period)
        time_alg1 = time_alg1 + time.clock() - start
        
        start = time.clock()
        gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha, beta_1, num_epochs, monitor_period)
        time_alg2 = time_alg2+  time.clock() - start
        
        start = time.clock()
        gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha, beta_2, num_epochs, monitor_period)
        time_alg3 = time_alg3+  time.clock() - start
    
    
    time_alg1 = time_alg1/5
    time_alg2 = time_alg2/5
    time_alg3 = time_alg3/5
    
    
    print("the average time of GD: ",time_alg1)
    print("the average time of GD with momentum, beta= 0.9: ",time_alg2)
    print("the average time of GD with momentum, beta= 0.99: ",time_alg3)

def problem_2_error_driver():
    Xs_tr, Ys_tr, Xs_te, Ys_te = load_MNIST_dataset()
    
    d, n = numpy.shape(Xs_tr)
    c, n = numpy.shape(Ys_tr)
    
    W0 = numpy.random.rand(c,d)
    B = 600
    gamma = 0.0001
    alpha = 0.2
    beta_1 = 0.9
    beta_2 = 0.99
    monitor_period = 10
    num_epochs = 10
    
    W_SGD = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, num_epochs, monitor_period)
    W_SGDM_B1 = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta_1, B, num_epochs, monitor_period)
    W_SGDM_B2 = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta_2, B, num_epochs, monitor_period)
    
    W_SGD_tr_err = []
    W_SGDM_B1_tr_err = []
    W_SGDM_B2_tr_err = []
    
    
    
    if len(W_SGD) != len(W_SGDM_B1) or len(W_SGD) != len(W_SGDM_B2) or len(W_SGDM_B1) != len(W_SGDM_B2):
        print("length of arrays not equal")
    
    index= numpy.linspace(1,10,len(W_SGD))
    
    for i in range(len(W_SGD)):
        W_SGD_tr_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, W_SGD[i]))
        W_SGDM_B1_tr_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, W_SGDM_B1[i]))
        W_SGDM_B2_tr_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, W_SGDM_B2[i]))

    W_SGD_te_err = []
    W_SGDM_B1_te_err = []
    W_SGDM_B2_te_err = []

    for i in range(len(W_SGD)):
        W_SGD_te_err.append(multinomial_logreg_error(Xs_te, Ys_te, W_SGD[i]))
        W_SGDM_B1_te_err.append(multinomial_logreg_error(Xs_te, Ys_te, W_SGDM_B1[i]))
        W_SGDM_B2_te_err.append(multinomial_logreg_error(Xs_te, Ys_te, W_SGDM_B2[i]))
        
    W_SGD_loss = []
    W_SGDM_B1_loss = []
    W_SGDM_B2_loss = []
    
    for i in range(len(W_SGD)):
        W_SGD_loss.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, W_SGD[i]))
        W_SGDM_B1_loss.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, W_SGDM_B1[i]))
        W_SGDM_B2_loss.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, W_SGDM_B2[i]))


    plt.figure()
    plt.plot(index,  W_SGD_tr_err, 'r-')
    plt.plot(index,  W_SGDM_B1_tr_err, 'b-')
    plt.plot(index,  W_SGDM_B2_tr_err, 'g-')
    plt.ylabel('Percent error')
    plt.xlabel('Epoch number')
    plt.title('SGD vs. SGD with momentum, training error')
    plt.savefig('train_err_SGD.pdf')

    plt.figure()
    plt.plot(index,  W_SGD_te_err, 'r-')
    plt.plot(index,  W_SGDM_B1_te_err, 'b-')
    plt.plot(index,  W_SGDM_B2_te_err, 'g-')
    plt.ylabel('Percent error')
    plt.xlabel('Epoch number')
    plt.title('SGD vs. SGD with momentum, testing error')
    plt.savefig('test_err_SGD.pdf')

    plt.figure()
    plt.plot(index,  W_SGD_loss, 'r-')
    plt.plot(index,  W_SGDM_B1_loss, 'b-')
    plt.plot(index,  W_SGDM_B2_loss, 'g-')
    plt.ylabel('Percent error')
    plt.xlabel('Epoch number')
    plt.title('SGD vs. SGD with momentum, training loss')
    plt.savefig('loss_SGD.pdf')


def problem_2_time_driver():
    Xs_tr, Ys_tr, Xs_te, Ys_te = load_MNIST_dataset()
        
    d, n = numpy.shape(Xs_tr)
    c, n = numpy.shape(Ys_tr)
    
    W0 = numpy.random.rand(c,d)
    B = 600
    gamma = 0.0001
    alpha = 0.2
    beta_1 = 0.9
    beta_2 = 0.99
    monitor_period = 10
    num_epochs = 10
    
    time_alg1 = 0
    time_alg2 = 0
    time_alg3 = 0
    
    
    for i in range(5):
        start = time.clock()
        sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, num_epochs, monitor_period)
        time_alg1 = time_alg1 + time.clock() - start
        
        start = time.clock()
        sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta_1, B, num_epochs, monitor_period)
        time_alg2 = time_alg2+  time.clock() - start
        
        start = time.clock()
        sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta_2, B, num_epochs, monitor_period)
        time_alg3 = time_alg3+  time.clock() - start
    
    
    time_alg1 = time_alg1/5
    time_alg2 = time_alg2/5
    time_alg3 = time_alg3/5
    
    
    print("the average time of SGD: ",time_alg1)
    print("the average time of SGD with momentum, beta= 0.9: ",time_alg2)
    print("the average time of SGD with momentum, beta= 0.99: ",time_alg3)

def problem_3_error_driver():
    Xs_tr, Ys_tr, Xs_te, Ys_te = load_MNIST_dataset()
    
    d, n = numpy.shape(Xs_tr)
    c, n = numpy.shape(Ys_tr)
    
    W0 = numpy.random.rand(c,d)
    B = 600
    gamma = 0.0001
    alpha_1 = 0.2
    alpha_2 = 0.01
    rho1 = 0.9
    rho2 = 0.999
    monitor_period = 10
    num_epochs = 10
    eps = .00001
    W_SGD = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha_1, B, num_epochs, monitor_period)
    W_ADAM = adam(Xs_tr, Ys_tr, gamma, W0, alpha_2, rho1, rho2, B, eps, num_epochs, monitor_period)
   
    
    W_SGD_tr_err = []
    W_ADAM_tr_err = []
   
   
   
    index= numpy.linspace(1,10,len(W_SGD))
    
    for i in range(len(W_SGD)):
        W_SGD_tr_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, W_SGD[i]))
        W_ADAM_tr_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, W_ADAM[i]))


    W_SGD_te_err = []
    W_ADAM_te_err = []


    for i in range(len(W_SGD)):
        W_SGD_te_err.append(multinomial_logreg_error(Xs_te, Ys_te, W_SGD[i]))
        W_ADAM_te_err.append(multinomial_logreg_error(Xs_te, Ys_te, W_ADAM[i]))

        
    W_SGD_loss = []
    W_ADAM_loss = []

    for i in range(len(W_SGD)):
        W_SGD_loss.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, W_SGD[i]))
        W_ADAM_loss.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, W_ADAM[i]))


    plt.figure()
    plt.plot(index,  W_SGD_tr_err, 'r-')
    plt.plot(index,  W_ADAM_tr_err, 'b-')

    plt.ylabel('Percent error')
    plt.xlabel('Epoch number')
    plt.title('SGD vs. ADAM, training error')
    plt.savefig('train_err_ADAM.pdf')

    plt.figure()
    plt.plot(index,  W_SGD_te_err, 'r-')
    plt.plot(index,  W_ADAM_te_err, 'b-')
    plt.ylabel('Percent error')
    plt.xlabel('Epoch number')
    plt.title('SGD vs. ADAM, testing error')
    plt.savefig('test_err_ADAM.pdf')

    plt.figure()
    plt.plot(index,  W_SGD_loss, 'r-')
    plt.plot(index,  W_ADAM_loss, 'b-')
    plt.ylabel('Percent error')
    plt.xlabel('Epoch number')
    plt.title('SGD vs. ADAM, training loss')
    plt.savefig('loss_ADAM.pdf')

def problem_3_time_driver():
    Xs_tr, Ys_tr, Xs_te, Ys_te = load_MNIST_dataset()
    
    d, n = numpy.shape(Xs_tr)
    c, n = numpy.shape(Ys_tr)
    
    W0 = numpy.random.rand(c,d)
    B = 600
    gamma = 0.0001
    alpha_1 = 0.2
    alpha_2 = 0.01
    rho1 = 0.9
    rho2 = 0.999
    monitor_period = 10
    num_epochs = 10
    eps = .00001
    
    time_alg1 = 0
    time_alg2 = 0

    
    
    for i in range(5):
        start = time.clock()
        sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha_1, B, num_epochs, monitor_period)
        time_alg1 = time_alg1 + time.clock() - start
        
        start = time.clock()
        adam(Xs_tr, Ys_tr, gamma, W0, alpha_2, rho1, rho2, B, eps, num_epochs, monitor_period)
        time_alg2 = time_alg2+  time.clock() - start
        
    print("the average time of SGD: ",time_alg1)
    print("the average time of ADAM: ",time_alg2)

if __name__ == "__main__":
    #problem_1_error_driver()
    #problem_1_time_driver()
    #problem_2_error_driver()
    #problem_2_time_driver()
    #problem_3_error_driver()
    problem_3_time_driver()

