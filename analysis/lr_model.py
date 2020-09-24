import nltk 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import nltk
from nltk.corpus import stopwords
import string
import pickle
import json
import requests

# Code adapted from Coursera 

def add_to_list(filepath):
    '''
    DESCRIPTION
    -------------
    Function takes in a CSV filepath, and returns the words in the CSV file,
    excluding stop words contained in the NLTK stopwords 'english' corpus.
        
    PARAMETERS
    -------------
    filepath: str
        Specify CSV filepath

    RETURNS
    -------------
    specific_list: list 
    '''

    # Initate lists of stop words and specific list that will be returned 
    english_stopwords = stopwords.words('english')
    specific_list = []

    # Open file
    with open(filepath, newline='') as csvfile:
            data = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in data:
                word = (row[0].split(',')[0].lower())

                # Removing stop words and punctuation 
                if (word not in english_stopwords and 
                    word not in string.punctuation):
                    specific_list.append(word)

    return specific_list

def build_datasets(filepathList):
    '''
    DESCRIPTION
    -------------
    Takes in a list of filepaths for CSV files. Iterates through filepaths, and calls the 
    add_to_list() function for each filepath. 
    Each CSV file is appended to the all_items list. 
    Each item in all_items is iterated through, and appended to the list_type
    list to get each word as a single item in the final list.
    
    PARAMETERS
    -------------
    filepathList: list
        List containing filepaths to the desired CSV files

    RETURNS
    -------------
    list_type: list 
    '''
    
    all_items = []
    for filepath in filepathList:
        all_items.append(add_to_list(filepath))

    list_type = []
    for item in all_items:
        for i in item:
            list_type.append(i)
    
    return list_type

def train_test_split(fraction, list_type, split_type):
    '''
    DESCRIPTION
    -------------
    Split data into two pieces, one for training (~80%) and one for testing (~20% in validation set) 
    
    PARAMETERS
    -------------
    fraction: float 
        Will split train and test dataset into x % train, 100 - x % test.
        
    list_type: list

    split_type: str
        Accepts 'test' or 'train'

    RETURNS
    -------------
    split_type: list  
    '''

    training_fraction = int(len(list_type) * fraction)

    if split_type == 'train':
        split_type = list_type[:training_fraction]
    elif split_type == 'test':
        split_type = list_type[training_fraction:]
    else:
        TypeError

    return split_type

def build_y_output(pos_list, neg_list):
    '''
    DESCRIPTION
    -------------
    Creates output (y) positive and negative labels according to length of each
    positive and negative list 
    
    PARAMETERS
    -------------
    pos_list: list 
        
    neg_list: list

    RETURNS
    -------------
    list_y: array  
        A  numpy array of output labels for each positive and negative list 
    '''
    list_y = np.append(np.ones((len(pos_list), 1)), np.zeros((len(neg_list), 1)), axis=0)
    return list_y

def word_freq(train_x, train_y):
    """
    DESCRIPTION
    -------------
    Build word frequency dictionary.
        
    PARAMETERS
    -------------
    train_x: list 
        
    train_y: array
        An m x 1 array with the performance of each conversation (either 1 for good, 0 for bad)

    RETURNS
    -------------
    freqs: dict
        A dictionary mapping each (word, label) pair to its frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    labelslist = np.squeeze(train_y).tolist()

    # Start with an empty dictionary and populate it by looping over words.
    freqs = {}
    
    for y, word in zip(labelslist, train_x):
        pair = (word, y)
        if pair in freqs:
            freqs[pair] += 1
        else:
            freqs[pair] = 1    

    return freqs

def word_counts(train_x, train_y):
    """
    DESCRIPTION
    -------------
    Returns a list representing our table of word counts. 

    PARAMETERS
    -------------
    train_x: list 
        
    train_y: array
        An m x 1 array with the performance of each conversation (either 1 for good, 0 for bad)

    RETURNS
    -------------
    word_count_data: list
        Each element consist of a sublist with this pattern: [<word>, <positive_count>, <negative_count>]
    """
 
    word_count_data = []
    
    # loop through our selected words
    for word in train_x[100:150]:
    
    # initialize positive and negative counts
        pos = 0
        neg = 0
        
        freqs = word_freq(train_x, train_y)

        # retrieve number of positive counts
        if (word, 1) in freqs:
            pos = freqs[(word, 1)]
            
        # retrieve number of negative counts
        if (word, 0) in freqs:
            neg = freqs[(word, 0)]
            
        # append the word counts to the table
        word_count_data.append([word, pos, neg])

    return word_count_data

def plot_word_count_data(word_count_data):
    """
    DESCRIPTION
    -------------
    Plots a scatter graph of the words according to where they are on the line,
    either on the 'positive' side or 'negative' side. 

    PARAMETERS
    -------------
    word_count_data: list
        Each element consist of a sublist with this pattern: [<word>, <positive_count>, <negative_count>]

    RETURNS
    -------------
    None 
    """

    fig, ax = plt.subplots(figsize = (8, 8))
    
    # convert positive raw counts to logarithmic scale. we add 1 to avoid log(0)
    x = np.log([x[1] + 1 for x in word_count_data])  
    
    # do the same for the negative counts
    y = np.log([x[2] + 1 for x in word_count_data]) 
    
    # Plot a dot for each pair of words
    ax.scatter(x, y)  

    # assign axis labels
    plt.xlabel("Log Positive count")
    plt.ylabel("Log Negative count")

    # Add the word as the label at the same position as you added the points just before
    for i in range(0, len(word_count_data)):
        ax.annotate(word_count_data[i][0], (x[i], y[i]), fontsize=12)

    # Plot the red line that divides the 2 areas.
    ax.plot([0, 9], [0, 9], color = 'red') 

    plt.show()

def feature_extraction(words, freqs):
    '''
    DESCRIPTION
    -------------
    Given a list of words, the function extracts two features and stores them in a matrix:
    1. The first feature is the number of positive words in the list.
    2. The second feature is the number of negative words in the list.

    PARAMETERS
    -------------
    words: list
        a list of words said in conversation that have been processed and had stopwords removed 
    freqs: dict
        A dictionary corresponding to the frequencies of each tuple (word, label)

    RETURNS
    -------------
        x: vector
            A feature vector of dimension (1,3)
    '''

    x = np.zeros((1, 3)) 
    
    #bias term is set to 1
    x[0,0] = 1 
    
    # Loop through each word in the list of words
    for word in words:
        # Increment the word count for the positive label 1
        x[0,1] += freqs.get((word, 1.0),0)
        
        # Increment the word count for the negative label 0
        x[0,2] += freqs.get((word, 0.0),0)
        
    assert(x.shape == (1, 3))
    return x    


# Training Logistic Regression Model

# Sigmoid function from coursera 
def sigmoid(a): 
    '''
    Input:
        a: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of a
    '''

    # calculate the sigmoid of a
    h = 1/(1 + np.exp(-a))
    return h

# To train the model: stack the features for all training examples into a matrix X. Call gradientDescent
def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    DESCRIPTION
    -------------
    Outputs theta, the final weight vector. 

    PARAMETERS
    -------------
    x: matrix
        Matrix of features which is (m,n+1) 
    y: vector
        Corresponding labels of the input matrix x, dimensions (m,1)
    theta: vector 
        Weight vector of dimension (n+1,1)
    alpha: float
        Specifies the learning rate
    num_iters: int 
        The number of iterations you want to train your model for

    RETURNS
    -------------
    J: float
        The final cost
    theta: vector 
        Final weight vector

    '''
    # get 'm', the number of rows in matrix x
    m = x.shape[0]     
    for i in range(0, num_iters):

        # get z, the dot product of x and theta
        z = np.dot(x,theta)

        # get the sigmoid of h
        h = sigmoid(z)
        
        # calculate the cost function
        J = -1./m * (np.dot(y.transpose(), np.log(h)) + np.dot((1-y).transpose(),np.log(1-h)))                                                    
        
        # update the weights theta
        theta = theta - (alpha/m) * np.dot(x.transpose(),(h-y))
        
    J = float(J)
    return J, theta

def predict_conversation(words, freqs, theta):
    '''
    DESCRIPTION
    -------------
    Predict whether a conversation is postiive or negative. Given sentences said 
    in a conversation, process it, then extract the features. 
    Apply the model's learned weights on the features to get the logits. 
    Apply the sigmoid to the logits to get the prediction (a value between 0 and 1)

    PARAMETERS 
    -------------
    words: list
        A list of words
    freqs: dict
        A dictionary corresponding to the frequencies of each tuple (word, label)
    theta: vector 
        (3,1) vector of weights
    
    RETURN
    -------------
    y_pred: float 
        The probability of the input (list of words) being positive or negative
    '''
    
    # extract the features of the words and store it into x
    x = feature_extraction(words, freqs)
    
    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x,theta))
        
    return y_pred

# Test logisitic regression function ----------------------------
def test_logistic_regression(test_x, test_y, freqs, theta):
    '''
    PARAMETERS 
    -------------
    test_x: list
        A list of words
    test _y: vector
        (m, 1) vector with the corresponding labels for the list of words
    freqs: dict
        A dictionary with the frequency of each pair (or tuple)
    theta: vector
        Weight vector of dimension (3, 1)

    RETURNS 
    -------------
    accuracy: float 
        (# of dialogues classified correctly) / (total # of dialogues)
    '''
    
    # the list for storing predictions
    y_hat = []
    
    for word in test_x:
        # get the label prediction for the tweet
        y_pred = predict_conversation(word, freqs, theta)
        
        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1)
        else:
            # append 0 to the list
            y_hat.append(0)

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    accuracy = (y_hat==np.squeeze(test_y)).sum()/len(test_x)

    return accuracy

trainingdata_location = '../dcapp/trainingdata/'

all_negative = build_datasets([trainingdata_location + 'dc2-stt-l.csv', trainingdata_location + 'dc4-stt-l.csv', trainingdata_location + 'dc5-stt-l.csv'])
all_positive = build_datasets([trainingdata_location + 'dc1-stt-l.csv', trainingdata_location + 'dc3-stt-l.csv'])

train_pos = train_test_split(0.8, all_positive, 'train')
test_pos = train_test_split(0.8, all_positive, 'test')

train_neg = train_test_split(0.8, all_negative, 'train')
test_neg = train_test_split(0.8, all_negative, 'test')

train_x = train_pos + train_neg 
test_x = test_pos + test_neg

train_y = build_y_output(train_pos, train_neg)
test_y = build_y_output(test_pos, test_neg)

freqs = word_freq(train_x, train_y)

word_count_data = word_counts(train_x, train_y)

# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= feature_extraction(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
# print(f"The cost after training is {J:.8f}.")
# print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")

tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
# print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")