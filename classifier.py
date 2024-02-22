# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
import numpy as np

class LinearRegressionClassifier:
    def __init__(self):
        self.weights = None # 25 features and 1 dummy feature
        self.alpha = 0.01

        self.reset()

    def reset(self):
        # Weight list will consist of values for 25 features + 1 dummy feature for w0
        self.weights = []
        for i in range(0, 26):
            self.weights.append(1)

    def hfunction(self, data):
        h = 0
        # Param data is assumed to be a certain instance of training data with x_j,0 = 1 appended to the end
        for i in range(0, 26):
            h += self.weights[i] * data[i]

        return h

    def fit(self, data, target):
        # Update 26 weights (25 features, 1 dummy)
        for i in range(0, 26):
            temp = 0 # Running total for summation
            # Batch gradient descent
            # j is index of a training data set 
            for j in range(0, len(data)):
                self.training = data[j] + [1]
                temp += (target[j] - self.hfunction(self.training + [1])) * self.training[i] # Sigma( ( y_j - hw(x_j) ) * x_j,i )

            temp *= self.alpha
            # Update weight w_i 
            self.weights[i] += temp
            

    def predict(self, data, legal=None):
        temp = 0
        for i in range(0, 25):
            temp += data[i] * self.weights[i]

        temp += self.weights[25] # + w0
        
        return 3 if temp >= 3 else 0 if temp <= 0 else round(temp)

class NaiveBayesClassifier:
    def __init__(self):
        self.pv = [] # array to hold p(v_i)
        self.v = [] # will hold occurances of v

        # arrays that will hold conditional probabilities for each feature in the training data. one for each target.
        self.a0 = []
        self.a1 = []
        self.a2 = []
        self.a3 = []

        self.reset()

    # Called when a game is over - clean something up
    def reset(self):
        self.pv = [0, 0, 0, 0]
        self.v = [0, 0, 0, 0]

        # first row is for when the feature is set to 0, second row for 1
        self.a0 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        
        self.a1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        
        self.a2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        
        self.a3 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    # Training function. Data is in the form of a feature vector and target is the resultant direction in number form
    def fit(self, data, target):
        self.v = np.bincount(np.array(target)) # -> [28 33 26 39] number of instances of 0 1 2 3 in their respetive indicies
        self.pv = list(map(lambda x: x / len(target), self.v)) # Map each value to the probability of that index occuring in the training data

        # count up occurances
        for fv, trgt in zip(data, target):
            for f, i in zip(fv, range(0, 25)):
                match trgt:
                    case 0:
                        self.a0[f][i] += 1
                    case 1:
                        self.a1[f][i] += 1
                    case 2:
                        self.a2[f][i] += 1
                    case 3:
                        self.a3[f][i] += 1
                    case _:
                        print("stinker")

        # Calculate p(a|v) for each feature
        self.a0 = [[x / self.v[0] for x in i] for i in self.a0]
        self.a1 = [[x / self.v[1] for x in i] for i in self.a1]
        self.a2 = [[x / self.v[2] for x in i] for i in self.a2]
        self.a3 = [[x / self.v[3] for x in i] for i in self.a3]

    # Needs to return a number (0,1,2,3) direction
    def predict(self, data, legal=None):
        
        # p(v) for each v is held in list self.pv
        # p(a|v) for each a is held in 4 different lists for each v
        # perform product notation to calculate probability of each v
        for v, i in zip(data, range(0, 25)):
            self.pv[0] *= self.a0[v][i]

        for v, i in zip(data, range(0, 25)):
            self.pv[1] *= self.a1[v][i]

        for v, i in zip(data, range(0, 25)):
            self.pv[2] *= self.a2[v][i]

        for v, i in zip(data, range(0, 25)):
            self.pv[3] *= self.a3[v][i]

        return np.argmax(self.pv)

class Classifier:
    def __init__(self):
        self.clist = [] # Will hold all the classifiers being used

        self.nb_classifier = NaiveBayesClassifier()
        self.clist.append(self.nb_classifier)

        self.linear_classifier = LinearRegressionClassifier()
        self.clist.append(self.linear_classifier)

    # Called when a game is over - clean something up
    def reset(self):
        # Call reset method on every classifier
        for c in self.clist:
            c.reset()
        
    # Training function. Data is in the form of a feature vector and target is the resultant direction in number form
    def fit(self, data, target):
        # Call fit method on every classifier
        for c in self.clist:
            c.fit(data, target)

    # Needs to return a number (0,1,2,3) direction
    def predict(self, data, legal=None):
        tally = [0, 0, 0, 0]

        # predict() should only return integers 0-3.
        # Let each classifier vote on which action to take
        for c in self.clist:
            tally[c.predict(data, legal)] += 1

        # Checking for legal actions is done by getAction() in classifierAgents and by api.makeMove()
        # argmax will return the first index of the max value in the list (in case of duplicates)
        return np.argmax(tally)
