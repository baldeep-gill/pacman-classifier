# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
import numpy as np

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

        self.a0 = [[x / self.v[0] for x in i] for i in self.a0]
        self.a1 = [[x / self.v[1] for x in i] for i in self.a1]
        self.a2 = [[x / self.v[2] for x in i] for i in self.a2]
        self.a3 = [[x / self.v[3] for x in i] for i in self.a3]

    # Needs to return a number (0,1,2,3) direction
    def predict(self, data, legal=None):
        
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
        self.clist = []

        self.nb_classifier = NaiveBayesClassifier()
        self.clist.append(self.nb_classifier)

    # Called when a game is over - clean something up
    def reset(self):
        for c in self.clist:
            c.reset()
        
    # Training function. Data is in the form of a feature vector and target is the resultant direction in number form
    def fit(self, data, target):
        for c in self.clist:
            c.fit(data, target)

    # Needs to return a number (0,1,2,3) direction
    def predict(self, data, legal=None):
        tally = [0, 0, 0, 0]

        for c in self.clist:
            tally[c.predict(data, legal)] += 1

        return np.argmax(tally)
