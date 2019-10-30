import numpy as np

class KNN:
    def __init__ (self, k):
       #allows for different k values
        self.k = k

    def train(self, X, y):
        """
        X is N x D where each row is an example.
        Y is 1-dimension of size N
        """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """
        X is N x D where each row is an example we wish to predict label for
        """
        num_test = X.shape[0]
        
        #lets make sure that the output type matches the input type
        YPred = np.zeros((num_test,2), dtype = self.ytr.dtype)
   
        for i in range(num_test):
            """
            #L1 - Manhattan formula to find the distance between train and test
            https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
            """
            dist_l1 = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)

            """
            This will be for the Euclidean distance        
            #L2 - Euclidean distance 
            dist_l2 = np.reshape(np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis=1)), (-1, 1))
            """
            dist_l2 = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis=1))
            
            """
            finds closest k indicies  
            https://stackoverflow.com/questions/16817948/i-have-need-the-n-minimum-index-values-in-a-numpy-array
            https://docs.scipy.org/doc/numpy-1.8.0/reference/generated/numpy.argpartition.html
            """
            top_k_ind_l1 = np.argsort(dist_l1)[:self.k]
            top_k_ind_l2 = np.argsort(dist_l2)[:self.k]
           
            """
            np.unique combines the unique labels and tallys each
            then amax will return the one with the highest score
            https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html
            https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
            voting:
                if there is a tie, grab the closest of the highest
                we are not using any weighted metrics due to sample sizes

            """
            #print('L1 dist for k=', self.k, 'image', i, ' is:', self.ytr[top_k_ind_l1])
            #print('L2 dist for k=', self.k, 'image', i, ' is:', self.ytr[top_k_ind_l2])

            u_label_l1, u_count_l1 = np.unique(self.ytr[top_k_ind_l1], return_counts=True)
            u_label_l2, u_count_l2 = np.unique(self.ytr[top_k_ind_l2], return_counts=True)

            #high_vote_l1 = u_label_l1[np.argmax(u_count_l1)]
            #high_vote_l2 = u_label_l2[np.argmax(u_count_l2)]
            """
            take the label with the highest vote
            """
            YPred[i][0] = u_label_l1[np.argmax(u_count_l1)]
            YPred[i][1] = u_label_l2[np.argmax(u_count_l2)]
            #print('L1 prediction for k=', self.k, "image', i, ' is:', YPred[i][0])
            #print('L2 prediction for k=' self.k, "image', i, ' is:', YPred[i][1])
            
        return YPred

    def predict_l1(self, X):
        """
        This is the same as predict() but only for L1 - Manhatten
        Therefore comments are removed
        """
        num_test = X.shape[0]
        YPred = np.zeros((num_test), dtype = self.ytr.dtype) 
        for i in range(num_test):

            dist_l1 = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)
            top_k_ind_l1 = np.argsort(dist_l1)[:self.k]
            #print('L1 dist for k=', self.k, 'image', i, 'is:', self.ytr[top_k_ind_l1])
            u_label_l1, u_count_l1 = np.unique(self.ytr[top_k_ind_l1], return_counts=True)        
            YPred[i] = u_label_l1[np.argmax(u_count_l1)]
            #print('L1 prediction for image', i, ' is:', YPred[i])
        return YPred


    
    def predict_l2(self, X):
        """
        This is the same as predict() but only for L1 - Manhatten
        Therefore comments are removed
        """
        num_test = X.shape[0]
        YPred = np.zeros((num_test), dtype = self.ytr.dtype)
        for i in range(num_test):
            dist_l2 = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis=1))
            top_k_ind_l2 = np.argsort(dist_l2)[:self.k]
            #print('L2 dist for k=', self.k, 'image', i, ' is:', self.ytr[top_k_ind_l2])
            u_label_l2, u_count_l2 = np.unique(self.ytr[top_k_ind_l2], return_counts=True)
            high_vote_l2 = u_label_l2[np.argmax(u_count_l2)]
            YPred[i] = u_label_l2[np.argmax(u_count_l2)]
            #print('L2 prediction for image', i, ' is:', YPred[i])
        return YPred
