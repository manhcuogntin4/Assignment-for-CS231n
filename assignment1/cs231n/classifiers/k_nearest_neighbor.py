import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    for i in range(num_test):
    	for j in range(num_train):
    		dists[i,j]=np.sum(np.sqrt(np.abs((X[i]*X[i]-self.X_train[j]*self.X_train[j]))))
    #print dists.shape
    #print X.shape
    #print self.X_train.shape
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
    	X2=X[i]*X[i]
    	X_train2=self.X_train*self.X_train
    	Delta=np.sqrt(np.abs(X2-X_train2))
    	dists[i:]=np.sum(Delta,axis=1)
    	#np.sum(np.sqrt(np.abs((X[i]*X[i]-self.X_train[j]*self.X_train[j]))))
    	
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    X2=X.reshape((X.shape[0]*X.shape[1]),1)
    X_train2=self.X_train*self.X_train
    Delta=np.sqrt(np.abs(X2-X_train2,axis=1),axis=1)
    dists=np.sum(Delta,axis=1)
    return dists


  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):

      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      min_row=np.argsort(dists[i,:])[:k]
      closest_y=self.y_train[min_row]
      d={}
      for m in closest_y:
      	if m not in d:
      		d[m]=1
      	else:
      		d[m]+=1
      d_max=sorted(d.items(), key=lambda kv: kv[1])
      y_pred[i]=d_max[0][0]
      #print closest_y
      #y_pred[]

      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      #pass
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      #pass
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

def main():
	X_train=np.array([[1,2,3],[4,4,4],[1,3,3]])
	X=np.array([[2,2,4],[4,4,5]])
	#X=X.tranform()
	y_train=np.array([0,1,0])
	classifier=KNearestNeighbor()
	classifier.train(X_train,y_train)
	dists=classifier.compute_distances_two_loops(X)
	dists_one=classifier.compute_distances_one_loop(X)
	t=classifier.predict_labels(dists,1)
	print t
	print dists
	print dists

if __name__ == "__main__":
    # execute only if run as a script
    main()