# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

def sigmoid_activation(x):
	# compute the sigmoid activation value for a given input
    #start your code here
    return 1 / (1+ np.exp(-x))
    # end code
	

def predict(X, W):
    # take the dot product between our features and weight matrix
    #start your code here
    dotp = np.dot(X,W)
    # end code
    
    # apply a step function to threshold (=0.5) the outputs to binary class labels
    #start your code here
    preds=np.heaviside(dotp, 0.5)
    # end code

    # return the predictions
    return preds  #preds = 0 or 1
	

epochs = 50
alpha = 0.01

# generate a 2-class classification problem with 1,000 data points, where each data point is a 2D feature vector
# X: data
# y: label
#start your code here
(X,y)=make_moons(n_samples=1000, noise = 0.15)
y=y.reshape(y.shape[0],1)
# end code



# insert a column of 1's as the last entry in the feature
# matrix -- this little trick allows us to treat the bias
# as a trainable parameter within the weight matrix
X = np.c_[X, np.ones((X.shape[0]))]

# partition the data into training and testing splits using 50% of
# the data for training and the remaining 50% for testing
(trainX, testX, trainY, testY) = train_test_split(X, y,
	test_size=0.5, random_state=42)

# initialize our weight matrix and list of losses
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []

# loop over the desired number of epochs
for epoch in np.arange(0, epochs):
	# take the dot product between our features `X` and the weight
	# matrix `W`, then pass this value through our sigmoid activation
	# function, thereby giving us our predictions on the dataset
    
    #start your code here
    sig = sigmoid_activation(np.dot(trainX, W))
    # end code
    
    
    
    # now that we have our predictions, we need to determine the
    # `error`, which is the difference between our predictions and the true values
    # loss: loss value for each iteration
    #start your code here
    error = sig - trainY
    loss = np.sum(pow(error, 2))
    # end code
        
    losses.append(loss)

    # the gradient descent update is the dot product between our
    # features and the error of the predictions
    #start your code here
    update = np.dot(trainX.T, error)
    # end code

    # in the update stage, all we need to do is "nudge" the weight
    # matrix in the negative direction of the gradient (hence the
    # term "gradient descent" by taking a small step towards a set
    # of "more optimal" parameters
    #start your code here
    W-= update*alpha
    # end code
    
    # check to see if an update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}" .format(epoch,
              loss))
    
# evaluate our modelcle
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY[:,0], s=30)

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()