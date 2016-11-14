import numpy as np
import pickle
import matplotlib.pyplot as plt
import numpy as np

dataDir = "../data/"
# output = np.load(dataDir + "output.npy")
# print output.shape
# print output[0]
# x = output[0]
# x_normed = x + np.amax(-x)
# print(x_normed)

testLabels = pickle.load( open( dataDir + "yTest.p", "rb" ) )[:1000]
print testLabels.shape
print type(testLabels[0][0])
print testLabels[0]


# plt.hist(output[0])
# plt.hist(testLabels[0])

# plt.title("Gaussian Histogram")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.show()