import numpy
import math


def parseLocation(array):
    x = []
    y = []
    for i in range(0, len(array), 2):
        x.append(array[i])
        y.append(array[i + 1])
    return x, y


predictions = numpy.random.rand(10, 136)
test_labels = numpy.random.rand(10, 136)

print predictions.shape

newPredictions = []
newLabels = []
error = []

# print predictions.shape

(length, _) = predictions.shape


for index in range(length):
    pred = predictions[index]
    x, y =parseLocation(pred)
    predPts = zip(x, y)
    newPredictions.append(predPts)


    label = test_labels[index]

    x, y =parseLocation(label)
    labelPts = zip(x, y)   
    newLabels.append(labelPts)


    interocular_distance = math.sqrt((labelPts[37][0] - labelPts[46][0])**2 + (labelPts[37][1] - labelPts[46][1])**2)


    cum = 0
    for i in range(68):
        cum = cum + math.sqrt((predPts[i][0] - labelPts[i][0])**2 + (predPts[i][1] - labelPts[i][1])**2);

    error.append(cum/(68*interocular_distance))

print error

# for i in range(len(predictions)):
#     detected_points      = predictions[i]
#     ground_truth_points  = test_labels[i]

#     interocular_distance = norm(test_labels(37,:)-test_labels(46,:));
    
#     sum=0;
#     for j=1:num_of_points
#         sum = sum+norm(detected_points(j,:)-ground_truth_points(j,:));
#     end
#     error_per_image(i) = sum/(num_of_points*interocular_distance);
# end

# end
