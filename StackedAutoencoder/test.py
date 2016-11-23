import numpy as np

a = np.asarray([[1, 2], [0, 3]])
b = np.asarray([[5, 2], [1, 3]])
# a = np.asarray([[1, 2]])
# b = np.asarray([[5, 2]])

print a
# print a.shape
print ",,,,,,,,,,,,,,,,,,,,"
print b
print ",,,,,,,,,,,,,,,,,,,,"
# print b.shape
# print np.dot(a, b)
print ",,,,,,,,,,,,,,,,,,,,"

print np.cross(a, b)