import json
import matplotlib.pyplot as plt
import numpy as np


with open("output.txt", "r") as infile:
    mylist = json.load(infile)

# myList = [[1,2],[3,3],[4,4],[5,2]]
# np.array(myList).dump(open('array.txt', 'wb'))

#convert from list zu nparray
a = np.asarray(mylist)


# a = np.load(open('array.txt', 'rb'))

plt.figure()
plt.ylabel('highest Value')
plt.xlabel('Episode')

plt.plot(a[:,0], a[:,1], 'r.-')
plt.show()

