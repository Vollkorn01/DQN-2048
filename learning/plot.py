import json
import matplotlib.pyplot as plt
import numpy as np

with open('./learning/data//output.txt', "r") as infile:
    mylist = json.load(infile)


#convert from list zu nparray
a = np.asarray(mylist)

plt.figure()
plt.ylabel('highest Value')
plt.xlabel('Episode')

plt.plot(a[:,0], a[:,1], 'r.-')
plt.show()

