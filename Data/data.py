import numpy as np
mydata = np.loadtxt("mycsv.csv", delimiter=",", dtype=int)
print(mydata)

def normalise(myInput):
  mini = np.amin(myInput)
  myInput -= mini
  maxi = np.amax(myInput)
  myInput = myInput / maxi
  myInput *= 100
  return myInput


print(normalise(mydata))