import numpy as np
import matplotlib.pyplot as plt

def normalise(data):
  max=np.amax(data)
  min=np.amin(data)
  dataRange=max-min
  data -= min
  data /= dataRange
  return data

np.set_printoptions(suppress=True)

anomalies=np.loadtxt("climateGraph.txt",skiprows=5, delimiter="     ")

anomalies=anomalies[:,0:2]

anomalies[:,1] = normalise(anomalies[:,1])

print("normalised data")
print(anomalies)

# Graphs and charts

fig = plt.figure()
ax = fig.add_subplot(2,2,1)
fig.suptitle("title")
ax.plot(anomalies[:,0],anomalies[:,1])

cities = ["London",	"New York",	"Beijing",	"Sydney",	"Honolulu",	"Sao Paulo",	"Cape Town",	"Riyadh"]

cityData = [1.25,	0.41,	1.93,	0.31,	1.13,	1.64,	1.41,	1.93
]

ax2=fig.add_subplot(2,1,2)
ax2.bar(cities,cityData,color="yellow")

countries = ["China",	"USA","India",	"Russia",	"Japan",	"Germany",	"S Korea",	"Iran",	"Canada",	"S Arabia"]

co2 = np.array([9040,	4998,	2066,	1469,	1142,	730,	586,	552,	549,	531])
'''
co2 /= 32288
co2 *= 100
co2 = np.append()
'''

ax4 = fig.add_subplot(2,2,2)
ax4.pie(co2, labels=countries)

plt.show()
plt.close()
