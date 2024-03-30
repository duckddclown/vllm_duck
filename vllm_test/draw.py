import numpy as np
import matplotlib.pyplot as plt

hit_rate = np.array([0.446,1.399,2.126,2.767,3.034,3.936,4.511,5.397,6.285,6.712])
performance = [411.57,413.04,412.53,411.66,411.71,407.85,413.55,415.81,417.88,419.32]
prob = np.arange(1,11) / 10

plt.plot(prob,performance)
plt.xlabel("Probability of Selecting from Pool")
plt.ylabel("Items/s")

print("good")