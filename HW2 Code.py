import replit
import time
import random
import matplotlib.pyplot as plt
import statsmodels.api as sm

dice_avg = []
step = 1
i = 1
while i < 100000:
  dice_rolls = []
  for j in range(i):
    dice_rolls.append(random.randint(1,6))
  dice_avg.append(sum(dice_rolls)/float(len(dice_rolls)))
  i += step
  step += 1
plt.plot(dice_avg,'bo')
plt.savefig('plot.png')