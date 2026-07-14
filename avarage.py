import numpy as np
male_heights=np.array([168,172,190,159,170])
mean_=np.mean(male_heights)
var_=np.mean((male_heights-mean_)**2)
print("mean: ",mean_)
print("varience: ",var_)
