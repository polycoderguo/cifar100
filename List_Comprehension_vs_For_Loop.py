import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
raw_data = np.load('raw_train.npy')

def forloop_2(x):
	percentage = []	
	for img in x:
    		#color = set()
		#clr = [ str(p) for p in [img.reshape(1024,3)] ]	
		pix = img.reshape(1024,3)
		clr = [str(p) for p in pix]
		color = list(set(clr))
        	#print(len(color))
       		 #print(len(clr))
		per = len(color)/1024
		percentage.append(per)
	percentage = np.array(percentage)
	return percentage
def forloop_memory_2(x):
	percentage = np.zeros(x.shape[0])
	for i, img in enumerate(x):
		#color = set()
		#clr = [ str(p) for p in [img.reshape(1024,3)] for img in x]
		pix = img.reshape(1024,3)
		clr = [str(p) for p in pix]
		color = list(set(clr))
		#print(len(color))
		#print(len(clr))
		per = len(color)/1024
		percentage[i] = per
	return percentage

def list_comprehension_2(x):
	per = [len(set([str(pix) for pix in img.reshape(1024,3)]))/1024 for img in x]
	return np.array(per)


times = []
nums = [num for num in range(0,raw_data.shape[0], 600)]
for num in nums:
	start_time = time.time()
	result1 = forloop_2(raw_data[:num])
	times.append(time.time()-start_time)
df = pd.DataFrame(nums, columns={'nums'})
df['time_1'] = times
times = []
for num in nums:	
	start_time = time.time()
	result2 = forloop_memory_2(raw_data[:num])
	times.append(time.time()-start_time)
        #df = pd.DataFrame(nums, columns={'nums'})
df['time_2'] = times
times = []
for num in nums:
	start_time = time.time()
	result3 = list_comprehension_2(raw_data[:num])
	times.append(time.time()-start_time)
	#df = pd.DataFrame(nums, columns={'nums'})
df['time_3'] = times


df.to_csv('List_Comprehension_vs_For_Loop.csv')


