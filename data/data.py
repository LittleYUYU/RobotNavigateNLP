# script for analyzing dataset

import matplotlib.pyplot as plt

def helperLen(file_path):
	""" a helper function for counting length."""
	lenStat = []
	with open(file_path) as f:
		line = f.readline().strip()
		while(line != '\n' and line != ''):
			leng = len(line.split())
			# if leng in lenStat.keys():
			# 	lenStat[leng] += 1
			# else:
			# 	lenStat[leng] = 1
			lenStat.append(leng)
			line = f.readline().strip()
	return lenStat


def getBucketInfo(flag):
	""" statistic information for deciding bucket sizes."""
	lenPair = []
	path = "data0/" + flag + "/data"
	srce_file = path + ".srce"
	trgt_file = path + ".trgt"
	lenSrce = helperLen(srce_file)
	lenTrgt = helperLen(trgt_file)
	lenPair.extend(zip(lenSrce, lenTrgt))
	return lenPair

if 1:
	l1 = getBucketInfo("train")
	l2 = getBucketInfo("dev")
	l3 = getBucketInfo("test")

	lenAll = l1
	lenAll += l2
	lenAll += l3

	pair0 = zip(*lenAll)[0]
	pair1 = zip(*lenAll)[1]

	print "srce min = ", min(pair0), " max = ", max(pair0)
	print "trgt min = ", min(pair1), " max = ", max(pair1)

	#plot
	plt.scatter(pair0, pair1)
	plt.show()