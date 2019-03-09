import math


def bhatt_distance(a, b):
	"""Caculate to Bhattacharyya Distance"""
	if not len(a) == len(b):
		raise ValueError("a and b must be of the same size")
    
    score = -math.log(sum(math.sqrt(u * w) for u, w in zip(a, b)))
    return score

def getNeibor(data, dataSet, e):
	"""Get the ε-Area"""
    res = []
    for i in range(shape(dataSet)[0]):
        if b_distance(data , dataSet[i])<e:
            res.append(i)
    return res

def DBSCAN(dataSet , e , minPts):
	"""Density-Based Spatial Clustering of Applications with Noise algo"""
    coreObjs = {}# Initialize the core object(index)
    C = {}
    n = shape(dataSet)[0]
    # Find all core objects, the key is the index of the core object, and the \
	# value is the inde of the object in the ε-neighborhood.
    for i in range(n):
        neibor = getNeibor(dataSet[i] , dataSet , e)
        if len(neibor)>=minPts:
            coreObjs[i] = neibor
    oldCoreObjs = coreObjs.copy()
    k = 0 # Initialize the number of clusters
    notAccess = range(n)#Initialize unvisited sample collection (index)
    while len(coreObjs)>0:
        OldNotAccess = []
        OldNotAccess.extend(notAccess)
        cores = coreObjs.keys()
        # Randomly select a core object
        randNum = random.randint(0,len(cores))
        core = cores[randNum]
        queue = []
        queue.append(core)
        notAccess.remove(core)
        while len(queue)>0:
            q = queue[0]
            del queue[0]
            if q in oldCoreObjs.keys() :
                delte = [val for val in oldCoreObjs[q] if val in notAccess]#Δ = N(q)∩Γ
                queue.extend(delte)# Add the sample in Δ to the queue Q
                notAccess = [val for val in notAccess if val not in delte]#Γ = Γ\Δ
        k += 1
        C[k] = [val for val in OldNotAccess if val not in notAccess]
        for x in C[k]:
            if x in coreObjs.keys():
                del coreObjs[x]
    return C
