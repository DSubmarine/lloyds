import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import defaultdict

def clusterpoints(X, newcenters):
    clusters  = {}
    for x in X:
        bestnewcenterskey = min([(i[0], np.linalg.norm(x-newcenters[i[0]])) \
                    for i in enumerate(newcenters)], key=lambda t:t[1])[0]
        try:
            clusters[bestnewcenterskey].append(x)
        except KeyError:
            clusters[bestnewcenterskey] = [x]
    return clusters
 
def repositioncentre(newcenters, clusters):
    newnewcenters = []
    jcl = 0
    keys = sorted(clusters.keys())
    for k in keys:
        temp = np.mean(clusters[k], axis = 0)
        for point in clusters[k]:
            jcl = jcl + ((point[0] - temp[0])**2 + (point[1] - temp[1])**2)
        newnewcenters.append(temp)
    return (jcl,newnewcenters)
 

def findcenters(X, K):
    oldcenters = X[np.random.choice(X.shape[0], K, replace=False), :]
    newcenters = X[np.random.choice(X.shape[0], K, replace=False), :]
    jcl = int()
    while not (set([tuple(a) for a in newcenters]) == set([tuple(a) for a in oldcenters])):
        oldcenters = newcenters
        clusters = clusterpoints(X, newcenters)
        jcl, newcenters = repositioncentre(oldcenters, clusters)
    return (jcl, newcenters)

def main():
    df = pd.read_csv('collegedataset.txt',sep = "\t", header = None)
    df.columns = ['X','Y','timestamp']
    jcllist = list()
    number = list()
    uniquenodes = list(df['X'].unique())
    randomnodes = random.sample(uniquenodes,100)
    newdf = df[df['X'].isin(randomnodes)]
    tlist = list()
    timedict = defaultdict(list)
    
    for index,row in newdf.iterrows():
        timedict[row['X']].append(row['timestamp'])
        tlist.append(row['timestamp']/100000)
    
    df = pd.DataFrame() 
    df['X'] = tlist
    df['Y'] = np.zeros_like(tlist)
    
    for K in range(1,6):
        minjcl = 1000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001000000000
        for i in range(1,200):
            jcl, centers = findcenters(df.values, K)
            if(minjcl > jcl):
                minjcl = jcl
        jcllist.append(minjcl)
        number.append(K)
        print("im at "+str(K)+" with "+str(minjcl))
    plt.plot(number,jcllist)
        
        
          
if __name__ == "__main__":
    main()