import time,os
import numpy as np
import scipy as sp
import sys


from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.mixture import  GMM
import cPickle
from sklearn.naive_bayes import MultinomialNB

def find_max_index(mylist):
	maxV=-sys.maxint
	indmax=-1
	for i in xrange(len(mylist)):
		if maxV< mylist[i]:
			indmax=i
			maxV= mylist[i]
	return indmax


def read_numbers_form_column(filePath):
	data = [int(line.strip()) for line in open(filePath, 'r')]
	return data


def find_max_index(mylist):
	maxV=-sys.maxint
	indmax=-1
	for i in xrange(len(mylist)):
		if maxV< mylist[i]:
			indmax=i
			maxV= mylist[i]
	return indmax

def read_numbers_form_column(filePath):
	data = [int(line.strip()) for line in open(filePath, 'r')]
	return data


def clutering_advs(filePathC,filePathW, algoCluster,numT=10):
	#classes and weights
	cls=read_numbers_form_column(filePathC)
	wts=read_numbers_form_column(filePathW)

	clwt=  [   i   for i in sorted(zip(cls,wts), key=lambda t:t[1]) ]

	wts=np.array(wts)

	scores=np.zeros(numT)
	for i in xrange(numT):
		algoCluster.set_params(n_components=i+1)
		algoCluster.fit(wts)
		scores[i]=algoCluster.bic(wts)
	print scores


	maxIndex=find_max_index(-scores)
	print 'finally there are '+ str(maxIndex+1)+ ' clusters'

	algoCluster.set_params(n_components=maxIndex+1)
	algoCluster.fit(wts)

	#get clusterId of each user
	clusterIds=algoCluster.predict(wts)

	


	clusters= {}
	clustersWt={}

	len_cluster=maxIndex+1
	for i in xrange(len_cluster):
		clusters[i]=[]
		clustersWt[i]=[]


	#assign user to cluster
	for i in xrange(len(cls)):
		clusters[clusterIds[i]].append(cls[i])
		clustersWt[clusterIds[i]].append(wts[i])

	for key,value in clustersWt.items():
		clustersWt[key]=sorted(value)

	# print clusters
	# print clustersWt

	# import matplotlib.pyplot as plt
	# plt.plot(list(range(1,numT+1)), scores,'go')
	# plt.show()

	#statistics of weights in each cluster
	means=np.zeros(len_cluster)
	for i in xrange(len_cluster):
		means[i]=sum(clustersWt[i])/float(len(clustersWt[i]))
	# print means

	#delect the most big and most small

	G=find_max_index(means)
	S=find_max_index(-means)

	to_drop=[G,S]

	del clusters[G]
	del clusters[S]
	del clustersWt[G]
	del clustersWt[S]
	return clusterIds,clusters,clustersWt




'''
stock the matrix in scipy sparse matrix format, reduce from 600M to 12M
'''

foldPath='../data/matrix 2014_04_13'

def change_full_to_sparse_and_stock(foldPath,pickleX,pickleY):
	cTok=''
	filenames=[]
	for file in os.listdir(foldPath):
		if file.startswith('matrix'):
			filenames.append(foldPath+'/'+file)

	lenF= len(filenames)

	i=0
	X=''
	Y=''

	for file in filenames:
		f=open(file)
		f.readline()
		data=np.loadtxt(f)
		# y=  data[:,-1].astype(int).reshape(data.shape[0],1)
		y=  data[:,-1].astype(int)

		#find colomn to keep, delete userid [0], score [-2]
		if i==0:
			cTok= [j for j in xrange(data.shape[1])]
			del cTok[-1]
			del cTok[0]

		data= data[:, cTok]

		# X.append(data)
		# Y.append(y)


		#how many nb of domain he has each person has visited
		#print np.sum(data,1)
		#print data[0],y
		print 'file ',i
		if i!=0:
			X= sp.sparse.vstack((X,sp.sparse.coo_matrix(data)))
			Y= np.hstack((Y,y))
		else:
			X=sp.sparse.coo_matrix(data)
			Y=y

		# print X.shape
		# print Y.shape
		
		print 'done stock petit '+ str(i)
		
		i+=1

	cPickle.dump(X, open(pickleX, "w"))
	cPickle.dump(Y, open(pickleY, "w"))

	print 'done stock'






def change_full_to_stock(foldPath,pickleX,pickleY):
	cTok=''
	filenames=[]
	for file in os.listdir(foldPath):
		if file.startswith('matrix'):
			filenames.append(foldPath+'/'+file)

	lenF= len(filenames)

	i=0
	X=''
	Y=''

	for file in filenames:
		f=open(file)
		f.readline()
		data=np.loadtxt(f)
		# y=  data[:,-1].astype(int).reshape(data.shape[0],1)
		y=  data[:,-1].astype(int)

		#find colomn to keep, delete userid [0], score [-2]
		if i==0:
			cTok= [j for j in xrange(data.shape[1])]
			del cTok[-1]
			del cTok[0]

		data= data[:, cTok]

		# X.append(data)
		# Y.append(y)


		#how many nb of domain he has each person has visited
		#print np.sum(data,1)
		#print data[0],y
		print 'file ',i
		if i!=0:
			X= np.vstack((X,data))
			Y= np.hstack((Y,y))
		else:
			X=data
			Y=y

		# print X.shape
		# print Y.shape
		
		print 'done stock petit '+ str(i)
		
		i+=1


	# cPickle.dump(X, open(pickleX, "w"))
	# cPickle.dump(Y, open(pickleY, "w"))

	print 'done stock'
	return X,Y





'''
phase training algorithm of classification:
'''

# tuned_parameters = {'C':[0.01,0.1,1]}
tuned_parameters = {'alpha':[0.1,1]}
test_size=0.3
score = 'precision'
ppfile='post_porba.csv'



def clf_return_score_and_proba(X_train, X_test, Y_train, Y_test,tuned_parameters,ppfile ):
	

	print("# Tuning hyper-parameters for %s" % score)
	print()

	clf = GridSearchCV(MultinomialNB(alpha=1), tuned_parameters, cv=5, scoring=score)
	# clf=MultinomialNB(alpha=1)
	st=time.time()
	
	clf.fit(X_train, Y_train)

	st2=time.time()
	
	print 'train use: ', st2-st , ' time'
	
	# print("Best parameters set found on development set:")
	# print()
	# print(clf.best_estimator_)
	# # print()
	# print("Grid scores on development set:")
	# print()
	# for params, mean_score, scores in clf.grid_scores_:
	#     print("%0.3f (+/-%0.03f) for %r"
	#           % (mean_score, scores.std() / 2, params))
	# print()

	# print("Detailed classification report:")
	# print()
	# print("The model is trained on the full development set.")
	# print("The scores are computed on the full evaluation set.")
	# print()
	# y_true, y_pred = Y_test, clf.predict(X_test)

	#print(classification_report(y_true, y_pred))
	print()

	pp= clf.predict_proba(X_test)

	st3=time.time()
	print 'predict proba use: ', st3-st2 , ' time'


	pc= clf.predict(X_test)

	perf= sum( pc==Y_test)/float(len(Y_test))
	np.savetxt(ppfile, pp, delimiter=",")
	return perf 



'''
phase clustering advertisers to regroup users in group:
'''
filenameC='all_classes.txt'
filenameW='all_weights.txt'
filePathC=foldPath+'/'+filenameC
filePathW=foldPath+'/'+filenameW




# clusterIds,clusters,clustersWt= clutering_advs(filePathC,filePathW, algoCluster,numT=10)
# print clusterIds
# print clusters
# print clustersWt





# 


# def regroup_by_given_clusters(clusters, X,Y):
# 	X_dict= {}
# 	Y_dict= {}

# 	if len(X) != len(Y):
# 		print 'error: X length not correspond Y length'
# 		break
	
# 	for i in xrange(len(X)):
		


if __name__=="__main__":

	pickleX=foldPath+'/X.pickle'
	pickleY=foldPath+'/Y.pickle'

	#comment it if we have already stocked the data in pickle format
	change_full_to_sparse_and_stock(foldPath,pickleX,pickleY)

	'''
	reading the X and Y from cPickle 
	'''
	X = cPickle.load(open(pickleX))
	Y= 	cPickle.load(open(pickleY))	
	print X.shape, Y.shape

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)

	pc,perf=clf_return_score_and_proba(X,Y, test_size,tuned_parameters,foldPath+'/'+ppfile)
	print perf
	# algoCluster=GMM(n_components=1, covariance_type='full',  random_state=None,n_iter=10000)