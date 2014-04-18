import time,os
import numpy as np
import scipy as sp
import sys
import gensim


from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.mixture import  GMM
import cPickle
import sklearn.naive_bayes  #MultinomialNB
import sklearn.svm  #SVC

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


def clutering_advs(filePathC,filePathW, algoCluster,numT=10, drop=False):
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
	if (drop):
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
def change_full_to_sparse_and_stock(foldPath,pickleX,pickleY):
	cTok=''
	filenames=[]
	usr_ids=[]
	for file in os.listdir(foldPath):
		if file.startswith('matrix'):
			filenames.append(foldPath+'/'+file)

	lenF= len(filenames)

	i=0
	X=''
	Y=''

	for file in filenames:
		f=open(file)
		
		# store the md5
		# -----------------------
		if i!=0:
			f.readline()
		else:
			md5=f.readline()
			md5s= md5.strip().split()
			#remove de string 'User_id','Score','CONV_id'
			del md5s[0]
			del md5s[-1]
			del md5s[-1]
		#--------------------------


		data=np.loadtxt(f)

		usr_ids.extend(data[:,0])

		# y=  data[:,-1].astype(int).reshape(data.shape[0],1)
		y=  data[:,-1].astype(int)

		#find colomn to keep, delete userid [0], score [-2]
		if i==0:
			cTok= [j for j in xrange(data.shape[1])]
			del cTok[-1]
			del cTok[-1]
			del cTok[0]
		data= data[:, cTok]

		# X.append(data)
		# Y.append(y)


		#how many nb of domain he has each person has visited
		#print np.sum(data,1)
		#print data[0],y

		if i!=0:
			X= sp.sparse.vstack((X,sp.sparse.csr_matrix(data).astype(int)))
			Y= np.hstack((Y,y))
		else:
			X=sp.sparse.csr_matrix(data).astype(int)
			Y=y

		# print X.shape
		# print Y.shape
		
		print 'done small stock '+ str(i)
		
		i+=1
		
	#store the information: X, Y, usr_ids, md5(domain+deplacement)
	cPickle.dump(usr_ids, open(pickleUsrids, "w"))
	cPickle.dump(md5s, open(pickleMd5s, "w"))
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

def clf_return_score_and_proba(X,Y, test_size,tuned_parameters,ppfile,score,algo,cv):
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)

	print("# Tuning hyper-parameters for %s" % score)
	print()
	if cv>0:
		clf = GridSearchCV(algo(), tuned_parameters, cv=cv, scoring=score,refit=True)
		
		st=time.time()
		
		clf.fit(X_train, Y_train)

		st2=time.time()
		
		print 'train use: ', st2-st , ' time'
		
		print("Best parameters set found on development set:")
		print(clf.best_estimator_)

		# print("Grid scores on development set:")
		
		for params, mean_score, scores in clf.grid_scores_:
		    print("%0.3f (+/-%0.03f) for %r"
		          % (mean_score, scores.std() / 2, params))
	else:
		clf=algo()
		st=time.time()
		clf.fit(X_train, Y_train)
		st2=time.time()

	# print("Detailed classification report:")
	# 
	# print("The model is trained on the full development set.")
	# print("The scores are computed on the full evaluation set.")
	
	# y_true, y_pred = Y_test, clf.predict(X_test)

	#print(classification_report(y_true, y_pred))
	print()

	pp= clf.predict_proba(X_test)

	st3=time.time()
	print 'predict proba use: ', st3-st2 , ' time'


	pc= clf.predict(X_test)

	perf= sum( pc==Y_test)/float(len(Y_test))
	np.savetxt(ppfile, pp, delimiter=",")

	print pp

	return pc,perf 



'''
phase clustering advertisers to regroup users in group:
'''

def find_key_if_value_in_dict(mydict, v):
	res=''
	for key, value in mydict.items():
		if v in value:
			res=key
			break
	return res

def regroup_by_given_clusters(cl_dict, X,Y):
	X_dict= {}
	Y_dict= {}

	list_id_per_cl={}

	for cl_id in cl_dict:
		list_id_per_cl[cl_id]=[] 

	if X.shape[0] != len(Y):
		print 'error: X length not correspond Y length'
	
	for i in xrange(len(Y)):
		cl=find_key_if_value_in_dict(cl_dict,Y[i])
		# print cl
		if cl !='':
			list_id_per_cl[cl].append(i)

	# print list_id_per_cl

	for cl_id in cl_dict:
		#seperate X by row with index
		X_dict[cl_id]=X[list_id_per_cl[cl_id],:]
		Y_dict[cl_id]=Y[list_id_per_cl[cl_id],]


	# print X_dict, Y_dict
	return X_dict, Y_dict

#@param: X: a sparse matrix format, for ex. csr
#@return: X_red: reduced matrix
#		  per_red: percentage reduced

#problem: should i use X directly or another format
def reduce_nonuseful_attr(X):
	n,m=X.shape
	#sum X in column:0, row:1
	nonZeroCount=(X.sum(0)!=0)
	indNonZero=np.asarray(nonZeroCount).reshape(-1)
	

	print len(indNonZero)

	per_red= (m-sum(indNonZero))/float(m)

	X_red= X[:,indNonZero]

	print type(X_red), X_red.shape

 	return X_red,per_red

if __name__=="__main__":

	foldPath='C:/Users/tradelab/Documents/donnes/2014-04-07'
	
	pickleX=foldPath+'/X.pickle'
	pickleY=foldPath+'/Y.pickle'
	pickleMd5s=foldPath+'/md5s.pickle'
	pickleUsrids=foldPath+'/usr_ids.pickle'

	"""
	----------------------store and read information----------------------------------------'
	"""
	#store info.  X is in sparse matrix format: coo
	# change_full_to_sparse_and_stock(foldPath,pickleX,pickleY)

	#reading info 
	X = cPickle.load(open(pickleX))
	Y= 	cPickle.load(open(pickleY))
	md5s = cPickle.load(open(pickleMd5s))
	usr_ids=cPickle.load(open(pickleUsrids))

	print 'X type is ', type(X)

	print 'Y type is ',	type(Y)
	print 'X shape is ', X.shape
	print 'Y shape is ', Y.shape
	print 'mds length is ', len(md5s)
	print 'usr_ids length is ',	len(usr_ids)

	#important: when store, X is coo format
	#			shold be changed to csr format for slicing
	X=X.tocsr()
	print 'change X type to ', type(X), ' for slicing'



	"""
	----------------------train and test----------------------------------------'
	"""
	# tuned_parameters = {'C':[0.01,0.1,1]}
	tuned_parameters = {'alpha':[0.01,0.1,1,10,100]}
	test_size=0.3
	score = 'accuracy'
	ppfile='post_porba.csv'
	algo=getattr(sklearn.naive_bayes,'MultinomialNB')
	pc,perf=clf_return_score_and_proba(X,Y, test_size,tuned_parameters,foldPath+'/'+ppfile,score,algo,cv=0)
	
	print 'perf with all data in a pool ', perf
	
	
	"""
	----------------------clustering----------------------------------------'
	"""
	filenameC='all_classes.txt'
	filenameW='all_weights.txt'
	filePathC=foldPath+'/'+filenameC
	filePathW=foldPath+'/'+filenameW
	algoCluster=GMM(n_components=1, covariance_type='full',  random_state=0,n_iter=10000)

	clusterIds,clusters,clustersWt= clutering_advs(filePathC,filePathW, algoCluster,numT=10)
	print clusterIds
	print clusters
	print clustersWt


	"""
	----------------------regroupment----------------------------------------'
	"""
	X_dict, Y_dict=regroup_by_given_clusters(clusters, X,Y)
	per_red={}
	for cl in X_dict:
		print type(X_dict[cl])
		X_dict[cl],per_red[cl]=reduce_nonuseful_attr(X_dict[cl])
	print 'the matrix has reduced '
	print per_red



	pc_dict={}
	perf_dict={}
	for cl in X_dict:
		print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
		# print X_dict[cl],Y_dict[cl]
		# print type(X_dict[cl]),X_dict[cl].shape
		# print type(Y_dict[cl]),Y_dict[cl].shape
		if cl==0:
			pc_dict[cl],perf_dict[cl]=clf_return_score_and_proba(X_dict[cl],Y_dict[cl], test_size,tuned_parameters,ppfile,score,algo,cv=0)
	print perf_dict
