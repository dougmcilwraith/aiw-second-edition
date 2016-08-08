import pdb
import csv
import re
import math
import numpy as np
from numpy import linalg as LA

import recsys.algorithm
from recsys.algorithm.factorize import SVD
from recsys.datamodel.data import Data
from recsys.datamodel.user import User
from recsys.datamodel.item import Item
from recsys.evaluation.prediction import RMSE, MAE
from recsys.utils.svdlibc import SVDLIBC

#Helpers

def read_user_data_from_ratings(data_file):
    data = Data()
    format = {'col':0, 'row':1, 'value':2, 'ids': 'int'}    
    data.load(dat_file, sep='::', format=format)
    
    userdict = {}
    for d in data.get():
        if d[2] in userdict:
            user = userdict[d[2]] 
        else:
            user = User(d[2]) 
        
        user.add_item(d[1],d[0])
        userdict[d[2]] = user
    return userdict

def read_item_data(filename):
    itemdict = {}
    f = open(item_file,'r')
    for r in f:
        p = r.split('::')
        if p[0] in itemdict:
            print "Duplicate!", p[0]
        else:
            item = Item(p[0])
        idat = {}
        if len(p)>1:
            idat['Title']  = p[1] 
        if len(p)>2: 
            idat['Genres'] = p[2]
        item.add_data(idat)
        itemdict[p[0]] = item
    return itemdict

def items_reviewed(user_id, userdict):
    return [x[0] for x in userdict[user_id].get_items()]

def get_score_item_reviewed(user_id,item_id, userdict):
    #return the first score in the list that matches the id
    return [x[1] for x in userdict[user_id].get_items() if x[0]==int(item_id)][0] 

def get_name_item_reviewed(user_id,userdict,itemdict):
    l =  [(x[0],itemdict[str(x[0])].get_data()['Title'],itemdict[str(x[0])].get_data()['Genres'],x[1]) for x in userdict[user_id].get_items()]
    return sorted(l,key=lambda x: x[3], reverse=True)

#Pass a list of (id, title, filmcategory,rating) and a filter_value (0,5]
#Returns a  list of [category,#count of ratings>filter_value],
def movies_by_category(movie_list,filter_value):
    d = {}
    for x in movie_list:
        if x[3]>filter_value:
            if str(x[2]) in d:
                d[str(x[2])]+=1
            else:
                d[str(x[2])]=1
    dictlist = []
    for key, value in d.iteritems():
        temp = [key,value]
        dictlist.append(temp)
        
    return dictlist

#End Helpers

#3.2
def similarity(user_id_a,user_id_b,sim_type=0):
    user_a_tuple_list = userdict[user_id_a].get_items()
    user_b_tuple_list = userdict[user_id_b].get_items()
    common_items=0
    sim = 0.0
    for t1 in user_a_tuple_list:
        for t2 in user_b_tuple_list:
            if (t1[0] == t2[0]):
                common_items += 1
                sim += math.pow(t1[1]-t2[1],2)
    if common_items>0:
        sim = math.sqrt(sim/common_items)
        sim = 1.0 - math.tanh(sim)
        if sim_type==1:
            max_common = min(len(user_a_tuple_list),len(user_b_tuple_list))
            sim = sim * common_items / max_common
    print "User Similarity between",names[user_id_a],"and",names[user_id_b],"is", sim
    return sim #If no common items, returns zero

#3.1
# load movielens data
dat_file = 'ratings-11.dat'
item_file = 'movies-11.dat'

names = ['Frank','Constantine','Catherine']

userdict = read_user_data_from_ratings(dat_file) #Build a userdict with users and ratings 
itemdict = read_item_data(item_file) #Build an item, info dict 

similarity(0,1,sim_type=0)
similarity(0,1,sim_type=1)
similarity(0,2,sim_type=0)
similarity(1,2,sim_type=0)
similarity(2,1,sim_type=0)
similarity(0,0,sim_type=0)
similarity(0,0,sim_type=1)

#3.7
class RatingCountMatrix:
    user_id_a = None
    user_id_b = None
    matrix = None
    
    #Instantiate with two users and the total possible number of ratings (eg 5 in the movielens case)
    def __init__(self, user_id_a, user_id_b):
        num_rating_values = max([x[0] for x in data])
        self.user_id_a = user_id_a
        self.user_id_b = user_id_b
        self.matrix = np.empty((num_rating_values,num_rating_values,))
        self.matrix[:] = 0
        self.calculate_matrix(user_id_a,user_id_b)
        
    def get_shape(self):
        a = self.matrix.shape
        return a
    
    def get_matrix(self):
        return self.matrix
    
    def calculate_matrix(self,user_id_a, user_id_b):
        for item in items_reviewed(user_id_a, userdict):
            if int(item) in items_reviewed(user_id_b, userdict):
                i = get_score_item_reviewed(user_id_a,item, userdict)-1 #Need to subtract 1 as indexes are 0 to 4 (5 items)
                j = get_score_item_reviewed(user_id_b,item, userdict)-1
                self.matrix[i][j] +=1
                
        
    #Total number of items that the users have both ranked        
    def get_total_count(self):
        return self.matrix.sum()
    
    #Total number of items that they both agree on
    def get_agreement_count(self):
       return np.trace(self.matrix) #sum across the diagonal

#3.6
class SimilarityMatrix:
   
    similarity_matrix = None
    
    def __init__(self):
        self.build()
   
    def build(self):
        self.similarity_matrix = np.empty((len(userdict),len(userdict),))
    
        for u in range(0,len(userdict)):
            for v in range(u+1,len(userdict)):
                rcm = RatingCountMatrix(int(u),int(v))
                if(rcm.get_agreement_count()>0):
                    self.similarity_matrix[u][v] = rcm.get_agreement_count()/rcm.get_total_count()
                else:
                    self.similarity_matrix[u][v] = 0
            self.similarity_matrix[u][u]=1
            
    def get_user_similarity(self,user_id1, user_id2):
        return self.similarity_matrix[min(user_id1,user_id2),max(user_id1,user_id2)] # Due to upper traingular form 

# 3.5:
def predict_rating(user_id, item_id): 
    estimated_rating = None;
    similarity_sum = 0;
    weighted_rating_sum = 0;
    
    if (int(item_id) in items_reviewed(user_id,userdict)):
        return get_score_item_reviewed(user_id,item_id,userdict)
    else:
        for u in userdict.keys():
            if (int(item_id) in items_reviewed(u,userdict)):
                item_rating = get_score_item_reviewed(u,item_id,userdict)
                user_similarity = similarity_matrix.get_user_similarity(user_id,u)
                weighted_rating = user_similarity * item_rating
                weighted_rating_sum += weighted_rating
                similarity_sum += user_similarity
                
        if (similarity_sum > 0.0):
            estimated_rating = weighted_rating_sum / similarity_sum
    
    return estimated_rating

# 3.4:
def recommend(user_id, top_n): 
    #[(item,value),(item1, value1)...]
    recommendations = []
    for i in itemdict.keys():
        if (int(i) not in items_reviewed(int(user_id),userdict)):
            recommendations.append((i,predict_rating(user_id, i))) #only get those not predicted.
    recommendations.sort(key=lambda t: t[1], reverse=True)
    return recommendations[:top_n]

#3.3:
data = Data()
format = {'col':0, 'row':1, 'value':2, 'ids': 'int'}
    # About format parameter:
    #   'row': 1 -> Rows in matrix come from column 1 in ratings.dat file
    #   'value': 2 -> Values (Mij) in matrix come from column 2 in ratings.dat file
    #   'ids': int -> Ids (row and col ids) are integers (not strings)
data.load(dat_file, sep='::', format=format)

similarity_matrix = SimilarityMatrix()
recommend(0,10)
recommend(1,10)
recommend(2,10)

##################
#Now we do SVD
##################

#3.8
svd = SVD()
recsys.algorithm.VERBOSE = True

dat_file = './ml-1m/ratings.dat'
item_file = './ml-1m/movies.dat'

data = Data()
data.load(dat_file, sep='::', format={'col':0, 'row':1, 'value':2, 'ids': int})

items_full = read_item_data(item_file)
user_full = read_user_data_from_ratings(dat_file)

svd.set_data(data)

#3.9
k = 100
svd.compute(k=k, min_values=10, pre_normalize=None, mean_center=True, post_normalize=True)
films = svd.recommend(10,only_unknowns=True, is_row=False) #movies that user 10 should see (That they haven't rated)

#3.10
[items_full[str(x[0])].get_data() for x in films]

#3.11
get_name_item_reviewed(10,user_full,items_full)

#3.12
items_full[str(2628)].get_data()
users_for_star_wars = svd.recommend(2628,only_unknowns=True)
users_for_star_wars

#3.13
movies_reviewed_by_sw_rec  =[get_name_item_reviewed(x[0],user_full,items_full) for x in users_for_star_wars]
movies_flatten = [movie for movie_list in movies_reviewed_by_sw_rec for movie in movie_list]
movie_aggregate = movies_by_category(movies_flatten, 3)
movies_sort = sorted(movie_aggregate,key=lambda x: x[1], reverse=True)
movies_sort

#3.14
from recsys.evaluation.prediction import RMSE
err = RMSE()
for rating, item_id, user_id in data.get():
    try:
        prediction = svd.predict(item_id, user_id)
        err.add(rating, prediction)
    except KeyError, k:
        continue

print 'RMSE is ' + str(err.compute())
