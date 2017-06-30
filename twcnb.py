import string
from string import digits

import numpy as np
import random as random

# running the TWCNB algorithm
# from http://machinelearning.wustl.edu/mlpapers/paper_files/icml2003_RennieSTK03.pdf

big_file = open("validation_training.txt")
test_file = open("validation_test.txt")
papers = big_file.read().split("FEDERALIST No.")
disputed = test_file.read().split("FEDERALIST No.")

del papers[0]
del disputed[0]
# make them all lower case:
papers = map(str.lower, papers)
disputed = map(str.lower, disputed)
# classify each paper:
# 0 = hamilton, 1 = madison
paper_author = [0] * len(papers)


#for each paper: want to classify them. Also just remove stuff up to the name
a_names = ["hamilton", "madison", "jay", "madison, with hamilton"]

def remove_author_name(words, i):
    r = -1
    try:
        r = words.index("jay")
        paper_author[i] = 2
    except:
        try:
            r = words.index("madison, with hamilton")
            paper_author[i] = 3
        except:
            try:
                r = words.index("hamilton")
                paper_author[i] = 0
            except:
                try:
                    r = words.index("madison")
                    paper_author[i] = 1
                except:
                    pass
    return r

def remove_author_disputed(words, i):
    r = -1
    try:
        r = words.index("jay")
    except:
        try:
            r = words.index("madison, with hamilton")
        except:
            try:
                r = words.index("hamilton")
            except:
                try:
                    r = words.index("madison")
                except:
                    pass
    return r

print "Tokenizing text documents to set all to lower case and other punctuation"

for i in range(len(papers)):
    findex = remove_author_name(papers[i],i)
    papers[i] = papers[i][findex + len(a_names[paper_author[i]]):]
    papers[i] = papers[i].translate(None, string.punctuation) # get rid of punctuation
    papers[i] = papers[i].translate(None, digits) # get rid of numbers
    #print "author of paper " , i+1 , " is ", a_names[paper_author[i]]


for i in range(len(disputed)):
    findex = remove_author_disputed(disputed[i],i)
    disputed[i] = disputed[i][findex + 7:]
    disputed[i] = disputed[i].translate(None, string.punctuation)
    disputed[i] = disputed[i].translate(None, digits)

print "Finished normalizing"
print len(disputed)
print len(papers)

# now we have got the table of authors, and all the words. Need now to build up our word bag.
print "Building word list..."
word_list = []
#get word list for non-disputed papers (training)
for i in range(len(papers)):
    paper_tokens = papers[i].split()
    #print paper_tokens
    for j in range(len(paper_tokens)):
        if (paper_tokens[j] not in word_list):
            word_list.append(paper_tokens[j])
print "Finished building word list: no. of unique words: ", len(word_list)
# Don't need to add tokens for non-training.
##for i in range(len(disputed)):
##    paper_tokens = disputed[i].split()
##    #print paper_tokens	
##    for j in range(len(paper_tokens)):
##        if (paper_tokens[j] not in word_list):
##            word_list.append(paper_tokens[j])

#now we have word list which contains the list of all distinct words.

# counts the number of words for term j (word) in document i.
def number_of_occurrences(i, j):
    num_words = 0
    paper_tokens = i.split()
    for i in range(len(paper_tokens)):
        if (paper_tokens[i] == j):
            num_words = num_words + 1
    return num_words

def occurs(i,j):
    paper_tokens = i.split()
    for i in range(len(paper_tokens)):
        if (paper_tokens[i] == j):
            return True
    return False

# counts the number of training documents that contain the word j
def number_train_documents_contains(j):
    num_documents = 0
    for i in range(len(papers)):
        occ = occurs(papers[i],j)
        if occ == True:
            num_documents = num_documents + 1
    return num_documents

num_docs_feature_j = []
def create_num_train_doc_cache():
# We see that number_train_documents_contains is very slow, but it is global. So we create a cache.
    #f = open('num_train_doc2.txt', 'w')
    #print "Creating num_train_doc file"
    for i in range(len(word_list)):
        num_docs_feature_j.append(number_train_documents_contains(word_list[i]))
        #print >> f, num_docs_feature_j[i] , " ", word_list[i]
        #print i

create_num_train_doc_cache()

# we cached in file: num_train_doc2.txt
##file_cache = open("num_train_doc2.txt")
##tokens = file_cache.read().split()
##for i in range(len(tokens)):
##    if i % 2 == 0:
##        num_docs_feature_j.append(int(tokens[i]))

# now want to generate unnormalized features for each document:
Xu = np.zeros((len(papers), len(word_list)))
num_train_doc = len(papers)
Xu_test = np.zeros((len(disputed), len(word_list)))
num_test_doc = len(disputed)


def create_normalized_features_2():
    print "Creating normalized features using tf-idf algorithm 2 for training set"
    print "Parsing document: ",
    for j in range(len(papers)):
        print j, " ",
        for i in range(len(word_list)):
            nij = number_of_occurrences(papers[j], word_list[i])
            if (nij == 0):
                Xu[j][i] = 0 
            else:
                # calc the left term:
                tf = np.log(nij + 1) # part 1, TF transform.
                # right:
                idf = (float(num_train_doc))/(float(num_docs_feature_j[i])) # part 2: IDF transform.
                idf = np.log(idf)
                Xu[j][i] = tf * idf
        # now need normalize:
        denom = 0
        for k in range(len(word_list)):
            #nij = number_of_occurrences(papers[j], word_list[k])
            nij = Xu[j][k]
            denom = denom + nij * nij
        denom = np.sqrt(denom)
        for i in range(len(word_list)): # part 3, length norm.
            Xu[j][i] = Xu[j][i] / denom


# FOR SECOND ALGORITHM!!
create_normalized_features_2()

###print list of features to a file
##features_file = open('features_tf_idf_2.txt', 'w')
##for i in range(len(papers)):
##    for j in range(len(word_list)):
##        print >> features_file, Xu[i][j], " ",
##    print >> features_file

##features_file_test = open('idf_test.txt', 'w')
##for i in range(len(disputed)):
##    print i
##    for j in range(len(word_list)):
##        print >> features_file_test, Xu_test[i][j], " ",
##    print >> features_file_test
##features_file_test.close()

##features_file_test = open('features_tf_idf_test_2.txt', 'w')
##for i in range(len(disputed)):
##    for j in range(len(word_list)):
##        print >> features_file_test, Xu_test[i][j], " ",
##    print >> features_file_test


# If file exists: can just fill up Xu with them.
##cur_num = 0
##features_file = open('features_tf_idf_2.txt')
##tokens = features_file.read().split()
##for i in range(len(papers)):
##    for j in range(len(word_list)):
##        Xu[i][j] = tokens[cur_num]
##        cur_num = cur_num + 1

# If file exists: can just fill up Xu_test with them.
##cur_num = 0
##features_file = open('features_tf_idf_test_2.txt')
##tokens = features_file.read().split()
##for i in range(len(disputed)):
##    for j in range(len(word_list)):
##        Xu_test[i][j] = tokens[cur_num]
##        cur_num = cur_num + 1
##
##
##
### do naive bayes.
### section 4.4 of http://machinelearning.wustl.edu/mlpapers/paper_files/icml2003_RennieSTK03.pdf
##
# we have 2 classes. 0= hamilton, 1 = madison
# do CNB estimate:
num_authors = len(a_names)
class_parameter_vector = np.zeros((num_authors, len(word_list)))

complement_xu = np.zeros((num_authors, len(word_list)))
for c in range(num_authors):
    print "Parsing class: ", c
    for i in range(len(word_list)):
        #print i, " ",
        for j in range(len(papers)):
            if (paper_author[j] != c):
                complement_xu[c][i] = complement_xu[c][i] + Xu[j][i]
        complement_xu[c][i] = complement_xu[c][i] + 1
    denom = 0
    for j in range(len(papers)):
        if (paper_author[j] != c):
            for k in range(len(word_list)):
                denom = denom + Xu[j][k]
    denom = denom + len(word_list)
    for i in range(len(word_list)):
        class_parameter_vector[c][i] = float(complement_xu[c][i]) / float(denom) # part 4
        class_parameter_vector[c][i] = np.log(class_parameter_vector[c][i]) # part 5

##    for i in range(len(word_list)):
##    print i, " ",
##    numer = 0
##    for j in range(len(papers)):
##        if paper_author[j] != c:
##            numer = numer + Xu[j][i]
##    numer = numer + 1
##    class_parameter_vector[c][i] = float(numer)/float(denom) # part 4.
##    denom = 0
##    for j in range(len(papers)):
##        if paper_author[j] != c:
##            for k in range(len(word_list)):
##                denom = denom + Xu[j][k]
##    denom = denom + len(word_list)
##    print "Done with denom"     

##
##class_param_vector_file = open('class_param_vector_file.txt', 'w')
##for i in range(4):
##    for j in range(len(word_list)):
##        print >> class_param_vector_file, class_parameter_vector[i][j], " ",
##    print >> class_param_vector_file

##cur_num = 0
##class_param_vector_file = open('class_param_vector_file.txt')
##tokens = class_param_vector_file.read().split()
##for i in range(4):
##    for j in range(len(word_list)):
##        class_parameter_vector[i][j] = tokens[cur_num]
##        cur_num = cur_num + 1

### now need to take log of em. (finds Wci in the paper)
##for i in range(2):
##    for j in range(len(word_list)): # part 5
##        class_parameter_vector[i][j] = np.log(class_parameter_vector[i][j])

# now normalize:
for i in range(num_authors):
    denom = 0
    for k in range(len(word_list)):
        denom = denom + np.absolute(class_parameter_vector[i][k]) # part 6
    for j in range(len(word_list)):
        class_parameter_vector[i][j] = class_parameter_vector[i][j]/float(denom)
Xu_test = np.zeros((len(disputed), len(word_list)))
### now do test documents:
for i in range(len(disputed)):
    for j in range(len(word_list)):
        Xu_test[i][j] = number_of_occurrences(disputed[i],word_list[j])


### we have 2 classes. 0 = hamilton, 1 = madison
for k in range(len(disputed)):
    print k
    c_1 = np.dot(Xu_test[k], class_parameter_vector[0])
    c_2 = np.dot(Xu_test[k], class_parameter_vector[1])
    c_3 = np.dot(Xu_test[k], class_parameter_vector[2])
    c_4 = np.dot(Xu_test[k], class_parameter_vector[3])
    print "hamilton: " , c_1
    print "madison: " , c_2
    print "jay: " , c_3
    print "mix: " , c_4
    c_lowest = np.minimum(c_4,np.minimum(c_3,np.minimum(c_2, c_1)))
    if (c_lowest==c_4):
        print "mix"
    if (c_lowest==c_3):
        print "jay"
    if (c_lowest==c_2):
        print "madison"
    if (c_lowest==c_1):
        print "hamilton"

