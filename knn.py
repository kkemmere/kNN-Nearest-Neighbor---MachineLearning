#
# Initial Idea/Framework: Shimin Li, Modified by Kevin Kemmerer
#
# Summary: k-NN algorithm, CS366 Project
#
# Useage: Compute if an article is in a certain category using machine learning kNN
#
#
import copy
import math
import re
import string
import sys

from os import listdir
from os.path import isfile, join

# Global Variables
k = 16 # K value is the number of articles in the training data
training_folder = '/Users/tj8200pw/Documents/CS366(Internet of Things)/FinalAssign/training/'
minn_folder     = training_folder + 'Minnesota/'
health_folder   = training_folder + 'Health/'
tech_folder = training_folder + 'Tech/'

def remove_punctuation(s):
     return re.sub(r'[^\w\s]','',s) # Changed remove punc function

def file_list(folder):
    return [f for f in listdir(folder) if isfile(join(folder, f))]

# More file lists can be added depending on how much training data you want to use.
# Be sure to change K value depending on the nunmber of articles used in training.
def all_file_list():
    minn_files = file_list(minn_folder)
    for i in range(len(minn_files)):
        minn_files[i] = minn_folder + minn_files[i]

    health_files = file_list(health_folder)
    for i in range(len(health_files)):
        health_files[i] = health_folder + health_files[i]

    tech_files = file_list(tech_folder)             # added for tech files
    for i in range(len(tech_files)):
        tech_files[i] = tech_folder + tech_files[i]

    return minn_files + health_files + tech_files

# Reads any file, removes punctuation and splits the text
def file_to_word_list(f):
    fr        = open(f, 'r')
    text_read = fr.read()
    text      = remove_punctuation(text_read)

    return text.split()

def get_vocabularies(all_files):
    voc = {}
    for f in all_files:
        words = file_to_word_list(f)
        for w in words:
            voc[w] = 0

    return voc

# Loading training data into training_data array
def load_training_data():
    all_files = all_file_list()
    voc = get_vocabularies(all_files)
    
    training_data = []

    for f in all_files:
        tag = f.split('/')[7]
        point = copy.deepcopy(voc)

        words = file_to_word_list(f)
        for w in words:
            point[w] += 1

        d = {'tag':tag, 'point':point}
        training_data.append(d)

    #print(training_data)   # test to make sure training data is correct
    return training_data

# Calculates distance vector from input to training data
def get_distance(p1, p2):
    sq_sum = 0

    for w in p1:
        if w in p2:
            sq_sum += (p1[w]-p2[w])*(p1[w]-p2[w])
        else:
            sq_sum += p1[w]*p1[w]

    return math.sqrt(sq_sum)

# This function is implemented for seeing insights of training data
# training_data is an array that contains ('tag','point') tag = Health, Minnesota or Tech. point = all words in article
# and how often they show up.
def show_distances(training_data):
    for i in range(len(training_data)):
        for j in range(i+1, len(training_data)):
            print('d('+str(i)+','+str(j)+')=', end='')
            print(get_distance(training_data[i]['point'], training_data[j]['point']), end=' ')
        print()
    for i in range(len(training_data)):
        print(training_data[i]['tag'])

# This function compares training data array to input txt file array, computes distance and puts it in dist_list array
def test(training_data, txt_file):
    dist_list = []
    txt       = {}
    item      = {}
    max_i     = 0

    words = file_to_word_list(txt_file)
    for w in words:
        if w in txt:
            txt[w] += 1
        else:
            txt[w]  = 1
    #print(txt)     # for testing
    #print(item)    # for testing
    print("How related your article is to the current training data (0.0 is identical): ") # for testing+format

    for pt in training_data:
        item['tag'] = pt['tag']
        item['distance'] = get_distance(pt['point'], txt)
        print(item['distance'])     # for testing+format

        if len(dist_list) < k:
            dist_list.append(copy.deepcopy(item))
        else:
            for i in range(1, k):
                if dist_list[i]['distance'] > dist_list[max_i]['distance']:
                    max_i = i
            if dist_list[max_i]['distance'] > item['distance']:
                dist_list[max_i] = item

    vote_result = {}

    #print(dist_list)    # for testing
    for d in dist_list:
        if d['tag'] in vote_result:
            vote_result[d['tag']] += 1
        else:
            vote_result[d['tag']]  = 1

    print() # for pretty print out format
    print("Total number of articles used for comparisons: "+str(vote_result))  # for testing+format
    sorted_items = sorted(dist_list, key=lambda items: items['distance'])   # Sorting dist_list by distance
    # print(sorted_items)    # for testing

    # Sorting dist_list such that the result is the article with the lowest distance.
    # Meaning the result is the training section which the article is most similar to

    result = sorted_items[0]['tag']  # Result is the 'tag' with the lowest distance i.e the most similar

    # Removed voting because it was not taking into account how similar
    # the article was to the training articles. This was only taking into account
    # the number of articles in the dist_list

    # for vote in vote_result:
    #   if vote_result[vote] > vote_result[result]:
    #        result = vote

    # Voting can be implemented in a better way

    return result
    
def main(txt):
    td = load_training_data()
   # show_distances(td)        # for test usage only
    print("Your article's category according to our training data is: " + test(td, txt))

testing = input("Which article would you like to test? ")
print() # for pretty print out format
main(testing)
