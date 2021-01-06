# MIT 6.034 Lab 6: k-Nearest Neighbors and Identification Trees
# Written by 6.034 Staff

from api import *
from data import *
import math
log2 = lambda x: math.log(x, 2)
INF = float('inf')


################################################################################
############################# IDENTIFICATION TREES #############################
################################################################################


#### Part 1A: Classifying points ###############################################

def id_tree_classify_point(point, id_tree):
    """Uses the input ID tree (an IdentificationTreeNode) to classify the point.
    Returns the point's classification."""
    if id_tree.is_leaf():
        return id_tree.get_node_classification()
    else:
        classify_result = id_tree.apply_classifier(point)
        return id_tree_classify_point(point,classify_result)



#### Part 1B: Splitting data with a classifier #################################

def split_on_classifier(data, classifier):
    """Given a set of data (as a list of points) and a Classifier object, uses
    the classifier to partition the data.  Returns a dict mapping each feature
    values to a list of points that have that value."""
    classification_split = {}
    
    for point in data:
        classification = classifier.classify(point)
        if classification in classification_split:
            classification_split[classification].append(point)
        else:
            classification_split[classification] = [point]
    
    return classification_split


#### Part 1C: Calculating disorder #############################################

def branch_disorder(data, target_classifier):
    """Given a list of points representing a single branch and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the branch."""
    disorder = 0
    final_classifications = split_on_classifier(data, target_classifier)
    num_total = len(data)
    
    for final_class in final_classifications:
        num_class = len(final_classifications[final_class])
        proportion = num_class/num_total
        disorder -= proportion*log2(proportion)
        
    return disorder


def average_test_disorder(data, test_classifier, target_classifier):
    """Given a list of points, a feature-test Classifier, and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the feature-test stump."""
    test_disorder = 0
    test_classifications = split_on_classifier(data, test_classifier)
    num_total = len(data)
    
    for test_branch in test_classifications:
        num_branch = len(test_classifications[test_branch])
        disorder = branch_disorder(test_classifications[test_branch],target_classifier)
        proportion = num_branch/num_total
        test_disorder += proportion*disorder
    
    return test_disorder

## To use your functions to solve part A2 of the "Identification of Trees"
## problem from 2014 Q2, uncomment the lines below and run lab6.py:

# for classifier in tree_classifiers:
#     print(classifier.name, average_test_disorder(tree_data, classifier, feature_test("tree_type")))


#### Part 1D: Constructing an ID tree ##########################################

def find_best_classifier(data, possible_classifiers, target_classifier):
    """Given a list of points, a list of possible Classifiers to use as tests,
    and a Classifier for determining the true classification of each point,
    finds and returns the classifier with the lowest disorder.  Breaks ties by
    preferring classifiers that appear earlier in the list.  If the best
    classifier has only one branch, raises NoGoodClassifiersError."""

    min_disorder = INF
    best_classifier = None
    
    for test_classify in possible_classifiers:
        disorder = average_test_disorder(data,test_classify,target_classifier)
        if disorder < min_disorder:
            min_disorder = disorder
            best_classifier = test_classify

    if len(split_on_classifier(data, best_classifier)) <= 1:
        raise NoGoodClassifiersError("Best Classifier does not separate data.")
    else:
        return best_classifier



## To find the best classifier from 2014 Q2, Part A, uncomment:
# print(find_best_classifier(tree_data, tree_classifiers, feature_test("tree_type")))

def construct_greedy_id_tree(data, possible_classifiers, target_classifier, id_tree_node=None):
    """Given a list of points, a list of possible Classifiers to use as tests,
    a Classifier for determining the true classification of each point, and
    optionally a partially completed ID tree, returns a completed ID tree by
    adding classifiers and classifications until either perfect classification
    has been achieved, or there are no good classifiers left."""
    if id_tree_node == None:
        id_tree_node = IdentificationTreeNode(target_classifier)
        
    if branch_disorder(data, target_classifier) == 0:
        classification = target_classifier.classify(data[0])
        id_tree_node.set_node_classification(classification)
    else:
        try:
            best_classifier = find_best_classifier(data,possible_classifiers,target_classifier)
            features = split_on_classifier(data, best_classifier)
            id_tree_node.set_classifier_and_expand(best_classifier, features)
            
            branches = id_tree_node.get_branches()
            new_possible_classifiers = []
            for classifier in possible_classifiers:
                if classifier != best_classifier:
                    new_possible_classifiers.append(classifier)
            
            for branch in branches:
                construct_greedy_id_tree(features[branch],new_possible_classifiers,target_classifier,branches[branch])
                   
        except NoGoodClassifiersError:
            pass

    return id_tree_node
    


## To construct an ID tree for 2014 Q2, Part A:
# print(construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type")))

## To use your ID tree to identify a mystery tree (2014 Q2, Part A4):
# tree_tree = construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type"))
# print(id_tree_classify_point(tree_test_point, tree_tree))

## To construct an ID tree for 2012 Q2 (Angels) or 2013 Q3 (numeric ID trees):
# print(construct_greedy_id_tree(angel_data, angel_classifiers, feature_test("Classification")))
# print(construct_greedy_id_tree(numeric_data, numeric_classifiers, feature_test("class")))


#### Part 1E: Multiple choice ##################################################

ANSWER_1 = 'bark_texture'
ANSWER_2 = 'leaf_shape'
ANSWER_3 = 'orange_foliage'

ANSWER_4 = [2,3]
ANSWER_5 = [3]
ANSWER_6 = [2]
ANSWER_7 = 2

ANSWER_8 = 'No'
ANSWER_9 = 'No'


#### OPTIONAL: Construct an ID tree with medical data ##########################

## Set this to True if you'd like to do this part of the lab
DO_OPTIONAL_SECTION = False

if DO_OPTIONAL_SECTION:
    from parse import *
    medical_id_tree = construct_greedy_id_tree(heart_training_data, heart_classifiers, heart_target_classifier_discrete)


################################################################################
############################# k-NEAREST NEIGHBORS ##############################
################################################################################

#### Part 2A: Drawing Boundaries ###############################################

BOUNDARY_ANS_1 = 3
BOUNDARY_ANS_2 = 4

BOUNDARY_ANS_3 = 1
BOUNDARY_ANS_4 = 2

BOUNDARY_ANS_5 = 2
BOUNDARY_ANS_6 = 4
BOUNDARY_ANS_7 = 1
BOUNDARY_ANS_8 = 4
BOUNDARY_ANS_9 = 4

BOUNDARY_ANS_10 = 4
BOUNDARY_ANS_11 = 2
BOUNDARY_ANS_12 = 1
BOUNDARY_ANS_13 = 4
BOUNDARY_ANS_14 = 4


#### Part 2B: Distance metrics #################################################

def dot_product(u, v):
    """Computes dot product of two vectors u and v, each represented as a tuple
    or list of coordinates.  Assume the two vectors are the same length."""
    dot_prod = 0
    for n in range(len(u)):
        dot_prod += u[n]*v[n]
        
    return dot_prod


def norm(v):
    "Computes length of a vector v, represented as a tuple or list of coords."
    norm = 0
    for n in v:
        norm += n*n
    return norm**(1/2)


def euclidean_distance(point1, point2):
    "Given two Points, computes and returns the Euclidean distance between them."
    distance = 0
    coord1 = point1.coords
    coord2 = point2.coords
    for n in range(len(coord1)):
        distance += (coord1[n] - coord2[n])**2
    return distance**(1/2)


def manhattan_distance(point1, point2):
    "Given two Points, computes and returns the Manhattan distance between them."
    distance = 0
    coord1 = point1.coords
    coord2 = point2.coords
    for n in range(len(coord1)):
        distance += abs(coord1[n] - coord2[n])
    return distance


def hamming_distance(point1, point2):
    "Given two Points, computes and returns the Hamming distance between them."
    distance = 0
    coord1 = point1.coords
    coord2 = point2.coords
    for n in range(len(coord1)):
        if coord1[n] != coord2[n]:
            distance += 1
    return distance
    
    raise NotImplementedError

def cosine_distance(point1, point2):
    """Given two Points, computes and returns the cosine distance between them,
    where cosine distance is defined as 1-cos(angle_between(point1, point2))."""
    coord1 = point1.coords
    coord2 = point2.coords
    distance = 1 - dot_product(coord1,coord2)/(norm(coord1)*norm(coord2))
    return distance


#### Part 2C: Classifying points ###############################################

def get_k_closest_points(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns a list containing the k points
    from the data that are closest to the test point, according to the distance
    metric.  Breaks ties lexicographically by coordinates."""
    distances = {}
    for d in data:
        distances[str(d)] = distance_metric(point,d)
    return sorted(data,key = lambda data:(distances[str(data)], data.coords))[:k]


def knn_classify_point(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns the classification of the test
    point based on its k nearest neighbors, as determined by the distance metric.
    Assumes there are no ties."""
    neighbors = get_k_closest_points(point, data, k, distance_metric)
    classification_count = {}
    for n in neighbors:
        classification = n.classification
        if classification in classification_count:
            classification_count[classification] += 1
        else:
            classification_count[classification] = 1
    return max(classification_count, key = lambda classification:classification_count[classification])



## To run your classify function on the k-nearest neighbors problem from 2014 Q2
## part B2, uncomment the line below and try different values of k:
# print(knn_classify_point(knn_tree_test_point, knn_tree_data, 1, euclidean_distance))


#### Part 2C: Choosing k #######################################################

def cross_validate(data, k, distance_metric):
    """Given a list of points (the data), an int 0 < k <= len(data), and a
    distance metric (a function), performs leave-one-out cross-validation.
    Return the fraction of points classified correctly, as a float."""
    correct_classification = 0
    
    for test in data:
        train_set = data.copy()
        train_set.remove(test)
        classification = knn_classify_point(test, train_set, k, distance_metric)
        if classification == test.classification:
            correct_classification +=1
    return correct_classification/len(data)


def find_best_k_and_metric(data):
    """Given a list of points (the data), uses leave-one-out cross-validation to
    determine the best value of k and distance_metric, choosing from among the
    four distance metrics defined above.  Returns a tuple (k, distance_metric),
    where k is an int and distance_metric is a function."""
    distances = [euclidean_distance, manhattan_distance, hamming_distance, cosine_distance]
    results = {}
    for k in range(1,len(data)):
        for distance_metric in distances:
            accuracy = cross_validate(data, k, distance_metric)
            results[(k,distance_metric)] = accuracy
    return max(results, key = lambda result:results[result])


## To find the best k and distance metric for 2014 Q2, part B, uncomment:
# print(find_best_k_and_metric(knn_tree_data))


#### Part 2E: More multiple choice #############################################

kNN_ANSWER_1 = 'Overfitting'
kNN_ANSWER_2 = 'Underfitting'
kNN_ANSWER_3 = 4

kNN_ANSWER_4 = 4
kNN_ANSWER_5 = 1
kNN_ANSWER_6 = 3
kNN_ANSWER_7 = 3


#### SURVEY ####################################################################

NAME = 'Luke Chiang'
COLLABORATORS = 'N/A'
HOW_MANY_HOURS_THIS_LAB_TOOK = '10 hours'
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
