# MIT 6.034 Lab 9: Boosting (Adaboost)
# Written by 6.034 staff

from math import log as ln
from utils import *


#### Part 1: Helper functions ##################################################

def initialize_weights(training_points):
    """Assigns every training point a weight equal to 1/N, where N is the number
    of training points.  Returns a dictionary mapping points to weights."""
    init_weight = make_fraction(1,len(training_points))
    weights = {}
    for point in training_points:
        weights[point] = init_weight
    return weights


def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    """Given a dictionary mapping training points to their weights, and another
    dictionary mapping classifiers to the training points they misclassify,
    returns a dictionary mapping classifiers to their error rates."""
    error_rates = {}
    for classifier in classifier_to_misclassified:
        error_rate = 0
        for point in classifier_to_misclassified[classifier]:
            error_rate += point_to_weight[point]
        error_rates[classifier] = error_rate
    return error_rates
    raise NotImplementedError

def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    """Given a dictionary mapping classifiers to their error rates, returns the
    best* classifier, or raises NoGoodClassifiersError if best* classifier has
    error rate 1/2.  best* means 'smallest error rate' if use_smallest_error
    is True, otherwise 'error rate furthest from 1/2'."""
    if use_smallest_error:
        best_classifier = min(classifier_to_error_rate, key = lambda classifier:classifier_to_error_rate[classifier])
    else:
        best_classifier = max(classifier_to_error_rate, key = lambda classifier:abs(make_fraction(1,2)-classifier_to_error_rate[classifier]))
    
    if classifier_to_error_rate[best_classifier] == make_fraction(1,2):
        raise NoGoodClassifiersError
    else:
        return best_classifier


def calculate_voting_power(error_rate):
    """Given a classifier's error rate (a number), returns the voting power
    (aka alpha, or coefficient) for that classifier."""
    if error_rate == 0:
        return INF
    elif error_rate == 1:
        return -INF
    else:
        return make_fraction(1,2) * ln((1-error_rate)/error_rate)


def get_overall_misclassifications(H, training_points, classifier_to_misclassified):
    """Given an overall classifier H, a list of all training points, and a
    dictionary mapping classifiers to the training points they misclassify,
    returns a set containing the training points that H misclassifies.
    H is represented as a list of (classifier, voting_power) tuples."""
    misclassified_points = []
    for point in training_points:
        output = 0
        for classifier in H:
            if point in classifier_to_misclassified[classifier[0]]:
                output += classifier[1]*-1
            else:
                output += classifier[1]*1
        if output <= 0:
            misclassified_points.append(point)
    return set(misclassified_points)


def is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance=0):
    """Given an overall classifier H, a list of all training points, a
    dictionary mapping classifiers to the training points they misclassify, and
    a mistake tolerance (the maximum number of allowed misclassifications),
    returns False if H misclassifies more points than the tolerance allows,
    otherwise True.  H is represented as a list of (classifier, voting_power)
    tuples."""
    misclassified_points = get_overall_misclassifications(H, training_points, classifier_to_misclassified)
    if len(misclassified_points) <= mistake_tolerance:
        return True
    else:
        return False


def update_weights(point_to_weight, misclassified_points, error_rate):
    """Given a dictionary mapping training points to their old weights, a list
    of training points misclassified by the current weak classifier, and the
    error rate of the current weak classifier, returns a dictionary mapping
    training points to their new weights.  This function is allowed (but not
    required) to modify the input dictionary point_to_weight."""
    new_weights = {}
    for point in point_to_weight:
        if point not in misclassified_points:
            new_weights[point] = make_fraction(1,2)*make_fraction(1,1-error_rate)*point_to_weight[point]
        else:
            new_weights[point] = make_fraction(1,2)*make_fraction(1,error_rate)*point_to_weight[point]
    return new_weights



#### Part 2: Adaboost ##########################################################

def adaboost(training_points, classifier_to_misclassified,
             use_smallest_error=True, mistake_tolerance=0, max_rounds=INF):
    """Performs the Adaboost algorithm for up to max_rounds rounds.
    Returns the resulting overall classifier H, represented as a list of
    (classifier, voting_power) tuples."""
    H = []
    point_to_weight = initialize_weights(training_points)
    num_round = 1
    
    while (not is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance)) and num_round <= max_rounds:
        classifier_to_error_rate = calculate_error_rates(point_to_weight, classifier_to_misclassified)
        
        # Use a try/except block? to handle error
        try:
            best_classifier = pick_best_classifier(classifier_to_error_rate, use_smallest_error)
        except:
            break
        voting_power = calculate_voting_power(classifier_to_error_rate[best_classifier])
        H.append((best_classifier,voting_power))
        point_to_weight = update_weights(point_to_weight, classifier_to_misclassified[best_classifier], classifier_to_error_rate[best_classifier])
        num_round += 1
    return H
    
    raise NotImplementedError


#### SURVEY ####################################################################

NAME = 'Luke Chiang'
COLLABORATORS = 'None'
HOW_MANY_HOURS_THIS_LAB_TOOK = 2
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
