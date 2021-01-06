# MIT 6.034 Lab 8: Support Vector Machines
# Written by 6.034 staff

from svm_data import *
from functools import reduce


#### Part 1: Vector Math #######################################################

def dot_product(u, v):
    """Computes the dot product of two vectors u and v, each represented 
    as a tuple or list of coordinates. Assume the two vectors are the
    same length."""
    product = 0
    for n in range(len(u)):
        product += u[n]*v[n]
    return product


def norm(v):
    """Computes the norm (length) of a vector v, represented 
    as a tuple or list of coords."""
    norm = 0
    for n in v:
        norm += n**2
    return norm**0.5



#### Part 2: Using the SVM Boundary Equations ##################################

def positiveness(svm, point):
    """Computes the expression (w dot x + b) for the given Point x."""
    x = point.coords
    w = svm.w
    b = svm.b
    return dot_product(w,x) + b



def classify(svm, point):
    """Uses the given SVM to classify a Point. Assume that the point's true
    classification is unknown.
    Returns +1 or -1, or 0 if point is on boundary."""
    result = positiveness(svm,point)
    if result > 0:
        return 1
    elif result < 0:
        return -1
    else:
        return 0


def margin_width(svm):
    """Calculate margin width based on the current boundary."""
    return 2/norm(svm.w)


def check_gutter_constraint(svm):
    """Returns the set of training points that violate one or both conditions:
        * gutter constraint (positiveness == classification, for support vectors)
        * training points must not be between the gutters
    Assumes that the SVM has support vectors assigned."""
    points = svm.training_points
    support_vectors = svm.support_vectors
    bad_points = []
    for point in points:
        result = positiveness(svm,point)
        if point in support_vectors and result != point.classification:
            bad_points.append(point)
        elif result < 1 and result > -1:
            bad_points.append(point)
    return set(bad_points)
    


#### Part 3: Supportiveness ####################################################

def check_alpha_signs(svm):
    """Returns the set of training points that violate either condition:
        * all non-support-vector training points have alpha = 0
        * all support vectors have alpha > 0
    Assumes that the SVM has support vectors assigned, and that all training
    points have alpha values assigned."""
    points = svm.training_points
    support_vectors = svm.support_vectors
    bad_alphas = []
    for point in points:
        if point in support_vectors and point.alpha <= 0:
            bad_alphas.append(point)
        elif point not in support_vectors and point.alpha != 0:
            bad_alphas.append(point)
    return set(bad_alphas)


def check_alpha_equations(svm):
    """Returns True if both Lagrange-multiplier equations are satisfied,
    otherwise False. Assumes that the SVM has support vectors assigned, and
    that all training points have alpha values assigned."""
    points = svm.training_points
    support_sum = 0
    vector_sum = None
    
    for point in points:
        y = point.classification
        alpha = point.alpha
        coord = point.coords
        support_sum += y*alpha
        if vector_sum == None:
            vector_sum = scalar_mult(y*alpha,coord)
        else:
            vector_sum = vector_add(vector_sum, scalar_mult(y*alpha,coord))
        
    if support_sum != 0 or vector_sum != svm.w:
        return False
    return True



#### Part 4: Evaluating Accuracy ###############################################

def misclassified_training_points(svm):
    """Returns the set of training points that are classified incorrectly
    using the current decision boundary."""
    points = svm.training_points
    misclassified_points = []
    
    for point in points:
        y = point.classification
        boundary_classify = classify(svm,point)
        if y != boundary_classify:
            misclassified_points.append(point)
    
    return set(misclassified_points)



#### Part 5: Training an SVM ###################################################

def update_svm_from_alphas(svm):
    """Given an SVM with training data and alpha values, use alpha values to
    update the SVM's support vectors, w, and b. Return the updated SVM."""
    points = svm.training_points
    
    support_vects = []
    w = None
    for point in points:
        alpha = point.alpha
        coord = point.coords
        y = point.classification
        
        if point.alpha > 0:
            support_vects.append(point)
        
        if w == None:
            w = scalar_mult(y*alpha,coord)
        else:
            w = vector_add(w, scalar_mult(y*alpha,coord))
    
    min_b = None
    max_b = None

    for support_vect in support_vects:
        y = support_vect.classification
        coord = support_vect.coords
        b = y - dot_product(w,coord)
        if y == -1:
            if min_b == None:
                min_b = b
            elif b < min_b:
                min_b = b
        else:
            if max_b == None:
                max_b = b
            elif b > max_b:
                max_b = b
    
    svm.support_vectors = support_vects
    svm.set_boundary(w, (min_b+max_b)/2)
    return svm



#### Part 6: Multiple Choice ###################################################

ANSWER_1 = 11
ANSWER_2 = 6
ANSWER_3 = 3
ANSWER_4 = 2

ANSWER_5 = ['A','D']
ANSWER_6 = ['A','B','D']
ANSWER_7 = ['A','B','D']
ANSWER_8 = []
ANSWER_9 = ['A','B','D']
ANSWER_10 = ['A','B','D']

ANSWER_11 = False
ANSWER_12 = True
ANSWER_13 = False
ANSWER_14 = False
ANSWER_15 = False
ANSWER_16 = True

ANSWER_17 = [1,3,6,8]
ANSWER_18 = [1,2,4,5,6,7,8]
ANSWER_19 = [1,2,4,5,6,7,8]

ANSWER_20 = 6


#### SURVEY ####################################################################

NAME = 'Luke Chiang'
COLLABORATORS = 'None'
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
