# MIT 6.034 Lab 5: Bayesian Inference
# Written by 6.034 staff

from nets import *


#### Part 1: Warm-up; Ancestors, Descendents, and Non-descendents ##############

def get_ancestors(net, var):
    "Return a set containing the ancestors of var"
    ancestors = net.get_parents(var)
    
    if ancestors != {}:
        for parent in ancestors:
            ancestors = ancestors.union(get_ancestors(net,parent))
    return ancestors


def get_descendants(net, var):
    "Returns a set containing the descendants of var"
    descendants = net.get_children(var)
    
    if descendants != {}:
        for child in descendants:
            descendants = descendants.union(get_descendants(net,child))
    return descendants


def get_nondescendants(net, var):
    "Returns a set containing the non-descendants of var"
    all_variables = set(net.get_variables())
    descendants = get_descendants(net, var) 
    descendants.update(var)    
    
    return all_variables.difference(descendants)



#### Part 2: Computing Probability #############################################

def simplify_givens(net, var, givens):
    """
    If givens include every parent of var and no descendants, returns a
    simplified list of givens, keeping only parents.  Does not modify original
    givens.  Otherwise, if not all parents are given, or if a descendant is
    given, returns original givens.
    """
    parents = net.get_parents(var)
    descendants = get_descendants(net,var)
    set_givens = set(givens)
    simp_givens = {}

    
    descendant_present = False
    for given in givens:
        if given in descendants:
            descendant_present = True
    
    if parents.issubset(set_givens) and (not descendant_present):
        for parent in parents:
            simp_givens[parent] = givens[parent]
        return simp_givens
    elif (not parents.issubset(simp_givens)) or  descendant_present:
        return givens

    
def probability_lookup(net, hypothesis, givens=None):
    "Looks up a probability in the Bayes net, or raises LookupError"
    for var in hypothesis:
        var = var
    
    if givens != None:
        givens = simplify_givens(net, var, givens)
    
    try:
        probability = net.get_probability(hypothesis, givens)
    except:
        raise LookupError
    
    return probability


def probability_joint(net, hypothesis):
    "Uses the chain rule to compute a joint probability"
    list_var = net.topological_sort()
    probability = None
    
    while list_var:
        new_var_hypothesis = list_var.pop()
        if new_var_hypothesis in hypothesis:
            new_hypothesis = {new_var_hypothesis:hypothesis[new_var_hypothesis]}
            
            new_givens = {}
            for giv in list_var:
                if giv != new_var_hypothesis:
                    new_givens[giv] = hypothesis[giv]

            if probability == None:
                probability = probability_lookup(net,new_hypothesis,new_givens) 
            else:
                probability = probability * probability_lookup(net,new_hypothesis,new_givens)
    return probability

    
def probability_marginal(net, hypothesis):
    "Computes a marginal probability as a sum of joint probabilities"
    all_var = net.get_variables()
    miss_var = []
    probability = None
    
    for var in all_var:
        if var not in hypothesis:
            miss_var.append(var)
    
    combo_probabilities = net.combinations(miss_var,hypothesis)
    
    for joint_hypothesis in combo_probabilities:
        if probability == None:
            probability = probability_joint(net, joint_hypothesis)
        else:
            probability += probability_joint(net, joint_hypothesis)
     
    return probability
    

def probability_conditional(net, hypothesis, givens=None):
    "Computes a conditional probability as a ratio of marginal probabilities"
    if givens != None:
        if hypothesis.keys() == givens.keys() and hypothesis == givens:
            return 1
        elif hypothesis.keys() == givens.keys():
            return 0
    
    try:
        probability = probability_lookup(net, hypothesis, givens)
    except:
        if givens == None:
            probability = probability_marginal(net,hypothesis)
        else:
            numerator_hypothesis = dict(givens, **hypothesis)
            probability = probability_marginal(net,numerator_hypothesis) / probability_marginal(net, givens)
    return probability

    
def probability(net, hypothesis, givens=None):
    "Calls previous functions to compute any probability"
    if givens == None:
        probability = probability_marginal(net, hypothesis)
    else:
        probability = probability_conditional(net, hypothesis, givens)
    return probability



#### Part 3: Counting Parameters ###############################################

def number_of_parameters(net):
    """
    Computes the minimum number of parameters required for the Bayes net.
    """
    num_param = 0
    all_var = net.topological_sort()
    
    for var in all_var:
        var_domain = net.get_domain(var)
        parents = net.get_parents(var)
        parent_domain = []
        if len(parents) > 0:
            for parent in parents:
                parent_domain.append(len(net.get_domain(parent)))
        num_param += (len(var_domain) -1) * product(parent_domain)
    return num_param



#### Part 4: Independence ######################################################

def is_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    otherwise False. Uses numerical independence.
    """
    var1_val = net.get_domain(var1)[0]
    var2_val = net.get_domain(var2)[0]
    
    if givens == None or len(givens)<1:
        joint_hypothesis = {var1:var1_val,var2:var2_val}
        var1_hypothesis = {var1:var1_val}
        var2_hypothesis = {var2:var2_val}
        
        prob_marginal = probability(net,joint_hypothesis)
        prob_var1 = probability(net,var1_hypothesis)
        prob_var2 = probability(net,var2_hypothesis)
        if approx_equal(prob_marginal, prob_var1*prob_var2):
            return True
    else:
        add_given = {var2:var2_val}
        new_givens = dict(givens, **add_given)
        var1_hypothesis = {var1:var1_val}
        
        prob_cond = probability(net,var1_hypothesis,givens)
        prob_cond_new = probability(net,var1_hypothesis,new_givens)
        if approx_equal(prob_cond, prob_cond_new):
            return True
        
    return False
    

    
def is_structurally_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    based on the structure of the Bayes net, otherwise False.
    Uses structural independence only (not numerical independence).
    """
    # Drawing ancestral graph of vars in problem statement
    all_orig_vars = net.topological_sort()
    prob_vars = [var1,var2]
    if givens != None:
        for var in givens:
            prob_vars.append(var)
    
    for var in all_orig_vars:
        if var not in prob_vars:
            childrens = get_descendants(net,var)
            for child in childrens:
                if child in prob_vars:
                    prob_vars.append(var)
    ancestral_graph = net.subnet(prob_vars)
    
    # Linking parents
    all_vars = ancestral_graph.topological_sort()
    for var in all_vars:
        parents = ancestral_graph.get_parents(var)
        if len(parents) > 1:
            for p1 in parents:
                for p2 in parents:
                    if p1 != p2:
                        ancestral_graph.link(p1, p2)
    
    # Disorienting graph
    ancestral_graph.make_bidirectional()
    
    # Delete givens
    if givens != None:
        new_net_vars = set(all_vars).difference(givens)
        final_net = ancestral_graph.subnet(list(new_net_vars))
    else:
        final_net = ancestral_graph
    
    # Find path
    if final_net.find_path(var1,var2) == None:
        return True
    return False



#### SURVEY ####################################################################

NAME = "Luke Chiang"
COLLABORATORS = "None"
HOW_MANY_HOURS_THIS_LAB_TOOK = "6"
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
