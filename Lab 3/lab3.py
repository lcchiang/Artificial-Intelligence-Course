# MIT 6.034 Lab 3: Constraint Satisfaction Problems
# Written by 6.034 staff

from constraint_api import *
from test_problems import get_pokemon_problem


#### Part 1: Warmup ############################################################

def has_empty_domains(csp) :
    """Returns True if the problem has one or more empty domains, otherwise False"""
    variables = csp.variables
    for var in variables:
        if len(csp.get_domain(var)) == 0:
            return True
    return False


def check_all_constraints(csp) :
    """Return False if the problem's assigned values violate some constraint,
    otherwise True"""
    assignments = csp.assignments
    constraints = csp.constraints
    for constraint in constraints:
        if constraint.var1 in assignments and constraint.var2 in assignments:
           if not constraint.check(assignments[constraint.var1], assignments[constraint.var2]):
               return False
    return True



#### Part 2: Depth-First Constraint Solver #####################################

def solve_constraint_dfs(problem) :
    """
    Solves the problem using depth-first search.  Returns a tuple containing:
    1. the solution (a dictionary mapping variables to assigned values)
    2. the number of extensions made (the number of problems popped off the agenda).
    If no solution was found, return None as the first element of the tuple.
    """
    agenda = [problem]
    num_extensions = 0
    

    while agenda:
        prob = agenda.pop(0)
        num_extensions +=1
        if has_empty_domains(prob):
            continue
        if check_all_constraints(prob):
            if len(prob.unassigned_vars) > 0:
                unass_var = prob.pop_next_unassigned_var()
                values = prob.get_domain(unass_var)
                new_probs = []
                for val in values:
                    new_prob = prob.copy()
                    new_prob.set_assignment(unass_var,val)
                    new_probs.append(new_prob)
                agenda = new_probs + agenda
            else:
                return(prob.assignments, num_extensions)
    
    return (None, num_extensions)
            

# QUESTION 1: How many extensions does it take to solve the Pokemon problem
#    with DFS?

# Hint: Use get_pokemon_problem() to get a new copy of the Pokemon problem
#    each time you want to solve it with a different search method.

print(solve_constraint_dfs(get_pokemon_problem())[1])
ANSWER_1 = solve_constraint_dfs(get_pokemon_problem())[1]


#### Part 3: Forward Checking ##################################################

def eliminate_from_neighbors(csp, var) :
    """
    Eliminates incompatible values from var's neighbors' domains, modifying
    the original csp.  Returns an alphabetically sorted list of the neighboring
    variables whose domains were reduced, with each variable appearing at most
    once.  If no domains were reduced, returns empty list.
    If a domain is reduced to size 0, quits immediately and returns None.
    """
    def check_constraint(var1, var2, val1, val2):
        """
        Checks whether there is a constraint violation between var1/val1 and var2/val2.
        Returns True if there isn't a violation or false if there is.
        """
        constraints = csp.constraints
        for constraint in constraints:
            if (constraint.var1 == var1 or constraint.var1 == var2) and (constraint.var2 == var1 or constraint.var2 == var2):
                if not constraint.check(val1, val2):
                    return False
        return True
    
    reduced_neighbors = []
    neighbors = csp.get_neighbors(var)
    unassig_var = csp.unassigned_vars
    values = csp.get_domain(var)
    remove_n_values = {}
    
    
    #Building dictionary of neighbors and values that conflict
    for n in neighbors:
        if n in unassig_var:
            n_values = csp.get_domain(n)
            remove_n_values[n] = []
            for n_val in n_values:
                constraint_check = []
                for val in values:
                    if check_constraint(var,n,val,n_val):
                        constraint_check.append(0)
                    else:
                        constraint_check.append(1)
                if sum(constraint_check) == len(constraint_check):
                    remove_n_values[n].append(n_val)
    
    #Remove values from neighbors if they conflict. Check if domain is reduced to 0.
    for n in remove_n_values:
        for n_val in remove_n_values[n]:
            csp.eliminate(n,n_val)
        if len(csp.get_domain(n)) == 0:
            return None
        if len(remove_n_values[n]) > 0:
            reduced_neighbors.append(n)
    
    return reduced_neighbors


# Because names give us power over things (you're free to use this alias)
forward_check = eliminate_from_neighbors

def solve_constraint_forward_checking(problem) :
    """
    Solves the problem using depth-first search with forward checking.
    Same return type as solve_constraint_dfs.
    """
    agenda = [problem]
    num_extensions = 0
    

    while agenda:
        prob = agenda.pop(0)
        num_extensions +=1
        if has_empty_domains(prob):
            continue
        if check_all_constraints(prob):
            if len(prob.unassigned_vars) > 0:
                unass_var = prob.pop_next_unassigned_var()
                values = prob.get_domain(unass_var)
                new_probs = []
                for val in values:
                    new_prob = prob.copy()
                    new_prob.set_assignment(unass_var,val)
                    new_probs.append(new_prob)
                for new_p in new_probs:
                    eliminate_from_neighbors(new_p,unass_var)
                agenda = new_probs + agenda
            else:
                return(prob.assignments, num_extensions)
    
    return (None, num_extensions)


# QUESTION 2: How many extensions does it take to solve the Pokemon problem
#    with DFS and forward checking?

print(solve_constraint_forward_checking(get_pokemon_problem())[1])
ANSWER_2 = solve_constraint_forward_checking(get_pokemon_problem())[1]


#### Part 4: Domain Reduction ##################################################

def domain_reduction(csp, queue=None) :
    """
    Uses constraints to reduce domains, propagating the domain reduction
    to all neighbors whose domains are reduced during the process.
    If queue is None, initializes propagation queue by adding all variables in
    their default order. 
    Returns a list of all variables that were dequeued, in the order they
    were removed from the queue.  Variables may appear in the list multiple times.
    If a domain is reduced to size 0, quits immediately and returns None.
    This function modifies the original csp.
    """
    if queue == None:
        queue = csp.get_all_variables()
    
    dequeued = []
    
    while queue:
        var = queue.pop(0)
        dequeued.append(var)
        reduced_neighbors = eliminate_from_neighbors(csp,var)
        if reduced_neighbors == None:
            return None
        for n in reduced_neighbors:
            if n not in queue:
                queue.append(n)
    return dequeued  



# QUESTION 3: How many extensions does it take to solve the Pokemon problem
#    with DFS (no forward checking) if you do domain reduction before solving it?

x = get_pokemon_problem()
domain_reduction(x)
print(solve_constraint_dfs(x)[1])
ANSWER_3 = 6


def solve_constraint_propagate_reduced_domains(problem) :
    """
    Solves the problem using depth-first search with forward checking and
    propagation through all reduced domains.  Same return type as
    solve_constraint_dfs.
    """
    agenda = [problem]
    num_extensions = 0
    

    while agenda:
        prob = agenda.pop(0)
        num_extensions +=1
        if has_empty_domains(prob):
            continue
        if check_all_constraints(prob):
            if len(prob.unassigned_vars) > 0:
                unass_var = prob.pop_next_unassigned_var()
                values = prob.get_domain(unass_var)
                new_probs = []
                for val in values:
                    new_prob = prob.copy()
                    new_prob.set_assignment(unass_var,val)
                    new_probs.append(new_prob)
                for new_p in new_probs:
                    queue = eliminate_from_neighbors(new_p,unass_var)
                    domain_reduction(new_p,queue)
                agenda = new_probs + agenda
            else:
                return(prob.assignments, num_extensions)
    
    return (None, num_extensions)



# QUESTION 4: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through reduced domains?
print(solve_constraint_propagate_reduced_domains(get_pokemon_problem())[1])
ANSWER_4 = solve_constraint_propagate_reduced_domains(get_pokemon_problem())[1]


#### Part 5A: Generic Domain Reduction #########################################

def propagate(enqueue_condition_fn, csp, queue=None) :
    """
    Uses constraints to reduce domains, modifying the original csp.
    Uses enqueue_condition_fn to determine whether to enqueue a variable whose
    domain has been reduced. Same return type as domain_reduction.
    """
    if queue == None:
        queue = csp.get_all_variables()
    
    dequeued = []
    while queue:
        var = queue.pop(0)
        dequeued.append(var)
        reduced_neighbors = eliminate_from_neighbors(csp,var)
        if reduced_neighbors == None:
            return None
        for n in reduced_neighbors:
            if enqueue_condition_fn(csp, n):
                if n not in queue:
                    queue.append(n)
    return dequeued  

def condition_domain_reduction(csp, var) :
    """Returns True if var should be enqueued under the all-reduced-domains
    condition, otherwise False"""
    return True


def condition_singleton(csp, var) :
    """Returns True if var should be enqueued under the singleton-domains
    condition, otherwise False"""
    domain = csp.get_domain(var)
    if len(domain) == 1:
        return True
    return False


def condition_forward_checking(csp, var) :
    """Returns True if var should be enqueued under the forward-checking
    condition, otherwise False"""
    return False



#### Part 5B: Generic Constraint Solver ########################################

def solve_constraint_generic(problem, enqueue_condition=None) :
    """
    Solves the problem, calling propagate with the specified enqueue
    condition (a function). If enqueue_condition is None, uses DFS only.
    Same return type as solve_constraint_dfs.
    """
    agenda = [problem]
    num_extensions = 0
    

    while agenda:
        prob = agenda.pop(0)
        num_extensions +=1
        if has_empty_domains(prob):
            continue
        if check_all_constraints(prob):
            if len(prob.unassigned_vars) > 0:
                unass_var = prob.pop_next_unassigned_var()
                values = prob.get_domain(unass_var)
                new_probs = []
                for val in values:
                    new_prob = prob.copy()
                    new_prob.set_assignment(unass_var,val)
                    new_probs.append(new_prob)
                if enqueue_condition != None:
                    for new_p in new_probs:
                        propagate(enqueue_condition, new_p, [unass_var])
                agenda = new_probs + agenda
            else:
                return(prob.assignments, num_extensions)
    
    return (None, num_extensions)


# QUESTION 5: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through singleton domains? (Don't
#    use domain reduction before solving it.)
print(solve_constraint_generic(get_pokemon_problem(),condition_singleton)[1])
ANSWER_5 = solve_constraint_generic(get_pokemon_problem(),condition_singleton)[1]


#### Part 6: Defining Custom Constraints #######################################

def constraint_adjacent(m, n) :
    """Returns True if m and n are adjacent, otherwise False.
    Assume m and n are ints."""
    if abs(m-n) == 1:
        return True
    else:
        return False
    
    raise NotImplementedError

def constraint_not_adjacent(m, n) :
    """Returns True if m and n are NOT adjacent, otherwise False.
    Assume m and n are ints."""
    return not constraint_adjacent(m,n)


def all_different(variables) :
    """Returns a list of constraints, with one difference constraint between
    each pair of variables."""
    list_constraints = []
    variable_pairs = []
    
    
    for var1 in variables:
        for var2 in variables:
            constraint_need = True
            for var_pair in variable_pairs:
                if var1 in var_pair and var2 in var_pair:
                    constraint_need = False
            if var1 != var2 and constraint_need:
                list_constraints.append(Constraint(var1,var2,constraint_different))
                variable_pairs.append([var1,var2])
    return list_constraints


#### SURVEY ####################################################################

NAME = 'Luke Chiang'
COLLABORATORS = 'None'
HOW_MANY_HOURS_THIS_LAB_TOOK = 3
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
