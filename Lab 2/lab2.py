# MIT 6.034 Lab 2: Games
# Written by 6.034 staff

from game_api import *
from boards import *
from toytree import GAME1

INF = float('inf')

# Please see wiki lab page for full description of functions and API.

#### Part 1: Utility Functions #################################################

def is_game_over_connectfour(board):
    """Returns True if game is over, otherwise False."""
    for chain in board.get_all_chains():
        if len(chain) >= 4:
            return True
        
    if board.count_pieces() == 42:
        return True
    return False


def next_boards_connectfour(board):
    """Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made."""
    next_moves = []
    if is_game_over_connectfour(board):
        return next_moves
    
    for col in range(7):
        if not board.is_column_full(col):
            next_moves.append(board.add_piece(col))
    return next_moves


def endgame_score_connectfour(board, is_current_player_maximizer):
    """Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie."""
    for chain in board.get_all_chains():
        if len(chain) >= 4:
            if is_current_player_maximizer:
                return -1000
            else:
                return 1000
    return 0
    


def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner."""
    num_pieces = board.count_pieces()
    if is_game_over_connectfour(board):
        return endgame_score_connectfour(board, is_current_player_maximizer) * (43- num_pieces)


def heuristic_connectfour(board, is_current_player_maximizer):
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""
    current_chain = board.get_all_chains(True)
    other_chain = board.get_all_chains(False)
    current_val = 0
    other_val = 0
    
    for chain in current_chain:
        if len(chain) == 3:
            current_val += 150
        elif len(chain) == 2:
            current_val += 25
        else:
            current_val += 5
    
    for chain in other_chain:
        if len(chain) == 3:
            other_val += 150
        elif len(chain) == 2:
            other_val += 25
        else:
            other_val += 5
    
    if is_current_player_maximizer:
        return current_val - other_val
    else:
        return other_val - current_val

# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot = ConnectFourBoard(),
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(snapshot = NEARLY_OVER,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot = BOARD_UHOH,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)


#### Part 2: Searching a Game Tree #############################################

# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

def dfs_maximizing(state) :
    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    agenda = [[state]]
    best_path = ([],-INF, 0)
    evaluation_counter = 0

    while (len(agenda) > 0):
        path = agenda.pop(0)
        if path[-1].is_game_over():
            score = path[-1].get_endgame_score(False)
            evaluation_counter += 1
            best_path = (best_path[0], best_path[1], evaluation_counter)
            if score > best_path[1]:
                best_path = (path, score, best_path[2])                
        else:
            children = path[-1].generate_next_states()
            new_paths = []
            for child in children:
                temp_path = path.copy()
                temp_path.append(child)
                new_paths.append(temp_path)
            agenda = new_paths + agenda
    return best_path
    



# Uncomment the line below to try your dfs_maximizing on an
# AbstractGameState representing the games tree "GAME1" from toytree.py:

# pretty_print_dfs_type(dfs_maximizing(GAME1))


def minimax_endgame_search(state, maximize=True) :
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Same return type as dfs_maximizing."""
    if type(state) != list:
        state = [[state]]
    leaf_node = []
    static_eval_count = 0
    
    while state:
        path = state.pop(0)
        if path[-1].is_game_over():
            score = path[-1].get_endgame_score(maximize)
            leaf_node.append((path,score,1))
        else:
            children = path[-1].generate_next_states()
            new_path = []
            for child in children:
                temp_path = path.copy()
                temp_path.append(child)
                new_path.append(temp_path)
            leaf_node.append(minimax_endgame_search(new_path,not maximize))

    for leaf in leaf_node:
        static_eval_count += leaf[2]
    new_leaf = []
    for leaf in leaf_node:
        new_leaf.append((leaf[0], leaf[1], static_eval_count))
        
    if not maximize:
        return max(new_leaf, key = lambda leaf:leaf[1])
    else:
        return min(new_leaf, key = lambda leaf:leaf[1])



# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

# pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))


def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True) :
    """Performs standard minimax search. Same return type as dfs_maximizing."""
    if type(state) != list:
        state = [[state]]
    leaf_node = []
    static_eval_count = 0
    
    while state:
        path = state.pop(0)
        if path[-1].is_game_over() or depth_limit == 0:
            if path[-1].is_game_over():
                score = path[-1].get_endgame_score(maximize)
                leaf_node.append((path,score,1))
            else:
                score = heuristic_fn(path[-1].snapshot, maximize)
                leaf_node.append((path,score,1))
        else:
            children = path[-1].generate_next_states()
            new_path = []
            for child in children:
                temp_path = path.copy()
                temp_path.append(child)
                new_path.append(temp_path)
            leaf_node.append(minimax_search(new_path, heuristic_fn, depth_limit -1, not maximize))

    for leaf in leaf_node:
        static_eval_count += leaf[2]
    new_leaf = []
    for leaf in leaf_node:
        new_leaf.append((leaf[0], leaf[1], static_eval_count)) 
        # new_leaf = [(leaf[0],leaf[1],static_eval_count)] + new_leaf
        
    if not maximize:
        return max(new_leaf, key = lambda leaf:leaf[1])
    else:
        return min(new_leaf, key = lambda leaf:leaf[1])


# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1. Try increasing the value of depth_limit to see what happens:


# pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=2))


def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True) :
    """"Performs minimax with alpha-beta pruning. Same return type 
    as dfs_maximizing."""
    global static_eval_count
    
    if type(state) != list:
        state = [state]
        static_eval_count = 0
    
    best_path = state
        
    if state[-1].is_game_over():
        static_eval_count += 1
        return (state,state[-1].get_endgame_score(maximize),static_eval_count)
    elif depth_limit == 0:
        static_eval_count += 1
        return (state,heuristic_fn(state[-1].snapshot, maximize),static_eval_count)
    else:
        if maximize:
            children = state[-1].generate_next_states()
            new_paths = []
            for child in children:
                new_paths = new_paths + [state + [child]]
            for new_path in new_paths:
                end_game = minimax_search_alphabeta(new_path,alpha,beta,heuristic_fn,depth_limit-1,not maximize)
                if end_game[1] > alpha:
                    alpha = max(alpha,end_game[1])
                    best_path = end_game[0]
                if alpha >= beta:
                    break
            return (best_path,alpha,static_eval_count)
        else:
            children = state[-1].generate_next_states()
            new_paths = []
            for child in children:
                new_paths = new_paths + [state + [child]]
            for new_path in new_paths:
                end_game = minimax_search_alphabeta(new_path,alpha,beta,heuristic_fn,depth_limit-1,not maximize)
                if end_game[1] < beta:
                    beta = min(beta,end_game[1])
                    best_path = end_game[0]
                if alpha >= beta:
                    break        
            return (best_path,beta,static_eval_count)




# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4. Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

# pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True) :
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. Returns anytime_value."""
    
    depth = 1
    anytime_value = AnytimeValue()
    
    while depth <= depth_limit:
        best_path = minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=heuristic_fn,
                             depth_limit=depth, maximize=True)
        anytime_value.set_value(best_path)
        depth += 1
    
    return anytime_value





# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4. Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

# progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4).pretty_print()


# Progressive deepening is NOT optional. However, you may find that 
#  the tests for progressive deepening take a long time. If you would
#  like to temporarily bypass them, set this variable False. You will,
#  of course, need to set this back to True to pass all of the local
#  and online tests.
TEST_PROGRESSIVE_DEEPENING = True
if not TEST_PROGRESSIVE_DEEPENING:
    def not_implemented(*args): raise NotImplementedError
    progressive_deepening = not_implemented


#### Part 3: Multiple Choice ###################################################

ANSWER_1 = '4'

ANSWER_2 = '1'

ANSWER_3 = '4'

ANSWER_4 = '5'


#### SURVEY ###################################################

NAME = 'Luke Chiang'
COLLABORATORS = 'Andrew Tresansky'
HOW_MANY_HOURS_THIS_LAB_TOOK = '8'
WHAT_I_FOUND_INTERESTING = 'Playing against the AI'
WHAT_I_FOUND_BORING = 'None'
SUGGESTIONS = 'None'
