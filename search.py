"""
In search.py, you will implement generic search algorithms
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

	print("Start:", problem.get_start_state().state)
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    "*** YOUR CODE HERE ***"
    # return dfs_helper(problem, problem.get_start_state(), list(), set())
    action_list = []
    visited_dict = dict()
    fringe = util.Stack()
    state = None
    fringe.push(((problem.get_start_state(), 0, 0), None))
    while not fringe.isEmpty():
        state = fringe.pop()
        if problem.is_goal_state(state[0][0]):
            break
        if state[0][0] not in visited_dict.keys():
            visited_dict[state[0][0]] = state
            for successor in problem.get_successors(state[0][0]):
                if successor[0] not in visited_dict.keys():
                    fringe.push((successor, state[0]))
    while state[0][0] is not problem.get_start_state():
        action_list.append(state[0][1])
        state = visited_dict[state[1][0]]
    return action_list[::-1]


# def dfs_help?er(problem, state, action_list, visited_list):
# if problem.is_goal_state(state):
#     return action_list
# for successor in problem.get_successors(state):
#     if successor[0] not in visited_list:
#         visited_list.add(successor[0])
#         temp_actions = dfs_helper(problem, successor[0],
#                                   action_list + [successor[1]],
#                                   visited_list)
#         if temp_actions:
#             return temp_actions
# return None


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    action_list = []
    visited_dict = dict()
    fringe = util.Queue()
    state = None
    fringe.push(((problem.get_start_state(), 0, 0), None))
    while not fringe.isEmpty():
        state = fringe.pop()
        if problem.is_goal_state(state[0][0]):
            break
        if state[0][0] not in visited_dict.keys():
            visited_dict[state[0][0]] = state
            for successor in problem.get_successors(state[0][0]):
                if successor[0] not in visited_dict.keys():
                    fringe.push((successor, state[0]))
    while state[0][0] is not problem.get_start_state():
        action_list.append(state[0][1])
        state = visited_dict[state[1][0]]
    return action_list[::-1]


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    "*** YOUR CODE HERE ***"
    action_list = []
    fringe = util.PriorityQueue()
    state = None
    fringe.push(problem.get_start_state(), 0)
    visited_dict = {problem.get_start_state(): [0, 0, 0]}
    while not fringe.isEmpty():
        state = fringe.pop()
        cost_to_state = visited_dict[state][2]
        if problem.is_goal_state(state):
            break
        for successor in problem.get_successors(state):
            if successor[0] not in visited_dict.keys():
                visited_dict[successor[0]] = [successor, state, cost_to_state + successor[2]]
                fringe.push(successor[0], cost_to_state + successor[2])
    while state is not problem.get_start_state():
        action_list.append(visited_dict[state][0][1])
        state = visited_dict[state][1]
    return action_list[::-1]


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    "*** YOUR CODE HERE ***"
    action_list = []
    fringe = util.PriorityQueue()
    state = None
    fringe.push(problem.get_start_state(), 0)
    visited_dict = {problem.get_start_state(): [0, 0, 0]}
    while not fringe.isEmpty():
        state = fringe.pop()
        cost_to_state = visited_dict[state][2]
        if problem.is_goal_state(state):
            break
        for successor in problem.get_successors(state):
            if successor[0] not in visited_dict.keys():
                visited_dict[successor[0]] = [successor, state,
                                              heuristic(state, problem) + cost_to_state + successor[2]]
                fringe.push(successor[0], heuristic(state, problem) + cost_to_state + successor[2])
    while state is not problem.get_start_state():
        action_list.append(visited_dict[state][0][1])
        state = visited_dict[state][1]
    return action_list[::-1]


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
