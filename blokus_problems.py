import numpy as np

from board import Board
from search import SearchProblem, ucs
import util


class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in
                state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.expanded = 0
        "*** YOUR CODE HERE ***"
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        return state.state[0][0] == 0 and state.state[-1][0] == 0 \
               and state.state[0][-1] == 0 and state.state[-1][-1] == 0

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for
                move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        return sum(action.piece.get_num_tiles() for action in actions)


def blokus_corners_heuristic(state, problem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """
    "*** YOUR CODE HERE ***"
    # return sum(1 if corner == -1 else 0 for corner in
    #            [state.state[-1][0], state.state[0][-1], state.state[-1][-1]])
    # valid_tiles = [(x, y) for x, y in (range(state.board_w), range(
    #     state.board_h)) if state.check_tile_legal(1, x, y)]
    # valid_tiles = []
    # for x in range(state.board_w):
    #     for y in range(state.board_h):
    #         if state.check_tile_legal(0, x, y):
    #             valid_tiles.append((x, y))
    # tile_dist = []
    # for x, y in valid_tiles:
    #     tile_dist.append([min(x, state.board_h - y) + state.board_h - y - x
    #                       if min(x, state.board_h - y) == x else x - (
    #                 state.board_h - y),
    #
    #                       min(state.board_w - x, state.board_h - y) +
    #                       state.board_h - (y + state.board_w - x) if min(
    #                           state.board_w - x, state.board_h - y) ==
    #                                                                  state.board_w - x else state.board_w - (
    #                               x + state.board_h - y),
    #                       min(y, state.board_w - x) - state.board_w - x - y
    #                       if min(y, state.board_w - x) == y else y - (
    #                               state.board_w - x)])
    # tile_dist = np.array(tile_dist)
    # return np.min(tile_dist[:, 0]) + np.min(tile_dist[:, 1]) + np.min(
    #     tile_dist[:, 2])
    w, h = state.board_w, state.board_h
    valid_tiles = []
    # if not can_be_valid_board(state):
    #     return h + w
    for x in range(w):
        for y in range(h):
            if state.check_tile_legal(0, x, y):
                if has_croos(x, y, state) and has_no_sides(x, y, state):
                    valid_tiles.append((x, y))
    if not valid_tiles:
        return h + w + min(h, w)
    tile_dist = []

    for x, y in valid_tiles:
        tile_dist.append([min(x, h - y) + (h - y - x if min(x, h - y) == x else x - (h - y)),
                          min(w - x, y) + (y - (w - x) if min(w - x, y) == (w - x) else w - (x + y)),
                          min(h - y, w - x) + (w - x - (h - y) if min(h - y, w - x) == h - y else h - y - (w - x))])
    tile_dist = np.array(tile_dist)
    return np.min(tile_dist[:, 0]) + np.min(tile_dist[:, 1]) + np.min(
        tile_dist[:, 2]) + sum(1 if corner == -1 else 0 for corner in
                               [state.state[-1][0], state.state[0][-1],
                                state.state[-1][-1]])
    # return sum(1 if corner == -1 else 0 for corner in
    #                            [state.state[-1][0], state.state[0][-1],
    #                             state.state[-1][-1]])


def can_be_valid_board(state):
    w, h = state.board_w, state.board_h
    if state.get_position(w-2, h-1) == 0 or state.get_position(w-1, h-2) == 0 or state.get_position(w-2, 0) == 0\
            or state.get_position(w-1, 1) == 0 or state.get_position(0, h-2) == 0 or state.get_position(1, h-1) == 0:
        return False
    return True


def has_croos(x, y, state):
    daigs_index = [(1, 1), (-1, -1), (-1, 1), (1, -1)]
    for diag in daigs_index:
        if 0 <= x + diag[0] < state.board_w and 0 <= y + diag[1] < state.board_h:
            if state.get_position(x + diag[0], y + diag[1]) == 0:
                return True
    return False


def has_no_sides(x, y, state):
    sides_index = [(0, 1), (-1, 0), (1, 0), (0, -1)]
    for side in sides_index:
        if 0 <= x + side[0] < state.board_w and 0 <= y + side[1] < state.board_h:
            if state.get_position(x + side[0], y + side[1]) != -1:
                return False
    return True


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0),
                 targets=[(0, 0)]):
        self.targets = targets.copy()
        self.expanded = 0
        "*** YOUR CODE HERE ***"
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        for target in self.targets:
            if state.get_position(target[1], target[0]) != 0:
                return False
        return True

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for
                move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        return sum(action.piece.get_num_tiles() for action in actions)


def blokus_cover_heuristic(state, problem):
    "*** YOUR CODE HERE ***"
    return sum(1 if state.get_position(target[1], target[0]) == -1 else 0 for target in problem.targets)


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0),
                 targets=(0, 0)):
        self.expanded = 0
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        """
        This method should return a sequence of actions that covers all target locations on the board.
        This time we trade optimality for speed.
        Therefore, your agent should try and cover one target location at a time. Each time, aiming for the closest uncovered location.
        You may define helpful functions as you wish.

        Probably a good way to start, would be something like this --

        current_state = self.board.__copy__()
        backtrace = []

        while ....

            actions = set of actions that covers the closets uncovered target location
            add actions to backtrace

        return backtrace
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0),
                 targets=(0, 0)):
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
