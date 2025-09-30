# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

    def isWall(self, state):
        """
            state: The current search state
            Return true if the current state is a wall
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem, initialHit=0, returnHit=False):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    startState = problem.getStartState()
    stack = util.Stack()
    visited = []

    stack.push((startState, 0, []))
    visited.append((startState, initialHit))

    while True:
        if stack.isEmpty():
            return []
        currentState, hitWalls, currentActions = stack.pop()
        if hitWalls > 2:
            continue
        if problem.isGoalState(currentState) and hitWalls >= 1 and hitWalls <= 2:
            if not returnHit:
                return currentActions
            else:
                return currentActions, hitWalls - initialHit

        for successor, action, stepCost in problem.getSuccessors(currentState):
            if problem.isWall(successor):
                nextState = (successor, hitWalls + 1)
            else:
                nextState = (successor, hitWalls)
            if nextState in visited:
                continue
            stack.push((nextState[0], nextState[1], currentActions + [action]))
            visited.append(nextState)

def breadthFirstSearch(problem, initialHit=0, returnHit=False):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    startState, _ = problem.getStartState()
    queue = util.Queue()
    visited = []

    queue.push((startState, 0, []))
    visited.append((startState, initialHit))

    while True:
        if queue.isEmpty():
            return []
        currentState, hitWalls, currentActions = queue.pop()
        if hitWalls > 2:
            continue
        if problem.isGoalState(currentState) and hitWalls >= 1 and hitWalls <= 2:
            if not returnHit:
                return currentActions
            else:
                return currentActions, hitWalls - initialHit

        for successor, action, stepCost in problem.getSuccessors(currentState):
            if problem.isWall(successor):
                nextState = (successor, hitWalls + 1)
            else:
                nextState = (successor, hitWalls)
            if nextState in visited:
                continue
            queue.push((nextState[0], nextState[1], currentActions + [action]))
            visited.append(nextState)


def uniformCostSearch(problem, initialHit=0, returnHit=False):
    """Search the node of least total cost first."""
    startState = problem.getStartState()
    queue = util.PriorityQueue()

    queue.push((startState, initialHit, []), 0)
    distance = {}
    state = (tuple(startState), initialHit)
    distance[state] = 0

    while True:
        if queue.isEmpty():
            return []
        currentState, hitWalls, currentActions = queue.pop()

        if hitWalls > 2:
            continue

        if problem.isGoalState(currentState) and hitWalls >= 1 and hitWalls <= 2:
            if not returnHit:
                return currentActions
            else:
                return currentActions, hitWalls - initialHit

        for successor, action, stepCost in problem.getSuccessors(currentState):

            if problem.isWall(successor):
                nextState = (successor, hitWalls + 1)
            else:
                nextState = (successor, hitWalls)
            state = (tuple(nextState[0]), nextState[1])
            costToGo = problem.getCostOfActions(currentActions + [action])
            if state not in distance or costToGo < distance[state]:
                queue.push((nextState[0], nextState[1], currentActions + [action]), costToGo)
                distance[state] = costToGo


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic, initialHit=0, returnHit=False):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    startState = problem.getStartState()
    queue = util.PriorityQueue()

    queue.push((startState, initialHit, []), 0) ## Current Position, Number of Hitting Walls, List of Actions
    distance = {}
    state = (tuple(startState), initialHit)
    distance[state] = 0
    while True:
        if queue.isEmpty():
            return []
        currentState, hitWalls, currentActions = queue.pop()

        if hitWalls > 2:
            continue

        if problem.isGoalState(currentState) and hitWalls >= 1 and hitWalls <= 2:
            if not returnHit:
                return currentActions
            else:
                return currentActions, hitWalls - initialHit

        for successor, action, stepCost in problem.getSuccessors(currentState):

            if problem.isWall(successor):
                nextState = (successor, hitWalls + 1)
            else:
                nextState = (successor, hitWalls)

            state = (tuple(nextState[0]), nextState[1])
            costToGo = problem.getCostOfActions(currentActions + [action]) + heuristic(successor, problem)

            if state not in distance or costToGo < distance[state]:
                queue.push((nextState[0], nextState[1], currentActions + [action]), costToGo)
                distance[state] = costToGo


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
