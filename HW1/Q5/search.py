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
import queue

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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST

    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    from util import Stack
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
    stack = Stack()
    visited = []
    start = problem.getStartState()
    visited.append((start,0))
    stack.push((start, 0, []))
    print(stack.list)

    if problem.isGoalState(problem.getStartState()):
        return []

    while True:
        if stack.isEmpty():
            return []
        print(stack.list)

        curPos, hit_walls, path = stack.pop()
        if hit_walls > 2:
            continue

        if problem.isGoalState(curPos) and hit_walls > 0:
            return path

        for successor, action, stepCost in problem.getSuccessors(curPos):
            if problem.isWall(successor):
                nextState = (successor, hit_walls+1)
            else:
                nextState = (successor, hit_walls)

            if nextState not in visited and nextState:
                stack.push((nextState[0], nextState[1], path+[action]))
                visited.append(nextState)

    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    visited = []
    posQueue = util.Queue()
    posQueue.push((problem.getStartState(), 0, []))
    visited.append((problem.getStartState(), 0))

    if problem.isGoalState(problem.getStartState()):
        return []

    while True:
        if posQueue.isEmpty():
            return []

        curPos, walls_hit, current_actions = posQueue.pop()

        if walls_hit > 2:
            continue

        if problem.isGoalState(curPos) and walls_hit > 0:
            return current_actions
        print(f"Walls hit: {walls_hit}")
        for successor, action, stepCost in problem.getSuccessors(curPos):
            if walls_hit < 3:
                if problem.isWall(successor):
                    new_walls_hit = walls_hit + 1
                    nextState = (successor, new_walls_hit, action)
                else:
                    nextState = (successor, walls_hit, action)

            if nextState not in visited and nextState:
                action = nextState[2]
                posQueue.push((nextState[0], nextState[1], current_actions + [action]))
                visited.append(nextState)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    priorityQueue = util.PriorityQueue()
    visited = []
    distance = {}

    distance[(start,0)] = 0
    print(distance)
    priorityQueue.push((start, 0, []), 0)
    visited.append((start, 0))

    while True:
        if priorityQueue.isEmpty():
            return []

        currentState, hitWalls, currentActions = priorityQueue.pop()

        if hitWalls > 2:
            continue

        if problem.isGoalState(currentState) and hitWalls > 0 and hitWalls < 3:
            return currentActions

        for successor, action, stepCost in problem.getSuccessors(currentState):
            if problem.isWall(successor):
                nextState = (successor, hitWalls+1)
            else:
                nextState = (successor, hitWalls)

            costToGo = problem.getCostOfActions(currentActions + [action])
            if costToGo < distance.get(nextState, float('inf')):
                priorityQueue.push((nextState[0], nextState[1], currentActions + [action]), costToGo)
                distance[nextState] = costToGo

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    priorityQueue = util.PriorityQueue()
    visited = []
    distance = {}

    priorityQueue.push((start, 0, []), priority=0)
    visited.append((start, 0))
    distance[(start,0)] = 0

    while True:
        if priorityQueue.isEmpty():
            continue

        currentState, hitWalls, currentActions = priorityQueue.pop()

        if hitWalls > 2:
            continue

        if problem.isGoalState(currentState) and hitWalls > 0:
            return currentActions

        for successor, action, stepCost in problem.getSuccessors(currentState):
            if problem.isWall(successor):
                nextState = (successor, hitWalls+1)
            else:
                nextState = (successor, hitWalls)

            print(successor)
            costToGo = problem.getCostOfActions(currentActions + [action]) + heuristic(successor, problem)

            if costToGo < distance.get(nextState, float('inf')):
                priorityQueue.push((nextState[0], nextState[1], currentActions + [action]), costToGo)
                distance[nextState] = costToGo


    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
