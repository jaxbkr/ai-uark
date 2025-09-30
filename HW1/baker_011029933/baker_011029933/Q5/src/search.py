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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    "*** YOUR CODE HERE ***"
    
    stack = util.Stack()
    visited = set()

    # (curPos, wallsHit)
    visited.add((problem.getStartState(),0))

    # (curPos, wallsHit, path)
    stack.push((problem.getStartState(),0,[]))

    if problem.isGoalState(problem.getStartState()):
        return []
    
    while True:
        if stack.isEmpty():
            return []
        
        curPos, wallsHit, path = stack.pop()
        if wallsHit > 2:
            continue

        if problem.isGoalState(curPos) and wallsHit > 0 and wallsHit <=2:
            return path
        
        for neighbor, dir, cost in problem.getSuccessors(curPos):
            if problem.isWall(neighbor):
                nextPos = (neighbor, wallsHit + 1)
            else:
                nextPos = (neighbor, wallsHit)

            if nextPos not in visited and nextPos:
                stack.push((nextPos[0],nextPos[1], path+[dir]))
                visited.add(nextPos)
    
    
    
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    visited = set()

    # (curPos, wallsHit)
    visited.add((problem.getStartState(),0))

    # (curPos, wallsHit, path)
    queue.push((problem.getStartState(),0,[]))

    if problem.isGoalState(problem.getStartState()):
        return []
    
    while True:
        if queue.isEmpty():
            return []
        
        curPos, wallsHit, path = queue.pop()
        if wallsHit > 2:
            continue

        if problem.isGoalState(curPos) and wallsHit > 0 and wallsHit <=2:
            return path
        
        for neighbor, dir, cost in problem.getSuccessors(curPos):
            if problem.isWall(neighbor):
                nextPos = (neighbor, wallsHit + 1)
            else:
                nextPos = (neighbor, wallsHit)

            if nextPos not in visited and nextPos:
                queue.push((nextPos[0],nextPos[1], path+[dir]))
                visited.add(nextPos)

    

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    priorityQueue = util.PriorityQueue()
    distance = dict()

    # [(pos), cost]
    distance[(problem.getStartState(), 0)] = 0

    # (curPos, wallsHit, path), priority
    priorityQueue.push((problem.getStartState(),0,[]),0)

    if problem.isGoalState(problem.getStartState()):
        return []
    
    while True:
        if priorityQueue.isEmpty():
            return []
        
        curPos, wallsHit, path = priorityQueue.pop()
        if wallsHit > 2:
            continue

        if problem.isGoalState(curPos) and wallsHit > 0 and wallsHit <=2:
            return path
        
        for neighbor, dir, cost in problem.getSuccessors(curPos):
            if problem.isWall(neighbor):
                nextPos = (neighbor, wallsHit + 1)
            else:
                nextPos = (neighbor, wallsHit)

            costOfNextPos = (problem.getCostOfActions(path + [dir]))
            if costOfNextPos < distance.get(nextPos, float('inf')):
                priorityQueue.push((nextPos[0],nextPos[1], path+[dir]), costOfNextPos)
                distance[nextPos] = costOfNextPos

    

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    priorityQueue = util.PriorityQueue()
    distance = dict()

    # [(pos), cost]
    distance[(problem.getStartState(), 0)] = 0
    # (curPos, wallsHit, path), priority
    priorityQueue.push((problem.getStartState(),0,[]), priority=0)

    if problem.isGoalState(problem.getStartState()):
        return []
    
    while True:
        if priorityQueue.isEmpty():
            return []
        
        curPos, wallsHit, path = priorityQueue.pop()
        if wallsHit > 2:
            continue

        if problem.isGoalState(curPos) and wallsHit > 0 and wallsHit <=2:
            return path
        
        for neighbor, dir, cost in problem.getSuccessors(curPos):
            if problem.isWall(neighbor):
                nextPos = (neighbor, wallsHit + 1)
            else:
                nextPos = (neighbor, wallsHit)

            costOfNextPos = (problem.getCostOfActions(path + [dir]) + heuristic(neighbor, problem))
            if costOfNextPos < distance.get(nextPos, float('inf')):
                priorityQueue.push((nextPos[0],nextPos[1], path+[dir]), costOfNextPos)
                distance[nextPos] = costOfNextPos
            


    
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
