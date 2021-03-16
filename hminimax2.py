from pacman_module.game import Agent
from pacman_module.pacman import Directions
import numpy as np
from pacman_module.util import PriorityQueue, manhattanDistance


class PacmanAgent(Agent):

    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args
        # ghost id
        self.GHOST = 1
        # pacman id
        self.PACMAN = 0

        #depth of game tree
        self.depth = 9
  
     
    def max_value(self, state, alpha, beta, depth):
        """
        Given a pacman game state, computes the MAX of the game.

        Arguments:
        ----------
        - `state`: A game state. See FAQ and class
                   `pacman.GameState`.
        - `depth`: the current depth 
        - `alpha`: alpha value 
        - `beta`: beta value 

        Return:
        -------
        - The maximum value of the hMinimax Algorithm with alpha beta pruning.
        """
        #Checking of cutoff
        if self.cutoff_test(state, depth):
            return self.evaluation_function(state)
        v = -np.inf #Initialization of value
        #computing of value if the next state is a legal action of the pacman
        for (next_node, _) in state.generatePacmanSuccessors():
            v = max(
                v, 
                self.min_value(
                    next_node,
                     alpha,
                      beta,
                       depth - 1))
                
            # the pruning
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, state, alpha, beta, depth):
        """
        Given a pacman game state, computes the MIN of the game.

        Arguments:
        ----------
        - `state`: A game state. See FAQ and class
                   `pacman.GameState`.
        - `depth`: the current depth 
        - `alpha`: alpha value 
        - `beta`: beta value 
        Return:
        -------
        - The minimum value of the hMinimax.
        """
        #Checking of cutoff
        if self.cutoff_test(state, depth):
            return self.evaluation_function(state)
        v = np.inf
        #computing of value if the next state is a legal action of the ghost
        for (next_node, _) in state.generateGhostSuccessors(self.GHOST):
            v = min(
                v, 
                self.max_value(
                    next_node,
                     alpha,
                      beta,
                       depth - 1))
                
            # the pruning
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def cutoff_test(self, state, depth):
        """
        Given a pacman game state, checks if the state is a cutoff state.

        Arguments:
        ----------
        - `state`: A game state. See FAQ and class
                   `pacman.GameState`.
        - `depth`: the current depth 


        Return:
        -------
        - True if the game is Lose or Won or Max is reached
        """
        return state.isWin() or state.isLose() or depth == 0

    def terminal_test(self, state):
        """
        Given a pacman game state, checks if the state is a terminal state

        Arguments:
        ----------
        - `state`: A game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - True if the game is Lose or Won,
          False otherwise.
        """
        if state.isWin() or state.isLose():
            return True
        return False

    def evaluation_function(self, state):
        """
        Given a pacman game state, computes the evaluation function 

        Arguments:
        ----------
        - `state`: A game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - `score`: The evalution function.
        """
        dots= state.getFood().asList()
        score = state.getScore() - self.heuristic(state, dots)
        return score

    

    def heuristic(self, state, foodState):
        """
        Given a pacman game and food state,returns the cost of heuristic in heuristic .

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                    `pacman.GameState`.
        - `foodState`: set that contains the coordinates a and b of every food

        Return:
        -------
        - the heuristic.
        """
        pacmanPosition = state.getPacmanPosition()
        mDistance = []
        
        for x, y in foodState:
            gCost = abs(x - pacmanPosition[0]) + abs(y - pacmanPosition[1])
            mDistance.append(gCost)
        
        if mDistance:
            heuristic = mDistance[0]
            for x in mDistance:
                if (x < heuristic):
                    heuristic = x

            return heuristic
        return 0  
    

    
    def get_key(self, state):
        """
        Given a pacman state, returns a key that represents the state

        Arguments:
        ----------
        - `state`: A game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - A key that represents the state.
        """
        return hash((state.getPacmanPosition(),
                     state.getFood(), state.getGhostPosition(self.GHOST)))

    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """
        #HMINIMAX algorithm

        # Initialization
        alpha = -np.inf
        beta = np.inf
        v = -np.inf
        best_action = None
        pacman_actions = state.generatePacmanSuccessors()
        for (next_node, action) in pacman_actions:
             # return immediatly the action if next_node is a Win state
            if next_node.isWin():
                return action
            score = self.min_value(next_node, alpha, beta, self.depth - 1)
            # Keep the action that bring the highest score
            if score > v:
                v = score
                best_action = action
            # Update alpha value
            alpha = max(alpha, v)
        return best_action

    
