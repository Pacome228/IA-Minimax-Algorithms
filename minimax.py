from pacman_module.game import Agent
from pacman_module.pacman import Directions
import numpy as np


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
        

    def max_value(self, state, closed):
        """
        Given a pacman game state, computes the MAX of the game .

        Arguments:
        ----------
        - `state`: A game state. See FAQ and class
                   `pacman.GameState`.
        - `closed`: A set that contains states already visited

        Return:
        -------
        - The maximum value of the Minimax Algorithm.
        """
        # Check if is terminal stateÒ
        if(state.isWin() or state.isLose()):
            return state.getScore()
        # Adds the current state in the closed
        closed.add(self.get_key(state))
        v = -np.inf #Initialization of value
        pacman_actions = state.generatePacmanSuccessors()
        for (next_node, _) in pacman_actions:
            # Check if nextnode not in closed
            if self.get_key(next_node) not in closed:
                v = max(v, self.min_value(next_node, closed.copy()))
        if abs(v) == np.inf:
            # there all node is visidted and the current state is the terminal
            v = state.getScore()
        # return the high value
        return v

    def min_value(self, state, closed):
        """
        Given a pacman game state, computes the MIN of the game .

        Arguments:
        ----------
        - `state`: A game state. See FAQ and class
                   `pacman.GameState`.
        - `closed`: A set that contains states already computed

        Return:
        -------
        - The minimum value of the Minimax Algorithm.
        """
        # Check if is the terminal state
        if (state.isWin() or state.isLose()):
            return state.getScore()
        # Adds the state in the closed
        closed.add(self.get_key(state))
        v = np.inf #initialisation of value
        for (next_node, _) in state.generateGhostSuccessors(self.GHOST):
            # Check if nextnode not in visited
            if self.get_key(next_node) not in closed:
                v = min(v, self.max_value(next_node, closed.copy()))
        if abs(v) == np.inf:
            # there all node is visidted and the current state is the terminal
            v = state.getScore()
        return v

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

    def utility(self, state):
        """
        Given a pacman game state, returns the score of the state.

        Arguments:
        ----------
        - `state`: A game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - The score of the state.
        """
        return state.getScore()

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
        return hash((state.getFood(), state.getPacmanPosition(),
                     state.getGhostPosition(self.GHOST)))

    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move corresponding to the
        Minimax algorithm.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """
        #MINIMAX algorithm

        # Set of already node visited
        closed = set()
        #all the legal actions of pacman.
        pacman_actions = state.generatePacmanSuccessors()
        # Add for the start node in the closed
        closed.add(self.get_key(state))
        v = -np.inf #Initialization of value
        #action to be returned at the end
        best_action = None 
        for (next_node, action) in pacman_actions :
            # return immediatly the action if next_node is a Win state
            if next_node.isWin():
                return action
            score = self.min_value(next_node, closed.copy())
            # Keep the action that bring the highest score
            if (score > v):
                v = score
                best_action = action
        return best_action


