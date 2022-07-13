from checkers import game
from node import Node
import copy
import numpy as np
import tqdm

class MCTAlgorithm():

  """
  Default class for MCTS Algorithm.
  """

  def __init__(self, game: game.Game, disable = False):
    self.root = Node(game, player=game.whose_turn())
    self.disable = disable


  def playout(self, state, player: int) -> int:
    """
    Creates random playout and returns result
    """
    playout_game = copy.deepcopy(state)

    while not playout_game.is_over():
      moves = playout_game.get_possible_moves()
      selected_move = moves[np.random.randint(0, len(moves))]
      playout_game.move(selected_move)

    winner = playout_game.get_winner()
    # If the winner is the same as the player, return 1 (win), else return 0
    return 1 if winner == player else 0

