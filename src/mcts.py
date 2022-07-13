import numpy as np
from checkers import game
from .node import Node, RaveNode
import copy
from tqdm import tqdm

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


class UCT(MCTAlgorithm):

  """
  Upper Confidence Bound applied to trees
  """

  def __init__(self, game: game.Game, n_iter: int=1000, C=.4):
    super().__init__(game)
    self.name = "UCT"
    self.n_iter = n_iter
    self.C = C


  def _ucb(self, child: Node, parent: Node) -> int:
    # Returns UCB score for a child node
    if len(child.results) == 0 :  # Si le noeud n'a pas encore été visité !
      return 10000
    mu_i = np.mean(child.results)
    return mu_i + self.C * (np.sqrt(np.log(len(parent.results)) / len(child.results)))  # Beware of division by 0 ?


  def _selection(self, node: Node) -> Node:
    """
    Selects a candidate child node from the input node.
    """
    if not node.is_leaf():  # Tant que l'on a pas atteint une feuille de l'arbre
      # Choisir le meilleur noeud enfant
      candidate_id = np.argmax([self._ucb(child, node) for child in node.get_children()])
      candidate = node.get_children()[candidate_id]
      return self._selection(candidate)

    return node


  def _expansion(self, node: Node) -> Node:
    """
    Unless L ends the game decisively (e.g. win/loss/draw) for either player,
    create one (or more) child nodes and choose node C from one of them.
    Child nodes are any valid moves from the game position defined by L.
    """
    if not node.is_terminal():
      node.children = [Node(copy.deepcopy(node.state),
                            player=node.state.whose_turn(),
                            move=m,
                            parent=node)
                             for m in node.state.get_possible_moves()]
      
      # Play the move for each child (updates the board in the child nodes)
      for child in node.children:
        child.state.move(child.move)
      return node.children[np.random.randint(0, len(node.children))]

    return node


  def _backpropagation(self, node: Node, result: int):
    """
    Backpropagates the result of a playout up the tree.
    """
    if node.parent is None:
      node.results.append(result)
      return "Done"
    node.results.append(result)  # Ajouter le résultat à la liste
    return self._backpropagation(node.parent, result)  # Fonction récursive

    
  def __call__(self, player: int):
    """
    Body of UCT
    """
    # Si il n'y a qu'un coup à jouer
    if len(self.root.state.get_possible_moves()) == 1:
      return Node(self.root.state, player=player, move=self.root.state.get_possible_moves()[0])

    for i in tqdm(range(self.n_iter), disable=self.disable):
      leaf_node = self._selection(self.root)
      expanded_node = self._expansion(leaf_node)
      result = self.playout(expanded_node.state, player=player)
      _ = self._backpropagation(expanded_node, result)

    best_move_id = np.argmax([self._ucb(child, self.root) for child in self.root.get_children()])
    best_move = self.root.get_children()[best_move_id]
    for child in self.root.get_children():
      print(f"Child : {child.move} Results : {np.mean(child.results)} Games played : {len(child.results)}")

    return best_move

class RAVE(MCTAlgorithm):
  """
  RAVE algorithm
  """

  def __init__(self, game: game.Game, n_iter: int=1000, C=.4, C_amaf=.8, b=1e-5):
    super().__init__(game)
    self.name = "RAVE"
    self.n_iter = n_iter
    self.list_nodes = []
    self.list_nodes.append(self.root)
    self.C = C
    self.C_amaf = C_amaf
    self.b = b


  def playout(self, state: game.Game, player: int):
    """
    Crée un playout aléatoire et retourne le résultat ainsi que la liste des moves
    """
    #print('rave playout')
    playout_game = copy.copy(state)
    list_moves = []

    while not playout_game.is_over():
      moves = playout_game.get_possible_moves()
      next_player = playout_game.whose_turn()
      selected_move = moves[np.random.randint(0, len(moves))]
      list_moves.append((selected_move, next_player))
      playout_game.move(selected_move)

    winner = playout_game.get_winner()
    # If the winner is the same as the player, return 1 (win), else return 0
    del playout_game  # No unnecesary garbage
    result = 1 if winner == player else 0
    # print(f"[Playout] Player : {player}, winner : {winner}, result : {result}")
    # print(f"Moves played during playout : {list_moves}")
    return result, list_moves

  def _ucb(self, child: Node, parent: Node) -> int:
    # Returns UCB score for a child node
    if len(child.results) == 0 :  # Si le noeud n'a pas encore été visité !
      return 10000
    mu_i = np.mean(child.results)
    return mu_i + self.C * (np.sqrt(np.log(len(parent.results)) / len(child.results)))  # Beware of division by 0 ?

  def _ucb_rave(self, child: Node, parent: Node) -> int:
    # Returns UCB RAVE score for a child node
    if len(child.results) == 0 or len(child.results_amaf) == 0:  # Si le noeud n'a pas encore été visité !
      return 10000
    
    """
    Formula :
    (1 - beta(ni, nî))*(wi/ni) + beta(ni, nî)*(wî/nî) + c*sqrt(ln(t)/ni)
    """
    
    ni = len(child.results)
    ni2 = len(child.results_amaf)
    beta = ni2 / (ni + ni2 + self.b*ni*ni2)
    wi_ni = np.nan_to_num(np.mean(child.results), 0)
    wi_ni2 = np.nan_to_num(np.mean(child.results_amaf), 0)
    # print((1-beta)*wi_ni + (beta)*wi_ni2 + C*np.sqrt(np.log(len(parent.results))/ni))
    return (1-beta)*wi_ni + (beta)*wi_ni2 + self.C_amaf*np.sqrt(np.log(len(parent.results))/ni)


  def _selection(self, node: Node) -> Node:
    """
    Selects a candidate child node from the input node.
    """
    if node.is_leaf():
      return node

    elif node.parent is None:
      candidate_id = np.argmax([self._ucb(child, node) for child in node.get_children()])
      candidate = node.get_children()[candidate_id]
      return self._selection(candidate)

    else:  # Tant que l'on a pas atteint une feuille de l'arbre
      # Choisir le meilleur noeud enfant
      candidate_id = np.argmax([self._ucb_rave(child, node) for child in node.get_children()])
      # for i, child in enumerate(node.get_children()):
      #   print(f"[Selection] [{i}] results = {np.mean(child.results)} ({len(child.results)})")
      #   print(f"[Selection] [{i}] results_amaf = {np.mean(child.results_amaf)} ({len(child.results_amaf)})")
      #   print(f"[Selection] [{i}] ucbrave value = {self._ucb_rave(child, node)}")
      # print(f"[Selection] : {candidate_id}")
      candidate = node.get_children()[candidate_id]
      return self._selection(candidate)

    

  def _expansion(self, node: Node) -> Node:
    """
    Unless L ends the game decisively (e.g. win/loss/draw) for either player,
    create one (or more) child nodes and choose node C from one of them.
    Child nodes are any valid moves from the game position defined by L.
    """
    if not node.is_terminal():
      # print(f"[RAVE] [Expansion] Possible moves : {node.state.get_possible_moves()}")
      node.children = [RaveNode(copy.deepcopy(node.state),
                                player=node.state.whose_turn(),
                                move=m,
                                parent=node)
                             for m in node.state.get_possible_moves()]
      for child in node.children:
        self.list_nodes.append(child)
      
      # Play the move for each child (updates the board in the child nodes)
      for child in node.children:
        # print(f"[RAVE] [Expansion] Child move : {child.move}")
        child.state.move(child.move)
      return node.children[np.random.randint(0, len(node.children))]

    return node


  def _backpropagation(self, node: Node, result: int, list_moves: list):
    """
    Backpropagates the result of a playout up the tree.
    """
    if node.parent is None:
      node.results.append(result)

      # Parcourir les moves du playout
      for move in list_moves:
        for node_2 in self.list_nodes:
          if node_2.move == move[0]:
            node_2.results_amaf.append(result)
          
      return "Done"

    node.results.append(result)  # Ajouter le résultat à la liste
    
    return self._backpropagation(node.parent, result, list_moves)  # Fonction récursive

    
  def __call__(self, player: int):
    """
    Body of RAVE
    """
    # Si il n'y a qu'un coup à jouer
    if len(self.root.state.get_possible_moves()) == 1:
      return RaveNode(self.root.state, player=player, move=self.root.state.get_possible_moves()[0])

    for i in tqdm(range(self.n_iter), disable=self.disable):
      leaf_node = self._selection(self.root)
      expanded_node = self._expansion(leaf_node)
      result, list_moves = self.playout(expanded_node.state, player=player)
      _ = self._backpropagation(node = expanded_node,
                                result = result,
                                list_moves = list_moves)

    for child in self.root.get_children():
      print(f"Child : {child.move} Results : {np.mean(child.results)} Games played : {len(child.results)}")
    best_move_id = np.argmax([self._ucb(child, self.root) for child in self.root.get_children()])
    best_move = copy.deepcopy(self.root.get_children()[best_move_id])
    for n in self.list_nodes:
      del n
    self.list_nodes = []

    return best_move

class GRAVE(RAVE):
  """
  GRAVE algorithm
  """

  def __init__(self, game: game.Game, n_iter: int=1000, C=.4, C_amaf=.8, b=1e-5, ref: int=10):
    super().__init__(game, n_iter, C, C_amaf, b)
    self.ref = ref
    self.name = "GRAVE"

  def _ucb_rave(self, child: Node, parent: Node, C=.9, b=1e-5) -> int:
    """
    If child doesn't have more than ref playouts, we go up the tree !
    """
    tref = child
    tref_parent = parent

    while len(tref.results) <= self.ref:
      if tref_parent.parent is None:  # Root of tree
        break
      tref = tref_parent
      tref_parent = tref_parent.parent

    # Returns UCB RAVE score for a child node
    if len(child.results) == 0 or len(child.results_amaf) == 0:  # Si le noeud n'a pas encore été visité !
      return 10000
    
    """
    Formula :
    (1 - beta(ni, nî))*(wi/ni) + beta(ni, nî)*(wî/nî) + c*sqrt(ln(t)/ni)
    """
    
    ni = len(child.results)
    ni2 = len(tref.results_amaf)
    beta = ni2 / (ni + ni2 + b*ni*ni2)
    wi_ni = np.nan_to_num(np.mean(child.results), 0)
    wi_ni2 = np.nan_to_num(np.mean(tref.results_amaf), 0)
    # print((1-beta)*wi_ni + (beta)*wi_ni2 + C*np.sqrt(np.log(len(parent.results))/ni))
    return (1-beta)*wi_ni + (beta)*wi_ni2 + C*np.sqrt(np.log(len(parent.results))/ni)