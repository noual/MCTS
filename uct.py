import mcts
from mcts import MCTAlgorithm

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
