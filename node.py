class Node():

  """
  Node class : represents a node in the game tree.
  """

  def __init__(self, state, player, move=None, parent=None):
    self.state = state  # L'état du plateau au moment de l'initialisation du noeud
    self.player = player  # Le joueur qui effectue le mouvement
    self.move = move
    self.parent = parent  # None pour le noeud racine
    self.children = []  # Les enfants du noeud
    self.results = []  # Ajouter 1 pour une victoire, ajouter 0 pour une défaite

  def get_children(self):
    return self.children

  def is_leaf(self):
    return True if len(self.children) == 0 else False

  def is_terminal(self):
    return self.state.is_over()

class RaveNode(Node):
  """
  Contient un attribut supplémentaire results_amaf
  qui contient les résultats amaf des playouts
  """

  def __init__(self, state, player, move=None, parent=None):
    super().__init__(state, player, move, parent)
    self.results_amaf = []