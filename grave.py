import rave
from rave import RAVE

class GRAVE(RAVE):
  """
  GRAVE GRAVE
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