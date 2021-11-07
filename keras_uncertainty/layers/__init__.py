from .dropconnect_layers import DropConnectDense, DropConnectConv2D
from .rbf_layers import RBFClassifier, add_gradient_penalty, add_l2_regularization, duq_training_loop
from .bayes_backprop_layers import BayesByBackpropDense
from .flipout_layers import FlipoutDense
from .stochastic_layers import SamplingSoftmax, StochasticDropout
#from .swag_layers import SWADiagonalDense
