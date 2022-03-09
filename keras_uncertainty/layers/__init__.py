from .dropconnect_layers import DropConnectDense, DropConnectConv1D, DropConnectConv2D
from .rbf_layers import RBFClassifier, add_gradient_penalty, add_l2_regularization, duq_training_loop
from .variational_layers import VariationalDense, VariationalConv1D, VariationalConv2D, VariationalConv3D
from .flipout_layers import FlipoutDense, FlipoutConv1D, FlipoutConv2D, FlipoutConv3D
from .stochastic_layers import SamplingSoftmax, StochasticDropout
#from .swag_layers import SWADiagonalDense
