from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.spatial.distance import euclidean
from torch.nn.modules.utils import _pair

from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes
from bindsnet.network.topology import Connection, LocalConnection


class TwoLayerNetwork(Network):
    # language=rst
    """
    Implements an ``Input`` instance connected to a ``LIFNodes`` instance with a
    fully-connected ``Connection``.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        dt: float = 1.0,
        wmin: float = 0.0,
        wmax: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        norm: float = 78.4,
    ) -> None:
        # language=rst
        """
        Constructor for class ``TwoLayerNetwork``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of neurons in the ``LIFNodes`` population.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param wmax: Maximum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param norm: ``Input`` to ``LIFNodes`` layer connection weights normalization
            constant.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.n_neurons = n_neurons
        self.dt = dt

        self.add_layer(Input(n=self.n_inpt, traces=True, tc_trace=20.0), name="X")
        self.add_layer(
            LIFNodes(
                n=self.n_neurons,
                traces=True,
                rest=-65.0,
                reset=-65.0,
                thresh=-52.0,
                refrac=5,
                tc_decay=100.0,
                tc_trace=20.0,
            ),
            name="Y",
        )

        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        self.add_connection(
            Connection(
                source=self.layers["X"],
                target=self.layers["Y"],
                w=w,
                update_rule=PostPre,
                nu=nu,
                reduction=reduction,
                wmin=wmin,
                wmax=wmax,
                norm=norm,
            ),
            source="X",
            target="Y",
        )


class DiehlAndCook2015(Network):
    # language=rst
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        exc: float = 22.5,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-3, 1e-2), # default: 1e-4, 1e-2
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
        inh_thresh: float = -40.0,
        exc_thresh: float = -52.0,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt

        # Layers
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        exc_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=exc_thresh,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        inh_layer = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=-60.0,
            reset=-45.0,
            thresh=inh_thresh,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,
        )

        # Connections
        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        exc_inh_conn = Connection(
            source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=self.exc
        )
        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        inh_exc_conn = Connection(
            source=inh_layer, target=exc_layer, w=w, wmin=-self.inh, wmax=0
        )

        # Add to network
        self.add_layer(input_layer, name="X")
        self.add_layer(exc_layer, name="Ae")
        self.add_layer(inh_layer, name="Ai")
        self.add_connection(input_exc_conn, source="X", target="Ae")
        self.add_connection(exc_inh_conn, source="Ae", target="Ai")
        self.add_connection(inh_exc_conn, source="Ai", target="Ae")
    
    def add_neurons(self, n_ext):
        """
        Method for ``DiehlAndCook2015``
        Adds neurons to the model
        :param n_ext: number of neurons to be added to network
        """
        print("in add_neurons method")
        input()

        # Modifying model
        # Network
        self.n_neurons += n_ext

        # Excitatory layer
        extension = torch.full((self.Ae.s.shape[0], n_ext), False) # Extend spike occurance with initialized value False
        self.Ae.s = torch.cat((self.Ae.s, extension), 1)

        extension = torch.full((self.Ae.x.shape[0], n_ext), 0)     # Extend spike traces with initialized value 0
        self.Ae.x = torch.cat((self.Ae.x, extension), 1)


        extension = torch.full((self.Ae.v.shape[0], n_ext), -65)   # Extend neuron valtages with initialized value -65
        self.Ae.v = torch.cat((self.Ae.v, extension), 1)

        extension = torch.full((n_ext,), 0)                           # Extend adaptive thresholds with initialized value 0
        self.Ae.theta = torch.cat((self.Ae.theta, extension), 0)


        extension = torch.full((self.Ae.refrac_count.shape[0], n_ext), 0)   # Extend refractory counts with initialized value 0
        self.Ae.refrac_count = torch.cat((self.Ae.refrac_count, extension), 1)





        # Inhibitory layer
        extension = torch.full((self.Ai.s.shape[0], n_ext), False) # Extend spike occurance with initialized value False
        self.Ai.s = torch.cat((self.Ai.s, extension), 1)

        extension = torch.full((self.Ai.v.shape[0], n_ext), -60) # Extend spike occurance with initialized value False
        self.Ai.v = torch.cat((self.Ai.v, extension), 1)

        extension = torch.full((self.Ai.refrac_count.shape[0], n_ext), 0)   # Extend refractory counts with initialized value 0
        self.Ai.refrac_count = torch.cat((self.Ai.refrac_count, extension), 1)




        # Input to excitatory connection
        extension = 0.3 * torch.rand(self.n_inpt, n_ext) # Extend weight values with randomized value
        self.X_to_Ae.w.data = torch.nn.Parameter(
            torch.cat((self.X_to_Ae.w.data, extension), 1), requires_grad = True
        )
        self.connections["X", "Ae"].source.shape = [self.n_neurons] # Extend source of connection to self.n_neurons
        self.connections["X", "Ae"].target.shape = [self.n_neurons] # Extend target of connection to self.n_neurons




        # Excitatory to inhibitory connection
        extension = self.exc * torch.ones(n_ext) # Extend weight values by initialized hyper parameter
        self.Ae_to_Ai.w.data = torch.nn.Parameter(
            torch.diag(
                torch.cat((torch.diag(self.Ae_to_Ai.w.data), extension), 0)
            ), requires_grad = True
        )
        self.connections["Ae", "Ai"].source.shape = [self.n_neurons] # Extend source of connection to self.n_neurons
        self.connections["Ae", "Ai"].target.shape = [self.n_neurons] # Extend target of connection to self.n_neurons




        # Inhibitory to excitatory connection
        self.Ai_to_Ae.w.data = torch.nn.Parameter( # Extend inhibitory weight values by initialized hyper parameter
            -self.inh * (
                torch.ones(self.n_neurons, self.n_neurons)
                - torch.diag(torch.ones(self.n_neurons))
            ),
            requires_grad = True
        )
        self.connections["Ai", "Ae"].source.shape = [self.n_neurons] # Extend source of connection to self.n_neurons
        self.connections["Ai", "Ae"].target.shape = [self.n_neurons] # Extend target of connection to self.n_neurons


    def reduce_neurons(self, indices_to_remove):
        """
        Method for ``DiehlAndCook2015``
        Reduces neurons from network
        :param indices_to_remove: list of indices of neurons to be removed
        """


        # Modifying model
        # Create mask to remove elements from tensor
        mask = torch.ones(self.n_neurons, dtype=bool)
        mask[indices_to_remove] = False

        # Network
        self.n_neurons -= len(indices_to_remove)

        # Excitatory layer
        self.Ae.s = self.Ae.s[:, mask]
        self.Ae.x = self.Ae.x[:, mask]
        self.Ae.v = self.Ae.v[:, mask]
        self.Ae.theta = self.Ae.theta[mask]
        self.Ae.refrac_count = self.Ae.refrac_count[:, mask]

        # Inhibitory layer
        self.Ai.s = self.Ai.s[:, mask]
        self.Ai.v = self.Ai.v[:, mask]
        self.Ai.refrac_count = self.Ai.refrac_count[:, mask]

        # Input to excitatory connection
        self.X_to_Ae.w.data = torch.nn.Parameter(self.X_to_Ae.w.data[:, mask], requires_grad = True)
        self.connections["X", "Ae"].source.shape = [self.n_neurons]
        self.connections["X", "Ae"].target.shape = [self.n_neurons]

        # Excitatory to inhibitory connection
        self.Ae_to_Ai.w.data = torch.nn.Parameter(self.Ae_to_Ai.w.data[mask, :], requires_grad = True)
        self.Ae_to_Ai.w.data = torch.nn.Parameter(self.Ae_to_Ai.w.data[:, mask], requires_grad = True)
        self.connections["Ae", "Ai"].source.shape = [self.n_neurons]
        self.connections["Ae", "Ai"].target.shape = [self.n_neurons]

        # Inhibitory to excitatory connection
        self.Ai_to_Ae.w.data = torch.nn.Parameter(self.Ai_to_Ae.w.data[mask, :], requires_grad = True)
        self.Ai_to_Ae.w.data = torch.nn.Parameter(self.Ai_to_Ae.w.data[:, mask], requires_grad = True)
        self.connections["Ai", "Ae"].source.shape = [self.n_neurons]
        self.connections["Ai", "Ae"].target.shape = [self.n_neurons]


    def merge_model(self, network_2):
        """
        Method for ``DiehlAndCook2015``
        Merges this network and given network
        :param network_2: Model to be merged
        """
        # Merge
        # Network
        self.n_neurons += network_2.n_neurons

        # Excitaory layer
        self.Ae.s = torch.cat((self.Ae.s, network_2.Ae.s), 1) # Merge spike occurances
        self.Ae.x = torch.cat((self.Ae.x, network_2.Ae.x), 1) # Merge spike traces
        self.Ae.v = torch.cat((self.Ae.v, network_2.Ae.v), 1) # Merge membrane potentials
        self.Ae.theta = torch.cat((self.Ae.theta, network_2.Ae.theta), 0) # Merge adaptive thresholds
        self.Ae.refrac_count = torch.cat((self.Ae.refrac_count, network_2.Ae.refrac_count), 1) # Merge refractory counts



        # Inhibitory layer
        self.Ai.s = torch.cat((self.Ai.s, network_2.Ai.s), 1) # Merge spike occurances
        self.Ai.v = torch.cat((self.Ai.v, network_2.Ai.v), 1) # Merge membrane potentials
        self.Ai.refrac_count = torch.cat((self.Ai.refrac_count, network_2.Ai.refrac_count), 1) # Merge refractory counts


        # Input to excitatory connection
        #  Merge weight values
        self.X_to_Ae.w.data = torch.nn.Parameter(
            torch.cat((self.X_to_Ae.w.data, network_2.X_to_Ae.w.data), 1), requires_grad = False
        )
        self.connections["X", "Ae"].source.shape = [self.n_neurons] # Extend source of connection to n_neurons
        self.connections["X", "Ae"].target.shape = [self.n_neurons] # Extend target of connection to n_neurons


        # Excitatory to inhibitory connection
        #  Merge weight values
        self.Ae_to_Ai.w.data = torch.nn.Parameter(
            torch.diag(
                torch.cat((torch.diag(self.Ae_to_Ai.w.data), torch.diag(network_2.Ae_to_Ai.w.data)), 0)
            ), requires_grad = False
        )
        self.connections["Ae", "Ai"].source.shape = [self.n_neurons] # Extend source of connection to n_neurons
        self.connections["Ae", "Ai"].target.shape = [self.n_neurons] # Extend target of connection to n_neurons


        # Inhibitory to excitatory connection
        #  Expand Ai to Ae weight metrix
        self.Ai_to_Ae.w.data = torch.nn.Parameter(
            -self.inh * (
                torch.ones(self.n_neurons, self.n_neurons)
                - torch.diag(torch.ones(self.n_neurons))
            ),
            requires_grad = False
        )
        self.connections["Ai", "Ae"].source.shape = [self.n_neurons] # Extend source of connection to n_neurons
        self.connections["Ai", "Ae"].target.shape = [self.n_neurons] # Extend target of connection to n_neurons


class DiehlAndCook2015v2(Network):
    # language=rst
    """
    Slightly modifies the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_ by removing
    the inhibitory layer and replacing it with a recurrent inhibitory connection in the
    output layer (what used to be the excitatory layer).
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: Optional[float] = 0.0,
        wmax: Optional[float] = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
        exc_thresh: float = -52.0,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015v2``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.inh = inh
        self.dt = dt

        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        self.add_layer(input_layer, name="X")

        output_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=exc_thresh,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        self.add_layer(output_layer, name="Y")

        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        input_connection = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        self.add_connection(input_connection, source="X", target="Y")

        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        recurrent_connection = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w,
            wmin=-self.inh,
            wmax=0,
        )
        self.add_connection(recurrent_connection, source="Y", target="Y")
    
    def reduce_neuron(self):
        print("In reduce_neuron()")


class IncreasingInhibitionNetwork(Network):
    # language=rst
    """
    Implements the inhibitory layer structure of the spiking neural network architecture
    from `(Hazan et al. 2018) <https://arxiv.org/abs/1807.09374>`_
    """

    def __init__(
        self,
        n_input: int,
        n_neurons: int = 100,
        start_inhib: float = 1.0,
        max_inhib: float = 100.0,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
        exc_thresh: float = -52.0,
    ) -> None:
        # language=rst
        """
        Constructor for class ``IncreasingInhibitionNetwork``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_input = n_input
        self.n_neurons = n_neurons
        self.n_sqrt = int(np.sqrt(n_neurons))
        self.start_inhib = start_inhib
        self.max_inhib = max_inhib
        self.dt = dt
        self.inpt_shape = inpt_shape

        input_layer = Input(
            n=self.n_input, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        self.add_layer(input_layer, name="X")

        output_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=exc_thresh,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        self.add_layer(output_layer, name="Y")

        w = 0.3 * torch.rand(self.n_input, self.n_neurons)
        input_output_conn = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        self.add_connection(input_output_conn, source="X", target="Y")

        # add internal inhibitory connections
        w = torch.ones(self.n_neurons, self.n_neurons) - torch.diag(
            torch.ones(self.n_neurons)
        )
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if i != j:
                    x1, y1 = i // self.n_sqrt, i % self.n_sqrt
                    x2, y2 = j // self.n_sqrt, j % self.n_sqrt

                    w[i, j] = np.sqrt(euclidean([x1, y1], [x2, y2]))
        w = w / w.max()
        w = (w * self.max_inhib) + self.start_inhib
        recurrent_output_conn = Connection(
            source=self.layers["Y"], target=self.layers["Y"], w=w
        )
        self.add_connection(recurrent_output_conn, source="Y", target="Y")


class LocallyConnectedNetwork(Network):
    # language=rst
    """
    Defines a two-layer network in which the input layer is "locally connected" to the
    output layer, and the output layer is recurrently inhibited connected such that
    neurons with the same input receptive field inhibit each other.
    """

    def __init__(
        self,
        n_inpt: int,
        input_shape: List[int],
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        n_filters: int,
        inh: float = 25.0,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: Optional[float] = 0.2,
        exc_thresh: float = -52.0,
    ) -> None:
        # language=rst
        """
        Constructor for class ``LocallyConnectedNetwork``. Uses ``DiehlAndCookNodes`` to
        avoid multiple spikes per timestep in the output layer population.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param input_shape: Two-dimensional shape of input population.
        :param kernel_size: Size of input windows. Integer or two-tuple of integers.
        :param stride: Length of horizontal, vertical stride across input space. Integer
            or two-tuple of integers.
        :param n_filters: Number of locally connected filters per input region. Integer
            or two-tuple of integers.
        :param inh: Strength of synapse weights from output layer back onto itself.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on ``Input`` to ``DiehlAndCookNodes``
            synapses.
        :param wmax: Maximum allowed weight on ``Input`` to ``DiehlAndCookNodes``
            synapses.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param norm: ``Input`` to ``DiehlAndCookNodes`` layer connection weights
            normalization constant.
        """
        super().__init__(dt=dt)

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)

        self.n_inpt = n_inpt
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_filters = n_filters
        self.inh = inh
        self.dt = dt
        self.theta_plus = theta_plus
        self.tc_theta_decay = tc_theta_decay
        self.wmin = wmin
        self.wmax = wmax
        self.norm = norm

        if kernel_size == input_shape:
            conv_size = [1, 1]
        else:
            conv_size = (
                int((input_shape[0] - kernel_size[0]) / stride[0]) + 1,
                int((input_shape[1] - kernel_size[1]) / stride[1]) + 1,
            )

        input_layer = Input(n=self.n_inpt, traces=True, tc_trace=20.0)

        output_layer = DiehlAndCookNodes(
            n=self.n_filters * conv_size[0] * conv_size[1],
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=exc_thresh,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        input_output_conn = LocalConnection(
            input_layer,
            output_layer,
            kernel_size=kernel_size,
            stride=stride,
            n_filters=n_filters,
            nu=nu,
            reduction=reduction,
            update_rule=PostPre,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            input_shape=input_shape,
        )

        w = torch.zeros(n_filters, *conv_size, n_filters, *conv_size)
        for fltr1 in range(n_filters):
            for fltr2 in range(n_filters):
                if fltr1 != fltr2:
                    for i in range(conv_size[0]):
                        for j in range(conv_size[1]):
                            w[fltr1, i, j, fltr2, i, j] = -inh

        w = w.view(
            n_filters * conv_size[0] * conv_size[1],
            n_filters * conv_size[0] * conv_size[1],
        )
        recurrent_conn = Connection(output_layer, output_layer, w=w)

        self.add_layer(input_layer, name="X")
        self.add_layer(output_layer, name="Y")
        self.add_connection(input_output_conn, source="X", target="Y")
        self.add_connection(recurrent_conn, source="Y", target="Y")