import argparse
import os
from time import time as t
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')  # Use the TkAgg backend
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from torchmetrics import PearsonCorrCoef

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--method", dest="method", default="mse")
parser.add_argument("--n_compression", dest="n_compression", default="0")

parser.set_defaults(plot=False, gpu=True, train=False)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu
method = args.method
n_compression = int(args.n_compression)

# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False
# print(device)
torch.set_num_threads(os.cpu_count() - 1)
# print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = 0  # gpu * 4 * torch.cuda.device_count()

if not train:
    update_interval = n_test

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

# Build network.
network = DiehlAndCook2015(
    n_inpt=784,
    n_neurons=64,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)

# Build network.
network_2 = DiehlAndCook2015(
    n_inpt=784,
    n_neurons=64,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)

# Load network_1
MODEL_PATH = Path("models")
MODEL_NAME = "model_1_64_neurons.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
network.load_state_dict(torch.load(f=MODEL_SAVE_PATH, map_location=torch.device(device)))
# print("Model 1 loaded")

# Load network_1 label assignments
assignments = torch.load("models/assinments_1_64_neurons.pth", map_location=torch.device(device))
# print("assignments loaded")
proportions = torch.load("models/proportions_1_64_neurons.pth", map_location=torch.device(device))
# print("proporttions loaded")


# Load network_2
MODEL_PATH = Path("models")
MODEL_NAME = "model_2_64_neurons.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
network_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH, map_location=torch.device(device)))
# print("Model 2 loaded")

# Load network_2 label assignments
assignments_2 = torch.load("models/assinments_2_64_neurons.pth", map_location=torch.device(device))
# print("assignments_2 loaded")
proportions_2 = torch.load("models/proportions_2_64_neurons.pth", map_location=torch.device(device))
# print("proporttions_2 loaded")

# Get weight for each model
weight_1 = network.X_to_Ae.w.transpose(0, 1)
weight_2 = network_2.X_to_Ae.w.transpose(0, 1)

# Function to plot 2D receptive field
def plot_receptive_field(network):
    input_exc_weights = network.X_to_Ae.w
    square_weights = get_square_weights(
        input_exc_weights.view(784, network.n_neurons), n_sqrt, 28
    )
    square_assignments = get_square_assignments(assignments, n_sqrt)
    weights_im = None
    weights_im = plot_weights(square_weights, im=weights_im)
    plt.pause(1e-8)
    input("Enter to close plot")

# Function to modify label assignment
def modify_label_assignment(assignments, proportions, indices_to_remove):
    """
    Removes specified elements from label assignments
    Args:
        indices_to_remove(list): list containing indices to remove
    Returns:
        None
    """
    mask = torch.ones(assignments.size(), dtype=bool)
    mask[indices_to_remove] = False
    assignments = assignments[mask]
    proportions = proportions[mask, :]
    return assignments, proportions



# Function to calculate Manhattan distance
def manhattan(w1, w2):
  return torch.sum( torch.abs(w1-w2) )

# Function to calculate Mean Squared Error
MSELoss = torch.nn.MSELoss()

# Function to calculate Cosine similarity
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)

# Function to calculate Pearson Correlation Coefficient
pearson = PearsonCorrCoef()

if method == "manhattan":
    sim = manhattan
elif method == "mse":
    sim = MSELoss
elif method == "cos":
    sim = cos
elif method == "corr":
    sim = pearson


# Get all iterations of two neurons from model 1 and 2
# Store difference with pair ex. (i, j, distance)
pairs = []
for i, w1 in enumerate(weight_1):
  for j, w2 in enumerate(weight_2):
    pairs.append((i, j, sim(w1, w2)))

# Sort pairs by distance
if method == "manhattan" or method == "mse":
    pairs.sort(key = lambda x:x[2])
elif method == "cos" or method == "corr":
    pairs.sort(key = lambda x:x[2], reverse=True)

for i in range(200):
    print(pairs[i])
input("break point after sorting pairs")
# Get indices of neurons to be removed from model 2
removes = set()
for i, j, _ in pairs:
    if len(removes) == n_compression:
        break
    removes.add(j)
    w1, w2 = network.X_to_Ae.w.data[i], network_2.X_to_Ae.w.data[j]
    network.X_to_Ae.w.data[i] = (w1 + w2) / 2

removes = list(removes)

# Remove neurons from model 2
network_2.reduce_neurons(removes)

# Modify label assignment for model 2 accordingly
assignments_2, proportions_2 = modify_label_assignment(assignments_2, proportions_2, removes)

# Merge model 1 and 2
network.merge_model(network_2)

# Merge label assignment
assignments = torch.cat( (assignments, assignments_2), 0 )
assignments.to(device)
proportions = torch.cat( (proportions, proportions_2), 0 )
proportions.to(device)

# Adjust number of neurons
n_neurons = 64 + 64 - n_compression

# Directs network to GPU
if gpu:
    network.to("cuda")

### Test merged model
# Record spikes during the simulation.
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)


# Neuron assignments and spike proportions.
n_classes = 10
# assignments = -torch.ones(n_neurons, device=device)
# proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(
    network.layers["Ae"], ["v"], time=int(time / dt), device=device
)
inh_voltage_monitor = Monitor(
    network.layers["Ai"], ["v"], time=int(time / dt), device=device
)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")


# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None


# Load MNIST data.
test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation.
spike_record = torch.zeros((1, int(time / dt), n_neurons), device=device)


# print("\nBegin testing\n")
network.train(mode=False)
start = t()


pbar = tqdm(total=n_test)
for step, batch in enumerate(test_dataset):
    if step >= n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time)

    # Add to spikes recording.
    spike_record[0] = spikes["Ae"].get("s").squeeze()

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()

"""
print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")
"""
acc = (accuracy["all"] / n_test)
print(f"Method: {method}, n_compression: {n_compression}, Accuracy: {acc}")
