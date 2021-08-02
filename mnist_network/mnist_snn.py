import os

import argparse

import numpy as np

from time import perf_counter

import torch
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from tqdm.auto import tqdm

from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.network.network import load

from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=0)
parser.add_argument("--update_steps", type=int, default=256)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=100)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--n_folds", type=int, default=6)
parser.set_defaults(gpu=False)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
batch_size = args.batch_size
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
update_steps = args.update_steps
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
gpu = args.gpu
n_folds = args.n_folds

update_interval = update_steps * batch_size

if gpu and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    if gpu:
        print("Could not use CUDA")
        gpu = False
torch.manual_seed(seed)
print("Running on Device = %s\n" % (device))

if not os.path.isfile("results_" + str(n_neurons) + "N.csv"):

    results_file = open("results_" + str(n_neurons) + "N.csv", "a+")

    results_file.write("Neurons,Batch Size,Epochs,")

    for fold in range(n_folds):
        results_file.write("Fold " + str(fold+1) + " Train,")
        results_file.write("Fold " + str(fold+1) + " Val,")
        if fold < (n_folds-1):
            results_file.write("Fold " + str(fold+1) + " Test,")
        else:
            results_file.write("Fold " + str(fold+1) + " Test\n")
else:
    results_file = open("results_" + str(n_neurons) + "N.csv", "a+")

if not os.path.isfile("execution_time_" + str(n_epochs) + "E.csv"):

    times_file = open("execution_time_" + str(n_epochs) + "E.csv", "a+")

    times_file.write("Neurons,Batch Size,Epochs,")

    for epoch in range(n_epochs):
        times_file.write("Epoch " + str(epoch+1) + ",")

    times_file.write("Eval Train,Eval Val,Eval Test\n")
else:
    times_file = open("execution_time_" + str(n_epochs) + "E.csv", "a+")

# Create the directory to store the networks
dirName = "networks/"
if not os.path.exists(dirName):
    os.mkdir(dirName)

# Load MNIST train data
train_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root="./data",
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Stratified K-Fold declaration
skfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

results_file.write(str(n_neurons) + "," +
                   str(batch_size) + "," + str(n_epochs) + ",")

times_file.write(str(n_neurons) + "," +
                   str(batch_size) + "," + str(n_epochs) + ",")

# K-Folds loop
for fold, (train_indices, val_indices) in enumerate(skfold.split(np.zeros(n_train), torch.narrow(train_dataset.targets, 0, 0, n_train))):

    print("*********** FOLD %s **********" % (fold + 1))

    # Build network
    network = DiehlAndCook2015(
        n_inpt=784,
        n_neurons=n_neurons,
        exc=exc,
        inh=inh,
        dt=dt,
        norm=78.4,
        nu=(1e-4, 1e-2),
        theta_plus=theta_plus,
        inpt_shape=(1, 28, 28),
    )

    # Directs network to GPU
    if gpu:
        network.to("cuda")

    # Neuron assignments and spike proportions.
    n_classes = 10
    assignments = -torch.ones(n_neurons, device=device)
    proportions = torch.zeros((n_neurons, n_classes), device=device)
    rates = torch.zeros((n_neurons, n_classes), device=device)

    # Set up monitors for spikes
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(
            network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
        )
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    spike_record = torch.zeros(
        (update_interval, int(time / dt), n_neurons), device=device)

    ################# TRAINING THE NETWORK #########################
    print("\nBegin training.\n")

    # Create a dataloader to iterate and batch data
    train_sampler = SubsetRandomSampler(train_indices)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=gpu,
        sampler=train_sampler
    )

    for epoch in range(n_epochs):

        if fold == 0:
            start = perf_counter()

        labels = []

        for step, batch in enumerate(tqdm(train_dataloader, desc="Batches processed")):
            # Get next input sample.
            inputs = {"X": batch["encoded_image"]}
            if gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            if (step % update_steps == 0 and step > 0):
                # Convert the array of labels into a tensor
                label_tensor = torch.tensor(labels, device=device)

                # Assign labels to excitatory layer neurons
                assignments, proportions, rates = assign_labels(
                    spikes=spike_record,
                    labels=label_tensor,
                    n_labels=n_classes,
                    rates=rates,
                )

                labels = []

            labels.extend(batch["label"].tolist())

            # Run the network on the input.
            network.run(inputs=inputs, time=time, input_time_dim=1)

            # Add to spikes recording.
            s = spikes["Ae"].get("s").permute((1, 0, 2))
            spike_record[
                (step * batch_size)
                % update_interval: (step * batch_size % update_interval)
                + s.size(0)
            ] = s

            network.reset_state_variables()  # Reset state variables.

        if fold == 0:
            end = perf_counter()
            times_file.write(str(end - start) + ",")

    print("\nTraining complete.")

    ############ SAVE THE NETWORK AND ITS FILES ####################

    dirName = "networks/" + str(n_neurons) + "N_" + str(batch_size) + "BS_" + str(
        n_epochs) + "E_" + str(n_folds) + "F/" + str(fold+1) + "F/"
    os.makedirs(dirName)

    network.save(dirName + "network.pt")

    torch.save(assignments, dirName + "assignments.pt")

    torch.save(proportions, dirName + "proportions.pt")

    ###################### LOAD THE NETWORK ########################

    # network = load(dirName + "network.pt", map_location="cpu", learning=False)

    # assignments = torch.load(dirName + "assignments.pt")

    # proportions = torch.load(dirName + "proportions.pt")

    # # Set up monitors for spikes
    # spikes = {}
    # for layer in set(network.layers):
    #     spikes[layer] = Monitor(
    #         network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    #     )
    #     network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    ################# EVALUATION TRAIN SET #########################

    # Train set accuracies
    accuracy = {"all": 0, "proportion": 0}

    print("\nBegin evaluation train set.\n")
    network.train(mode=False)

    folds_train_samples = len(train_indices)

    if fold == 0:
        start = perf_counter()

    for step, batch in enumerate(tqdm(train_dataloader, desc="Batches processed")):
        # Get next input sample.
        inputs = {"X": batch["encoded_image"]}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        # Add to spikes recording.
        spike_record = spikes["Ae"].get("s").permute((1, 0, 2))

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
        accuracy["all"] += float(torch.sum(label_tensor.long()
                                 == all_activity_pred).item())
        accuracy["proportion"] += float(
            torch.sum(label_tensor.long() == proportion_pred).item()
        )

        network.reset_state_variables()  # Reset state variables.

    if fold == 0:
        end = perf_counter()
        times_file.write(str(end - start) + ",")

    all_mean_accuracy = round((accuracy["all"] / folds_train_samples) * 100, 2)
    proportion_mean_accuracy = round(
        (accuracy["proportion"] / folds_train_samples) * 100, 2)

    print("\nAll accuracy train set: %.2f" % (all_mean_accuracy))
    print("Proportion weighting accuracy train set: %.2f" %
          (proportion_mean_accuracy))

    results_file.write(str(all_mean_accuracy) + ",")

    print("\nEvaluation train set complete.\n")
    #####################################################################

    ################# EVALUATION VALIDATION SET #########################
    val_sampler = SubsetRandomSampler(val_indices)
    val_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=gpu,
        sampler=val_sampler
    )

    # Validation set accuracies
    accuracy = {"all": 0, "proportion": 0}

    print("Begin evaluation validation set.\n")
    network.train(mode=False)

    folds_val_samples = len(val_indices)

    if fold == 0:
        start = perf_counter()

    for step, batch in enumerate(tqdm(val_dataloader, desc="Batches processed")):
        # Get next input sample.
        inputs = {"X": batch["encoded_image"]}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        # Add to spikes recording.
        spike_record = spikes["Ae"].get("s").permute((1, 0, 2))

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
        accuracy["all"] += float(torch.sum(label_tensor.long()
                                 == all_activity_pred).item())
        accuracy["proportion"] += float(
            torch.sum(label_tensor.long() == proportion_pred).item()
        )

        network.reset_state_variables()  # Reset state variables.

    if fold == 0:
        end = perf_counter()
        times_file.write(str(end - start) + ",")

    all_mean_accuracy = round((accuracy["all"] / folds_val_samples) * 100, 2)
    proportion_mean_accuracy = round(
        (accuracy["proportion"] / folds_val_samples) * 100, 2)

    print("\nAll accuracy validation set: %.2f" % (all_mean_accuracy))
    print("Proportion weighting accuracy validation set: %.2f \n" %
          (proportion_mean_accuracy))

    results_file.write(str(all_mean_accuracy) + ",")

    print("Evaluation validation set complete.\n")
    #####################################################################################

    ################# EVALUATION TEST SET #########################

    # Load MNIST test data
    test_dataset = MNIST(
        PoissonEncoder(time=time, dt=dt),
        None,
        root="./data",
        download=True,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
        ),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=gpu,
    )

    # Test set accuracies
    accuracy = {"all": 0, "proportion": 0}

    print("Begin evaluation test set.\n")
    network.train(mode=False)

    if fold == 0:
        start = perf_counter()

    for step, batch in enumerate(tqdm(test_dataloader, desc="Batches processed")):
        # Get next input sample.
        inputs = {"X": batch["encoded_image"]}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        # Add to spikes recording.
        spike_record = spikes["Ae"].get("s").permute((1, 0, 2))

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
        accuracy["all"] += float(torch.sum(label_tensor.long()
                                 == all_activity_pred).item())
        accuracy["proportion"] += float(
            torch.sum(label_tensor.long() == proportion_pred).item()
        )

        network.reset_state_variables()  # Reset state variables.

    if fold == 0:
        end = perf_counter()
        times_file.write(str(end - start) + "\n")

    all_mean_accuracy = round((accuracy["all"] / n_test) * 100, 2)
    proportion_mean_accuracy = round(
        (accuracy["proportion"] / n_test) * 100, 2)

    print("\nAll accuracy test set: %.2f" % (all_mean_accuracy))
    print("Proportion weighting accuracy test set: %.2f \n" %
          (proportion_mean_accuracy))

    if fold < (n_folds-1):
        results_file.write(str(all_mean_accuracy) + ",")
    else:
        results_file.write(str(all_mean_accuracy) + "\n")

    print("Evaluation test set complete.\n")
    #####################################################################################
