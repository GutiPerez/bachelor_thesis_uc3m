#############################
#   AUTHOR: Miguel Gutierrez Perez
#   LAST MODIFICATION DATE: 21/09/2021
#############################

########## IMPORTS ##########
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

#############################

########## ARGUMENTS PARSING ##########

# Define the argument parser.
parser = argparse.ArgumentParser()

# Add the desired arguments, defining their type (and default value for numeric arguments).
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

# Set the default values for boolean arguments.
parser.set_defaults(gpu=False)

# Parse the arguments and store them in a variable.
args = parser.parse_args()

# Store all arguments in variables.
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

#######################################

# Number of iterations to update the graphs.
update_interval = update_steps * batch_size

########## DEVICE SELECTION ##########

# Check if the 'gpu' argument is true and if the GPU is available to work with.
if gpu and torch.cuda.is_available():
    # Fix the GPU as the device to use.
    device = torch.device("cuda")
else:
    # Fix the CPU as the device to use.
    device = torch.device("cpu")
    if gpu:
        print("Could not use CUDA")
        gpu = False

######################################

# It exists more methods for establishing the seed, but this way the seed is fixed for all devices.
torch.manual_seed(seed)

# Print information about what device is being used.
print("Running on Device = %s\n" % (device))

######### CREATE/OPEN RESULTS FILE ##########

# Check if a results file already exists.
if not os.path.isfile("results_" + str(n_neurons) + "N.csv"):
    # If it does not exists, the file is created and then opened in append mode.
    results_file = open("results_" + str(n_neurons) + "N.csv", "a+")
    # Write the header of the file.
    results_file.write("Neurons,Batch Size,Epochs,")
    # A 'Train', 'Val' and 'Test' column is made for each fold.
    for fold in range(n_folds):
        results_file.write("Fold " + str(fold + 1) + " Train,")
        results_file.write("Fold " + str(fold + 1) + " Val,")
        # Check if it is the last fold.
        if fold < (n_folds - 1):
            # If it is not, write the 'Test' column normally.
            results_file.write("Fold " + str(fold + 1) + " Test,")
        else:
            # If it is, the 'Test' column must not be followed by a ','.
            results_file.write("Fold " + str(fold + 1) + " Test\n")
else:
    # If it exists, the file is opened in append mode.
    results_file = open("results_" + str(n_neurons) + "N.csv", "a+")

########################################

######### CREATE/OPEN EXECUTION TIME FILE ##########

# Check if a executions time file already exists.
if not os.path.isfile("execution_time_" + str(n_epochs) + "E.csv"):
    # If it does not exists, the file is created and then opened in append mode.
    times_file = open("execution_time_" + str(n_epochs) + "E.csv", "a+")
    # Write the header of the file.
    times_file.write("Neurons,Batch Size,Epochs,")
    # A columns is created per epoch.
    for epoch in range(n_epochs):
        times_file.write("Epoch " + str(epoch + 1) + ",")
    # Write the 'Eval' columns.
    times_file.write("Eval Train,Eval Val,Eval Test\n")
else:
    # If it exists, the file is opened in append mode.
    times_file = open("execution_time_" + str(n_epochs) + "E.csv", "a+")

###############################################

# Create the folder to store the network files if it does not exists.
dirName = "networks/"
if not os.path.exists(dirName):
    os.mkdir(dirName)

########## LOAD MNIST TRAIN DATASET ##########

train_dataset = MNIST(
    # The encoding applied to the input data.
    PoissonEncoder(time=time, dt=dt),
    None,
    # Folder where the dataset files are stored.
    root="./data",
    # Wheter to download the dataset (the download page usually fails, so using a local copy is recommended).
    download=False,
    # Wheter to use train or test files.
    train=True,
    # The transformation applied to the encoded input data.
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

##############################################

# Stratified K-Fold declaration. By shuffling, data is not split in folds sequentially.
skfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

# Write model information in the results file.
results_file.write(str(n_neurons) + "," +
                   str(batch_size) + "," + str(n_epochs) + ",")

# Write model information in the executions time file.
times_file.write(str(n_neurons) + "," +
                 str(batch_size) + "," + str(n_epochs) + ",")

########## STRATIFIED CROSS VALIDATION LOOP ##########

# The first argument for the 'split' function is only the number of training instances (that is why it is just an array of zeros).
# The second argument for the split are the target values of each train instance. The 'narrow' function is used so that the number of train instances can be changed as desired.
for fold, (train_indices, val_indices) in enumerate(skfold.split(np.zeros(n_train), torch.narrow(train_dataset.targets, 0, 0, n_train))):

    # Print the current fold number.
    print("*********** FOLD %s **********" % (fold + 1))

    ########## NETWORK CREATION ##########

    # Build network using the already created Dielh&Cook model.
    # More parameters can be set. For more information about them and their meaning, visit the model declaration.
    network = DiehlAndCook2015(
        n_inpt=784,  # Number of input neurons.
        n_neurons=n_neurons,  # Number of excitatory neurons.
        # Initial values of the excitatory-inhibitory connection weights.
        exc=exc,
        # Initial values of the inhibitory-excitatory connection weights.
        inh=inh,
        dt=dt,  # Step of the simulation.
        norm=78.4,  # Weight normalization factor.
        # Pre and post synaptic update learning rates, respectively.
        nu=(1e-4, 1e-2),
        # On-spike increment membrane threshold potential.
        theta_plus=theta_plus,
        # Shape of the input tensors (images of 28 x 28 pixels).
        inpt_shape=(1, 28, 28),
    )

    ######################################

    # Store the network in the GPU.
    if gpu:
        network.to("cuda")

    ########## EXCITATORY NEURONS ASSIGMENTS ##########

    # This section comes from the original code example.
    n_classes = 10
    assignments = -torch.ones(n_neurons, device=device)
    proportions = torch.zeros((n_neurons, n_classes), device=device)
    rates = torch.zeros((n_neurons, n_classes), device=device)

    ###################################################

    ########## SPIKES MONITORING ##########

    # They must be recorded regardless it is used or not. This section comes from the original code example.
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(
            network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
        )
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    spike_record = torch.zeros(
        (update_interval, int(time / dt), n_neurons), device=device)

    #######################################

    ########## TRAINING ##########

    print("\nBegin training.\n")

    # Create a sampler to sample the train data into a dataloader.
    # A 'SubsetRandomSampler' is used because it can load data according to a set of indices.
    train_sampler = SubsetRandomSampler(train_indices)

    # Dataloder for the training data.
    train_dataloader = DataLoader(
        train_dataset,  # Dataset to load.
        batch_size=batch_size,  # Batch size to use.
        num_workers=n_workers,  # CPU threads to use.
        pin_memory=gpu,  # True if using the GPU.
        sampler=train_sampler  # Sampler to use.
    )

    ########## EPOCH LOOP ##########

    for epoch in range(n_epochs):
        # If it is the first fold, record the start execution time.
        if fold == 0:
            start = perf_counter()

        ########## LEARNING LOGIC ##########

        # This section comes from the original code example.
        # 'update_steps' indicates in which iteration the accuracy is checked. This value highly affects the accuracy results.
        # 'tqdm' is used to show the progress of the loop.

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

        #################################

        # If it is the first fold, record the end execution time.
        if fold == 0:
            end = perf_counter()
            # Calculate the execution time and save it on the executon times file.
            times_file.write(str(end - start) + ",")

    ################################

    print("\nTraining complete.")

    ##############################

    ########## SAVE THE NETWORK FILES ##########

    # Create a folder for each fold.
    dirName = "networks/" + str(n_neurons) + "N_" + str(batch_size) + "BS_" + str(
        n_epochs) + "E_" + str(n_folds) + "F/" + str(fold+1) + "F/"
    os.makedirs(dirName)
    # Save the network file.
    network.save(dirName + "network.pt")
    # Save the assigments file.
    torch.save(assignments, dirName + "assignments.pt")
    # Save the proportions file.
    torch.save(proportions, dirName + "proportions.pt")

    ############################################

    ########## LOAD THE NETWORK ##########

    # Load the network on CPU and without learning.
    # network = load(dirName + "network.pt", map_location="cpu", learning=False)

    # Load the assigments file.
    # assignments = torch.load(dirName + "assignments.pt")

    # Load the proportions file.
    # proportions = torch.load(dirName + "proportions.pt")

    # # Set up monitors for spikes (again).
    # spikes = {}
    # for layer in set(network.layers):
    #     spikes[layer] = Monitor(
    #         network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    #     )
    #     network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    ######################################

    ########## EVALUATION TRAIN SET ##########

    # Train set accuracies
    accuracy = {"all": 0, "proportion": 0}

    print("\nBegin evaluation train set.\n")

    # Deactivate training (learning).
    network.train(mode=False)

    # Number of training samples.
    folds_train_samples = len(train_indices)

    # If it is the first fold, record the start execution time.
    if fold == 0:
        start = perf_counter()

    ########## INFERENCE LOGIC ##########

    # This section comes from the original code example.
    # 'tqdm' is used to show the progress of the loop.

    for step, batch in enumerate(tqdm(train_dataloader, desc="Batches processed")):
        # Get next input sample.
        inputs = {"X": batch["encoded_image"]}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)
        # Add to spikes recording.
        spike_record = spikes["Ae"].get("s").permute((1, 0, 2))
        # Convert the array of labels into a tensor.
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

    #####################################

    # If it is the first fold, record the end execution time.
    if fold == 0:
        end = perf_counter()
        times_file.write(str(end - start) + ",")

    ########## ACCURACY CALCULATION ##########

    all_mean_accuracy = round((accuracy["all"] / folds_train_samples) * 100, 2)
    proportion_mean_accuracy = round(
        (accuracy["proportion"] / folds_train_samples) * 100, 2)
    # Print accuracies
    print("\nAll accuracy train set: %.2f" % (all_mean_accuracy))
    print("Proportion weighting accuracy train set: %.2f" %
          (proportion_mean_accuracy))
    # Write accuracy
    results_file.write(str(all_mean_accuracy) + ",")

    ##########################################

    print("\nEvaluation train set complete.\n")

    ##########################################

    ########## EVALUATION VALIDATION SET ##########

    # Create a sampler to sample the validation data into a dataloader.
    # A 'SubsetRandomSampler' is used because it can load data according to a set of indices.
    val_sampler = SubsetRandomSampler(val_indices)

    # Dataloder for the validation data.
    val_dataloader = DataLoader(
        train_dataset,  # Dataset to load.
        batch_size=batch_size,  # Batch size to use.
        num_workers=n_workers,  # CPU threads to use.
        pin_memory=gpu,  # True if using the GPU.
        sampler=val_sampler  # Sampler to use.
    )

    # Validation set accuracies
    accuracy = {"all": 0, "proportion": 0}

    print("Begin evaluation validation set.\n")

    # Deactivate training (learning).
    network.train(mode=False)

    # Number of validation samples.
    folds_val_samples = len(val_indices)

    # If it is the first fold, record the start execution time.
    if fold == 0:
        start = perf_counter()

    ########## INFERENCE LOGIC ##########

    # This section comes from the original code example.
    # 'tqdm' is used to show the progress of the loop.

    for step, batch in enumerate(tqdm(val_dataloader, desc="Batches processed")):
        # Get next input sample.
        inputs = {"X": batch["encoded_image"]}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)
        # Add to spikes recording.
        spike_record = spikes["Ae"].get("s").permute((1, 0, 2))
        # Convert the array of labels into a tensor.
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

    #####################################

    # If it is the first fold, record the end execution time.
    if fold == 0:
        end = perf_counter()
        times_file.write(str(end - start) + ",")

    ########## ACCURACY CALCULATION ##########

    all_mean_accuracy = round((accuracy["all"] / folds_val_samples) * 100, 2)
    proportion_mean_accuracy = round(
        (accuracy["proportion"] / folds_val_samples) * 100, 2)
    # Print accuracies
    print("\nAll accuracy validation set: %.2f" % (all_mean_accuracy))
    print("Proportion weighting accuracy validation set: %.2f \n" %
          (proportion_mean_accuracy))
    # Write accuracy
    results_file.write(str(all_mean_accuracy) + ",")

    ##########################################

    print("Evaluation validation set complete.\n")

    ###############################################

    ########## EVALUATION TEST SET ##########

    # Load MNIST test data
    test_dataset = MNIST(
        # The encoding applied to the input data.
        PoissonEncoder(time=time, dt=dt),
        None,
        # Folder where the dataset files are stored.
        root="./data",
        # Wheter to download the dataset (the download page usually fails, so using a local copy is recommended).
        download=False,
        # Wheter to use train or test files.
        train=False,
        # The transformation applied to the encoded input data.
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
        ),
    )

    # Dataloder for the test data.
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,  # Batch size to use.
        num_workers=n_workers,  # CPU threads to use.
        pin_memory=gpu,  # True if using the GPU.
    )

    # Test set accuracies
    accuracy = {"all": 0, "proportion": 0}

    print("Begin evaluation test set.\n")

    # Deactivate training (learning).
    network.train(mode=False)

    # If it is the first fold, record the start execution time.
    if fold == 0:
        start = perf_counter()

    ########## INFERENCE LOGIC ##########

    # This section comes from the original code example.
    # 'tqdm' is used to show the progress of the loop.

    for step, batch in enumerate(tqdm(test_dataloader, desc="Batches processed")):
        # Get next input sample.
        inputs = {"X": batch["encoded_image"]}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)
        # Add to spikes recording.
        spike_record = spikes["Ae"].get("s").permute((1, 0, 2))
        # Convert the array of labels into a tensor.
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

    #####################################

    # If it is the first fold, record the end execution time.
    if fold == 0:
        end = perf_counter()
        times_file.write(str(end - start) + "\n")

    ########## ACCURACY CALCULATION ##########

    all_mean_accuracy = round((accuracy["all"] / n_test) * 100, 2)
    proportion_mean_accuracy = round(
        (accuracy["proportion"] / n_test) * 100, 2)
    # Print accuracies
    print("\nAll accuracy test set: %.2f" % (all_mean_accuracy))
    print("Proportion weighting accuracy test set: %.2f \n" %
          (proportion_mean_accuracy))
    # Write accuracy
    if fold < (n_folds-1):
        # If it is the last fold, the time must be followed by a ','.
        results_file.write(str(all_mean_accuracy) + ",")
    else:
        # If it is the last fold, the time must not be followed by a ','.
        results_file.write(str(all_mean_accuracy) + "\n")

    ##########################################

    print("Evaluation test set complete.\n")

    #########################################

######################################################
