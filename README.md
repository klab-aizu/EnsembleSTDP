# EnsembleSTDP
This is the source code for the paper:
- Title: EnsembleSTDP: Accelerating Distributed Spike Timing Dependent Plasticity Learning in Spiking Neural Networks


### Requirements
Python >=3.10, <3.12


## Set up
Below are the steps to set up the environment for this package.

#### Clone repository
```
git clone https://github.com/klab-aizu/EnsembleSTDP
```

#### Install the necessary libraries
```
pip install -r requirements.txt
```

## How to use scripts
Here is how to perform basic operations in ensemble learning
### Train sub-model
An SNN sub-model is defined by the following attributes

* model_number
* n_neurons (number of excitatory neurons)
* starting index of training images
* ending index of training images

For example, to train a sub-model 1 that has 100 neurons and trained from 0 to 6000 images, run:
```
python eth_mnist_train.py --model_number 1 --n_neurons 100 --start_image 0 --end_image 6000
```

### Test sub-model
After the training, sub-model’s classification accuracy can be tested by replacing "train" with "test" in the above command
```
python eth_mnist_test.py --model_number 1 --n_neurons 100 --start_image 0 --end_image 6000
```

### Merge sub-models
To merge and compress the trained sub-models, number of compression and similarity measurement need to specified. Before running the command below, make sure all the files of sub-models to be merged (model_*.pth, assignments_*.pth, and proportions_*.pth for each model) are in the models_to_merge folder. The example of the command to merge sub-models using MSE and compress 100 neurons:
```
python3 eth_mnist_merge.py --method mse --n_compression 100
```
* Mean Squired Error (MSE) -> --method mse
* Manhattan Distance -> --method manhattan
* Cosine Similarity -> --method cos
* Correlation Coefficient -> --method corr

### Plot 2D-Receptive field
To visualize the trained weights of a sub-model, run the following command.
The example of visualizing sub-model 1 with 100 neurons, trained from 0 to 6000 images
```
python eth_mnist_plot.py --model_number 1 --n_neurons 100 --start_image 0 --end_image 6000
```
The receptive field will be saved as receptive_field_model_1_100_neurons_0_to_6000.png

\*note that models trained on GPU cannot be plotted on CPU, or vice versa


### Training or testing multiple sub-models at once
There is a supplimental shell script to run the training, testing and merging on a series of command line arguments. For example, to train the following 5 sub-models at once,
* model_1_100_neurons_0_to_6000 (model 1 with 100 neurons trained from 0 to 6000 images)
* model_2_100_neurons_6000_to_12000
* model_3_100_neurons_12000_to_18000
* model_4_100_neurons_18000_to_24000
* model_5_100_neurons_24000_to_30000

first, set the parameters as follows in input.txt
1 0 6000
2 6000 12000
3 12000 18000
4 18000 24000
5 24000 30000

then edit shell.sh
```
#!/bin/bash
input_file="input_args.txt"
output_file="output_results.txt"

# Run program for each set of arguments in the input file
while read -r model_number start_image end_image ; do
    python eth_mnist_train.py --model_number "$model_number" --start_image "$start_image" --end_image "$end_iamge" --n_neurons 100 >> "$output_file"
done < "$input_file"
```
Finally run
```
./shell.sh
```




