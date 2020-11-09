# pretraining-learning-curves
This is the repository for the paper [When Do You Need Billions of Words of Pretraining Data?](https://github.com/nyu-mll/pretraining-learning-curves/blob/main/When%20Do%20You%20Need%20Billions%20of%20Words%20of%20Pretraining%20Data.pdf)

## Edge Probing
We use [jiant1](https://github.com/nyu-mll/jiant-v1-legacy) for our edge probing experiments. This [tutorial](https://github.com/nyu-mll/jiant-v1-legacy/blob/master/tutorials/setup_tutorial.md) can help you set up the environment and get started with jiant.

Below is an example of how to reproduce our dependency labelling experiment with roberta-base-1B-3, which is one of the MiniBERTas we probe.

#### Download and Preprocess the Data
The commands below help you get and tokenize the data for the dependency labelling task. Remember to change directory to the root of the jiant and activate your jiant environment first.
```bash
mkdir data

mkdir data/edges

probing/data/get_ud_data.sh data/edges/dep_ewt

python probing/get_edge_data_labels.py -o data/edges/dep_ewt/labels.txt -i data/edges/dep_ewt/*.json

python probing/retokenize_edge_data.py -t nyu-mll/roberta-base-1B-3  data/edges/dep_ewt/*.json
```

#### Run the Experiment
If you have not used jiant before, you will probably need to set two critical environment variables: 

```
$JIANT_PROJECT_PREFIX: the directory where logs and model checkpoints will be saved.

$JIANT_DATA_DIR: The data directory. Set it to PATH/TO/LOCAL/REPO/data
```

Now, you are ready to run the probing program:

```bash
python main.py –config_file jiant/config/edgeprobe/edgeprobe_miniberta.conf\ 
–overrides “exp_name=DL_tutorial, target_tasks=edges-dep-ud-ewt,\
transformers_output_mode=mix, input_module=nyu-mll/roberta-base-1B-3,\ 
target_train_val_interval=1000, batch_size=32, target_train_max_vals=130, lr=0.0005”
```

A logging message will be printed out after each validation. You should expect validation f1 to exceed 90 in only a few validations.

The final validation result will be printed after the experiment is finished, and can also be found in `$JIANT_PROJECT_PREFIX/DL_tutorial/results.tsv`. 
You should expect the final validation f1 to be around 95.

## Minimum Description Length Probing with Edge Probing tasks
For this experiment, we use this [fork](https://github.com/nyu-mll/online-code-for-edge-probing) of jiant1. 

## BLiMP
The code for our BLiMP experiments can be found [here](https://github.com/nyu-mll/mlm-scoring/tree/minibertas). You can already check [results](https://github.com/nyu-mll/mlm-scoring/tree/minibertas/examples/lingacc-blimp/results/nyu-mll) for our MiniBERTas.

If you want to rerun experiments on your own, we have prepared BLiMP data so you only need to include all dependencies for the environment and run scripts following the tutorial [here](https://github.com/nyu-mll/mlm-scoring/tree/minibertas/examples/lingacc-blimp). Note that when intalling dependencies CUDA version could be a problem when installing `mxnet`.

## SuperGLUE
We use [jiant2](https://github.com/nyu-mll/jiant) for our SuperGLUE experiments. Get started with jiant2 using this [guide](https://github.com/nyu-mll/jiant/tree/master/guides) and [examples](https://github.com/nyu-mll/jiant/tree/master/examples).
