# SpeechEnhancement
This is the repo for the paper [Test-Time Adaptation for Speech Enhancement via Domain Invariant Embedding Transformation](https://arxiv.org/abs/2509.04280).
## Usage
There are three main files for execution.
`main.py` runs training and then evaluates the trained model. 
`run_eval.py` only runs the evaluation routine.
`run_da.py` runs the adaptation and evaluation routine.
How training and evaluation are run is configured through the configuration management system that relies on [Hydra](https://hydra.cc/)
To specify a configuration file, use the `-cn` flag:
```bash
python main.py -cn=cmgan
```
chooses the configuration file `config/cmgan.yaml`.
#### Installing dependencies
To install the necessary dependencies, run
```bash
pip install -r requirements.txt
```
This installs all necessary python.
### Configuration Management
For a detailed overview of Hydra, checkout their website and documentation.
#### Structure
Hydra uses [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/) under the hood, which can be thought of as nested dictionaries.
The structure is dynamically built at runtime from the configuration that is defined in YAML files and from overwrites.
This means that different configurations will have different structures.
For example, when training a model end-to-end, there will only be one learning rate, whereas when training a GAN, there will be separate learning rates for the generator and discriminator.
This also means that the configuration should only contain values that are actually used.
Accessing the configuration works as as follows:
For the simple configuration
```YAML
model:
    name: CNN
    depth: 5
dataset: VBD
training:
    learning_rate: 1e-5
    epochs: 100
```
The number of epochs can be accessed by `cfg.training.epochs`. \
Hydra supports **modular** configurations. 
For example, multiple methods can use the same configuration of the training.
This can be specified by the defaults list
```YAML
defaults:
    - training: gan
    - _self_
training:
    lr: 1e-5
```
In this example the configuration of the training is given by the contents of `config/training/gan.yaml`.
Only the learning rate is changed to use a custom value.
As the modules of the configuration are typically shared between multiple methods, this means configurations do not have to be copied.
Keep in mind that changing a value of a default config will therefore change it in all parent modules.
As multiple modules in the default list can overwrite the same values, the order matters.
The key `_\_self\__` represents the current file.
Therefore, in this example the value `training->lr: 1e-5` is not overwritten by the default `training: gan` as it comes afterwards.
#### Overwrite Syntax
When running a script that uses hydra, the configuration can be changes in the command line.
To change an existing value, see the following example:
```bash
python main.py -cn=cmgan training.lr=1e-3
```
To add new keys to the configuration, run e.g.
```bash
python main.py -cn=cmgan +training.new_key="TEST"
```
The way hydra is used in this repo, this is rarely of use.

##### Loading checkpoints
Overwrite the configuration key `model.load` with a path that points to a saved model checkpoint.
### Datasets
- VoiceBank + DEMAND: Available for download at: [Link](https://datashare.ed.ac.uk/handle/10283/2791)
- EARS & WHAM: Available here: [Link](https://github.com/sp-uhh/ears_benchmark). In the paper V1 was used. 
- DNS: Available here: [Link](https://github.com/microsoft/DNS-Challenge/tree/v4dnschallenge_ICASSP2022). Disable synthetic RIRs to reproduce paper results.
- DEMAND: Available here: [Link](https://dcase-repo.github.io/dcase_datalist/datasets/scenes/demand.html)

It is recommended to downsample these datasets to 16kHz using `ffmpeg`.
