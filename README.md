# AdaFisher: Adaptive Second Order Optimizer using Fisher Information 
Official PyTorch implementation of the paper-related **AdaFisher** optimizer code from:
[Adaptive Second Order Optimizer using Fisher Information]()

---
We propose new second order optimization algorithm that incorporates the curvature information (FIM) 
in order to govern the gradient descent method. This optimizer uses the Adam/AdamW framework and incorporates as the second moment the Fisher 
Information. We call this new optimizer **AdaFisher**. AdaFisher is fast (roughly same time per iteration as Adam), performs better than existing sota, and generalizes well across experimental configurations for CNNs at this time.
For ViTs, we are currently doing experiments.

## Contents
This repository contains relevant code for using AdaFisher and other current sota optimizers. However, it does not reproduce the experiments present in the paper because this is an enhance version with new features.
Note that this repository is based on the RMSGD framework (paper branch). For more information, see the [RMSGD project](https://github.com/mahdihosseini/RMSGD/tree/paper).

## Usage
We provide numerous `config.yaml` files to replicate experimental configurations in [configs](Image Classification/configs). Running the code is as simple as 

```console
python train.py --config config.yaml --output **OUTPUT_DIR**
```
Run `python train.py --help` for other options.
You can laos use the [run](Image Classification/src/run.sh) file for training based on the optimizer and the config file.
## License
This project is released under the GNU General Public License v3.0. Please see the [LICENSE](LICENSE) file for more information.