# AdaFisher: Adaptive Second Order Optimization via Fisher Information

🚀 Official PyTorch implementation of the paper-related **AdaFisher** optimizer code. 
<center>
    <img src="imgs/AdaFisher.png" alt="Overview of Project" width="800"/>
</center>


## 🌟 Introduction

We introduce **AdaFisher**, a novel second-order optimization algorithm that leverages the Fisher Information Matrix (FIM) to enhance the gradient descent method. Integrating the curvature information provided by the FIM with the Adam/AdamW framework, AdaFisher achieves fast computation speeds—comparable to Adam—while outperforming existing SOTA optimizers in terms of performance and generalization across various experimental setups. Currently, AdaFisher is optimized for image classification tasks using Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) and Language Modelling. We also explore the structure of the Kronecker Factors, where their energy is mainly concentrated along the diagonal, in order to improve the time and space complexity of AdaFisher.

## 📁 Repository Contents
This project is organized into four main folders, each dedicated to a specific aspect of our research and development efforts.

### 🛠️ Optimizers

This section features the implementation of a variety of optimizers, including both well-established methods and novel optimizers developed during our research. Explore the optimizers in the [optimizers](optimizers) directory:

- **`AdaFisher`**: An optimizer developed as part of our research, focusing on second-order optimization techniques by integrating Fisher Information.
- **`AdaFisherW`**: A variant of AdaFisher, based on the AdamW framework, incorporating second-order optimization techniques through Fisher Information.
- **`KFAC`**: An optimizer that leverages second-order optimization techniques via Fisher Information. For more details, visit the [KFAC-Pytorch](https://github.com/alecwangcq/KFAC-Pytorch) repository.
- **`SGD`**: The standard Stochastic Gradient Descent optimizer. For further information, refer to the [ASDL](https://github.com/kazukiosawa/asdl) repository.
- **`AdaHessian`**: An optimizer that uses Hessian-based information to adjust learning rates. Additional information can be found in the [AdaHessian](https://github.com/amirgholami/adahessian) repository.
- **`Adam`**: The Adaptive Moment Estimation optimizer, renowned for its effectiveness in various deep learning tasks.
- **`AdamW`**: A variant of the Adam optimizer that includes weight decay for better regularization, widely used for its enhanced performance in deep learning tasks.


### 💡 Enhanced Kronecker Factor Analysis and Optimizer Trajectory Visualization on Loss Landscapes
<center>
    <img src="imgs/Weight_Trajectories_Loss_Landscape.png" alt="AdaFisher" width="50%">
</center>

---

This repository is dedicated to analyzing the structure of Kronecker factors within ResNet18 architectures, using the CIFAR-10 dataset, and visualizing the loss landscape and trajectories of various optimizers. It showcases innovative optimization techniques, including a novel optimizer named AdaFisher, and provides detailed insights into the dynamics of deep learning model training.

#### 📂 Structure 

- **`models/`**: Contains the implementation of the ResNet18 network.

#### 📓 Jupyter Notebooks 

- **`Kronecker_Factors_Analysis.ipynb`**: This notebook delves into the properties of Kronecker factors across various layers of a neural network using the ResNet18 model, specifically applied to the CIFAR-10 dataset. Key aspects of the analysis include:
    - **Gershgorin Circle Theorem**: Utilizes this theorem to estimate the eigenvalue bounds of Kronecker-factored approximations. This is critical for understanding the conditioning of layers and their stability during training.
    - **Frequency Field**: Examines the distribution of frequency components (FFT) in the Kronecker factors, which helps in identifying the dominant frequency patterns that influence learning dynamics and feature extraction capabilities of the network layers.
- **`Visualization_Kronecker_Factors.ipynb`**: Provides visualizations of Kronecker factors to better understand their structures and behavior across training.
- **`Visualization_Loss_Landscape.ipynb`**: Offers a visual exploration of the loss landscape and trajectories of various optimizers, crucial for understanding optimization paths and strategies.
- **`train.ipynb`**: Contains the training loops for the ResNet18 model using the different optimizers, including performance comparisons and analysis.

### 🖼️ Image Classification 
This repository provides all necessary code to employ AdaFisher and compare it with other leading SOTA optimizers. It is structured based on the RMSGD framework (paper branch). For additional details, refer to the [RMSGD project](https://github.com/mahdihosseini/RMSGD/tree/paper).

#### 🎯 Usage

To facilitate replication of our experimental results, we include multiple configuration files within the [configs](Image_Classification/configs) directory. To run a training session using these configurations, execute the following command:

```console
python train.py --config config.yaml
```
For a list of available command-line options, you can run:
```console
python train.py --help
```
##### 🔗 Integrating AdaFisher/AdaFisherW as an Optimizer
AdaFisher and AdaFisherW can be seamlessly integrated into your training pipeline like any standard optimizer. Here’s how to set it up with your model:
```python
from AdaFisher import AdaFisher # from AdaFisherW import AdaFisherW
...
optimizer = AdaFisher(network, ...) # AdaFisherW(network, ...) 
...
for input in data_loader:
    optimizer.zero_grad()
    output = network(input)
    optimizer.step()
```
**Important**: When initializing AdaFisher, pass the entire network object to the optimizer, not just its parameters. This ensures that the optimizer has full context of the model architecture, which is essential for leveraging the Fisher Information effectively.

#### 🌎 Training with Pretrained Weights
To train using pretrained weights, utilize the following command:
```console
python train.py --config config.yaml --pretrained 1
```
This allows the several optimizers to leverage previously learned weights, potentially improving convergence speed and final model accuracy for complex image classification tasks.

#### ☄️ Training ImageNet with a Single GPU
For training ResNet50 on ImageNet using a single GPU, we have prepared a set of bash scripts tailored to each optimizer. These scripts are conveniently located in the [Image_Classification/ImageNet](ImageNet) directory. Each script is pre-configured with the optimal settings for training, ensuring that you can start your training sessions efficiently and effectively.

### 📖 Language Modeling 

This section of the repository focuses on training various optimizers using the WikiText-2 dataset with a compact GPT-1 network.

#### 🎯 Usage

To replicate our experimental results accurately, we provide a series of executable bash scripts in the [Language_Model](Language_Model) directory. These scripts are pre-configured with all necessary settings to facilitate straightforward experimentation. 

For detailed information on all available command-line options, execute the following command:

```console
python train.py --help
```
##### 🔗 Integrating AdaFisher/AdaFisherW as an Optimizer
AdaFisher and AdaFisherW can be seamlessly integrated into your training pipeline like any standard optimizer. Here’s how to set it up with your model:
```python
from AdaFisher import AdaFisher # from AdaFisherW import AdaFisherW
...
optimizer = AdaFisher(network, ...) # AdaFisherW(network, ...) 
...
for input in data_loader:
    optimizer.zero_grad()
    output = network(input)
    optimizer.step()
```
**Important**: When initializing AdaFisher, pass the entire network object to the optimizer, not just its parameters. This ensures that the optimizer has full context of the model architecture, which is essential for leveraging the Fisher Information effectively.
## 🌟 Getting Started 

### Prerequisites
To set up the required environment, you can either use `pip` or `conda`.

#### Using pip
You can install all required packages using:
```bash
pip install -r requirements.txt
pip install notebook
```
With a Python version of 3.9 or later.

#### Using Conda
First, create a new Conda environment with Python:
```bash
conda create --n AdaFisher myenv python
```
Activate the environment:
```bash
conda activate AdaFisher
```
Then, install the required packages from the requirements.txt file:
```bash
pip install -r requirements.txt
conda install -n gg ipykernel --update-deps --force-reinstall
```

## License 📜

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) - see the [LICENSE](LICENSE) file for details.
