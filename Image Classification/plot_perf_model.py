import os
import json
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path


class Performance:
    def __init__(self, optimizers: list,
                 models: list,
                 file_names: list = None,
                 dataset: list = None,
                 type_acc: str = "top1"):
        self.optimizers = optimizers
        self.models = models
        self.file_names = file_names
        self.dataset = dataset
        self.type_acc = type_acc
        self.processing_files()
        self.results = {}
        for f in self.files:
            self.results[f] = get_data(f)

    def processing_files(self):
        all_files = []
        for opt in self.optimizers:
            for model in self.models:
                path = f"logs/{opt}/{model}"
                files_with_path = [f"{path}/{file}" for file in os.listdir(path) if file != ".DS_Store"]
                all_files += files_with_path
        if self.file_names is not None:
            all_files = [f for f in all_files if f.split('/')[-1] in self.file_names]
        if self.dataset is not None:
            all_files = [f for f in all_files if f.split('/')[-1].split('_')[4] in self.dataset]
        self.files = all_files

    def get_mean_std(self):
        if len(self.dataset) != 1:
            raise ValueError("Only one dataset must be specified")
        if len(self.optimizers) != 1:
            raise ValueError("Only one optimizer is supported")
        if len(self.models) != 1:
            raise ValueError("Only one model is supported")
        if len(self.results) != 5:
            raise ValueError("Only 5 files must be specified")

        list_best_accuracy = []
        for k in self.results.keys():
            test_accuracy = self.results[k]["test_accuracy1"]
            list_best_accuracy.append(max(test_accuracy))

        mean_value = np.mean(list_best_accuracy)
        std_value = np.std(list_best_accuracy)
        print(f"opt: {self.optimizers[0]} | model: {self.models[0]} | dataset: {self.dataset[0]} | Mean accuracy: "
              f"{round(mean_value * 100, 3)}, std accuracy: {round(std_value * 100, 3)}")

    def get_best_accuracy(self, verbose: bool = False):
        best_accuracy = 0
        best_details = ""
        best_file = ""
        for k in self.results.keys():
            if self.type_acc == "top1":
                test_accuracy = self.results[k]["test_accuracy1"]
            elif self.type_acc == "top5":
                test_accuracy = self.results[k]["test_accuracy5"]
            else:
                raise ValueError("Unrecognized type {}".format(self.type_acc))

            current_best_accuracy = max(test_accuracy)
            optimizer = k.split("_")[5]
            network = get_network(k)
            dataset = k.split("_")[4]
            best_details = (f"opt: {optimizer} | model: {network} | dataset: {dataset} | "
                            f"The best accuracy is obtained at epoch {test_accuracy.index(current_best_accuracy)} "
                            f"with {round(current_best_accuracy * 100, 3)}%")
            print(best_details)
            if current_best_accuracy > best_accuracy:
                best_accuracy = current_best_accuracy
                best_file = k
        if verbose:
            print(
                f"\nopt: {optimizer} | model: {network} | dataset: {dataset} | The best accuracy is obtained for file: "
                f"{best_file.split('/')[3]}")

    def plot_WCT_together(self):
        if self.file_names is None:
            raise ValueError("File names must be specified")
        if len(self.dataset) != 1:
            raise ValueError("Only one dataset must be specified")
        if len(self.models) != 1:
            raise ValueError("Only one model is supported")

        plt.figure(figsize=(10, 10))
        min_accuracy = float('inf')
        max_accuracy = float('-inf')
        for idx, k in enumerate(self.results.keys()):
            if self.type_acc == "top1":
                test_accuracy, tot_time = (self.results[k]["test_accuracy1"], self.results[k]["tot_time"])
            elif self.type_acc == "top5":
                test_accuracy, tot_time = (self.results[k]["test_accuracy5"], self.results[k]["tot_time"])
            else:
                raise ValueError("Unrecognized type {}".format(self.type_acc))
            # Update min and max accuracies
            min_accuracy = min(min_accuracy, min(test_accuracy))
            max_accuracy = max(max_accuracy, max(test_accuracy))
            optimizer = k.split("_")[5]
            c = define_color(optimizer)
            plt.plot(tot_time, test_accuracy, '-', color=c, label=f"{optimizer}")

        network = get_network(k)
        plt.title(f"Testing: {network.capitalize()} on {self.dataset[0]}")
        plt.xlabel("Time (s)")
        plt.ylabel("Accuracy")
        plt.grid()
        # Set ylim dynamically based on min and max accuracy, with some padding
        padding = (max_accuracy - min_accuracy) * 0.05  # Adding 5% padding to top and bottom
        plt.ylim(min_accuracy - padding, max_accuracy + padding)
        plt.legend()
        plt.tight_layout()
        output_path = Path(f"results/{self.models[0]}/{self.dataset[0]}").expanduser()
        if not output_path.exists():
            print(f"Info: Output dir {output_path} does not exist, building")
            output_path.mkdir(exist_ok=True, parents=True)
        plt.savefig(f"{output_path}/testaccuracyWCT")

    def plot_curves_together(self):
        if self.file_names is None:
            raise ValueError("File names must be specified")
        if len(self.dataset) != 1:
            raise ValueError("Only one dataset must be specified")
        if len(self.models) != 1:
            raise ValueError("Only one model is supported")

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        ax1, ax2 = axs.flatten()
        min_accuracy = float('inf')  # Initialize with a very high value for accuracy
        max_accuracy = float('-inf')  # Initialize with a very low value for accuracy
        for idx, k in enumerate(self.results.keys()):
            if self.type_acc == "top1":
                train_accuracy, test_accuracy, train_loss, valid_loss = (self.results[k]["train_accuracy1"],
                                                                         self.results[k]["test_accuracy1"],
                                                                         self.results[k]["train_loss"],
                                                                         self.results[k]["valid_loss"])
            elif self.type_acc == "top5":
                train_accuracy, test_accuracy, train_loss, valid_loss = (self.results[k]["train_accuracy5"],
                                                                         self.results[k]["test_accuracy5"],
                                                                         self.results[k]["train_loss"],
                                                                         self.results[k]["valid_loss"])
            else:
                raise ValueError("Unrecognized type {}".format(self.type_acc))
            min_accuracy = min(min_accuracy, min(test_accuracy))
            max_accuracy = max(max_accuracy, max(test_accuracy))
            optimizer = k.split("_")[5]
            c = define_color(
                optimizer)
            ax1.plot(train_loss, '-', color=c, label=f"{optimizer}")
            ax2.plot(test_accuracy, '-', color=c, label=f"{optimizer}")
        network = get_network(k)

        ax1.set_title(f"Training: {network.capitalize()} on {self.dataset[0]}")
        ax2.set_title(f"Testing: {network.capitalize()} on {self.dataset[0]}")
        ax1.set_xlabel("Epochs")
        ax2.set_xlabel("Epochs")
        ax1.set_yscale('log')
        ax1.set_ylabel("Loss")
        ax2.set_ylabel("Accuracy")
        ax1.set_xlim(0, 200)
        ax2.set_xlim(0, 200)
        # Set ylim dynamically based on min and max accuracy, with some padding
        padding = (max_accuracy - min_accuracy) * 0.05  # Adding 5% padding to top and bottom
        ax2.set_ylim(min_accuracy - 0.1, max_accuracy + 0.01)
        ax1.legend()
        ax2.legend()
        plt.tight_layout()

        output_path = Path(f"results/{self.models[0]}/{self.dataset[0]}").expanduser()
        if not output_path.exists():
            print(f"Info: Output dir {output_path} does not exist, creating it")
            output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_path}/testaccuracy_trainloss")

    def plot_curves_diff(self):
        if self.file_names is None:
            raise ValueError("File names must be specified")
        if len(self.dataset) != 2:
            raise ValueError("Only two datasets must be specified")
        if len(self.models) != 1:
            raise ValueError("Only one model is supported")

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        ax1, ax2, ax3, ax4 = axs.flatten()

        # Initialize with high and low values for dynamic ylim adjustments
        min_accuracy_dataset1 = float('inf')
        max_accuracy_dataset1 = float('-inf')
        min_accuracy_dataset2 = float('inf')
        max_accuracy_dataset2 = float('-inf')

        for idx, k in enumerate(self.results.keys()):
            if self.type_acc == "top1":
                train_accuracy, test_accuracy, train_loss, valid_loss = (self.results[k]["train_accuracy1"],
                                                                         self.results[k]["test_accuracy1"],
                                                                         self.results[k]["train_loss"],
                                                                         self.results[k]["valid_loss"])
            elif self.type_acc == "top5":
                train_accuracy, test_accuracy, train_loss, valid_loss = (self.results[k]["train_accuracy5"],
                                                                         self.results[k]["test_accuracy5"],
                                                                         self.results[k]["train_loss"],
                                                                         self.results[k]["valid_loss"])
            else:
                raise ValueError("Unrecognized type {}".format(self.type_acc))
            optimizer = k.split("_")[5]
            c = define_color(optimizer)
            if k.split("_")[4] == self.dataset[0]:
                ax1.plot(train_loss, '-', color=c, label="{} Training".format(optimizer))
                ax2.plot(test_accuracy, '-', color=c, label="{} Testing".format(optimizer))
                # Update min and max accuracies for dataset1
                min_accuracy_dataset1 = min(min_accuracy_dataset1, min(test_accuracy))
                max_accuracy_dataset1 = max(max_accuracy_dataset1, max(test_accuracy))

            elif k.split("_")[4] == self.dataset[1]:
                ax3.plot(train_loss, '-', color=c, label="{} Training".format(optimizer))
                ax4.plot(test_accuracy, '-', color=c, label="{} Testing".format(optimizer))
                # Update min and max accuracies for dataset2
                min_accuracy_dataset2 = min(min_accuracy_dataset2, min(test_accuracy))
                max_accuracy_dataset2 = max(max_accuracy_dataset2, max(test_accuracy))

        network = get_network(k)
        ax1.set_title(f"Training: {network.capitalize()} on {self.dataset[0]}")
        ax2.set_title(f"Testing:  {network.capitalize()} on {self.dataset[0]}")
        ax3.set_title(f"Training:  {network.capitalize()} on {self.dataset[1]}")
        ax4.set_title(f"Testing:  {network.capitalize()} on {self.dataset[1]}")
        ax1.set_xlabel("Epochs")
        ax2.set_xlabel("Epochs")
        ax3.set_xlabel("Epochs")
        ax4.set_xlabel("Epochs")
        ax1.set_yscale('log')
        ax3.set_yscale('log')
        ax1.set_ylabel("Loss")
        ax3.set_ylabel("Loss")
        ax2.set_ylabel("Accuracy")
        ax4.set_ylabel("Accuracy")

        ax1.set_xlim(0, 200)
        ax2.set_xlim(0, 200)
        ax3.set_xlim(0, 200)
        ax4.set_xlim(0, 200)

        # Dynamically adjust ylim for ax2 and ax4
        padding1 = (max_accuracy_dataset1 - min_accuracy_dataset1) * 0.05
        ax2.set_ylim(min_accuracy_dataset1 - padding1, max_accuracy_dataset1 + padding1)

        padding2 = (max_accuracy_dataset2 - min_accuracy_dataset2) * 0.05
        ax4.set_ylim(min_accuracy_dataset2 - padding2, max_accuracy_dataset2 + padding2)

        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        plt.tight_layout()

        output_path = Path(f"results/{self.models[0]}/{self.dataset[0]}_{self.dataset[1]}").expanduser()
        if not output_path.exists():
            print(f"Info: Output dir {output_path} does not exist, creating it")
            output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_path}/testaccuracy_trainloss")


def get_data(directory: str) -> dict:
    def read_losses(filename):
        with open(os.path.join(directory, filename), 'r') as file:
            return [float(number.strip()) for number in file.readlines()]

    train_loss = read_losses("train_loss.txt")
    valid_loss = read_losses("test_loss.txt")
    train_accuracy1 = read_losses("train_accuracy1.txt")
    train_accuracy5 = read_losses("train_accuracy5.txt")
    test_accuracy1 = read_losses("test_accuracy1.txt")
    test_accuracy5 = read_losses("test_accuracy5.txt")

    with open(os.path.join(directory, "tot_time.txt"), 'r') as file:
        tot_time_s = [float(number.strip()) for number in file.readlines()]
        tot_time = process_WCT(tot_time_s)

    # with open(os.path.join(directory, "mem_used.txt"), 'r') as file:
    #     memory_used = []
    #     for line in file:
    #         if line.strip():  # This ensures we skip empty lines
    #             try:
    #                 memory_used.append(float(line.split(":")[3]))
    #             except IndexError:
    #                 pass  # Handle the case where a line doesn't have enough elements

    with open(os.path.join(directory, "config.json"), 'r') as file:
        logs = json.load(file)

    return {
        "configs": logs,
        "train_accuracy1": train_accuracy1,
        "train_accuracy5": train_accuracy5,
        "test_accuracy1": test_accuracy1,
        "test_accuracy5": test_accuracy5,
        "tot_time": tot_time,
        "train_loss": train_loss,
        "valid_loss": valid_loss
        #"memory_used": memory_used
    }


def process_WCT(time_results):
    for i in range(1, len(time_results)):
        time_results[i] += time_results[i - 1]
    return time_results


def get_network(k) -> str:
    if k.split("_")[3][-1] != "r":
        return k.split("_")[3]
    elif k.split("_")[3][8:9].isdigit():
        return k.split("_")[3][:9]
    else:
        return k.split("_")[3][:8]


def define_color(k):
    if k == "AdaFisher":
        c = "blue"
    elif k == "AdaFisherW":
        c = "darkorange"
    elif k == "Adam":
        c = "black"
    elif k == "AdaHessian":
        c = "green"
    elif k == "AdamW":
        c = "slategrey"
    elif k == "Apollo":
        c = "cyan"
    elif k == "SAM":
        c = "yellow"
    elif k == "kfac":
        c = "magenta"
    elif k == "ekfac":
        c = "tomato"
    elif k == "Shampoo":
        c = "tomato"
    elif k == "SGD":
        c = "lightpink"
    else:
        raise ValueError("Error during assigning the color")
    return c


if __name__ == "__main__":
    optimizers = ["Adam", "AdaHessian", "kfac", "Shampoo", "AdaFisher"]
    models = ["resnet50"]
    datasets = ["TinyImageNet"]
    type_acc = "top1"
    file_names = [
        "date=2024-04-06-00-06-39_results_trial=4_resnet50_TinyImageNet_AdaFisher_weight_decay=0.0005_gamma=0.92_Lambda=0.001_TCov=100.0_CosineAnnealingLRT_max=200.0_LR=0.001",
        "date=2024-04-19-05-57-50_results_trial=8_resnet50_TinyImageNet_Shampoo_weight_decay=0.0005_damping=1e-12_momentum=0.9_curvature_update_interval=1.0_ema_decay=-1.0_clipping_norm=1.0_CosineAnnealingLRT_max=36.0_LR=0.3",
        "date=2024-04-18-12-48-53_results_trial=8_resnet50_TinyImageNet_kfac_weight_decay=0.0005_momentum=0.9_stat_decay=0.95_damping=0.003_kl_clip=0.001_TCov=10.0_TInv=100.0_batch_averaged=1.0_CosineAnnealingLRT_max=187.0_LR=0.001",
        "date=2024-04-18-14-49-35_results_trial=8_resnet50_TinyImageNet_Adam_weight_decay=0.0005_CosineAnnealingLRT_max=288.0_LR=0.001",
        "date=2024-04-18-17-19-21_results_trial=8_resnet50_TinyImageNet_AdaHessian_weight_decay=0.0005_hessian_power=1.0_CosineAnnealingLRT_max=89.0_LR=0.15"
    ]
    perf = Performance(optimizers=optimizers, models=models, dataset=datasets,
                       type_acc=type_acc, file_names=file_names)
    # perf.get_mean_std()
    # perf.get_best_accuracy(verbose=True)
    perf.plot_WCT_together()
    # perf.plot_curves_together()
    # perf.plot_curves_together()
