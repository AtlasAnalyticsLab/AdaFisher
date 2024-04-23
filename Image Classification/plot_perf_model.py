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
            if  "swin_t" in self.models:
                all_files = [f for f in all_files if f.split('/')[-1].split('_')[5] in self.dataset]
            else:

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
        return f"{best_file.split('/')[3]}"

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
            if "swin_t" in self.models:
                optimizer = k.split("_")[7]
            else:
                optimizer = k.split("_")[5]
            c = define_color(optimizer)
            plt.plot(tot_time, test_accuracy, '-', color=c, label=f"{optimizer}")
        if "swin_t" in self.models:
            network = "Tiny Swin"
        else:
            network = get_network(k)
        plt.title(f"Testing: {network.capitalize()} on {self.dataset[0]}")
        plt.xlabel("Time (s)")
        plt.ylabel("Accuracy")
        plt.grid()
        # Set ylim dynamically based on min and max accuracy, with some padding
        padding = (max_accuracy - min_accuracy) * 0.05  # Adding 5% padding to top and bottom
        plt.ylim(min_accuracy - padding, max_accuracy + padding)
        plt.legend(frameon=False)
        plt.tight_layout()
        output_path = Path(f"results/{self.models[0]}/{self.dataset[0]}").expanduser()
        if not output_path.exists():
            print(f"Info: Output dir {output_path} does not exist, building")
            output_path.mkdir(exist_ok=True, parents=True)
        plt.savefig(f"{output_path}/testaccuracyWCT")

    def plot_trainAcc_testLoss(self):
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
                tot_time, test_accuracy, train_loss = (self.results[k]["tot_time"],
                                                       self.results[k]["test_accuracy1"],
                                                       self.results[k]["train_loss"])

            elif self.type_acc == "top5":
                tot_time, test_accuracy, train_loss = (self.results[k]["tot_time"],
                                                       self.results[k]["test_accuracy5"],
                                                       self.results[k]["train_loss"])

            else:
                raise ValueError("Unrecognized type {}".format(self.type_acc))
            min_accuracy = min(min_accuracy, min(test_accuracy))
            max_accuracy = max(max_accuracy, max(test_accuracy))
            if "swin_t" in self.models:
                optimizer = k.split("_")[7]
            else:
                optimizer = k.split("_")[5]
            c = define_color(
                optimizer)
            ax1.plot(tot_time, train_loss, '-', color=c, label=f"{optimizer}")
            ax2.plot(tot_time, test_accuracy, '-', color=c, label=f"{optimizer}")
        if "swin_t" in self.models:
            network = "Tiny Swin"
        else:
            network = get_network(k)

        ax1.set_title(f"Training loss: {network.capitalize()} on {self.dataset[0]}", fontsize="xx-large",
                      fontweight='bold')
        ax2.set_title(f"Testing accuracy: {network.capitalize()} on {self.dataset[0]}", fontsize="xx-large",
                      fontweight='bold')
        ax1.set_xlabel("Time (s)", fontsize="x-large")
        ax2.set_xlabel("Time (s)", fontsize="x-large")
        ax1.set_yscale('log')
        # Set ylim dynamically based on min and max accuracy, with some padding
        ax2.set_ylim(min_accuracy - 0.1, max_accuracy + 0.01)
        ax1.legend(frameon=False)
        ax2.legend(frameon=False)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        output_path = Path(f"results/{self.models[0]}/{self.dataset[0]}").expanduser()
        if not output_path.exists():
            print(f"Info: Output dir {output_path} does not exist, creating it")
            output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_path}/testaccuracy_trainloss")

    def plot_train_test_losses(self):
        if self.file_names is None:
            raise ValueError("File names must be specified")
        if len(self.dataset) != 1:
            raise ValueError("Only one dataset must be specified")
        if len(self.models) != 1:
            raise ValueError("Only one model is supported")

        plt.figure(figsize=(10, 10))
        for idx, k in enumerate(self.results.keys()):
            if self.type_acc == "top1":
                tot_time, train_loss, valid_acc = (self.results[k]["tot_time"],
                                                   self.results[k]["train_loss"],
                                                   self.results[k]["test_accuracy1"])
            elif self.type_acc == "top5":
                tot_time, train_loss, valid_acc = (self.results[k]["tot_time"],
                                                   self.results[k]["train_loss"],
                                                   self.results[k]["test_accuracy5"])
            else:
                raise ValueError("Unrecognized type {}".format(self.type_acc))
            if "swin_t" in self.models:
                optimizer = k.split("_")[7]
            else:
                optimizer = k.split("_")[5]
            c = define_color(
                optimizer)
            plt.plot(tot_time, train_loss, '-', color=c, label=f"{optimizer} | Training")
            plt.plot(tot_time, valid_acc, '--', color=c, label=f"{optimizer} | Testing")
        if "swin_t" in self.models:
            network = "Tiny Swin"
        else:
            network = get_network(k)

        plt.title(f"Training loss & Testing Accuracy: {network.capitalize()} on {self.dataset[0]}")
        plt.xlabel("Time")
        plt.yscale('log')
        plt.ylabel("Loss")
        plt.grid()
        # Set ylim dynamically based on min and max accuracy, with some padding
        plt.legend()
        plt.tight_layout()

        output_path = Path(f"results/{self.models[0]}/{self.dataset[0]}").expanduser()
        if not output_path.exists():
            print(f"Info: Output dir {output_path} does not exist, creating it")
            output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_path}/test_train_losses")


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
        # "memory_used": memory_used
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


def get_file_names(optimizers: list, datasets: list, models: list, type_acc: str):
    files_names = []
    if "AdaFisher" in optimizers:
        perf = Performance(optimizers=["AdaFisher"], models=models, dataset=datasets,
                           type_acc=type_acc)
        file_best_AdaFisher = perf.get_best_accuracy(verbose=False)
        files_names.append(file_best_AdaFisher)
        optimizers = [item for item in optimizers if item != "AdaFisher"]
    elif "AdaFisherW" in optimizers:
        perf = Performance(optimizers=["AdaFisherW"], models=models, dataset=datasets,
                           type_acc=type_acc)
        file_best_AdaFisher = perf.get_best_accuracy(verbose=False)
        files_names.append(file_best_AdaFisher)
        optimizers = [item for item in optimizers if item != "AdaFisherW"]

    perf = Performance(optimizers=optimizers, models=models, dataset=datasets,
                       type_acc=type_acc)
    for f in perf.files:
        if "swin_t" in models:
            if int(f.split('_')[3][-1]) == 8:
                files_names.append(f.split('/')[3])
        else:
            if int(f.split('_')[2][-1]) == 8:
                files_names.append(f.split('/')[3])
    return files_names


if __name__ == "__main__":
    optimizers = ["AdamW", "AdaHessian", "kfac", "Shampoo", "AdaFisherW"]
    models = ["swin_t"]
    datasets = ["CIFAR10"]
    type_acc = "top1"
    file_names = get_file_names(optimizers, datasets, models, type_acc)
    perf = Performance(optimizers=optimizers, models=models, dataset=datasets,
                       type_acc=type_acc, file_names=file_names)
    # perf.get_mean_std()
    # perf.get_best_accuracy(verbose=True)
    perf.plot_WCT_together()
    perf.plot_trainAcc_testLoss()
    perf.plot_train_test_losses()
