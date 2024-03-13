import os
import json
from matplotlib import pyplot as plt


def get_data(name_file):
    path = "logs/" + name_file
    with open(os.path.join(path, "train_loss.txt"), 'r') as f:
        train_loss = [float(number.strip()) for number in f.readlines()]
    with open(os.path.join(path, "test_loss.txt"), 'r') as f:
        valid_loss = [float(number.strip()) for number in f.readlines()]
    with open(os.path.join(path, "train_accuracy1.txt"), 'r') as f:
        train_accuracy1 = [float(number.strip()) for number in f.readlines()]
    with open(os.path.join(path, "train_accuracy5.txt"), 'r') as f:
        train_accuracy5 = [float(number.strip()) for number in f.readlines()]
    with open(os.path.join(path, "test_accuracy1.txt"), 'r') as f:
        test_accuracy1 = [float(number.strip()) for number in f.readlines()]
    with open(os.path.join(path, "test_accuracy5.txt"), 'r') as f:
        test_accuracy5 = [float(number.strip()) for number in f.readlines()]
    with open(os.path.join(path, "tot_time.txt"), 'r') as f:
        tot_time_s = [float(number.strip()) for number in f.readlines()]
        tot_time = process_WCT(tot_time_s)
    with open(os.path.join(path, "mem_used.txt"), 'r') as f:
        memory_used = []
        text = f.readlines()
        for l in text:
            if l == '/n':
                continue
            else:
                memory_used.append(float(l.split(":")[3]))
    with open(os.path.join(path, "config.json"), 'r') as f:
        logs = json.load(f)

    return {"logs": logs, "train_accuracy1": train_accuracy1, "train_accuracy5": train_accuracy5,
            "test_accuracy1": test_accuracy1, "test_accuracy5": test_accuracy5, "tot_time": tot_time,
            "train_loss": train_loss, "valid_loss": valid_loss, "memory_used": memory_used}


def process_WCT(time_results):
    for i in range(1, len(time_results)):
        time_results[i] += time_results[i - 1]
    return time_results


def get_best_accuracy(results, type: str = "top1"):
    for k in results.keys():
        if type == "top1":
            train_accuracy, test_accuracy = results[k]["train_accuracy1"], results[k]["test_accuracy1"]
        elif type == "top5":
            train_accuracy, test_accuracy = results[k]["train_accuracy5"], results[k]["test_accuracy5"]
        else:
            raise ValueError("Unrecognized type {}".format(type))
        optimizer = k.split("_")[5]
        print(f"Optimizer: {optimizer} | The best accuracy is obtained at epoch {test_accuracy.index(max(test_accuracy))} "
              f"with {round(max(test_accuracy) * 100, 2)} %")


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
        c = "navy"
    elif k == "SGD":
        c = "lightpink"
    else:
        raise ValueError("Error during assigning the color")
    return c


def plot_curves_diff(results, loc, dataset: list, type: str = "top1"):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    ax1, ax2, ax3, ax4 = axs.flatten()
    for idx, k in enumerate(results.keys()):
        if type == "top1":
            train_accuracy, test_accuracy, train_loss, valid_loss = (results[k]["train_accuracy1"],
                                                                     results[k]["test_accuracy1"],
                                                                     results[k]["train_loss"],
                                                                     results[k]["valid_loss"])
        elif type == "top5":
            train_accuracy, test_accuracy, train_loss, valid_loss = (results[k]["train_accuracy5"],
                                                                     results[k]["test_accuracy5"],
                                                                     results[k]["train_loss"],
                                                                     results[k]["valid_loss"])
        else:
            raise ValueError("Unrecognized type {}".format(type))
        optimizer = k.split("_")[5]
        c = define_color(optimizer)
        if k.split("_")[4] == dataset[0]:
            ax1.plot(train_loss, '-', color=c, label="{} Training".format(optimizer))
            ax2.plot(test_accuracy, '-', color=c, label="{} Testing".format(optimizer))

        elif k.split("_")[4] == dataset[1]:
            ax3.plot(train_loss, '-', color=c, label="{} Training".format(optimizer))
            ax4.plot(test_accuracy, '-', color=c, label="{} Testing".format(optimizer))

    network = k.split("_")[3] if k.split("_")[3][-1] != "r" else k.split("_")[3][:8]
    ax1.set_title(f"Training: {network.capitalize()} on {dataset[0]}")
    ax2.set_title(f"Testing:  {network.capitalize()} on {dataset[0]}")
    ax3.set_title(f"Training:  {network.capitalize()} on {dataset[1]}")
    ax4.set_title(f"Testing:  {network.capitalize()} on {dataset[1]}")
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

    # ax2.set_ylim(0.85, 0.95)
    # ax4.set_ylim(0.6, 0.75)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    plt.tight_layout()
    plt.savefig(f"results/{loc}/testaccuracy_trainloss_{dataset[0]}_{dataset[1]}_{network.capitalize()}")


def plot_curves_together(results, loc, dataset: list, type: str = "top1"):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    ax1, ax2 = axs.flatten()
    for idx, k in enumerate(results.keys()):
        if type == "top1":
            train_accuracy, test_accuracy, train_loss, valid_loss = (results[k]["train_accuracy1"],
                                                                     results[k]["test_accuracy1"],
                                                                     results[k]["train_loss"],
                                                                     results[k]["valid_loss"])
        elif type == "top5":
            train_accuracy, test_accuracy, train_loss, valid_loss = (results[k]["train_accuracy5"],
                                                                     results[k]["test_accuracy5"],
                                                                     results[k]["train_loss"],
                                                                     results[k]["valid_loss"])
        else:
            raise ValueError("Unrecognized type {}".format(type))

        optimizer = k.split("_")[5]
        c = define_color(optimizer)
        ax1.plot(train_loss, '-', color=c, label="{} Training".format(optimizer))
        ax2.plot(test_accuracy, '-', color=c, label="{} Testing".format(optimizer))

    network = k.split("_")[3] if k.split("_")[3][-1] != "r" else k.split("_")[3][:8]
    ax1.set_title(f"Training: {network.capitalize()} on {dataset[0]}")
    ax2.set_title(f"Testing: {network.capitalize()} on {dataset[0]}")
    ax1.set_xlabel("Epochs")
    ax2.set_xlabel("Epochs")
    ax1.set_yscale('log')
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    ax1.set_xlim(0, 200)
    ax2.set_xlim(0, 200)
    # ax2.set_ylim(0.25, 0.55)
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"results/{loc}/testaccuracy_trainloss_{dataset[0]}_{network.capitalize()}")


def plot_WCT_together(results, loc, dataset: list, type: str = "top1"):
    plt.figure(figsize=(10, 10))
    for idx, k in enumerate(results.keys()):
        if type == "top1":
            test_accuracy, tot_time = (results[k]["test_accuracy1"], results[k]["tot_time"])
        elif type == "top5":
            test_accuracy, tot_time = (results[k]["test_accuracy5"], results[k]["tot_time"])
        else:
            raise ValueError("Unrecognized type {}".format(type))

        optimizer = k.split("_")[5]
        c = define_color(optimizer)
        plt.plot(tot_time, test_accuracy, '-', color=c, label="{} Testing".format(optimizer))

    network = k.split("_")[3] if k.split("_")[3][-1] != "r" else k.split("_")[3][:8]

    plt.title(f"Testing: {network.capitalize()} on {dataset[0]}")
    plt.xlabel("Time (s)")
    plt.ylabel("Accuracy")
    # plt.ylim(0.25, 0.55)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{loc}/testaccuracyWCT_{dataset[0]}_{network.capitalize()}")


def main(models_to_plot: list, loc: str, dataset: list, type: str) -> None:
    path = "logs"
    results = {}
    files = os.listdir(path)
    for f in files:
        if f in models_to_plot:
            results[f] = get_data(f)

    get_best_accuracy(results)
    # If you have same network for all experiments but on two different datasets
    # plot_curves_diff(results = results,
    #                  loc = loc, # loc on the results folder
    #                  dataset = dataset,
    #                  type = type)
    plot_curves_together(results = results,
                         loc = loc,
                         dataset = dataset,
                         type = type)
    plot_WCT_together(results = results,
                         loc = loc,
                         dataset = dataset,
                         type = type)


if __name__ == "__main__":
    models_to_plot = ["date=2024-02-08_results_trial=0_resnet18Cifar_CIFAR10_kfac_weight_decay=0.0005_momentum=0.9_stat_decay=0.95_damping=0.003_kl_clip=0.001_TCov=10.0_TInv=100.0_batch_averaged=1.0_CosineAnnealingLRT_max=200.0_LR=0.001",
                      "date=2024-02-08_results_trial=0_resnet18Cifar_CIFAR10_Adam_weight_decay=0.0005_CosineAnnealingLRT_max=200.0_LR=0.001",
                      "date=2024-02-08_results_trial=0_resnet18Cifar_CIFAR10_AdaFisher_weight_decay=0.0005_beta3=0.91_Lambda=0.003_TCov=10.0_CosineAnnealingLRT_max=200.0_LR=0.01",
                      "date=2024-02-08_results_trial=0_resnet18Cifar_CIFAR10_AdaFisherW_weight_decay=0.0005_beta3=0.91_Lambda=0.003_CosineAnnealingLRT_max=200.0_LR=0.01"]
    loc = "Resnet18"
    dataset = ["CIFAR10"]
    type = "top1"
    main(models_to_plot = models_to_plot, loc = loc, dataset=dataset, type=type)
