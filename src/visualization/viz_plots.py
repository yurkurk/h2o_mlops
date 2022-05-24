
import pandas as pd
import click
import configs
import warnings

warnings.filterwarnings(action="ignore")
import matplotlib.pyplot as plt


ID = "PassengerId"
TARGET = "Survived"
SEED = 2022


def plot_hist(df, variable, output_path):
    plt.figure(figsize=(15, 10))
    plt.hist(df[variable], color="darkred")
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist ".format(variable))
    # plt.show()
    plt.savefig(f"{output_path}plot_hist_{variable}.png")


# @click.command()
# @click.argument("dataframe_path", type=click.Path())
# @click.argument("output_path", type=click.Path())
def viz_plots(dataframe_path: str, output_path: str):
    """

    Create histograms for each feature.

    :param dataframe_path: path to raw data.
    :param output_path: path to dir where plots will be saved.
    :return:

    """
    train = pd.read_csv(dataframe_path)

    str_list = []
    num_list = []
    for colname, colvalue in train.iteritems():
        if colname == TARGET or colname == ID:
            continue
        if type(colvalue[1]) == str:
            str_list.append(colname)
        else:
            num_list.append(colname)

    for col in num_list:
        plot_hist(train, col, output_path)


if __name__ == "__main__":
    output = 'reports/figures/'
    viz_plots(configs.TRAIN_PATH, output)
