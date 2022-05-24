import numpy as np
import pandas as pd
import click
from sklearn.preprocessing import StandardScaler
import configs
import warnings

warnings.filterwarnings(action="ignore")


def check_null_fill_data(df):
    for col in df.columns:
        if col == configs.TARGET:
            continue

        if len(df.loc[df[col].isnull()]) != 0:
            if df[col].dtype == "float64" or df[col].dtype == "int64":
                df.loc[df[col].isnull(), col] = df[col].median()
            else:
                df.loc[df[col].isnull(), col] = df[col].mode()[0]


def srt_list_num_list(train_test_df):
    str_list = []
    num_list = []
    for colname, colvalue in train_test_df.iteritems():
        if colname == configs.TARGET or colname == configs.ID:
            continue

        if type(colvalue[1]) == str:
            str_list.append(colname)
        else:
            num_list.append(colname)
    return str_list, num_list


@click.command()
@click.argument("train_path", type=click.Path())
@click.argument("test_path", type=click.Path())
@click.argument("output_path_train", type=click.Path())
@click.argument("output_path_test", type=click.Path())
def auto_preprocess(
    train_path: str, test_path: str, output_path_train: str, output_path_test: str
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_len = len(train_df)
    train_test = pd.concat(objs=[train_df, test_df], axis=0).reset_index(drop=True)
    train_test = train_test.drop(configs.DROP_COLS, axis=1)
    check_null_fill_data(train_test)
    str_list, num_list = srt_list_num_list(train_test)
    scaler = StandardScaler()
    train_test[num_list] = scaler.fit_transform(train_test[num_list])
    train_test = pd.get_dummies(train_test, columns=str_list)
    train = train_test[:train_len]
    test = train_test[train_len:]
    test.drop(labels=[configs.TARGET], axis=1, inplace=True)

    train["Fare"] = np.log(train["Fare"])
    test["Fare"] = np.log(test["Fare"])
    train["SibSp"] = np.log(train["SibSp"])
    test["SibSp"] = np.log(test["SibSp"])
    train["Parch"] = np.log(train["Parch"])
    test["Parch"] = np.log(test["Parch"])

    # train.to_csv(f"{output_path_train}train_processed.csv", index=False)
    train.to_csv(output_path_train, index=False)
    # test.to_csv(f"{output_path_test}test_processed.csv", index=False)
    test.to_csv(output_path_test, index=False)
    print(train.head())


if __name__ == "__main__":
    auto_preprocess()
