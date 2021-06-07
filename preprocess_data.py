import pandas as pd


# datasets: https://www.kaggle.com/crawford/gene-expression

def preprocess_data():
    train_data = pd.read_csv("src/main/resources/data_set_ALL_AML_train.csv")
    test_data = pd.read_csv("src/main/resources/data_set_ALL_AML_independent.csv")
    labels = pd.read_csv("src/main/resources/actual.csv", index_col="patient")

    cols = [col for col in test_data.columns if 'call' in col]
    test = test_data.drop(cols, 1)
    cols = [col for col in train_data.columns if 'call' in col]
    train = train_data.drop(cols, 1)

    patients = [str(i) for i in range(1, 73, 1)]
    df_all = pd.concat([train, test], axis=1)[patients]
    df_all = df_all.T

    df_all["patient"] = pd.to_numeric(patients)
    labels["cancer"] = pd.get_dummies(labels.cancer, drop_first=True)

    return pd.merge(df_all, labels, on="patient")


def main():
    data = preprocess_data()
    data.to_csv("src/main/resources/data_set.csv", index=False)


if __name__ == "__main__":
    main()
