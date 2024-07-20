import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt


def preproc_data_period(period):
    postfix = {
        "Wakefulness": "W",
        "N2": "N",
        "N3": "D",
        "REM": "R"}[period]
    data = []
    for root, dirs, files in os.walk(f"{period}_AllRegions"):
        for file in files:
            region = file.split("_")[0]
            npz_name = f"npz_folder/{period}_AllRegions_{region}_{postfix}.npz"
            data.append(np.load(npz_name)['X'].mean(axis=0))
    data = np.stack(data)
    data = pd.DataFrame(data)
    return data


def preproc_data_region(region):
    region = "Cuneus"
    w_file_name = f"npz_folder/Wakefulness_AllRegions_{region}_W.npz"
    n2_file_name = f"npz_folder/N2_AllRegions_{region}_N.npz"
    n3_file_name = f"npz_folder/N3_AllRegions_{region}_D.npz"
    rem_file_name = f"npz_folder/REM_AllRegions_{region}_R.npz"

    w_data = np.load(w_file_name)
    n2_data = np.load(n2_file_name)
    n3_data = np.load(n3_file_name)
    rem_data = np.load(rem_file_name)
    df = pd.DataFrame.from_dict({
        'Wakefulness': w_data,
        'N2': n2_data,
        'N3': n3_data,
        'REM': rem_data
    })
    return df


def fit(df):
    x = df.iloc[:,0:4] #x為所有特徵資料
    model = KMeans(n_clusters=3, n_init='auto',random_state=1) #預計分為三群，迭代次數由模型自行定義
    model.fit(x) #建立模型
    plt.scatter(df['petal length (cm)'],df['petal width (cm)'], c=model.labels_)
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.show()


if __name__ == "__main__":
    # print(preproc_data_period("N2"))
    for root, dirs, files in os.walk(f"N2_AllRegions"):
        for file in files:
            region = file.split("_")[0]

            w_file_name = f"npz_folder/Wakefulness_AllRegions_{region}_W.npz"
            n2_file_name = f"npz_folder/N2_AllRegions_{region}_N.npz"
            n3_file_name = f"npz_folder/N3_AllRegions_{region}_D.npz"
            rem_file_name = f"npz_folder/REM_AllRegions_{region}_R.npz"
            w_data = np.load(w_file_name)['X']
            n2_data = np.load(n2_file_name)['X']
            n3_data = np.load(n3_file_name)['X']
            rem_data = np.load(rem_file_name)['X']
            w = np.fft.fft(w_data).mean(axis=0)[1000: 6000]
            n = np.fft.fft(n2_data).mean(axis=0)[1000: 6000]
            d = np.fft.fft(n3_data).mean(axis=0)[1000: 6000]
            r = np.fft.fft(rem_data).mean(axis=0)[1000: 6000]
            # print(x.shape)
            plt.plot(np.abs(w))
            plt.plot(np.abs(n))
            plt.plot(np.abs(d))
            plt.plot(np.abs(r))
            plt.show()

