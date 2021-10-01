# cluster substrates

from pathlib import Path
import pandas as pd


# testing clustering for substrates
# for testing only: deoxy, arylation, suzuki
def data_loader(*datasets):
    for dataset in datasets:
        if dataset == "deoxy":
            fp = Path.cwd().parent / 'data/deoxy/substrate.csv'
        else:
            fp = ''

    df = pd.read_csv(fp)

    return df


if __name__ == '__main__':
    tp = data_loader('deoxy')
    print(tp)

