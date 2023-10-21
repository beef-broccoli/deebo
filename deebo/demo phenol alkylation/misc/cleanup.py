import pandas as pd

def cleanup(df):

    df['yield'] = df['yield'].apply(lambda x: x/100 if x>=0 else 0)

    return df

if __name__ == '__main__':
    for i in range(4):
        if i == 0:
            n = 36
        else:
            n = 18
        df = pd.read_csv(f'../backups/round{i+1}-{n}/proposed_experiments_filled.csv')
        df = cleanup(df)
        df.to_csv(f'../backups/round{i+1}-{n}/proposed_experiments_clean.csv')