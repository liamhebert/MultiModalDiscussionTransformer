import pandas as pd

def main():
    df = pd.read_csv('slurs/kurrek.2020.slur-corpus.csv')
    df = df[['id', 'link_id', 'gold_label']]
    df['id'] = df['id'].str[3:]
    df['link_id'] = df['link_id'].str[3:]
    df = df.rename({'gold_label': 'label'}, axis=1)
    df.to_parquet('slurs-processed.parquet')

if __name__ == '__main__':
    main()