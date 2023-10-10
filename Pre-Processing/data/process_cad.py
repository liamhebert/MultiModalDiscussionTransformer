import pandas as pd

def main(): 
    ## load data in 
    df = pd.read_csv('cad/cad_v1_1.tsv', sep='\t')

    # keep only data that was used in training
    df = df[(df['split'] == 'train') | (df['split'] == 'test') | (df['split'] == 'dev')]
    
    #df.loc[df['info_id.link'].isna(), 'info_id.link'] = df[df['info_id.link'].isna()]['info_id'].str[:6]
    
    df = df[['info_id.link', 'info_id', 'annotation_Primary']]
    df['info_id'] = df['info_id'].str.replace('-post', '')
    df['info_id'] = df['info_id'].str.replace('-title', '')
    grouped = df.groupby('info_id').agg(list)
    def get_real_label(x):
        vals = {}
        for y in x:
            if y not in vals:
                vals[y] = 0
            vals[y] += 1
        vals = sorted([(key, value) for key, value in vals.items() if key != 'Neutral'], key= lambda x: x[1])
        if len(vals) == 0:
            return 'Neutral'
        else:
            return vals[-1][0]
    df = df.drop('annotation_Primary', axis=1).join(grouped['annotation_Primary'].apply(get_real_label), on='info_id')   
    df = df.drop_duplicates(subset='info_id')   

    df = df.rename({'annotation_Primary': 'label', 'info_id': 'id', 'info_id.link': 'link_id'}, axis=1)          
    df.to_parquet('cad-processed.parquet')

if __name__ == '__main__':
    main()