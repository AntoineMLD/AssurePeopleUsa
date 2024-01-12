import pandas as pd

# Ajout de la méthode split_regions
def split_regions(df):
    for region in ['east', 'west', 'north', 'south']:
        df['is_' + region] = (df['region'] == region).astype(int)
    return df.drop('region', axis=1)

# Ajout de la méthode categorize_bmi
def categorize_bmi(df):
    df['bmi_category'] = pd.cut(df['bmi'], 
                                bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')],
                                labels=['underweight', 'normal weight', 'overweight', 'obesity class I', 'obesity class II', 'obesity class III'])
    return df.drop('bmi', axis=1)
