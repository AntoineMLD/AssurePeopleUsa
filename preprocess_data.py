import pandas as pd

# Ajout de la méthode categorize_bmi
def categorize_bmi_and_interaction(df):
    df['bmi_category'] = pd.cut(df['bmi'], 
                                bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')],
                                labels=['underweight', 'normal weight', 'overweight', 'obesity class I', 'obesity class II', 'obesity class III'])
    df['smoker_num'] = df['smoker'].map({'yes': 1, 'no': 0})
    df['bmi_smoker_interaction'] = df['bmi'] * df['smoker_num']
    return df