import os
import pandas as pd
import numpy as np

def read_data(input_file):
    input_path = os.path.join(os.getcwd(), input_file)
    df = pd.read_csv(input_path, header=None, sep='\s+', names=['seq_name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class'])
    df['class'].replace(['cp', 'im', "imS", "imL", "imU", "om", "omL", "pp"], [1, 2, 3, 4, 5, 6, 7, 8], inplace=True)
    print(df.to_markdown())


if __name__ == '__main__':
    read_data('ecoli.data')