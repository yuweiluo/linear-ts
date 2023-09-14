from scipy.io import arff
import pandas as pd

data, meta = arff.loadarff('../datasets/EEG.arff')
df = pd.DataFrame(data)

print(df.head())