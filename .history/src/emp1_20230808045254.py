from scipy.io import arff
import pandas as pd

data, meta = arff.loadarff('path/to/yourfile.arff')
df = pd.DataFrame(data)

print(df.head())