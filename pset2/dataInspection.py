import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat

# Import data from MATLAB and analyze the structure of the data 
mat = loadmat("data_updateBN2-1.mat")
data = mat["data"]
print(data.shape)

columns = ["SF", "F", "BD", "V", "Z", "R", "PL"]
df = pd.DataFrame(data, columns=columns)

for col in df.columns:
    print(
        f"{col}:"
        f"{df[col].nunique()}"
    )

for col in ["SF", "F", "BD", "V", "Z", "R"]:
    values = np.sort(df[col].unique())

    print(f"\n{col}")
    print(values)

plt.figure()
plt.hist(df["PL"], bins=50)
plt.xlabel("Propagation Loss (PL)")
plt.ylabel("Count")
plt.title("PL Distribution")

plt.show()

print(df["PL"].describe())