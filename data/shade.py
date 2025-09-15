import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/alan/PyCharmMiscProject/Flow_Matching/data/eulerian/experiments.csv")

df = df[df["key"].isin([1,2,3,4])]


stats = df.groupby("key")["w2"].agg(["mean", "std"]).reset_index()


plt.figure(figsize=(7,5))

x = stats["key"]
y = stats["mean"]
shade = stats["std"]

plt.plot(x, y, color="C0", marker="o", label="w2 (mean)")
plt.fill_between(x, y - shade, y + shade, color="C0", alpha=0.2)

plt.xlabel("epoch")
plt.ylabel("w2")
plt.title("w2 across epochs")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


