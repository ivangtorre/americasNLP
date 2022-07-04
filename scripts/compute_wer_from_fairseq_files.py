import pandas as pd
from jiwer import wer
###################################################
hypotesis = pd.read_csv("hypo.word-checkpoint_best.pt-test_rtve_small.txt", header=None)
referencia = pd.read_csv("ref.word-checkpoint_best.pt-test_rtve_small.txt", header=None)
####################################################
def preprocess(df):
    df["text"] = df[0].str.split("(", expand=True)[0]
    df["order"] = df[0].str.split("(", expand=True)[1]
    df["order"] = df["order"].str.split("None-", expand=True)[1].str.split(")", expand=True)[0]
    return df[["text", "order"]]


hypotesis = preprocess(hypotesis)
referencia = preprocess(referencia)

df = pd.merge(hypotesis, referencia, how='inner', on="order")
df.columns = ['hypotesis', 'order', 'referencia']
df.order = df.order.astype(int)
df = df.sort_values(by="order")
df.reset_index(inplace=True, drop=True)
df['counts'] = df['referencia'].str.count(' ') + 1


def fab(row):
    return wer(row['referencia'], row['hypotesis'])


df["wer"] = df.apply(fab, axis=1)
df["wer_counts"] = df["counts"] * df["wer"]


print(df["wer"])
print("------------------------------------------------")
MEAN_WER = df["wer_counts"].sum()/df["counts"].sum()
print("MEAN WER: " + str(MEAN_WER))
print("------------------------------------------------")
