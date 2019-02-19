import pandas as pd

imgdf = pd.read_csv('../metanew.csv',sep="\t")
print(imgdf.columns.values)

for img_index in range(len(imgdf)):
    print(imgdf.iloc[img_index]['image_id'],\
                imgdf.iloc[img_index]['object_id'],
                imgdf.iloc[img_index]['sample_type'])