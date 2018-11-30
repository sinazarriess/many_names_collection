import pandas as pd
from collections import Counter
from random import shuffle

def get_old_taboos(bdf):

    im2taboo = {}

    for rx,row in bdf.iterrows():
        for ix in ['0','1','2','3','4']:
            im = row['img_%s_url'%ix]
            taboo = row['taboolist_%s'%ix]

            im2taboo[im] = taboo

    return im2taboo

# add 1 new taboo word which was the most common in the
# answers of the previous round
# random in case of ties
def get_new_taboos_most_common(rdf):
    im2taboo = {}

    for ix in ['0','1','2','3','4']:
        images = set(rdf['Input.img_%s_url'%ix])
        for im in images:
            #print(im)
            imdf = rdf[rdf['Input.img_%s_url'%ix] == im]
            taboos = set(imdf['Input.taboolist_%s'%ix])
            if len(taboos) > 0:
                names = Counter([x.lower() for x in imdf['Answer.inputbox_objname-%s'%ix] if x != '{}'])
                names = Counter({n:names[n] for n in names if (names[n] > 1) and not (n in taboos)})
           
                if len(names) > 0:
                    most_common = names.most_common(1)[0][0]
                    taboos = list(taboos)
                    taboos.append(most_common)
           
                #print("Taboos:",taboos)
                #print("Names:",names)
                im2taboo[im] = taboos

    return im2taboo



def make_new_table(old_taboos,new_taboos,table_id):
    for im in old_taboos:
        if im not in new_taboos:
            new_taboos[im] = [old_taboos[im]]
    print("New taboos:",len(new_taboos.keys()))

    new_sort = list(new_taboos.keys())
    shuffle(new_sort)

    new_df_rows = []

    new_images = []
    for hit in range(0,60,5):
        images = [new_sort[hit+image] for image in range(5)]
        print([hit+image for image in range(5)])
        new_images += images
        taboos = [", ".join(new_taboos[new_sort[hit+image]]) for image in range(5)]
        new_df_rows.append(images+taboos)

    print("New images,",len(set(new_images)))

    new_df = pd.DataFrame(new_df_rows,columns=['img_%d_url'%d for d in range(5)] + \
        ['taboolist_%d'%d for d in range(5)])

    new_df.to_csv("pilot-amt_table_%d.csv"%table_id,index=False)




batchdf = pd.read_csv("pilot-amt_table.csv")
old_taboos = get_old_taboos(batchdf)
print("Old taboos:",len(old_taboos.keys()))

resultdf = pd.read_csv("Batch_3451914_batch_results.csv")
new_taboos = get_new_taboos_most_common(resultdf)
make_new_table(old_taboos,new_taboos,2)









