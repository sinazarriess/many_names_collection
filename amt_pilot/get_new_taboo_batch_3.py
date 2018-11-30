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

def get_new_taboos_listofdf(cdf):

    im2taboolists = {}
    im2taboowords = {}
    im2names = {}

    for rx,row in cdf.iterrows():
        for ix in ['0','1','2','3','4']:
            im = row['Input.img_%s_url'%ix]
            taboo = str(row['Input.taboolist_%s'%ix])
            #print("Taboo",str(taboo))
            if taboo != "nan":
                taboowords = taboo.split(", ")
                name = row['Answer.inputbox_objname-%s'%ix]

                if not im in im2taboolists:
                    im2taboolists[im] = []
                    im2taboowords[im] = []
                    im2names[im] = []
                for t in taboowords:
                    if not t in im2taboowords[im]:
                        im2taboowords[im].append(t)
                if not taboo in im2taboolists[im]:
                    im2taboolists[im].append(taboo)
                im2names[im].append(name.lower())

    new_taboos = {}    
    for im in im2names:
        #print(im)
        names = [n for n in im2names[im] if (not n == '{}') \
        and (not n in im2taboowords[im])]
        #print("Taboo lists:",im2taboolists[im])
        #print("Names:",Counter(names))
        tl = im2taboolists[im]
        shuffle(tl)
        most_common = Counter(names).most_common(1)[0][0]
        freq = Counter(names).most_common(1)[0][1]
        if freq > 1:
            new_taboos[im] = ", ".join([tl[0],most_common])
        else:
            new_taboos[im] = tl[0]

            print("No new tabooword")
            print("Names:",Counter(names))

        print("New:",new_taboos[im])

        #print("Tokens:",len(names))
        #print("Types:",len(set(names)))
    return new_taboos

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
        taboos = [new_taboos[new_sort[hit+image]] for image in range(5)]
        new_df_rows.append(images+taboos)

    print("New images,",len(set(new_images)))

    new_df = pd.DataFrame(new_df_rows,columns=['img_%d_url'%d for d in range(5)] + \
        ['taboolist_%d'%d for d in range(5)])

    new_df.to_csv("pilot-amt_table_%d.csv"%table_id,index=False)




batchdf = pd.read_csv("pilot-amt_table.csv")
old_taboos = get_old_taboos(batchdf)
print("Old taboos:",len(old_taboos.keys()))

resultdf1 = pd.read_csv("collected_data/Batch_3451914_batch_results.csv")
resultdf2 = pd.read_csv("collected_data/Batch_3452735_batch_results_approved.csv")
print("R2",len(resultdf2))
resultdf2 = resultdf2[resultdf2.Reject != "x"]
print("R2 approved",len(resultdf2))

resultdf = pd.concat([resultdf2,resultdf1])

new_taboos = get_new_taboos_listofdf(resultdf)
print(len(new_taboos))
make_new_table(old_taboos,new_taboos,3)








