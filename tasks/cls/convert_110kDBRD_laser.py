import os, pandas as pd 

DATA_DIR='data/cls-acl10-unprocessed/110kDBRD'
NEW_DIR = 'data/cls-acl10-unprocessed/nl/books'
for part in ['unsup','test','train']:
    revs_dict = {"labels":[],"text":[]}
    if part=="unsup":
        TEXT_DIR = f"{DATA_DIR}/{part}"
        for txt in os.listdir(TEXT_DIR)[:1000]:
            with open(f"{TEXT_DIR}/{txt}","r") as f:
                text = f.read()
                # use -1 to indicate no label
                revs_dict["labels"].append(-1)
                revs_dict["text"].append(text)
    else:
        for i, label in enumerate(['neg','pos']):
            TEXT_DIR = f"{DATA_DIR}/{part}/{label}"
            for txt in os.listdir(TEXT_DIR)[:1000]:
                with open(f"{TEXT_DIR}/{txt}","r") as f:
                    text = f.read()
                    revs_dict["labels"].append(i)
                    revs_dict["text"].append(text)

    
    # dict to dataframe
    revs_df = pd.DataFrame.from_dict(revs_dict)
    # shuffle the data
    revs_df = revs_df.sample(frac=1).reset_index(drop=True) 

    if part=="train":
        file_size = revs_df.shape[0]
        print("Use last 10% of training set as dev set")
        train_dev_split = int(file_size*0.9)  
        with open(f"{NEW_DIR}/train.txt","w") as f:
            for i in revs_df.index[:train_dev_split]:
                label = revs_df.at[i,"labels"]
                text = revs_df.at[i,"text"].replace("\n","")
                f.write("{}\t{}\n".format(label,text))

        with open(f"{NEW_DIR}/dev.txt","w") as f:
            for i in revs_df.index[train_dev_split:]:
                label = revs_df.at[i,"labels"]
                text = revs_df.at[i,"text"].replace("\n","")
                f.write("{}\t{}\n".format(label,text))
    # for test and unsupervised set
    else:
        with open(f"{NEW_DIR}/{part}.txt","w") as f:
            for i in revs_df.index:
                label = revs_df.at[i,"labels"]
                text = revs_df.at[i,"text"].replace("\n","")
                f.write("{}\t{}\n".format(label,text))