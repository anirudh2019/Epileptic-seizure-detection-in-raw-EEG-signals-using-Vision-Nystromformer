import os, random
import pandas as pd, numpy as np
import scipy.io
from tqdm.auto import tqdm
from datapipeline_utils import Kfold_crossval, split_df
    
seed = 1
random.seed(seed)
np.random.seed(seed)

sample_size = 256
old_data_dir = "./IIT_Delhi_EEG_dataset/"
new_data_dir = f"./IIT_Delhi_{sample_size}/"
num_folds = 5
data_split = {"train": 0.75, "val": 0.25}
overlap = 0.5

def extract_data(overlap = 0.5):
    for subset in tqdm(["ictal", "interictal"]):
        if not os.path.exists(new_data_dir+subset+"/"):
            os.makedirs(new_data_dir+subset+"/")

        for sample in sorted(os.listdir(old_data_dir+subset+"/")):
            mat = scipy.io.loadmat(old_data_dir+subset+"/"+sample)[subset].T[0]
            if subset=="interictal":
                for num, i in enumerate(range(0,1024,sample_size)):
                    pd.DataFrame(mat[i:i+sample_size]).to_csv(new_data_dir+subset+"/"+sample[:-4]+f"_{num+1}.csv", index=False, header = False)
            else: #Apply Overlapping Window for seizure samples
                for num, i in enumerate(range(0,mat.shape[0],int(sample_size*(1-overlap)))):
                    # BOUNDARY CONDITION
                    if i+sample_size-1>=mat.shape[0]:
                        pd.DataFrame(mat[mat.shape[0]-sample_size:mat.shape[0]]).to_csv(new_data_dir+subset+"/"+sample[:-4]+f"_{num+1}.csv", index=False, header = False)
                    else:
                        pd.DataFrame(mat[i:i+sample_size]).to_csv(new_data_dir+subset+"/"+sample[:-4]+f"_{num+1}.csv", index=False, header = False)
    return

def create_df():
    df = pd.DataFrame(columns=["edf_dir","label"])    
    for subset in ["ictal", "interictal"]:
        hmap = {"edf_dir":[]}
        hmap["edf_dir"].extend(list(map(lambda x: subset+"/"+x, sorted(os.listdir(new_data_dir+subset)))))
        subset_df = pd.DataFrame.from_dict(hmap, orient="columns")
        subset_df.insert(1,"label", [0 if subset=="interictal" else 1 for _ in range(subset_df.shape[0])])
        df = pd.concat([df, subset_df], axis = 0)
    df.to_csv(new_data_dir+"total_data.csv", index = False)
    return df

def train_val_test_split(df):
    folds = Kfold_crossval(df, num_folds)
    for fold, (train_df, test_df) in enumerate(folds):     
        fold_path = new_data_dir+f"fold-{fold+1}"+"/"
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)
        train_df, val_df = split_df(train_df, data_split = data_split)
        train_df.to_csv(fold_path+"train.csv", index=False)
        val_df.to_csv(fold_path+"val.csv", index=False)
        test_df.to_csv(fold_path+"test.csv", index=False)
    return

extract_data(overlap = 0.5)
train_val_test_split(create_df())