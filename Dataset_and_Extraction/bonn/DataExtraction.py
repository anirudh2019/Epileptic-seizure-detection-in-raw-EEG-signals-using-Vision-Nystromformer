import os, random
import pandas as pd, numpy as np
from tqdm.auto import tqdm
from datapipeline_utils import Kfold_crossval, split_df

seed = 1
random.seed(seed)
np.random.seed(seed)

sample_size = 256
old_data_dir = "./bonn_dataset/"
new_data_dir = f"./bonn_{sample_size}/"
num_folds = 5
data_split = {"train": 0.75, "val": 0.25}
overlap = 0.5

def extract_data(overlap = 0.5):
    for subset in tqdm(["A", "B", "C", "D", "E"]):
        if not os.path.exists(new_data_dir+subset+"/"):
            os.makedirs(new_data_dir+subset+"/")

        for sample in sorted(os.listdir(old_data_dir+subset+"/")):
            sample_data = pd.read_csv(old_data_dir+subset+"/"+sample, header=None, dtype=float)
            if subset!="E":
                for num, i in enumerate(range(0,4096,sample_size)):
                    sample_data.iloc[i:i+sample_size,:].to_csv(new_data_dir+subset+"/"+sample[:-4]+f"_{num+1}.csv", index=False, header = False)
            else: #Apply Overlapping Window for E (seizure samples)
                for num, i in enumerate(range(0,sample_data.shape[0],int(sample_size*(1-overlap)))):
                    # BOUNDARY CONDITION
                    if i+sample_size-1>=sample_data.shape[0]:
                        sample_data.iloc[sample_data.shape[0]-sample_size : sample_data.shape[0],:].to_csv(new_data_dir+subset+"/"+sample[:-4]+f"_{num+1}.csv", index=False, header = False)
                    else:
                        sample_data.iloc[i:i+sample_size,:].to_csv(new_data_dir+subset+"/"+sample[:-4]+f"_{num+1}.csv", index=False, header = False)
    return

def get_df_for_case(case):
    case_df = pd.DataFrame(columns=["edf_dir","label"])    
    for subset in case.split("_")[0]:
        hmap = {"edf_dir":[]}
        hmap["edf_dir"].extend(list(map(lambda x: subset+"/"+x, sorted(os.listdir(new_data_dir+subset)))))
        subset_df = pd.DataFrame.from_dict(hmap, orient="columns")
        subset_df.insert(1,"label", [0 for _ in range(subset_df.shape[0])])
        case_df = pd.concat([case_df, subset_df], axis = 0)
    
    hmap = {"edf_dir":[]}
    hmap["edf_dir"].extend(list(map(lambda x: "E/"+x, sorted(os.listdir(new_data_dir+"E")))))
    subset_df = pd.DataFrame.from_dict(hmap, orient="columns")
    subset_df.insert(1,"label", [1 for _ in range(subset_df.shape[0])])
    case_df = pd.concat([case_df, subset_df], axis = 0)
    if not os.path.exists(new_data_dir+"cases/"+case):
        os.makedirs(new_data_dir+"cases/"+case)  
    case_df.to_csv(new_data_dir+"cases/"+case+f"/{case}.csv", index = False)
    return case_df

extract_data(overlap = 0.5)
cases = ["A_E", "B_E", "C_E", "D_E", "ACD_E", "BCD_E", "ABCD_E"]
for case in cases:  
    case_df = get_df_for_case(case)
    folds = Kfold_crossval(case_df, num_folds)
    for fold, (train_df, test_df) in enumerate(folds):     
        fold_path = new_data_dir+"cases/"+case+"/"+f"fold-{fold+1}"+"/"
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)
        train_df, val_df = split_df(train_df, data_split)
        train_df.to_csv(fold_path+"train.csv", index=False)
        val_df.to_csv(fold_path+"val.csv", index=False)
        test_df.to_csv(fold_path+"test.csv", index=False)