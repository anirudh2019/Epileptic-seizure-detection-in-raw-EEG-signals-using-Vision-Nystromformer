import os, re, gzip, numpy as np, random
import pandas as pd
from tqdm.auto import tqdm
from pyedflib import highlevel
from datapipeline_utils import split_df, Kfold_crossval

seed = 1
random.seed(seed)
np.random.seed(seed)

old_data_dir = "./seizure_edfs/"
summary_dir = "./summary_txts/"

subjects = ["chb01","chb02","chb03","chb04","chb05","chb06","chb07","chb08","chb09","chb10","chb11","chb12","chb13","chb14","chb15","chb16","chb17","chb18","chb19","chb20","chb21","chb22","chb23","chb24"]
standard_channels = ['FP1-F7', 'F7-T7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FZ-CZ', 'CZ-PZ', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8','T8-P8', 'P8-O2', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8']                                                                                                        


def get_edf_info(edf):
    subject = edf[:5]
    subjectno = int(subject[-2:])
    edfno = edf[5:]

    f = open(summary_dir + subject + "-summary.txt", 'r')
    file_contents = f.read()
    file_list = file_contents.split('\n')

    # Getting edfinfo
    ind = 0
    num_seizures = 0

    if subjectno != 24:
        ind = file_list.index('File Name: ' + edf+".edf")
        num_seizures = list(map(int, re.findall(r'\d+', file_list[ind + 3])))[0]
    else:
        if edfno in [
                "_01", "_03", "_04", "_06", "_07", "_09", "_11", "_13", "_14",
                "_15", "_17", "_21"
        ]:
            ind = file_list.index('File Name: ' + edf+".edf")
            num_seizures = list(map(int, re.findall(r'\d+',
                                                    file_list[ind + 1])))[0]
        else:
            num_seizures = 0

    edfinfo = {"edf": edf, "num_seizures": num_seizures, "start": [], "end": [], "sez_dur": [], "tot_sez_dur": 0}

    if subjectno!=24:
        for i in range(edfinfo["num_seizures"]):
            edfinfo["start"].append(list(map(int, re.findall(r'\d+', file_list[ind + 2 * i + 4].split(":")[1])))[0])
            edfinfo["end"].append(list(map(int, re.findall(r'\d+', file_list[ind + 2 * i + 5].split(":")[1])))[0])
    else:
        if edfno in ["_01","_03","_04","_06","_07","_09","_11","_13","_14","_15","_17","_21"]:
            for i in range(edfinfo["num_seizures"]):
                edfinfo["start"].append(list(map(int, re.findall(r'\d+', file_list[ind + 2 * i + 2].split(":")[1])))[0])
                edfinfo["end"].append(list(map(int, re.findall(r'\d+', file_list[ind + 2 * i + 3].split(":")[1])))[0])

    for i in range(edfinfo["num_seizures"]):
        edfinfo["sez_dur"].append(edfinfo["end"][i] - edfinfo["start"][i])
        edfinfo["tot_sez_dur"] += edfinfo["end"][i] - edfinfo["start"][i]

    return edfinfo

def extract_seizures(signals, edfinfo, seizure_dir, dur, overlap, per_subj_sez_count):
    edf = edfinfo["edf"]
    subject = edf[:5]
    ### Extracting seizure segments from this edf file
    for i in range(edfinfo["num_seizures"]):
        seizure_seg = signals[:, edfinfo["start"][i]*256 : edfinfo["end"][i]*256]
        for j in range(0,seizure_seg.shape[1], int(dur*(1-overlap)*256)):          
            f = gzip.GzipFile(seizure_dir + f"{edf}_{i+1}_{j+1}"+ ".npy.gz", "w")
            if j+dur*256-1>=seizure_seg.shape[1]:
                np.save(file=f, arr= seizure_seg[:, seizure_seg.shape[1] - dur*256: seizure_seg.shape[1]])
            else:
                np.save(file=f, arr= seizure_seg[:, j: j + dur*256])
            f.close()
            per_subj_sez_count[subject] = per_subj_sez_count.get(subject,0) + 1    
    return

def extract_nonseizures(signals, edfinfo, nonseizure_dir, dur, per_subj_sez_count):
    edf = edfinfo["edf"]
    subject = edf[:5]        
    seizure_indices = []
    for i in range(edfinfo["num_seizures"]):
        seizure_indices.extend(list(range(edfinfo["start"][i]*256, edfinfo["end"][i]*256)))  
    seizure_indices = set(seizure_indices)

    ### Extracting non-seizure segments from this edf file
    if per_subj_sez_count[subject]!=0:        
        for i in range(0, signals.shape[1], dur*256):
            if i in seizure_indices or i+dur*256-1 in seizure_indices:
                continue
            nonseizure_seg = signals[:, i:i+dur*256]
            if nonseizure_seg.shape[1]<dur*256: # This happens at end of the edf file
                break 
            f = gzip.GzipFile(nonseizure_dir + f"{edf}_{i+1}"+ ".npy.gz", "w")
            np.save(file=f, arr= nonseizure_seg)
            f.close()
            per_subj_sez_count[subject] -= 1
            if per_subj_sez_count[subject]==0:
                break
    return

per_subj_sez_count = {}
def makedataset(dur, overlap):
    new_data_dir = f"./CHBMIT_{dur}s_{overlap}OW/"
    if not os.path.exists(new_data_dir):
      os.makedirs(new_data_dir)
    print(f"Duration = {dur}, Overlap = {overlap}")
    print(">> Extracting seizure samples from edf files using sliding window")
    for edf in tqdm(sorted(os.listdir(old_data_dir))):
        if edf[-4:]!=".edf":
            continue
        edf_dir = old_data_dir+edf
        edf = edf[:-4]
        subject = edf[:5]
        seizure_dir = new_data_dir+subject+"/seizure/"
        nonseizure_dir = new_data_dir+subject+"/nonseizure/"
        if not os.path.exists(new_data_dir+subject):
            os.makedirs(seizure_dir)
            os.makedirs(nonseizure_dir)
        signals, _, _ = highlevel.read_edf(edf_file = edf_dir, ch_names = standard_channels)
        edfinfo = get_edf_info(edf)
        extract_seizures(signals, edfinfo, seizure_dir, dur, overlap, per_subj_sez_count)
    
    print("\n>> Extracting non-seizure samples from edf files using sliding window")
    for edf in tqdm(sorted(os.listdir(old_data_dir))):
        if edf[-4:]!=".edf":
            continue
        edf_dir = old_data_dir+edf
        edf = edf[:-4]
        subject = edf[:5]
        nonseizure_dir = new_data_dir+subject+"/nonseizure/"
        if per_subj_sez_count[subject]==0:        
            continue
        signals, _, _ = highlevel.read_edf(edf_file = edf_dir, ch_names = standard_channels)
        edfinfo = get_edf_info(edf)
        extract_nonseizures(signals, edfinfo, nonseizure_dir, dur, per_subj_sez_count)
    return new_data_dir

def create_csvs(new_data_dir):
    print("\n>> Creating csv files")
    for subject in tqdm(sorted(os.listdir(new_data_dir))):
        if subject[-4:]==".csv":
          continue
        dct = {"edf_dir": [], "label": []}
        for edf in sorted(os.listdir(new_data_dir + subject + "/seizure/")):
            dct["edf_dir"].append(subject + "/seizure/" + edf)
            dct["label"].append(1)
        for edf in sorted(os.listdir(new_data_dir + subject + "/nonseizure/")):
            dct["edf_dir"].append(subject + "/nonseizure/" + edf)
            dct["label"].append(0)
        df = pd.DataFrame.from_dict(dct, orient="columns")
        df.to_csv(new_data_dir + subject + ".csv", index=False)
    return

def train_val_test(new_data_dir, num_folds = 6, data_split = {"train": 0.75, "val": 0.25}):
    print("\n>> Applying 6-fold Cross Validation then splitting Train into Train and Val(75%,25%)")
    for subject in subjects:
        df = pd.read_csv(new_data_dir+subject+".csv")
        folds = Kfold_crossval(df, num_folds)
        for fold,(train_df, test_df) in enumerate(folds):     
            fold_path = new_data_dir+subject+"/"+f"fold-{fold+1}"+"/"
            if not os.path.exists(fold_path):
                os.makedirs(fold_path)  
            train_df, val_df = split_df(train_df, data_split)
            train_df.to_csv(fold_path+"train.csv", index=False)
            val_df.to_csv(fold_path+"val.csv", index=False)
            test_df.to_csv(fold_path+"test.csv", index=False)
    return

new_data_dir = makedataset(dur=1, overlap=0.75)
create_csvs(new_data_dir)
train_val_test(new_data_dir, num_folds = 6, data_split = {"train": 0.75, "val": 0.25})
print("\nFINISHED!!!")