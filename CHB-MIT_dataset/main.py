##############  IMPORT MODULES
import os, random, itertools
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_pipeline import EdfDataset
from trainer import train_model, binary_acc, count_parameters
from VIT import ViT
from nystrom_attention import Nystromformer

##############  Some Variables
IST = pytz.timezone('Asia/Kolkata')
datetime_ist = datetime.now(IST)
timestamp = datetime_ist.strftime('%Y%m%d_%H%M%S')
print("Timestamp: ", timestamp)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)
##############  HYPERPARAMETERS
seed = 1
loss_fn = nn.CrossEntropyLoss()
num_epochs = 75
# Optimizer: Adam
num_classes = 2
attn_values_residual = True
pos_emb_scaling = True
attn_dropout = 0.2
ff_dropout = 0.3
hp_ranges = { "lr": [5e-3], 
            "batch_size": [32], 
            "img_size": [(21,256)],
            "patch_size": [(1,32)], 
            "depth": [3], 
            "num_heads": [4],
            "embed_dim_scale": [2], 
            "ff_mult": [4], 
            "num_landmarks": [32]}

# subjects = ["chb01","chb02","chb03","chb04","chb05","chb06","chb07","chb08","chb09","chb10","chb11","chb12","chb13","chb14","chb15","chb16","chb17","chb18","chb19","chb20","chb21","chb22","chb23","chb24"]
subjects = [ "chb01", "chb02"]
model_name = "VisNysT"
dataset_name = "CHBMIT_1s_0.75OW"
num_folds = 6
dataset_dir = "./"+dataset_name+"/"
save_dir = f"./results/{dataset_name}/{model_name}/"
# import zipfile, os
# if not os.path.exists(f"./{dataset_name}"):
#     with zipfile.ZipFile(f"./{dataset_name}.zip", 'r') as zip_ref:
#         zip_ref.extractall(f"./{dataset_name}")
##############  MAIN FUNCTION
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
print("model name: ",model_name)
run_path = save_dir+timestamp+"/"
all_subjs_hp_configs_df_list = []

for subject_name in subjects:
    hp_configs = {"num_conf": [], "seq_len": [], "embed_dim": [], "num_params": [], "lr": [], "batch_size": [], "img_size": [], "patch_size": [], "depth": [], "num_heads": [], "embed_dim_scale": [], "ff_mult": [], "num_landmarks": [], "val_accuracy": [], "val_hmean": [], "val_sensitivity": [], "val_specificity": [], "test_accuracy": [], "test_hmean": [], "test_sensitivity": [], "test_specificity": [], "best_epoch": []}
    subj_path = run_path+subject_name+"/"
    num_configs = len(list(itertools.product(hp_ranges["lr"], hp_ranges["batch_size"], hp_ranges["img_size"], hp_ranges["patch_size"], hp_ranges["depth"], hp_ranges["num_heads"], hp_ranges["embed_dim_scale"], hp_ranges["ff_mult"], hp_ranges["num_landmarks"])))

    for num_conf, (lr, batch_size, img_size, patch_size, depth, num_heads, embed_dim_scale, ff_mult, num_landmarks) in enumerate(itertools.product(hp_ranges["lr"], hp_ranges["batch_size"], hp_ranges["img_size"], hp_ranges["patch_size"], hp_ranges["depth"], hp_ranges["num_heads"], hp_ranges["embed_dim_scale"], hp_ranges["ff_mult"], hp_ranges["num_landmarks"])):
        num_conf+=1
        conf_path = subj_path+f"{num_conf}"+"/"
        seq_len = (img_size[0]//patch_size[0]) * (img_size[1]//patch_size[1])
        embed_dim = int((patch_size[0] * patch_size[1]) * embed_dim_scale)
    
        fold_results = {"val_accuracy": [], "val_hmean": [], "val_sensitivity": [], "val_specificity": [], "test_accuracy": [], "test_hmean": [], "test_sensitivity": [], "test_specificity": [], "best_epoch": []}
        for fold in range(num_folds):
            fold_path = conf_path+f"/fold-{fold+1}/"
            if not os.path.exists(fold_path):
                os.makedirs(fold_path)  
            
            train_df = pd.read_csv(dataset_dir+subject_name+f"/fold-{fold+1}/train.csv")
            val_df = pd.read_csv(dataset_dir+subject_name+f"/fold-{fold+1}/val.csv")
            test_df = pd.read_csv(dataset_dir+subject_name+f"/fold-{fold+1}/test.csv")
            
            train_dataset = EdfDataset(train_df, dataset_dir)
            val_dataset = EdfDataset(val_df, dataset_dir)
            test_dataset = EdfDataset(test_df, dataset_dir)

            train_dl = DataLoader(train_dataset, batch_size = batch_size, shuffle= True)
            val_dl = DataLoader(val_dataset, batch_size = batch_size, shuffle= True)
            test_dl = DataLoader(test_dataset, batch_size = batch_size, shuffle= False)

            efficient_transformer = Nystromformer(
                                                    dim = embed_dim,
                                                    depth = depth,
                                                    dim_head = embed_dim//num_heads,
                                                    heads = num_heads,
                                                    num_landmarks = num_landmarks,
                                                    attn_values_residual = attn_values_residual,
                                                    attn_dropout = attn_dropout,
                                                    ff_dropout = ff_dropout,
                                                    ff_mult = ff_mult
                                                )

            model = ViT(
                        pos_emb_scaling = pos_emb_scaling,
                        dim = embed_dim,
                        image_size = img_size,
                        patch_size = patch_size, 
                        num_classes = num_classes,
                        transformer = efficient_transformer
                    )

            num_params = count_parameters(model, False)
            print(f"\n*****{subject_name} {num_configs} {num_folds}*****")
            print(f"===>>> {num_conf}: ", f"{patch_size}, embed_dim_scale={embed_dim_scale}, ff_mult={ff_mult}, depth={depth}, lr={lr}, num_landmarks={num_landmarks}, batch_size={batch_size}", ": ")            
            print(f"fold-{fold+1}")
            print("Sequence Length: ", seq_len)
            print("Embed size: ", embed_dim)
            print("Number of parameters: ", num_params, "\n")        

            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr= lr)

            # TRAIN THE MODEL
            val_accuracy, val_sensitivity, val_specificity, val_hmean, best_model_state_dict, best_epoch = train_model(device, model, num_epochs, train_dl, val_dl, loss_fn, optimizer, fold_path)
            
            fold_results["val_accuracy"].append(val_accuracy)
            fold_results["val_hmean"].append(val_hmean)
            fold_results["val_sensitivity"].append(val_sensitivity)
            fold_results["val_specificity"].append(val_specificity)
            fold_results["best_epoch"].append(best_epoch)

            # TEST THE MODEL
            print("\nTesting...")
            model.load_state_dict(best_model_state_dict)
            torch.save(best_model_state_dict, fold_path+f"fold_{fold+1}_saved_model.pt")
            model.train(False)
            test_conf_mat = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
            test_accuracy = 0

            for data, label, _ in tqdm(test_dl):
                data = data.to(device)
                label = label.type(torch.LongTensor)
                label = label.to(device)    
                with torch.no_grad():
                    test_output = model(data)
            
                tn,fp,fn,tp = confusion_matrix(label.to('cpu').detach().numpy(), torch.argmax(torch.softmax(test_output, dim = 1), dim = 1).to('cpu').detach().numpy(), labels=[0,1]).ravel()
                test_conf_mat["tn"]+=tn
                test_conf_mat["fp"]+=fp
                test_conf_mat["fn"]+=fn
                test_conf_mat["tp"]+=tp
            
                acc = binary_acc(torch.argmax(torch.softmax(test_output, dim = 1), dim = 1), label)
                test_accuracy += acc / len(test_dl)
            
            test_specificity = test_conf_mat["tn"]/(test_conf_mat["tn"]+test_conf_mat["fp"])
            test_sensitivity = test_conf_mat["tp"]/(test_conf_mat["tp"]+test_conf_mat["fn"])
            test_hmean = (2*test_sensitivity*test_specificity)/(test_sensitivity+test_specificity)
            torch.cuda.empty_cache()             

            fold_results["test_accuracy"].append(round(test_accuracy.item(),5)*100)
            fold_results["test_hmean"].append(round(test_hmean,5)*100)
            fold_results["test_sensitivity"].append(round(test_sensitivity,5)*100)
            fold_results["test_specificity"].append(round(test_specificity,5)*100)

            all_folds_df = pd.DataFrame.from_dict(fold_results, orient="columns")
            all_folds_df.to_csv(conf_path+'fold_results.csv', index=False)

            print(f"TEST ANALYSIS: ")
            print(f"Accuracy: {round(test_accuracy.item(),5)*100}")
            print(f"Sensitivity: {round(test_sensitivity, 5)*100} - Specificity: {round(test_specificity, 5)*100}")
            print(f"Harmonic mean of sensitivivty and specificity : {round(test_hmean,5)*100}")

            _, ax = plt.subplots()
            sn.set(font_scale=1.4)
            sn.heatmap([[test_conf_mat["tn"], test_conf_mat["fp"]],[test_conf_mat["fn"],test_conf_mat["tp"]]], annot=True, annot_kws={"size": 20}, cmap="YlGnBu", ax= ax) # font size
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.savefig(fold_path+'test_confmat.png', bbox_inches='tight')
            plt.show()

        hp_configs["num_conf"].append(num_conf)
        hp_configs["lr"].append(lr)
        hp_configs["batch_size"].append(batch_size)
        hp_configs["img_size"].append(img_size)
        hp_configs["patch_size"].append(patch_size)
        hp_configs["embed_dim_scale"].append(embed_dim_scale)
        hp_configs["ff_mult"].append(ff_mult)
        hp_configs["depth"].append(depth)
        hp_configs["num_heads"].append(num_heads)
        hp_configs["num_landmarks"].append(num_landmarks)
        hp_configs["embed_dim"].append(embed_dim)
        hp_configs["seq_len"].append(seq_len)
        hp_configs["num_params"].append(num_params)
        hp_configs["val_accuracy"].append(round(sum(fold_results["val_accuracy"])/num_folds, 2))
        hp_configs["val_specificity"].append(round(sum(fold_results["val_specificity"])/num_folds, 2))
        hp_configs["val_sensitivity"].append(round(sum(fold_results["val_sensitivity"])/num_folds, 2))
        hp_configs["val_hmean"].append(round(sum(fold_results["val_hmean"])/num_folds, 2))
        hp_configs["test_accuracy"].append(round(sum(fold_results["test_accuracy"])/num_folds, 2))
        hp_configs["test_specificity"].append(round(sum(fold_results["test_specificity"])/num_folds, 2))
        hp_configs["test_sensitivity"].append(round(sum(fold_results["test_sensitivity"])/num_folds, 2))
        hp_configs["test_hmean"].append(round(sum(fold_results["test_hmean"])/num_folds, 2))
        hp_configs["best_epoch"].append(tuple(fold_results["best_epoch"]))

        all_hp_configs_df = pd.DataFrame.from_dict(hp_configs, orient="columns")
        all_hp_configs_df.to_csv(subj_path+subject_name+'_results.csv', index=False)
    
    all_hp_configs_df.insert(0,"subject", [subject_name for _ in range(all_hp_configs_df.shape[0])])
    all_subjs_hp_configs_df_list.append(all_hp_configs_df)
    all_subjs_hp_configs_df = pd.concat(all_subjs_hp_configs_df_list, axis = 0)
    all_subjs_hp_configs_df.to_csv(run_path+'all_subjs_hps_results.csv', index=False)