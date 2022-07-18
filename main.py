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
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
import sklearn.metrics as metrics

from data_pipeline import EdfDataset
from trainer import train_model, binary_acc, count_parameters
from VIT import ViT

##############  Some Variables
IST = pytz.timezone('Asia/Kolkata')
datetime_ist = datetime.now(IST)
timestamp = datetime_ist.strftime('%Y%m%d_%H%M%S')
print("Timestamp: ", timestamp)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

######################* Trial Setup
attention_type = "nystrom"
attn_mode = "s+t"
dataset_name = "IIT_Delhi_256" #"CHBMIT_1s_0.75OW" # "bonn_256" "IIT_Delhi_256"
dataset_dir = "./"+dataset_name+"/"
save_dir = f"./results/{dataset_name}/"
seed = 1
loss_fn = nn.CrossEntropyLoss()
num_epochs = 75
batch_size = 32

if dataset_name == "CHBMIT_1s_0.75OW":
    subjects = ["chb01","chb02","chb03","chb04","chb05","chb06","chb07","chb08","chb09","chb10","chb11","chb12","chb13","chb14","chb15","chb16","chb17","chb18","chb19","chb20","chb21","chb22","chb23","chb24"]
    num_folds = 5
    lr = 5e-3
    img_size = (21,256)
    attn_values_residual = True
    attn_values_residual_conv_kernel = 33
    attn_dropout = 0.2
    ff_dropout = 0.3
    num_heads = 4
    depth = 3
    if attn_mode == "s+t":
        patch_size = (1,32)
    elif attn_mode == "s":
        patch_size = (1,256)
    elif attn_mode == "t":
        patch_size = (21,1)
    embed_dim_scale = 2
    ff_mult = 4
    num_landmarks = 32

elif dataset_name == "bonn_256":
    subjects = ["A_E", "B_E", "C_E", "D_E", "ACD_E", "BCD_E", "ABCD_E"]
    num_folds = 5
    lr = 5e-3
    img_size = (1,256)
    attn_values_residual = True
    attn_values_residual_conv_kernel = 33
    attn_dropout = 0.2
    ff_dropout = 0.3
    num_heads = 4
    depth = 3
    patch_size = (1,32)
    embed_dim_scale = 2
    ff_mult = 4
    num_landmarks = 8

else:
    subjects = ["IIT_Delhi_256"]
    num_folds = 5
    lr = 5e-3
    img_size = (1,256)
    attn_values_residual = True
    attn_values_residual_conv_kernel = 33
    attn_dropout = 0.2
    ff_dropout = 0.3
    num_heads = 4
    depth = 3
    patch_size = (1,32)
    embed_dim_scale = 2
    ff_mult = 4
    num_landmarks = 8

##############  MAIN FUNCTION
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
run_path = save_dir+timestamp+"/"
subj_results = {"val_accuracy": [], "val_hmean": [], "val_sensitivity": [], "val_specificity": [], "val_auroc": [], "test_accuracy": [], "test_hmean": [], "test_sensitivity": [], "test_specificity": [], "test_auroc": [], "best_epoch": []}

for subject_name in subjects:
    val_fpr_tot = []
    val_tpr_tot = []
    test_fpr_tot = []
    test_tpr_tot = []
    subj_path = run_path+subject_name+"/"
    seq_len = (img_size[0]//patch_size[0]) * (img_size[1]//patch_size[1])
    embed_dim = int((patch_size[0] * patch_size[1]) * embed_dim_scale)

    fold_results = {"val_accuracy": [], "val_hmean": [], "val_sensitivity": [], "val_specificity": [], "val_auroc": [], "test_accuracy": [], "test_hmean": [], "test_sensitivity": [], "test_specificity": [], "test_auroc": [], "best_epoch": []}
    for fold in range(num_folds):
        fold_path = subj_path+f"/fold-{fold+1}/"
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)  

        if dataset_name != "bonn_256":
            train_df = pd.read_csv(dataset_dir+subject_name+f"/fold-{fold+1}/train.csv")
            val_df = pd.read_csv(dataset_dir+subject_name+f"/fold-{fold+1}/val.csv")
            test_df = pd.read_csv(dataset_dir+subject_name+f"/fold-{fold+1}/test.csv")
        else:
            train_df = pd.read_csv(dataset_dir+"cases/"+subject_name+f"/fold-{fold+1}/train.csv")
            val_df = pd.read_csv(dataset_dir+"cases/"+subject_name+f"/fold-{fold+1}/val.csv")
            test_df = pd.read_csv(dataset_dir+"cases/"+subject_name+f"/fold-{fold+1}/test.csv")
        
        if dataset_name != "IIT_Delhi_256":
            train_dataset = EdfDataset(train_df, dataset_dir)
            val_dataset = EdfDataset(val_df, dataset_dir)
            test_dataset = EdfDataset(test_df, dataset_dir)
        else:
            train_dataset = EdfDataset(train_df, dataset_dir+"IIT_Delhi_256/")
            val_dataset = EdfDataset(val_df, dataset_dir+"IIT_Delhi_256/")
            test_dataset = EdfDataset(test_df, dataset_dir+"IIT_Delhi_256/")

        train_dl = None
        count_samples = np.array([train_df[train_df["label"]==0].shape[0], train_df[train_df["label"]==1].shape[0]])
        if count_samples[0]!=count_samples[1]:
            print("Dataset is imbalanced, so using WeightedRandomSampler")
            num_samples = int(count_samples.max()*2)
            class_weight = num_samples/count_samples
            samples_weight = np.array([class_weight[t] for t in train_df["label"]])
            samples_weight=torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(samples_weight, num_samples)
            train_dl = DataLoader(train_dataset, batch_size = batch_size, sampler = sampler)
        else:
            train_dl = DataLoader(train_dataset, batch_size = batch_size, shuffle= True)
        val_dl = DataLoader(val_dataset, batch_size = batch_size, shuffle= False)
        test_dl = DataLoader(test_dataset, batch_size = batch_size, shuffle= False)
        
        model = ViT(
                    dim = embed_dim,
                    image_size = img_size,
                    patch_size = patch_size,
                    depth = depth,
                    num_heads = num_heads,
                    num_landmarks = num_landmarks,
                    attn_values_residual = attn_values_residual,
                    attn_values_residual_conv_kernel = attn_values_residual_conv_kernel,
                    attn_dropout = attn_dropout,
                    ff_dropout = ff_dropout,
                    ff_mult = ff_mult,
                    attention = attention_type
                )
        
        num_params = count_parameters(model, False)
        print(f"\n*****{subject_name} {num_folds}*****")
        print(f"{patch_size}, embed_dim_scale={embed_dim_scale}, ff_mult={ff_mult}, depth={depth}, lr={lr}, num_landmarks={num_landmarks}, batch_size={batch_size}", ": ")            
        print(f"fold-{fold+1}")
        print("Sequence Length: ", seq_len)
        print("Embed size: ", embed_dim)
        print("Number of parameters: ", num_params, "\n")

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr= lr)

        # TRAIN THE MODEL
        val_accuracy, val_sensitivity, val_specificity, val_hmean, val_auroc, val_fpr, val_tpr, best_model_state_dict, best_epoch = train_model(device, model, num_epochs, train_dl, val_dl, loss_fn, optimizer, fold_path)

        fold_results["val_accuracy"].append(val_accuracy)
        fold_results["val_hmean"].append(val_hmean)
        fold_results["val_sensitivity"].append(val_sensitivity)
        fold_results["val_specificity"].append(val_specificity)
        fold_results["val_auroc"].append(val_auroc)
        fold_results["best_epoch"].append(best_epoch)
        val_fpr_tot.append(val_fpr)
        val_tpr_tot.append(val_tpr)

        # TEST THE MODEL
        print("\nTesting...")
        model.load_state_dict(best_model_state_dict)
        torch.save(best_model_state_dict, fold_path+f"fold_{fold+1}_saved_model.pt")
        model.train(False)
        test_conf_mat = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
        test_accuracy = 0

        tot_labels = []
        tot_preds = []
        for data, label, _ in tqdm(test_dl):
            data = data.to(device)
            label = label.type(torch.LongTensor)
            label = label.to(device)    
            tot_labels.extend(label.tolist())
            with torch.no_grad():
                test_output = model(data)
            probs = torch.softmax(test_output, dim=1)
            preds = probs[:, 1].detach()
            tot_preds.extend(preds.tolist())

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
        fpr, tpr, threshold = metrics.roc_curve(tot_labels, tot_preds)
        test_auroc = metrics.auc(fpr, tpr)
        test_fpr_tot.append(fpr)
        test_tpr_tot.append(tpr)
        torch.cuda.empty_cache()             

        fold_results["test_accuracy"].append(round(test_accuracy.item(),5)*100)
        fold_results["test_hmean"].append(round(test_hmean,5)*100)
        fold_results["test_sensitivity"].append(round(test_sensitivity,5)*100)
        fold_results["test_specificity"].append(round(test_specificity,5)*100)
        fold_results["test_auroc"].append(test_auroc)
    
        all_folds_df = pd.DataFrame.from_dict(fold_results, orient="columns")
        all_folds_df.to_csv(subj_path+'fold_results.csv', index=False)

        print(f"TEST ANALYSIS: ")
        print(f"Accuracy: {round(test_accuracy.item(),5)*100}, AUROC: {round(test_auroc,3)}")
        print(f"Sensitivity: {round(test_sensitivity, 5)*100} - Specificity: {round(test_specificity, 5)*100}")
        print(f"Harmonic mean of sensitivivty and specificity : {round(test_hmean,5)*100}")

        _, ax = plt.subplots()
        sn.set(font_scale=1.4)
        sn.heatmap([[test_conf_mat["tn"], test_conf_mat["fp"]],[test_conf_mat["fn"],test_conf_mat["tp"]]], annot=True, annot_kws={"size": 20}, cmap="YlGnBu", ax= ax, fmt='g') # font size
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.savefig(fold_path+'test_confmat.png', bbox_inches='tight')
        plt.show()

    subj_results["val_accuracy"].append(round(sum(fold_results["val_accuracy"])/num_folds, 2))
    subj_results["val_specificity"].append(round(sum(fold_results["val_specificity"])/num_folds, 2))
    subj_results["val_sensitivity"].append(round(sum(fold_results["val_sensitivity"])/num_folds, 2))
    subj_results["val_hmean"].append(round(sum(fold_results["val_hmean"])/num_folds, 2))
    subj_results["val_auroc"].append(round(sum(fold_results["val_auroc"])/num_folds, 3))
    subj_results["test_accuracy"].append(round(sum(fold_results["test_accuracy"])/num_folds, 2))
    subj_results["test_specificity"].append(round(sum(fold_results["test_specificity"])/num_folds, 2))
    subj_results["test_sensitivity"].append(round(sum(fold_results["test_sensitivity"])/num_folds, 2))
    subj_results["test_hmean"].append(round(sum(fold_results["test_hmean"])/num_folds, 2))
    subj_results["test_auroc"].append(round(sum(fold_results["test_auroc"])/num_folds, 3))
    subj_results["best_epoch"].append(tuple(fold_results["best_epoch"]))

    all_subjs_df = pd.DataFrame.from_dict(subj_results, orient="columns")
    all_subjs_df.to_csv(run_path+'subj_results.csv', index=False)

    plt.title('Receiver Operating Characteristic')
    for i in range(num_folds):
        plt.plot(val_fpr_tot[i], val_tpr_tot[i], label = f'ROC curve for fold-{i+1} (area = {round(fold_results["val_auroc"][i],3)})')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(subj_path+f"{subject_name}_val_roc.png")
    plt.clf()

    plt.title('Receiver Operating Characteristic')
    for i in range(num_folds):
        plt.plot(test_fpr_tot[i], test_tpr_tot[i], label = f'ROC curve for fold-{i+1} (area = {round(fold_results["test_auroc"][i],3)})')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(subj_path+f"{subject_name}_test_roc.png")
    plt.clf()