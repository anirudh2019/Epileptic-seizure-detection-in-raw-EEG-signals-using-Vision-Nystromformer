from tqdm.auto import tqdm
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from prettytable import PrettyTable

def binary_acc(y_pred_tag, y_test):
  correct_results_sum = (y_pred_tag == y_test).sum().float()
  acc = correct_results_sum/y_test.shape[0]
  return acc

def count_parameters(model, pr_table = True):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params    
    if pr_table:
        print(f"\nTotal Trainable Params: {total_params}")
        print(table)
    return total_params

def train_model(device, model, num_epochs, train_dl, val_dl, loss_fn, optimizer, conf_path):
    loss_list = []
    val_loss_list = []

    best_model_state_dict = None
    best_epoch_val_loss = 999999
    best_epoch_val_accuracy = 0
    best_epoch_val_specificity = 0
    best_epoch_val_sensitivity = 0
    best_epoch_val_hmean = 0
    best_val_conf_mat = None
    best_epoch = -1
    
    print("Training...")
    for epoch in tqdm(range(num_epochs)):
      model.train(True)
      epoch_loss = 0
      epoch_accuracy = 0
      
      for data, label, _ in train_dl:
          data = data.to(device)
          label = label.type(torch.LongTensor)
          label = label.to(device)
          output = model(data)
      
          optimizer.zero_grad()
          loss = loss_fn(output, label)
          loss.backward()
          optimizer.step()

          acc = binary_acc(torch.argmax(torch.softmax(output, dim = 1), dim = 1), label)
          epoch_accuracy += acc / len(train_dl)
          epoch_loss += loss.detach() / len(train_dl)

      torch.cuda.empty_cache()
      
      model.eval()
      val_conf_mat = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
      epoch_val_accuracy = 0
      epoch_val_loss = 0

      for data, label, _ in val_dl:
          data = data.to(device)
          label = label.type(torch.LongTensor)
          label = label.to(device)
                
          with torch.no_grad():
              val_output = model(data)
              
          val_loss = loss_fn(val_output, label)
          tn,fp,fn,tp = confusion_matrix(label.to('cpu').detach().numpy(), torch.argmax(torch.softmax(val_output, dim = 1), dim = 1).to('cpu').detach().numpy(), labels=[0,1]).ravel()
          val_conf_mat["tn"]+=tn
          val_conf_mat["fp"]+=fp
          val_conf_mat["fn"]+=fn
          val_conf_mat["tp"]+=tp
      
          acc = binary_acc(torch.argmax(torch.softmax(val_output, dim = 1), dim = 1), label)
          epoch_val_accuracy += acc / len(val_dl)
          epoch_val_loss += val_loss.detach() / len(val_dl)

      epoch_val_specificity = val_conf_mat["tn"]/(val_conf_mat["tn"]+val_conf_mat["fp"])
      epoch_val_sensitivity = val_conf_mat["tp"]/(val_conf_mat["tp"]+val_conf_mat["fn"])
      epoch_val_hmean = (2*epoch_val_sensitivity*epoch_val_specificity)/(epoch_val_sensitivity+epoch_val_specificity)
      torch.cuda.empty_cache()
      

      # Track best performance, and save the model's state
      if ((round(epoch_val_hmean, 5)*100 > best_epoch_val_hmean) or (round(epoch_val_hmean, 5)*100 == best_epoch_val_hmean and epoch_val_loss < best_epoch_val_loss)):
            best_epoch_val_accuracy = round(epoch_val_accuracy.item(), 5)*100
            best_epoch_val_loss = epoch_val_loss.item()
            best_epoch_val_specificity = round(epoch_val_specificity, 5)*100
            best_epoch_val_sensitivity = round(epoch_val_sensitivity, 5)*100
            best_epoch_val_hmean = round(epoch_val_hmean, 5)*100
            best_val_conf_mat = val_conf_mat
            best_model_state_dict = model.state_dict()
            best_epoch = epoch + 1

      loss_list.append(epoch_loss)
      val_loss_list.append(epoch_val_loss)

    print(f"***VALIDATION ANALYSIS***")
    print(f"Best epoch: {best_epoch}")
    print(f"Accuracy: {best_epoch_val_accuracy}")
    print(f"Sensitivity: {best_epoch_val_sensitivity} - Specificity: {best_epoch_val_specificity}")
    print(f"Harmonic mean of sensitivivty and specificity: {best_epoch_val_hmean}")

    loss_list = list(map(lambda x: x.to('cpu').detach().item(), loss_list))
    val_loss_list = list(map(lambda x: x.to('cpu').detach().item(), val_loss_list))

    plt.plot(list(range(num_epochs)), loss_list, label = "train")
    plt.plot(list(range(num_epochs)), val_loss_list, label = "val")
    plt.title("Learning curve")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(conf_path+'Learning curve.png', bbox_inches='tight')
    plt.show()

    _, ax = plt.subplots()
    sn.set(font_scale=1.4)
    sn.heatmap([[best_val_conf_mat["tn"],best_val_conf_mat["fp"]],[best_val_conf_mat["fn"],best_val_conf_mat["tp"]]], annot=True, annot_kws={"size": 20}, cmap="YlGnBu", ax= ax) # font size
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(conf_path+'val_confmat.png', bbox_inches='tight')
    plt.show()

    return best_epoch_val_accuracy, best_epoch_val_sensitivity, best_epoch_val_specificity, best_epoch_val_hmean, best_model_state_dict, best_epoch
