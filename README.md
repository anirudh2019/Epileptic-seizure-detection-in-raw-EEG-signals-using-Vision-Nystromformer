# Epileptic seizure detection in raw EEG signals using Vision Nystromformer

<h2>Introduction</h2>

Epilepsy is a chronic brain disease characterized by persistent susceptibility to cause recurrent seizures. Electroencephalography (EEG) is a neuroimaging technique measuring the electrophysiological activity of the cerebral cortex. EEG has been commonly used to diagnose and treat patients with epilepsy.

<h3>Main Contributions of this study</h3>

- Using Transformer model directly with <b>raw EEG signals</b> without any removal of noise and artifacts.
- Using Transformer with <b>very few parameters</b> and as <b>small size</b> as possible.
- Implementing <b>linear Time and Space complexity</b> of Attention mechanism in Transformer using Nystrom Attention mechanism.

<h3>Why Transformer instead of CNN and RNN?</h3>
First, due to high temporal resolution, EEG signals are usually extremely long sequences. The sequence models, e.g., RNNs and LSTMs, process the EEG signals sequentially, namely, they train the data at each time step one by one, which largely increases the training time for convergence. In addition, although some deep learning frameworks can capture temporal dependencies, such as RNN-based models for long-term dependencies and CNN-based models for neighboring interactions, they can only achieve limited performance when the sequences are extremely long.

<h2>Methodology</h2>

<h3>Model</h3>

![Alt text](https://github.com/anirudh2019/Epileptic-seizure-detection-in-raw-EEG-signals-using-Vision-Nystromformer/blob/master/Model.png "a title")

<h3>Datasets</h3>

Three datasets are used: CHB-MIT, Bonn and IIT-Delhi EEG datasets.

<h3>Training</h3>

- <b>Number of Trainable parameters</b> for CHB-MIT, Bonn and IIT-Delhi datasets are <b>162894, 7250, 2058 only</b> respectively which is <b>very less for a transformer model</b>.
- Before training, for reliable results, I performed 6-fold cross validation in which one fold is taken as test dataset and remaining folds are taken as train dataset. I further took 25% of train dataset as validation dataset and remaining 75% is taken as train dataset.
- I then trained the model for 75 epochs using Adam optimizer, batch size of 32 and cross entropy loss function.
- Hyperparameters for each dataset is as follows:

|Dataset|sequence length|Embedding dimension|learning rate|signal input size|patch size|depth|# attention heads|embedding dimension scale|Feedforward multiplier|Number of landmarks|
|---|---|---|---|---|---|---|---|---|---|---|
|CHB-MIT|168|64|0.005|(21,256)|(1,32)|3|4|2|4|32|
|Bonn|32|16|0.005|(1,256)|(1,8)|3|4|2|2|8|
|IIT-Delhi|32|8|0.001|(1,128)|(1,4)|3|4|2|2|8|

<h2>Results:</h2>

- Four metrics are considered: Accuracy, Sensitivity, Specificity and harmonic mean of sensitivity and specificity.

<h3>IIT-Delhi dataset</h3>

|val_accuracy|val_hmean|val_sensitivity|val_specificity|test_accuracy|test_hmean|test_sensitivity|test_specificity|
|------------|---------|---------------|---------------|-------------|----------|----------------|----------------|
|98.52       |98.6     |98.21          |99.01          |96.77        |96.95     |96.02           |97.98           |

<h3>Bonn dataset</h3>
  
|subject|val_accuracy|val_hmean|val_sensitivity|val_specificity|test_accuracy|test_hmean|test_sensitivity|test_specificity|
|-------|------------|---------|---------------|---------------|-------------|----------|----------------|----------------|
|A_E    |99.97       |99.97    |100            |99.93          |99.63        |99.62     |99.56           |99.69           |
|B_E    |99.44       |99.44    |99.41          |99.48          |99           |98.99     |98.56           |99.44           |
|C_E    |99.2        |99.2     |99.38          |99.03          |98.47        |98.46     |98.31           |98.62           |
|D_E    |98.06       |98.05    |97.81          |98.3           |97.53        |97.51     |97              |98.06           |
|ACD_E  |98.32       |98.5     |98.89          |98.12          |98.16        |98.18     |98.25           |98.12           |
|BCD_E  |98.16       |98.18    |98.23          |98.14          |97.62        |97        |95.88           |98.21           |
|ABCD_E |98.27       |98.49    |98.85          |98.12          |98.11        |97.51     |96.56           |98.5            |

<h3>CHB-MIT dataset</h3>

|subject|val_accuracy|val_hmean|val_sensitivity|val_specificity|test_accuracy|test_hmean|test_sensitivity|test_specificity|
|-------|------------|---------|---------------|---------------|-------------|----------|----------------|----------------|
|chb01  |98.56       |98.85    |99.05          |98.64          |97.33        |97.3      |97.17           |97.46           |
|chb02  |100         |100      |100            |100            |100          |100       |100             |100             |
|chb03  |98.61       |98.6     |98.76          |98.46          |97.76        |97.73     |98.07           |97.45           |
|chb04  |98.08       |98.09    |98.73          |97.46          |96.58        |96.54     |97.69           |95.5            |
|chb05  |98.82       |98.78    |99.25          |98.32          |96.98        |96.97     |96.68           |97.36           |
|chb06  |94.73       |94.72    |95.18          |94.27          |91.52        |91.84     |92.81           |91.01           |
|chb07  |99.48       |99.48    |99.57          |99.38          |98.51        |98.49     |98.31           |98.69           |
|chb08  |97.02       |97.01    |97.02          |97.02          |94.75        |94.39     |97.39           |92              |
|chb09  |99.86       |99.85    |100            |99.71          |99.74        |99.73     |99.91           |99.55           |
|chb10  |98.78       |98.75    |98.61          |98.88          |97.99        |97.98     |97.6            |98.38           |
|chb11  |99.07       |99.07    |99.06          |99.08          |98.18        |98.16     |98.11           |98.26           |
|chb12  |95.99       |95.97    |95.45          |96.5           |95.71        |95.62     |95.17           |96.08           |
|chb13  |97.91       |97.91    |97.68          |98.14          |96.15        |96.08     |97.56           |94.71           |
|chb14  |94.31       |94.29    |93.37          |95.26          |91.66        |91.38     |94.66           |88.45           |
|chb15  |98.65       |98.65    |98.39          |98.91          |98.36        |98.35     |98.07           |98.64           |
|chb16  |97.37       |97.24    |96.55          |97.99          |96.75        |96.68     |96.01           |97.46           |
|chb17  |98.33       |98.33    |98.84          |97.82          |97.75        |97.68     |98.29           |97.1            |
|chb18  |99.17       |99.15    |99.06          |99.24          |98.81        |98.73     |98.58           |98.89           |
|chb19  |97.44       |97.28    |98.05          |96.53          |94.56        |94.27     |97.67           |91.52           |
|chb20  |97.43       |97.29    |98.44          |96.19          |95.63        |95.95     |97.7            |94.3            |
|chb21  |95.58       |95.74    |95.07          |96.48          |93.63        |93.56     |95.6            |91.81           |
|chb22  |99.57       |99.56    |99.71          |99.41          |99.36        |99.32     |99.63           |99.02           |
|chb23  |98.91       |98.87    |99.11          |98.63          |97.82        |97.89     |98              |97.82           |
|chb24  |98.61       |98.58    |98             |99.18          |97.25        |97.33     |98.34           |96.39           |
|total  |98.01166667 |98.0025  |98.03958333    |97.97916667    |96.7825      |96.74875  |97.45916667     |96.16041667     |
