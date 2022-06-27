import pandas as pd
import math
import statistics

# This function is to measure randomness, we ensured more randomness.
def runsTest(l, l_median):
	runs, n1, n2 = 0, 0, 0
	for i in range(len(l)):
		if (l[i] >= l_median and l[i-1] < l_median) or \
				(l[i] < l_median and l[i-1] >= l_median):
			runs += 1
		if(l[i]) >= l_median:
			n1 += 1
		else:
			n2 += 1
	runs_exp = ((2*n1*n2)/(n1+n2))+1
	stan_dev = math.sqrt((2*n1*n2*(2*n1*n2-n1-n2))/ \
					(((n1+n2)**2)*(n1+n2-1)))
	z = (runs-runs_exp)/stan_dev
	return z


def shuffle_df(df):
  Z_cr = 1
  Z = 999999
  while Z>Z_cr:
    df = df.sample(frac=1)
    l = list(df.index)
    l_median= statistics.median(l)
    Z = abs(runsTest(l, l_median))

  return df.reset_index(drop = True)


def Kfold_crossval(df, k = 6):
  df_ictal = shuffle_df(df[df["label"]==1])
  df_nonictal = shuffle_df(df[df["label"]==0])
  folds = []

  N1 = df_ictal.shape[0]
  x1 = N1//k
  i1 = 0
  N2 = df_nonictal.shape[0]
  x2 = N2//k
  i2 = 0

  for fold in range(k):
    train_df = pd.DataFrame(columns=["edf_dir","label"])
    test_df = pd.DataFrame(columns=["edf_dir","label"])
    
    if fold!=k-1:
      df_ictal_test = df_ictal.iloc[i1:i1+x1]
      df_ictal_train = df_ictal.drop(list(range(i1,i1+x1)), inplace = False)

      df_nonictal_test = df_nonictal.iloc[i2:i2+x2]
      df_nonictal_train = df_nonictal.drop(list(range(i2,i2+x2)), inplace = False)
    else:
      df_ictal_test = df_ictal.iloc[i1:]
      df_ictal_train = df_ictal.drop(list(range(i1,N1)), inplace = False)

      df_nonictal_test = df_nonictal.iloc[i2:]
      df_nonictal_train = df_nonictal.drop(list(range(i2,N2)), inplace = False)

    train_df = pd.concat([train_df, df_ictal_train, df_nonictal_train], axis = 0)
    test_df = pd.concat([test_df, df_ictal_test, df_nonictal_test], axis = 0)

    train_df = shuffle_df(train_df)
    test_df = shuffle_df(test_df)
    folds.append((train_df, test_df))

    i1+=x1
    i2+=x2
  
  return folds


def split_df(df, data_split = {"train": 0.75, "val": 0.25}):

  df_ictal = df[df["label"]==1]
  df_nonictal = df[df["label"]==0]

  df_ictal = shuffle_df(df_ictal)
  df_nonictal = shuffle_df(df_nonictal)
  
  train_df = pd.DataFrame(columns=["edf_dir","label"])
  val_df = pd.DataFrame(columns=["edf_dir","label"])
  
  #For ictal
  t11 = int(data_split["train"]*df_ictal.shape[0])
  train_df = pd.concat([train_df, df_ictal.iloc[0:t11,:]], axis = 0)
  val_df = pd.concat([val_df, df_ictal.iloc[t11:,:]], axis = 0)

  #For nonictal
  t01 = int(data_split["train"]*df_nonictal.shape[0])
  train_df = pd.concat([train_df, df_nonictal.iloc[0:t01,:]], axis = 0)
  val_df = pd.concat([val_df, df_nonictal.iloc[t01:,:]], axis = 0)
  
  train_df = shuffle_df(train_df)
  val_df = shuffle_df(val_df)

  return train_df, val_df