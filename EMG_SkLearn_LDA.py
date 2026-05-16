# -*- coding: utf-8 -*-
import numpy as np
import os
import kagglehub
import pandas as pd
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Train = True # This trains so fast there is no reason to save the model for this practice.

# =============================================================================
# Download the Dataset
# =============================================================================
# Download latest version of the UCI EMG Signal for Gesture Recognition Dataset
path = kagglehub.dataset_download("sojanprajapati/emg-signal-for-gesture-recognition")

print("Path to dataset files:", path)

# Just dont add another csv to this file and we're good...
# Also the file is a quarter gig so make sure to delete after this project...
for fname in os.listdir(path):
    if fname.endswith(".csv"):
        fpath = os.path.join(path, fname)
        data = pd.read_csv(fpath)

# =============================================================================
# Remove the Dataset
# =============================================================================
# Removes the dataset file and upstreme directories. Do not run if you have other datasets from kaggle uploaded by sojanprajapati.
# if os.path.exists(path):
#     parent = os.path.join(path, *([os.pardir])*4)
#     os.remove(parent)
#     print(f"{parent} has been removed.")
# else:
#     print(f"The file {path} does not exist.")



# =============================================================================
# Separate the Data into Training, Validation, and Testing Sets
# =============================================================================
# Validate on 1-30 test 2 second gesture, final eval on subjects 31-36

# Determine stretches of data by class shift.
# Since each gesture is separated by class 0 this allows for determination of whether the gesture observed
# Is the individual's first, second, etc. time performing the gesture
df_1 = data.copy()
df_1["class_changed"] = df_1.groupby("label")["class"].transform(lambda x: x.ne(x.shift()))
df_1["segment_id"] = df_1.groupby("label")["class_changed"].cumsum()


# Remove resting state and gesture 7 as gesture 7 is missing from some tests
gesture_df = df_1[(df_1["class"] != 0) & (df_1["class"] != 7)].copy()

# Rank by segment_id in ascending order to split gesture performance into training vs validation data
gesture_df["series"] = gesture_df.groupby(["label", "class"])["segment_id"].transform(
    lambda x: pd.factorize(x, sort=True)[0] + 1)

# Filter to series 1-3 and subjects 1–30
train_df = gesture_df[(gesture_df["label"] < 31)].copy()

# Filter to subjects 31-36
test_df = gesture_df[(gesture_df["label"] >= 31)].copy()


# Extracts center window_size of the group. Please note the floor operator // may cause unexpected windows with odd numbers so they are precluded
def center_window(group, window_size):
    assert window_size % 2 == 0, "window_size must be even"
    n = len(group)
    mid = n // 2
    half = window_size // 2
    return group.iloc[mid - half : mid + half]

# Loop through classes and group by label and segment_id. Apply center_window to each group
# Stich data back together and put in a df storing the centered data for that gesture id concated together
# Ignore any weirdness here. I am being lazy and adapting existing code
def windowed_data(df, window_size):
    result = {}
    df = df.drop(["class_changed"], axis=1)
    for gesture_class in sorted(df["class"].unique()):
        gesture_df = df[df["class"] == gesture_class]
        trimmed = gesture_df.groupby(["label", "segment_id"], group_keys=False).apply(
            center_window, window_size=window_size, include_groups=False)
        result[gesture_class] = trimmed.reset_index(drop=True)
    return result

# List of channels for later use in pyTorch
channels = ["channel1", "channel2", "channel3", "channel4", "channel5", "channel6", "channel7", "channel8"]

# Creating gesture seperated and windowed datasets
train_windows = windowed_data(train_df, window_size=400)
test_windows  = windowed_data(test_df,  window_size=400)

# =============================================================================
# Data Sets for Flattened Data
# =============================================================================

# Labels
y_train = np.concatenate([[g] * (len(train_windows[g]) // 400) for g in range(1, 7)])
y_test  = np.concatenate([[g] * (len(test_windows[g])  // 400) for g in range(1, 7)])

# Flatten data
X_train = np.vstack([train_windows[g][channels].values.reshape(-1, 3200) for g in range(1, 7)])
X_test  = np.vstack([test_windows[g][channels].values.reshape(-1, 3200)  for g in range(1, 7)])

# =============================================================================
# LDA on Flattened Data
# =============================================================================

print("LDA, Flattened Data")

# Instantiate the classifier
lda = LinearDiscriminantAnalysis()

# Fit the model to the training set
lda.fit(X_train, y_train)

# If you wanted to check for non-gesture signals you can use a probability threshold like this
#probs = lda.predict_proba(X_test)
#max_prob = probs.max(axis=1)

# Testing the model
y_pred = lda.predict(X_test)

# This is where you would reject predictions outside of a certain confidence level
#y_pred[max_prob < 0.6] = -1 # -1 = rejected or unknown

print(f"Test accuracy: {accuracy_score(y_test, lda.predict(X_test)):.1%}")

# =============================================================================
# Confusion Matrix
# =============================================================================

cm = np.zeros((6, 6), dtype=int)

# If prediction is the same as the true value add one to that cell in the 6x6 confusion matrix
for true_g, pred_g in zip(y_test, y_pred):
    cm[true_g - 1][pred_g - 1] += 1
    
# Diag/Total of cm matrix gives accuracy percentage
accuracy = cm.trace()/cm.sum()

# Reframing to pandas df to simplify printing of cm matrix
col_names = ["Pred1",  "Pred2",  "Pred3",  "Pred4",  "Pred5",  "Pred6"]
row_names = ["True 1",  "True 2",  "True 3",  "True 4",  "True 5",  "True 6"]
df = pd.DataFrame(cm, index=row_names, columns=col_names)

print(f"Overall Accuracy: {accuracy:.1%}\n")
print("Confusion Matrix:")
print(df)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# =============================================================================
# Data Sets for Hand Crafted Features
# =============================================================================

def extract_features(block):
    rms = np.sqrt(np.mean(block ** 2, axis = 0))
    mav = np.mean(np.abs(block), axis = 0)
    zcr = np.sum(np.diff(np.sign(block)) != 0, axis = 0).astype(float)
    wl = np.sum(np.abs(np.diff(block, axis = 0)), axis = 0)
    
    return np.concatenate([rms, mav, zcr, wl])

def append_crafted_features(windows):
    temp = []
    for g in range (1,7):
        n_blocks = len(windows[g]) // 400
        for i in range(n_blocks):
            window = windows[g][channels].values[i*400 : (i+1)*400]  # (400, 8)
            features = extract_features(window)  # (32,)
            
            temp.append(features)
    
    return np.array(temp)

X_train = append_crafted_features(train_windows)
X_test = append_crafted_features(test_windows)

y_train = np.concatenate([[g] * (len(train_windows[g]) // 400) for g in range(1, 7)])
y_test = np.concatenate([[g] * (len(test_windows[g]) // 400) for g in range(1, 7)])

# =============================================================================
# LDA on Hand Crafted Features
# =============================================================================

print("LDA, Crafted Features")

# Instantiate the classifier with n_estimators = 10 (default)
hlda = LinearDiscriminantAnalysis()

# Fit the model to the training set
hlda.fit(X_train, y_train)

# If you wanted to check for non-gesture signals you can use a probability threshold like this
#probs = hlda.predict_proba(X_test)
#max_prob = probs.max(axis=1)

# Testing the model
y_pred = hlda.predict(X_test)

# This is where you would reject predictions outside of a certain confidence level
#y_pred[max_prob < 0.6] = -1 # -1 = rejected or unknown

print(f"Test accuracy: {accuracy_score(y_test, hlda.predict(X_test)):.1%}")

# =============================================================================
# Confusion Matrix
# =============================================================================

cm = np.zeros((6, 6), dtype=int)

# If prediction is the same as the true value add one to that cell in the 6x6 confusion matrix
for true_g, pred_g in zip(y_test, y_pred):
    cm[true_g - 1][pred_g - 1] += 1
    
# Diag/Total of cm matrix gives accuracy percentage
accuracy = cm.trace()/cm.sum()

# Reframing to pandas df to simplify printing of cm matrix
col_names = ["Pred1",  "Pred2",  "Pred3",  "Pred4",  "Pred5",  "Pred6"]
row_names = ["True 1",  "True 2",  "True 3",  "True 4",  "True 5",  "True 6"]
df = pd.DataFrame(cm, index=row_names, columns=col_names)

print(f"Overall Accuracy: {accuracy:.1%}\n")
print("Confusion Matrix:")
print(df)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))