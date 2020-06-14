from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from tkinter import *
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import os
import numpy as np
import pandas as pd
import warnings

root = Tk()
warnings.filterwarnings('ignore')


def browse_dir_path():
    chosen_dir = filedialog.askdirectory()
    dir_path_box.insert(0, chosen_dir)


def is_valid_path(sp):
    if not (os.path.isfile(sp)):
        return False
    return True


def is_number(n):
    if n == "0":
        return False
    else:
        for i in range(len(n)):
            if not n[i].isdigit():
                return False
        return True


def is_not_empty_file(f, _type):
    if _type == "xlsx":
        try:
            pd.ExcelFile(f)
        finally:
            return False
    return True


def is_build_allowed():
    dp = dir_path.get()
    bn = clusters_num.get()
    structure_path = dp + "/Dataset.xlsx"
    error_msg = ""
    if bn and dp:
        if not is_valid_path(structure_path):
            error_msg += "ERROR: Wrong dir or missing files.\n"
        else:
            if not is_not_empty_file(structure_path, "xlsx"):
                error_msg += "ERROR: Some files are empty.\n"
        if not is_number(bn):
            error_msg += "ERROR: Illegal number of clusters k.\n"
        if error_msg == "":
            build.config(state='normal')
        else:
            messagebox.showerror(root.title(), error_msg + "\nPLEASE TRY AGAIN...")
            build.config(state='disabled')
    else:
        build.config(state='disabled')

    cluster.config(state='disabled')


def fill_missing_values(df):
    global attribute_dic

    for key in attribute_dic.keys():
        if attribute_dic[key] == ['NUMERIC']:
            df[key].fillna(df[key].mean(), inplace=True)
        else:  # categorical values
            df[key].fillna(df[key].mode()[0], inplace=True)
    return df


def normalize_numeric_values(df):
    global attribute_dic

    for key in attribute_dic.keys():
        if attribute_dic[key] == ['NUMERIC']:
            df[key] = (df[key] - df[key].min()) / (df[key].max() - df[key].min())
    return df


def preprocess():
    df = pd.read_excel(dir_path_box.get() + "/Dataset.xlsx")  # Loading train file

    df = df.drop(['country'], axis=1)

    # fill missing values
    df = fill_missing_values(df)

    # Normalize numeric values values
    df = normalize_numeric_values(df)

    print("Loading the Data frame and building the model COMPLETED.")
    print(" *** Number of clusters k = " + str(int(runs_box.get())))
    messagebox.showinfo(root.title(), "Preprocessing completed successfully!")


def draw_scatter(df):
    x = df["Social support"]
    y = df["Generosity"]
    plt.scatter(x, y, c=df["Cluster"], cmap='viridis')
    plt.xlabel("social_support")
    plt.ylabel("Generosity")
    plt.title("K Means Clustering")
    plt.show()


def draw_map(df):
    pass


def run_model():
    parent = tkinter.Tk()  
    parent.overrideredirect(1) 
    parent.withdraw()  
    try:
        num_of_clusters = int(cluster_box.get())
        num_of_runs = int(runs_box.get())
        df_no_country = df.drop(['country'], axis=1)
        labels = KMeans(n_clusters=num_of_clusters, n_init=num_of_runs, random_state=4).fit_predict(df_no_country)
        df["Cluster"] = labels
        draw_scatter(df)
        draw_map(df)
    except Exception as e:
        print(e)


# Following dir path and bins number:
dir_path = StringVar(root)
clusters_num = StringVar(root)
runs_num = StringVar(root)
dir_path.trace("w", is_build_allowed)
runs_num.trace("w", is_build_allowed)
clusters_num.trace("w", is_build_allowed)

# GUI Structure:
# -------------
# 1
root.title("K Means Clustering")

# 2
dir_path_label = Label(root)
dir_path_label["text"] = "Directory Path"
dir_path_label.grid(row=0, column=0, sticky='e', padx=(20, 10), pady=(20, 5))

# 3
dir_path_box = Entry(root, textvariable=dir_path)
dir_path_box["width"] = 50
dir_path_box.grid(row=0, column=1, sticky='w', padx=(0, 10), pady=(20, 5))

# 4
browse = Button(root)
browse["text"] = "Browse"
browse["command"] = browse_dir_path
browse.grid(row=0, column=2, padx=(10, 20), pady=(20, 5))

# 5
cluster_label = Label(root)
cluster_label["text"] = "Number of clusters k"
cluster_label.grid(row=1, column=0, sticky='e', padx=(20, 10), pady=(5, 10))

# 6
cluster_box = Entry(root, textvariable=clusters_num)
cluster_box.grid(row=1, column=1, sticky='w', pady=(5, 10))

# 7
runs_label = Label(root)
runs_label["text"] = "Number of Runs"
runs_label.grid(row=2, column=0, sticky='e', padx=(20, 10), pady=(5, 10))

# 8
runs_box = Entry(root, textvariable=runs_num)
runs_box.grid(row=2, column=1, sticky='w', pady=(5, 10))

# 9
build = Button(root)
build["text"] = "Pre-process"
build["command"] = preprocess
build["width"] = 20
build.grid(row=2, column=2, pady=(0, 10))
build.config(state='disabled')

# 10
cluster = Button(root)
cluster["text"] = "Cluster"
cluster["command"] = run_model
cluster["width"] = 20
cluster.grid(row=3, column=2, pady=(0, 20))
cluster.config(state='disabled')

# Global Variables
attribute_dic = {}  # attribute_dic = mapping between each attribute and its possible values
bins = {}  # bins = mapping between each numeric attribute and its intervals that define its discretization
train = pd.DataFrame()  # train data-frame

# MAIN:
# ----
print("-----------------------")
print("Clustering Application")
print("-----------------------")
root.mainloop()
