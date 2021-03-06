import os
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import warnings
from tkinter import *
from PIL import ImageTk, Image
from matplotlib import pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py
from sklearn import preprocessing
from sklearn.cluster import KMeans
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = Tk()
warnings.filterwarnings('ignore')


def browse_file_path():
    chosen_file = filedialog.askopenfile(parent=root, mode='rb', title='Choose a file')
    if chosen_file is not None:
        chosen_file.close()
        file_path.set(chosen_file.name)


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
            pd.read_excel(f)
        finally:
            return False
    return True


def is_build_allowed(*args):
    try:
        cn = cluster_box.get()
        structure_path = file_path.get()
        error_msg = ""
        if cn and structure_path:
            if not is_valid_path(structure_path):
                error_msg += "ERROR: Wrong dir or missing files.\n"
            else:
                if is_not_empty_file(structure_path, "xlsx"):
                    error_msg += "ERROR: Some files are empty.\n"
            if not is_number(cn):
                error_msg += "ERROR: Illegal number of clusters k.\n"
            if error_msg == "":
                build.config(state='normal')
            else:
                messagebox.showerror(root.title(), error_msg + "\nPLEASE TRY AGAIN...")
                build.config(state='disabled')
        else:
            build.config(state='disabled')
        cluster.config(state='disabled')
    except Exception as e:
        raise e


def fill_missing_values():
    global df
    df = df.fillna(df.mean())


def standardization():
    global df
    global df_no_country
    try:
        df_no_country = df.drop(['country'], axis=1)

        standard = preprocessing.StandardScaler()
        standard_df = standard.fit_transform(df_no_country)

        df_no_country = pd.DataFrame(standard_df, columns=df_no_country.columns)
        df = pd.concat([pd.DataFrame(df["country"]), df_no_country], axis=1)

        df = df.groupby(['country'], as_index=False).mean()
        df = df.drop(['year'], axis=1)
    except Exception as e:
        raise e


def preprocess():
    try:
        global df

        # Loading xlsx file
        df = pd.read_excel(file_path.get())

        # fill missing values
        fill_missing_values()

        # Normalize numeric values
        standardization()

        print("Loading the Data frame and building the model COMPLETED.")
        print(" *** Number of clusters k = " + str(int(cluster_box.get())))
        messagebox.showinfo(root.title(), "Preprocess completed successfully!")
        cluster.config(state='normal')
    except Exception as e:
        raise e


# noinspection PyUnresolvedReferences
def draw_scatter():
    global df
    figure = plt.Figure(figsize=(4, 4), dpi=100)
    ax = figure.add_subplot(111)
    ax.set_xlabel('social_support')
    ax.set_ylabel('Generosity')
    ax.set_title('Scatter K-Means Clustering ')
    ax.scatter(df["Social support"], df["Generosity"], c=df['Cluster'], cmap='viridis')
    scatter = FigureCanvasTkAgg(figure, root)
    scatter.get_tk_widget().grid(row=4, column=1)


def draw_map():
    global df
    py.sign_in('serfati', 'i7Q02m0PRgUQypNuHtbE')
    text_labels = df['country'].astype(str) + "<br>Cluster Number: " + df['Cluster'].astype(str)

    choropleth = go.Choropleth(z=df['Cluster'],
                               locations=df['country'],
                               locationmode='country names',
                               colorscale='Viridis',
                               marker_line_color='black',
                               marker_line_width=0.5,
                               text=text_labels)

    layout = go.Layout(title='K-Means Clustering Visualization',
                       title_x=0.5,
                       geo=dict(
                           showframe=False,
                           showcoastlines=False,
                           projection_type='equirectangular'
                       ))

    figure = go.Figure(data=[choropleth], layout=layout)
    # py.plot(figure)
    py.image.save_as(figure, filename='choromap.png')  # TODO remove comment before assign!
    img = Image.open("./choromap.png")
    img = img.resize((500, 400), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.grid(row=4, column=2)


def run_model():
    global df
    global df_no_country
    try:
        df_no_country = df.drop(['country'], axis=1)
        num_of_clusters = int(cluster_box.get())
        num_of_runs = int(runs_box.get())
        labels = KMeans(n_clusters=num_of_clusters, n_init=num_of_runs, random_state=4).fit_predict(df_no_country)
        df["Cluster"] = labels
        print()
        draw_scatter()
        draw_map()
        messagebox.showinfo(root.title(), "Clustering process completed successfully!")
    except Exception as e:
        raise e


def quit_program():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()


# Following dir path and bins number:
file_path = StringVar(root)
clusters_num = StringVar(root)
runs_num = StringVar(root)
file_path.trace("w", is_build_allowed)
runs_num.trace("w", is_build_allowed)
clusters_num.trace("w", is_build_allowed)

# GUI Structure:
# -------------
# 1
root.title("K Means Clustering")

# 2
file_path_label = Label(root)
file_path_label["text"] = "Directory Path"
file_path_label.grid(row=0, column=0, sticky='e', padx=(20, 10), pady=(20, 5))

# 3
file_path_box = Entry(root, textvariable=file_path)
file_path_box["width"] = 50
file_path_box.grid(row=0, column=1, sticky='w', padx=(0, 10), pady=(20, 5))

# 4
browse = Button(root)
browse["text"] = "Browse"
browse["command"] = browse_file_path
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

# 11
df = pd.DataFrame()
df_no_country = pd.DataFrame()

# MAIN:
# ----
print("-----------------------")
print("Clustering Application")
print("-----------------------")
root.protocol("WM_DELETE_WINDOW", quit_program)
root.mainloop()
