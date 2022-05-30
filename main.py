from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
sns.set()


def KMeansClusterPlot(df, firstColumn, secondColumn, nrClusters, title=""):

    # remove rows containing NaN values
    df.dropna(inplace=True)

    # scale values
    scaler = MinMaxScaler()

    scaler.fit(df[[firstColumn]])
    df[firstColumn] = scaler.transform(df[[firstColumn]])

    scaler.fit(df[[secondColumn]])
    df[secondColumn] = scaler.transform(df[[secondColumn]])

    # KMeans
    km = KMeans(n_clusters=nrClusters)
    predicted = km.fit_predict(df[[firstColumn, secondColumn]])
    df['cluster'] = predicted

    df1 = df[df.cluster == 0]
    df2 = df[df.cluster == 1]
    df3 = df[df.cluster == 2]
    df4 = df[df.cluster == 3]
    df5 = df[df.cluster == 4]

    plt.scatter(df1[firstColumn], df1[secondColumn], color='red')
    plt.scatter(df2[firstColumn], df2[secondColumn], color='blue')
    plt.scatter(df3[firstColumn], df3[secondColumn], color='yellow')
    plt.scatter(df4[firstColumn], df4[secondColumn], color='green')
    plt.scatter(df5[firstColumn], df5[secondColumn], color='orange')

    plt.xlabel(firstColumn)
    plt.ylabel(secondColumn)
    plt.title(title)
    plt.show()



def AgglomerativClusterPlot(df, firstColumn, secondColumn, nrClusters, title=""):

    df.dropna(inplace=True)

    cluster = AgglomerativeClustering(n_clusters=nrClusters, affinity='euclidean', linkage='ward')
    cl = cluster.fit_predict(df)
    df['cluster'] = cl

    df1 = df[df.cluster == 0]
    df2 = df[df.cluster == 1]
    df3 = df[df.cluster == 2]
    df4 = df[df.cluster == 3]
    df5 = df[df.cluster == 4]

    plt.scatter(df1[firstColumn], df1[secondColumn], color='red')
    plt.scatter(df2[firstColumn], df2[secondColumn], color='blue')
    plt.scatter(df3[firstColumn], df3[secondColumn], color='yellow')
    plt.scatter(df4[firstColumn], df4[secondColumn], color='green')
    plt.scatter(df5[firstColumn], df5[secondColumn], color='orange')

    plt.xlabel(firstColumn)
    plt.ylabel(secondColumn)
    plt.title(title)
    plt.show()


def BirchPlot(df, firstColumn, secondColumn, nr_clusters, title=""):

    df.dropna(inplace=True)

    model = Birch(branching_factor=50, n_clusters=nr_clusters, threshold=1)
    model.fit(df)

    pred = model.predict(df)
    plt.scatter(df[:, 0], df[:, 1], c=pred)
    plt.xlabel(firstColumn)
    plt.ylabel(secondColumn)
    plt.title(title)
    plt.plot()


def DbscanPlot(df, firstColumn, secondColumn, eps, title=""):
    df.dropna(inplace=True)

    x = df.loc[:500, [firstColumn, secondColumn]].values

    dbscan = DBSCAN(eps=eps, min_samples=4).fit(x)
    labels = dbscan.labels_

    plt.scatter(x[:, 0], x[:, 1], c=labels, cmap="plasma")
    plt.xlabel(firstColumn)
    plt.ylabel(secondColumn)
    plt.title(title)
    plt.show()


def MeanShiftPlot(df, firstColumn, secondColumn, title=""):
    df.dropna(inplace=True)

    x = df.loc[:500, [firstColumn, secondColumn]].values

    clustering = MeanShift(bandwidth=2).fit(x)
    labels = clustering.labels_

    plt.scatter(x[:, 0], x[:, 1], c=labels, cmap="plasma")
    plt.xlabel(firstColumn)
    plt.ylabel(secondColumn)
    plt.title(title)
    plt.show()


def GaussianMixturePlot(df, firstColumn, secondColumn, nr=4, title=""):

    df.dropna(inplace=True)

    x = df.loc[:500, [firstColumn, secondColumn]].values

    gmm = GaussianMixture(n_components=nr)
    gmm.fit(x)

    labels = gmm.predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis')
    plt.xlabel(firstColumn)
    plt.ylabel(secondColumn)
    plt.title(title)
    plt.show()




# read from csvs
dataframe_dishes = pd.read_csv('Dish.csv')
dataframe_menu = pd.read_csv('Menu.csv')
dataframe_menuitem = pd.read_csv('MenuItem.csv')
dataframe_menupage = pd.read_csv('MenuPage.csv')


# MENU : page_count and dish_count
df_menu_page_dish = dataframe_menu.iloc[:, 18:20]
KMeansClusterPlot(df_menu_page_dish, 'page_count', 'dish_count', 4)

# DISHES : menus_appeared and lowest_price
df_dishes_menus_price = dataframe_dishes.iloc[:, 3:8:4]
KMeansClusterPlot(df_dishes_menus_price, 'menus_appeared', 'lowest_price', 2)

# DISHES : menus_appeared and times_appeared
df_dishes_menus_times = dataframe_dishes.iloc[:, 3:5]
KMeansClusterPlot(df_dishes_menus_times, 'menus_appeared', 'times_appeared', 2)

# MENU PAGE : full_height and full_width : KMeans
df_menupage_page_height = dataframe_menupage.iloc[:2000, 4:6]
KMeansClusterPlot(df_menupage_page_height, 'full_height', 'full_width', 3, 'KMeans')

# MENU ITEM : price and dish_id
df_menuitem_price_dish_id = dataframe_menuitem.iloc[:, 2:5:2]
KMeansClusterPlot(df_menuitem_price_dish_id, 'price', 'dish_id', 2)

############################################

# MENU PAGE : page_number and full_width
df_menupage_page_width = dataframe_menupage.iloc[:, 2:6:3]
KMeansClusterPlot(df_menupage_page_width, 'page_number', 'full_width', 2)

# MENU PAGE : page_number and full_height
df_menupage_page_height = dataframe_menupage.iloc[:, 2:5:2]
KMeansClusterPlot(df_menupage_page_height, 'page_number', 'full_height', 2)

# MENU ITEM : xpos and ypos (nu are nici cea mai mica treaba)
df_menuitem_xpos_ypos = dataframe_menuitem.iloc[:, 7:9]
KMeansClusterPlot(df_menuitem_xpos_ypos, 'xpos', 'ypos', 2)



# MENU PAGE : full_height and full_width : Agglomerative
df_menupage_page_height = dataframe_menupage.iloc[:500, 4:6]
AgglomerativClusterPlot(df_menupage_page_height, 'full_height', 'full_width', 4, 'Agglomerative')

# MENU PAGE : full_height and full_width : KMeans
df_menupage_page_height = dataframe_menupage.iloc[:500, 4:6]
KMeansClusterPlot(df_menupage_page_height, 'full_height', 'full_width', 3, 'KMeans')

# MENU PAGE : full_height and full_width : Birch
df_menupage_page_height = dataframe_menupage.iloc[:500, 4:6]
KMeansClusterPlot(df_menupage_page_height, 'full_height', 'full_width', 3, 'Birch')

# MENU PAGE : full_height and full_width : DBSCAN
df_menupage_page_height = dataframe_menupage.iloc[:500, 4:6]
DbscanPlot(df_menupage_page_height, 'full_height', 'full_width', 200, 'DBSCAN')

# MENU PAGE : full_height and full_width : MeanShift
df_menupage_page_height = dataframe_menupage.iloc[:500, 4:6]
MeanShiftPlot(df_menupage_page_height, 'full_height', 'full_width', 'MeanShift')


df_menupage_page_height = dataframe_menupage.iloc[:500, 4:6]
GaussianMixturePlot(df_menupage_page_height, 'full_height', 'full_width', 3, 'Gaussian')


##############################################################################################


# read csv
df = pd.read_csv('cars.csv')


# scale values
scaler = MinMaxScaler()

scaler.fit(df[['Cylinders']])
df['Cylinders'] = scaler.transform(df[['Cylinders']])

scaler.fit(df[['Horsepower']])
df['Horsepower'] = scaler.transform(df[['Horsepower']])

df = df[['Cylinders', 'Horsepower']]

AgglomerativClusterPlot(df, 'Cylinders', 'Horsepower', 3, 'Agglomerative')
KMeansClusterPlot(df, 'Cylinders', 'Horsepower', 3, 'KMeans')
DbscanPlot(df, 'Cylinders', 'Horsepower', 0.15, 'Dbscan')  # Destul de bun
MeanShiftPlot(df, 'Cylinders', 'Horsepower', 'MeanShift')
GaussianMixturePlot(df, 'Cylinders', 'Horsepower', 3, 'Gaussian')  # Destul de bun





df = pd.read_csv('cars.csv')

# scale values
scaler = MinMaxScaler()

scaler.fit(df[['Weight']])
df['Weight'] = scaler.transform(df[['Weight']])

scaler.fit(df[['Acceleration']])
df['Acceleration'] = scaler.transform(df[['Acceleration']])

df = df[['Weight', 'Acceleration']]

KMeansClusterPlot(df, 'Weight', 'Acceleration', 3, 'KMeans')
AgglomerativClusterPlot(df, 'Weight', 'Acceleration', 3, 'Agglomerative')
DbscanPlot(df, 'Weight', 'Acceleration', 0.1, 'Dbscan')
MeanShiftPlot(df, 'Weight', 'Acceleration', 'MeanShift')
GaussianMixturePlot(df, 'Weight', 'Acceleration', 3, 'Gaussian')




df = pd.read_csv('cars.csv')

# scale values
scaler = MinMaxScaler()

scaler.fit(df[['Weight']])
df['Weight'] = scaler.transform(df[['Weight']])

scaler.fit(df[['Acceleration']])
df['Acceleration'] = scaler.transform(df[['Acceleration']])

df = df[['Weight', 'Acceleration']]

KMeansClusterPlot(df, 'Weight', 'Acceleration', 3, 'KMeans')
AgglomerativClusterPlot(df, 'Weight', 'Acceleration', 3, 'Agglomerative')
DbscanPlot(df, 'Weight', 'Acceleration', 0.1, 'Dbscan')
MeanShiftPlot(df, 'Weight', 'Acceleration', 'MeanShift')
GaussianMixturePlot(df, 'Weight', 'Acceleration', 3, 'Gaussian')



df = pd.read_csv('cars.csv')

# scale values
scaler = MinMaxScaler()

scaler.fit(df[['MPG']])
df['MPG'] = scaler.transform(df[['MPG']])

scaler.fit(df[['Acceleration']])
df['Acceleration'] = scaler.transform(df[['Acceleration']])

df = df[['MPG', 'Acceleration']]

KMeansClusterPlot(df, 'MPG', 'Acceleration', 3, 'KMeans')
AgglomerativClusterPlot(df, 'MPG', 'Acceleration', 3, 'Agglomerative')
DbscanPlot(df, 'MPG', 'Acceleration', 0.1, 'Dbscan')  # Arata bine
MeanShiftPlot(df, 'MPG', 'Acceleration', 'MeanShift')
GaussianMixturePlot(df, 'MPG', 'Acceleration', 3, 'Gaussian')




df = pd.read_csv('cars.csv')

# scale values
scaler = MinMaxScaler()

scaler.fit(df[['MPG']])
df['MPG'] = scaler.transform(df[['MPG']])

scaler.fit(df[['Horsepower']])
df['Horsepower'] = scaler.transform(df[['Horsepower']])

df = df[['MPG', 'Horsepower']]

KMeansClusterPlot(df, 'MPG', 'Horsepower', 3, 'KMeans')
AgglomerativClusterPlot(df, 'MPG', 'Horsepower', 3, 'Agglomerative')
DbscanPlot(df, 'MPG', 'Horsepower', 0.1, 'Dbscan')  # Arata bine
MeanShiftPlot(df, 'MPG', 'Horsepower', 'MeanShift')
GaussianMixturePlot(df, 'MPG', 'Horsepower', 3, 'Gaussian')


df = pd.read_csv('cars.csv')

# scale values
scaler = MinMaxScaler()

scaler.fit(df[['Displacement']])
df['Displacement'] = scaler.transform(df[['Displacement']])

scaler.fit(df[['Horsepower']])
df['Horsepower'] = scaler.transform(df[['Horsepower']])

df = df[['Displacement', 'Horsepower']]
KMeansClusterPlot(df, 'Displacement', 'Horsepower', 4, 'KMeans')
AgglomerativClusterPlot(df, 'Displacement', 'Horsepower', 4, 'Agglomerative')
DbscanPlot(df, 'Displacement', 'Horsepower', 0.1, 'Dbscan')
MeanShiftPlot(df, 'Displacement', 'Horsepower', 'MeanShift')
GaussianMixturePlot(df, 'Displacement', 'Horsepower', 4, 'Gaussian')

