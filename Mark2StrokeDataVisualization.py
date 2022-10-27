import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import seaborn as sns

# id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke

from sklearn.preprocessing import OrdinalEncoder
def main():
    cleanup_nums = {"gender": {"Other": 3, "Male": 1, "Female": 0},
                    "ever_married": {"Yes": 1, "No": 0},
                    "work_type": {"Never_worked": 4, "children": 3, "Private": 2, "Self-employed": 1, "Govt_job": 0},
                    "Residence_type": {"Urban": 1, "Rural": 0},
                    "smoking_status": {"smokes": 4, "formerly smoked": 3, "never smoked": 2, "Unknown": 1},
                    }

    data = pd.read_csv("stroke.csv", header=0).replace(cleanup_nums)
    data["bmi"] = data["bmi"].fillna(data["bmi"].mean())
    print(data.head)
    columns = data.columns[1:-1]
    print(data.dtypes)
    m = LogisticRegression(solver='liblinear', random_state=0)

    y = data["stroke"]

    """
    for col in columns:
    
        sns.boxplot(data=data, x="stroke", y=col)
        plt.show()
    
    
        sns.displot(data[col],kde=True, alpha=0.1, color= 'g')
        plt.show()
    
    
    """

    for col in columns:
        X = data[col].values.reshape(-1, 1)
        m.fit(X, y)
        X0 = m.intercept_
        K = m.coef_
        #plt.scatter(X, y)
        model = lambda x, k, x0: 1 / (1 + math.e ** (-k * (x - x0)))
        # Now plot the model, making sure to divide x by 100 to transform it

        if (col == "bmi" or col == "age" or col == "avg_glucose_level"):
            xRange = np.linspace(1, 100, 100)
        elif (col == "work_type" or col == "smoking_status"):
            xRange = np.linspace(1, 4, 100)
        else:
            xRange = np.linspace(0, 1, 100)

        #plt.scatter(xRange, model(xRange, K, X0))
        #plt.title(col)
        #plt.show()

    # gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke
    yVal = columns[1]
    zVal = columns[7]
    xVal = columns[8]
    ax = plt.axes(projection='3d')
    ax.set_xlabel(xVal)
    ax.set_ylabel(yVal)
    ax.set_zlabel(zVal)
    zdata = data[zVal].values.reshape(-1, 1)
    xdata = data[xVal].values.reshape(-1, 1)
    ydata = data[yVal].values.reshape(-1, 1)
    ax.scatter3D(xdata, ydata, zdata, c=y, cmap='Greens')
    plt.show()

main()