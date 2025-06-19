import pandas as pd
from sklearn.preprocessing import LabelEncoder #kategaorik verileri sayiya cevirir
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def train_model():
    df = pd.read_csv("coffe_preferences.csv")

    # Label Encoding yani oop
    le_mood = LabelEncoder() 
    le_tol = LabelEncoder()
    le_pref = LabelEncoder()
    #fit_transform ile sayiya cevirir-hem fit hem transform
    df['Mood'] = le_mood.fit_transform(df['Mood'])
    df['Caffeine_Tolerance'] = le_tol.fit_transform(df['Caffeine_Tolerance'])
    df['Coffee_Preference'] = le_pref.fit_transform(df['Coffee_Preference'])

    X = df[['Age', 'Sleep_Hours', 'Mood', 'Caffeine_Tolerance']]
    y = df['Coffee_Preference']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    return model, le_mood, le_tol, le_pref, df #df-encode edilmis(ISLENMIS) pandas dataframe

