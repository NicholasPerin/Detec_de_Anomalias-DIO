import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
)

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

df["Amount_log"] = np.log1p(df["Amount"])  # Comprime valores de "Amount" para log

scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(
    df[["Amount"]]
)  # Valores passam a ter a mesma escala

x = df.drop("Class", axis=1)
y = df["Class"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)  # Treina o modelo em 1000 iterações
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_probs = model.predict_proba(x_test)[:, 1]


def classification_report_results():
    print(classification_report(y_test, y_pred))
    return


def roc_curve_plot():
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    print("AUC: ", roc_auc_score(y_test, y_probs))
    return


def precision_recall_plot():
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
    return


def undersampling():
    fraudes = df[df["Class"] == 1]
    normais = df[df["Class"] == 0].sample(len(fraudes), random_state=42)
    df_under = pd.concat([fraudes, normais])
    return df_under


def oversampling():
    smote = SMOTE()
    x_res, y_res = smote.fit_resample(x, y)
    return smote


def rf_pred():
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
    )
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)
    print(classification_report(y_test, y_pred_rf))


def pipeline():
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    threshold = 0.3
    y_pred_custom = (y_probs > threshold).astype(int)
    print(classification_report(y_test, y_pred_custom))
    return y_pred


xgb = XGBClassifier(scale_pos_weight=10, use_label_encoder=False, eval_metric="logloss")


def xgb_pred():
    xgb.fit(x_train, y_train)
    y_pred_xgb = xgb.predict(x_test)
    print(classification_report(y_test, y_pred_xgb))
    return


def importancias():
    importancias = xgb.feature_importances_
    plt.bar(range(len(importancias)), importancias)
    plt.title("Importância das variáveis")
    plt.show()
    return


def melhor_modelo():
    param_grid = {"max_depth": [3, 5], "n_estimators": [50, 100]}
    grid = GridSearchCV(
        XGBClassifier(eval_metric="logloss"), param_grid, scoring="recall", cv=3
    )
    grid.fit(x_train, y_train)
    print("Melhor modelo:", grid.best_params_)
    return


def shap():
    explainer = shap.Explainer(xgb)
    shap_values = explainer(x_test[:100])
    shap.plots.bar(shap_values)
    return
