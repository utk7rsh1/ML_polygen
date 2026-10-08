import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso

def generate_predictions():
    train_v1 = pd.read_csv('data/BT2024119_train_var1.csv')
    test_v1 = pd.read_csv('data/BT2024119_test_var1.csv')
    
    train_v2 = pd.read_csv('data/BT2024119_train_var2.csv')
    test_v2 = pd.read_csv('data/BT2024119_test_var2.csv')
    
    X1_train = train_v1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
    y1_train = train_v1['y'].values
    X1_test = test_v1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
    
    X2_train = train_v2[['x1', 'x2', 'x3']].values
    y2_train = train_v2['y'].values
    X2_test = test_v2[['x1', 'x2', 'x3']].values
    
    poly1 = PolynomialFeatures(degree=5, include_bias=False)
    scaler1 = StandardScaler()
    X1_poly_train = scaler1.fit_transform(poly1.fit_transform(X1_train))
    X1_poly_test = scaler1.transform(poly1.transform(X1_test))
    
    model1 = Lasso(alpha=0.00796, max_iter=25000, random_state=42)
    model1.fit(X1_poly_train, y1_train)
    y1_pred = model1.predict(X1_poly_test)
    
    poly2 = PolynomialFeatures(degree=11, include_bias=False)
    scaler2 = StandardScaler()
    X2_poly_train = scaler2.fit_transform(poly2.fit_transform(X2_train))
    X2_poly_test = scaler2.transform(poly2.transform(X2_test))
    
    model2 = Ridge(alpha=2.0, random_state=42)
    model2.fit(X2_poly_train, y2_train)
    y2_pred = model2.predict(X2_poly_test)
    
    pred_v1_df = pd.DataFrame({'y': y1_pred})
    pred_v2_df = pd.DataFrame({'y': y2_pred})
    
    assert len(pred_v1_df) == 1000
    assert len(pred_v2_df) == 1000
    assert list(pred_v1_df.columns) == ['y']
    assert list(pred_v2_df.columns) == ['y']
    assert pred_v1_df['y'].isnull().sum() == 0
    assert pred_v2_df['y'].isnull().sum() == 0
    
    pred_v1_df.to_csv('predictions/BT2024119_pred_var1.csv', index=False)
    pred_v2_df.to_csv('predictions/BT2024119_pred_var2.csv', index=False)

if __name__ == '__main__':
    generate_predictions()
