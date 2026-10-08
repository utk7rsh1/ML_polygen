import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score

def evaluate_degrees():
    train_v1 = pd.read_csv('data/BT2024119_train_var1.csv')
    X1 = train_v1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
    y1 = train_v1['y'].values
    
    train_v2 = pd.read_csv('data/BT2024119_train_var2.csv')
    X2 = train_v2[['x1', 'x2', 'x3']].values
    y2 = train_v2['y'].values
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    print("\nPhase 1: Turbine Power Score (var1)")
    print(f"{'Degree':<8}{'Features':<10}{'OLS MSE':<14}{'Ridge MSE':<14}{'Lasso MSE':<14}{'Lasso R2':<12}")
    
    for deg in range(1, 11):
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X1_poly = poly.fit_transform(X1)
        scaler = StandardScaler()
        X1_scaled = scaler.fit_transform(X1_poly)
        n_features = X1_poly.shape[1]
        
        ols_str = "N/A"
        if deg <= 6:
            ols_mse = -cross_val_score(LinearRegression(), X1_poly, y1, scoring='neg_mean_squared_error', cv=kf).mean()
            ols_str = f"{ols_mse:.4f}"
            
        ridge = Ridge(alpha=10.0, random_state=42)
        ridge_mse = -cross_val_score(ridge, X1_scaled, y1, scoring='neg_mean_squared_error', cv=kf).mean()
        
        lasso = Lasso(alpha=0.008, max_iter=25000, random_state=42)
        lasso_mse = -cross_val_score(lasso, X1_scaled, y1, scoring='neg_mean_squared_error', cv=kf).mean()
        lasso_r2 = cross_val_score(lasso, X1_scaled, y1, scoring='r2', cv=kf).mean()
        
        print(f"{deg:<8}{n_features:<10}{ols_str:<14}{ridge_mse:<14.4f}{lasso_mse:<14.4f}{lasso_r2:<12.4f}")
        
    print("\nPhase 2: Subterranean Thermal Mapping (var2)")
    print(f"{'Degree':<8}{'Features':<10}{'OLS MSE':<14}{'Ridge MSE':<14}{'Ridge R2':<14}")
    
    for deg in range(1, 21):
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X2_poly = poly.fit_transform(X2)
        scaler = StandardScaler()
        X2_scaled = scaler.fit_transform(X2_poly)
        n_features = X2_poly.shape[1]
        
        ols_str = "N/A"
        if deg <= 12:
            ols_mse = -cross_val_score(LinearRegression(), X2_poly, y2, scoring='neg_mean_squared_error', cv=kf).mean()
            ols_str = f"{ols_mse:.4f}"
            
        ridge = Ridge(alpha=2.0, random_state=42)
        ridge_mse = -cross_val_score(ridge, X2_scaled, y2, scoring='neg_mean_squared_error', cv=kf).mean()
        ridge_r2 = cross_val_score(ridge, X2_scaled, y2, scoring='r2', cv=kf).mean()
        
        print(f"{deg:<8}{n_features:<10}{ols_str:<14}{ridge_mse:<14.4f}{ridge_r2:<14.4f}")

if __name__ == '__main__':
    evaluate_degrees()
