from sklearn.tree import DecisionTreeRegressor
from misc import load_data, split_features_targets, train_test_split_data, train_model, evaluate_mse

def main():
    df = load_data()

    X, y = split_features_targets(df)

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    model = DecisionTreeRegressor(random_state=42)
    model = train_model(model, X_train, y_train)

    mse = evaluate_mse(model, X_test, y_test)
    print(f"Average Test MSE (DecisionTreeRegressor): {mse:.4f}")

if __name__ == "__main__":
    main()
