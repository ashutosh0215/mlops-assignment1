from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from misc import load_data, split_features_targets, train_test_split_data, train_model, evaluate_mse

def main():
    # Load data and split
    df = load_data()
    X, y = split_features_targets(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Create a pipeline: StandardScaler + KernelRidge
    model = make_pipeline(
        StandardScaler(),
        KernelRidge(alpha=0.1, kernel="rbf", gamma=0.1)
    )

    # Train and evaluate
    model = train_model(model, X_train, y_train)
    mse = evaluate_mse(model, X_test, y_test)

    print(f"Average Test MSE: {mse:.4f}")

if __name__ == "__main__":
    main()
