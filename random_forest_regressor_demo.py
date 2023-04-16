import mlflow
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# connect to mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlflow_tracking_examples")

# this is the magical stuff

mlflow.autolog(log_model_signatures=True, log_input_examples=True)

with mlflow.start_run(run_name="autolog_with_named_run") as run:
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)

    run_id = run.info.run_id

    mlflow.sklearn.log_model(
        sk_model=rf,
        artifact_path="sklearn-model",
        registered_model_name="my_registered_model_1"  # the magic is here
    )

# get model path from run id
# (run_id can also be retrieved using the API or the UI)
# model_path = f"runs:/{run_id}/model"
# print(f"Loading model from: {model_path}")

# load using sklearn flavor
# loaded_model = mlflow.sklearn.load_model(model_path)

model_name = "my_registered_model_1"
model_version = 1
model_path = f"models:/{model_name}/{model_version}"

loaded_model = mlflow.pyfunc.load_model(model_path)

print("Showing predictions")
print(loaded_model.predict(X_test))
