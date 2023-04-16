import pandas as pd
import mlflow
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# create an MLflow-compliant model by extending PythonModel
class TextAnalyzerModel(mlflow.pyfunc.PythonModel):

    def __init__(self):
        super().__init__()
        self._analyser = SentimentIntensityAnalyzer()

    def _preprocess(self):
        pass

    def _score(self, txt):
        prediction_scores = self._analyser.polarity_scores(txt)
        return prediction_scores

    def predict(self, context, model_input):
        model_output = model_input.apply(lambda col: self._score(col))
        return model_output


# connect to mlflow and set experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("sentiment_analysis")

# enable autolog
mlflow.autolog(log_model_signatures=True, log_input_examples=True)

model_artifact_path = "vader_model"
model = TextAnalyzerModel()

# execute run
with mlflow.start_run(run_name="Vader Sentiment Analysis") as run:
    mlflow.log_param("algorithm", "VADER")
    mlflow.pyfunc.log_model(artifact_path=model_artifact_path,
                            python_model=model)
    run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/vader_model"

    loaded_model = mlflow.pyfunc.load_model(model_uri)

    queries = [
        "This is a bad movie. You don't want to see it! :-)",
        "Ricky Gervais is smart, witty, and creative!!!!!! :D",
        "LOL, this guy fell off a chair while sleeping and snoring in a meeting",
        "Men shoots himself while trying to steal a dog, OMG",
        "Yay!! Another good phone interview. I nailed it!!",
        "This is INSANE! I can't believe it. How could you do such a horrible thing?"
    ]

    for q in queries:
        m_input = pd.DataFrame([q])
        scores = loaded_model.predict(m_input)
        print(f"<{q}> -- {str(scores[0])}")
