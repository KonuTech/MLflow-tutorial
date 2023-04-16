# MLflow-tutorial
https://blog.devgenius.io/mlflow-an-extended-hello-world-99739b68bf29

Quotes:

The main features of MLflow include:

* Experiment tracking: allows you to track and compare different runs of your machine learning models, including parameters, metrics, and artifacts (such as model files or data) associated with each run.
* Experiments and model packaging standardisation: allows you to specify environments, dependencies and similar requirements in a standardised way, so that experiments can be reproduced always in the same environment and models can be used seamlessly across different environments.
* Model management: allows you to store and manage your models in a central repository, including versioning and tagging.
* Model deployment: allows you to deploy your models in various environments, such as a REST API or a Docker container.

We have a complete ModelOps environment, providing tracking, reproducibility, standardisation, lineage, centralisation and evolution. The cherry on top of the cake comes with the final capability: serving.


Staging/Production/Archived

That gives you much better management capabilities. First, you can clearly distinguish which version is in production (Production), which is being tested (Staging), which has been decommissioned (Archived) and which has just been generated (None). This way, your production deployment can simply reload the model in production stage, no code changes required. And doing that is a piece of cake:


To serve a model through a REST API, straight out of model registry, on the command line, run the following:

mlflow models serve --env-manager conda --model-uri models:/my_registered_model_1/production --port 5001