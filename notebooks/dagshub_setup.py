import dagshub
import mlflow
dagshub.init(repo_owner='himadri06', repo_name='mlops-project', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/himadri06/mlops-project.mlflow')
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)