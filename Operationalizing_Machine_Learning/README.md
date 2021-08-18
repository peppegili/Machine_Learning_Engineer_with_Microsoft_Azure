# Operationalizing Machine Learning

## Overview
This project is part of the Udacity Azure ML Nanodegree. In this project, will use Azure to configure a cloud-based machine learning production model, deploy it, and consume it. Moreover, we will also create, publish, and consume a pipeline.

Both ***Azure ML Studio*** and ***Python SDK*** will be used in this project. The model will be trained using *AutoML* and the best model will be, finally, deployed and consumed through *Azure Container Instance (ACI)* and *REST endpoint* respectively.

The [aml-pipelines-with-automated-machine-learning-step.ipynb](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/aml-pipelines-with-automated-machine-learning-step.ipynb) jupyter notebook contains all the steps of the entire procedure.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Architectural Diagram](#architectural-diagram)
- [Key Steps](#key-steps)
  - [AutoML Model](#automl-model)
  - [Deploy the best model](#deploy-the-best-model)
  - [Enable logging](#enable-logging)
  - [Consume model endpoints](#consume-model-endpoints)
  - [Create and publish a pipeline](#create-and-publish-a-pipeline)
- [Screen Recording](#screen-recording)
- [Future Work](#future-work)

## Dataset
In this project a bank marketing [dataset](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) is used.
It contains phone calls from a direct marketing compaign of a Portoguese banking institution.

The dataset has a series of information (age, job, marital, education, etc...) for a total of 32950 observations, 20 features, and a target variable (y)
with two possible values: yes or no.
The task is addressed as a classification task and the goal is to predict if a client will subscribe a term deposit (y variable).

## Architectural Diagram
The following diagram shows all the steps of the entire process:

![Architectural Diagram](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/architectural_diagram.png)

  - **Authentication**: authentication is crucial for the continuous flow of operations. infact, when it is not set properly it requires human interaction and thus, the flow is       interrupted. So whenever possible, it's good to use authentication with automation (authentication types: key-based, token-based)
    
    A *Service Principal* is a user role with controlled permissions to access specific resources. Using a service principal is a great way to allow authentication while reducing     the scope of permissions, which enhances security.
    
    This step has been skipped since we are not authorized to create a security principal, using the lab udacity provided
  
  - **Auto ML Model**: AutoML is the process of automating the time-consuming, iterative tasks of machine learning model development. A classification model on bank marketing         [dataset](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) is built using AutoML, and the best model is determined       based on metrics collected (*AUC_weighted*)
  
  - **Deploy the best model**: the goal is to ship a model into production. The best performing model, the one with the highest metric value (*AUC_weighted*), is selected for         deployment. *Azure Container Instance* or *Azure Kubernetes Service* can be chosen in this step
  
  - **Enable logging**: Application Insights tool allows to detect anomalies, visualize performance and keep track of the deployed model. It can be enabled before or after a           deployment
  
  - **Consume model endpoints**: after model deployment, consuming information from it is a foundamental step. An HTTP API is exposed over the network so that interaction with         trained model can happen via HTTP requests
  
  - **Create and publish a pipeline**: pipelines allow to automate workflows, i.e. the entire process of train and deploy a model. Published pipelines allow external services to       interact with them in a simple and efficient way
  
  - **Documentation**: this is the last but not least step. A screencast video and a README file have been created, containing the description of the project and all the               performed steps

## Key Steps
### **Auto ML Model**

Automated ML includes all the tasks of machine learning model development, from loading dataset, creating pipeline and AutoML step, to start the training procedure:
    
- **Load data**: dataset has been loaded as tabularDataset and registered in workspace

  ```python
  found = False
  key = "BankMarketing Dataset"
  description_text = "Bank Marketing DataSet for Udacity Course 2"

  if key in ws.datasets.keys(): 
          found = True
          dataset = ws.datasets[key] 

  if not found:
          # Create AML Dataset and register it into Workspace
          example_data = 'https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv'
          dataset = Dataset.Tabular.from_delimited_files(example_data)        
          #Register Dataset in Workspace
          dataset = dataset.register(workspace=ws,
                                     name=key,
                                     description=description_text)

  df = dataset.to_pandas_dataframe()
  ```
  ![Dataset](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/dataset.png)
        
- **AutoML**: *AutoMLConfig*, *AutoMLStep*, *Pipeline* classes, responsible of the automated machine learning process, have been instantiated with the following parameters
 
  ```python
  # AutoML config
  automl_settings = {
      "experiment_timeout_minutes": 20,
      "max_concurrent_iterations": 5,
      "primary_metric" : 'AUC_weighted'
  }
  automl_config = AutoMLConfig(compute_target=compute_target,
                               task = "classification",
                               training_data=dataset,
                               label_column_name="y",   
                               path = project_folder,
                               enable_early_stopping= True,
                               featurization= 'auto',
                               debug_log = "automl_errors.log",
                               **automl_settings
                              )

  # AutoML step
  automl_step = AutoMLStep(
      name='automl_module',
      automl_config=automl_config,
      outputs=[metrics_data, model_data],
      allow_reuse=True)

  # Pipeline
  from azureml.pipeline.core import Pipeline
  pipeline = Pipeline(
      description="pipeline_with_automlstep",
      workspace=ws,    
      steps=[automl_step])
  ```
  After submitting the pipeline run to the experiment, results metrics have been collected and the best model, ***VotingEnsemble***, has been retrieved.

  The pipeline has been completed in 29m 22s.

  ![Pipeline](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/pipeline.png)

  ![Experiment](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/experiment.png)

  ![Experiment Graph](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/experiment_graph.png)

  ![Best model](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/best_model.png)
  
### **Deploy the best model**
The best model, the output of the above step, has been deployed to ***Azure Container Instance***, by clicking the button "Deploy" in *Model* tab under *Experiment* section. *Enable authentication* has been enabled.
    
![Model Deploy](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/model_deploy.png)
    
![Model Deploy Running](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/model_deploy_running.png)
    
![Model Deploy Completed](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/model_deploy_completed.png)
    
    
### **Enable logging**
Once the model has been deployed ("Deployment state" has become *Healthy*), a REST endpoint and a Swagger URI have been generated.
    
![Model Deploy Endpoint](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/model_deploy_endpoint.png)
    
***Application Insights enabled*** is false. So it has been enabled in order to retrieve logs, using the provided script
`logs.py`.
    
The script has been modifyied in order to correctly enable ***Application Insights***.
    
Moreover, a `config.json`, containing workspace info, has been retrieved from Azure ML Studio and placed in the same directory of the above script, before running it.
    
```python
from azureml.core import Workspace
from azureml.core.webservice import Webservice

# Requires the config to be downloaded first to the current working directory
ws = Workspace.from_config()

# Set with the deployment name
name = "model-deploy"

# Load existing web service
service = Webservice(name=name, workspace=ws)

# Enable Application Insights
service.update(enable_app_insights=True)

logs = service.get_logs()

for line in logs.split('\n'):
    print(line)
```
When the execution has been completed, ***Application Insights*** has been set to true:
    
![Logs Terminal](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/logs_terminal.png)
    
![Application Insights](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/application_insights.png)
    
![Application Insights Monitor](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/application_insights_monitor.png)
  
### **Consume model endpoints**

Endpoints allow other services to interact with deployed models. There are some interesting details to be aware when trying to use HTTP:
   
- **Swagger**: swagger is a tool that eases the documentation efforts of HTTP APIs. It helps to build, document, and consume RESTful web services. It further explains what types of HTTP requests that an API can consume, like `POST` and `GET`.

  Azure provides a `swagger.json` that is used to create a web site that documents the HTTP endpoint for a deployed model.
        
  The file has been downloaded (from Swagger URI) and saved in [swagger directory](https://github.com/peppegili/2_Operationalizing_Machine_Learning/tree/master/swagger) containing `swagger.sh` and `serve.py` scripts.
        
  ![Swagger Terminal](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/swagger_terminal.png)
        
  Then, `serve.py` and `swagger.sh` have been executed in order to start a python server on port 8000 and download the latest Swagger container and run it on port 9000, respectively. The documentation for the HTTP API of the model is shown below:
        
  ![Swagger](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/swagger.png)
        
  ![Swagger Post](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/swagger_post.png)
        
  ![Swagger Post 2](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/swagger_post_2.png)
       
        
- **Consume deployed services**: a deployed service can be consumed via an HTTP API. Users can initiate HTTP requestes, for example an input request, usually via an HTTP POST request. HTTP POST is a request method that is used to submit data. The HTTP GET is another commonly used request method. HTTP GET is used to retrieve information from a URL. The allowed requests methods and the different URLs exposed by Azure create a bi-directional flow of information.
The APIs exposed by Azure ML will use JSON (JavaScript Object Notation) to accept data and submit responses.
        
  The provided script `endpoint.py` has been executed in order to interact with the deployed model. It has been modified with the correct *scoring_uri* and *key* retrieved from the "*Consume*" tab of the endpoint:
  ```python
  import requests
  import json

  # URL for the web service, should be similar to:
  # 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
  scoring_uri = 'http://4c4af6b2-3cb1-40bf-9448-1f97233b5a54.southcentralus.azurecontainer.io/score'
  # If the service is authenticated, set the key or token
  key = 'OSqqbQTqY3bMc8eeFVUbGeqKf6WelvBg'

  # Two sets of data to score, so we get two results back
  data = {"data":
          [
            {
              "age": 17,
              "job": "blue-collar",
              "marital": "married",
              "education": "university.degree",
              "default": "no",
              "housing": "yes",
              "loan": "yes",
              "contact": "cellular",
              "month": "may",
              "day_of_week": "mon",
              "duration": 971,
              "campaign": 1,
              "pdays": 999,
              "previous": 1,
              "poutcome": "failure",
              "emp.var.rate": -1.8,
              "cons.price.idx": 92.893,
              "cons.conf.idx": -46.2,
              "euribor3m": 1.299,
              "nr.employed": 5099.1
            },
            {
              "age": 87,
              "job": "blue-collar",
              "marital": "married",
              "education": "university.degree",
              "default": "no",
              "housing": "yes",
              "loan": "yes",
              "contact": "cellular",
              "month": "may",
              "day_of_week": "mon",
              "duration": 471,
              "campaign": 1,
              "pdays": 999,
              "previous": 1,
              "poutcome": "failure",
              "emp.var.rate": -1.8,
              "cons.price.idx": 92.893,
              "cons.conf.idx": -46.2,
              "euribor3m": 1.299,
              "nr.employed": 5099.1
            },
        ]
      }
  # Convert to JSON string
  input_data = json.dumps(data)
  with open("data.json", "w") as _f:
      _f.write(input_data)

  # Set the content type
  headers = {'Content-Type': 'application/json'}
  # If authentication is enabled, set the authorization header
  headers['Authorization'] = f'Bearer {key}'

  # Make the request and display the response
  resp = requests.post(scoring_uri, input_data, headers=headers)
  print(resp.json())
  ```
  The script issues a `POST` request to the deployed model and gets a JSON response. A `data.json` file has been created once the script has been executed:

  ![Endpoint Test](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/endpoint_test.png)
  ![Endpoint Test 2](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/endpoint_test_2.png)
      
- **Benchmarking**: a benchmark is used to create a baseline or acceptable performance measure. Benchmarking HTTP APIs is used to find the average response time for a deployed model. One of the most significant metrics is the *response time* since Azure will timeout if the response times are longer than 60 seconds.
        
  [Apache Benchmark](https://httpd.apache.org/docs/2.4/programs/ab.html) is an easy and popular tool for benchmarking HTTP services.

  The `benchmark.sh` has been executed, once the correct endpoint and key have been compiled
  ```
  ab -n 10 -v 4 -p data.json -T 'application/json' -H 'Authorization: Bearer OSqqbQTqY3bMc8eeFVUbGeqKf6WelvBg' http://4c4af6b2-3cb1-40bf-9448-1f97233b5a54.southcentralus.azurecontainer.io/score

  ```
  The payload `data.json` has been required and used to HTTP POST to the endpoint.

  ![Benchmark](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/benchmark.png)
  ![Benchmark 2](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/benchmark_2.png)
        
  
### **Create and publish a pipeline**
*Python SDK* has been used to crate and publish a pipeline. A pipeline automate the entire training process and when a pipeline is published, a publich HTTP endpoint becomes available, allowing other services, including external ones, to interact with Azure Pipeline.
    
The following code has been used to publish the pipeline to the workspace:
```python
published_pipeline = pipeline_run.publish_pipeline(
    name="Bankmarketing Train", description="Training bankmarketing pipeline", version="1.0")

published_pipeline
```

![Pipeline Endpoint](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/pipeline_endpoint.png)
![Pipeline Endpoint 2](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/pipeline_endpoint_2.png)

The following code has been used to send a `POST` request to the endpoint. The endpoint is the URI that the SDK will use to communicate with it over HTTP:
```python
import requests

rest_endpoint = published_pipeline.endpoint
response = requests.post(rest_endpoint, 
                         headers=auth_header, 
                         json={"ExperimentName": "pipeline-rest-endpoint"}
                        )
```
Once all steps have been completed, the Pipeline will be triggered and available in Azure ML Studio
    
![Pipeline Endpoint 3](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/pipeline_endpoint_3.png)
![Pipeline Endpoint 4](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/pipeline_endpoint_4.png)
![Pipeline Endpoint 5](https://github.com/peppegili/2_Operationalizing_Machine_Learning/blob/master/img/pipeline_endpoint_5.png)
  
## Screen Recording
[Link](https://drive.google.com/file/d/1i4SaNovKVCzf2W8D6ghK6zWn95N5md9P/view?usp=sharing) to the video.

## Future Work
- Enhance the pipeline with other steps, i.e. preprocessing, feature engineering, etc.
- Handle class imbalance problem, when performing AutoML
- Try to add CI/CD for automated testing
- Show model output using an user frindly dashboard
