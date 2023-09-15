# mnist-image-ibm-ce
Train and use the MNIST image prediction model in the IBM Code Engine environment

This project creates and trains a machine learning model that predicts what number a handrawn digit (0-9) is. It is built using the code and data sets of the "hello world" of ML - the MNIST image dataset.

The `train-model` folder contains the code to create and train the model. It runs as a Job in IBM Cloud Engine and stores the output model in IBM Cloud Object Storage (COS).

The `digit-image-app` folder contains the code to run a web app where you can handraw a digit (using mousr or trackpad). The web app is an IBM Code Engine web application. The application loads with an `init()` that downloads the model file from IBM COS and the loads the model into memory.

    Note: this is for demonstration purposes only and is not really the type of architecture you use in production. Typically after creating the model is served on some ML infrastructure e.g. Seldon is used in the Open Data Hub open source project (which is the upstream OSS for Red Hat OpenShift Data Science).

## Local Apple Mac on Metal
For those using Apple Macbook and M1 chip you need to setup Tensorflow accordingly.
https://developer.apple.com/metal/tensorflow-plugin/

Tested with
```
python -m pip install tensorflow-macos==2.10.0

# tensorflow-macos        2.10.0
# tensorflow-metal        0.6.0
```

## IBM Cloud Object Storage
From your IBM Cloud account you need to create a COS bucket in your COS service and then get the credentials to the bucket. Then ensure you local environment has that key.
```
export COS_API_KEY_ID=xxx
```
This bucket contains the data files to train and test the model.
```

```

## Production server
Production Python apps should use a WSGI HTTP server. The `digit-image-app` uses  [Gunicorn](https://gunicorn.org).

Run and test app with Gunicorn locally.
```
pip install gunicorn

# -w for number of worker process numbers; 1 is fine for local dev
# -b for address:port to listen on i.e. browser connects to this, and gunicorn routes to the port the app listens on
# main:app where main is the main.py program and app is the entrypoint name of the application
gunicorn -w 1 -b :3000 main:app
```
In production we can scale the number of containers. In each container there will be a number of Gunicorn worker threads running. This could there be an environment variable injected at deployment time.

## Run as a container locally
When you run the container you need to inject the Cloud Object storage API key to the app running in the container can access the bucket.

Assuming you have full control over your development then a simple hack is to run as root so the container process can write to the file system when it is getting the model file.
```
podman build -t digit-image:latest .
podman run -u root -t -p 3000:8080 -e COS_API_KEY_ID=xxxxxx digit-image:latest
```

## IBM Cloud Engine
Login and set right targets for IBM Code Engine work
```
ibmcloud login  ...

ibmcloud target -r us-south
ibmcloud target -g <group name>
```

### Setup IBM Cloud Engine project
One-time to create an IBM Code Engine project
```
ibmcloud ce project create --name mnist-image-ibm-ce
```

### Configure Project
You will create a secret key for your COS bucket credentials. Then you setup various environment variables in a ConfigMap for you Code Engine project environment. This Secret and the ConfigMap is used in the build of your applications and jobs in Code Engine. If you look in the source code you will see these same parameters defaulted so strictly speaking you don't need them in the ConfigMap. 

```
# check project is there and select it before using it
ibmcloud ce project list
ibmcloud ce project select --name mnist-image-ibm-ce

# need secret
ibmcloud ce secret create --name caine-cos-api-key --from-literal COS_API_KEY_ID=xxxxxxx

# config map for variables
ibmcloud ce configmap create --name mnist-image-ibm-ce-cm \
    --from-literal COS_ENDPOINT=https://s3.eu-gb.cloud-object-storage.appdomain.cloud  \
    --from-literal COS_AUTH_ENDPOINT=https://iam.cloud.ibm.com/identity/token  \
    --from-literal COS_SERVICE_CRN=crn:v1:bluemix:public:cloud-object-storage:global:a/b71ac2564ef0b98f1032d189795994dc:875e3790-53c1-40b0-9943-33b010521174::  \
    --from-literal COS_STORAGE_CLASS=eu-gb-smart  \
    --from-literal H5_FILE_NAME=mnist-model.h5  \
    --from-literal TRAIN_CSV=mnist_train.csv  \
    --from-literal TEST_CSV=mnist_test.csv
```

### Train Model
Job to train image prediction model
```
# create app first time
ibmcloud ce job create --name train-model --src https://github.com/jeremycaine/mnist-image-ibm-ce --bcdr train-model --str dockerfile --env-from-secret caine-cos-api-key --env-from-configmap mnist-image-ibm-ce-cm --size large

# or, rebuild after git commit
ibmcloud ce job update --name train-model --rebuild

# when you want to delete it
ibmcloud ce job delete --name train-model
```

### Digit Image 
App to draw digit and prediction model to label its image
```
# create app first time
ibmcloud ce app create --name digit-image --src https://github.com/jeremycaine/mnist-image-ibm-ce --bcdr digit-image-app --str dockerfile --env-from-secret caine-cos-api-key --env-from-configmap mnist-image-ibm-ce-cm

# or, rebuild after git commit
ibmcloud ce app update --name digit-image --rebuild
```

## Test the Application
First, create and train the model in IBM Code Engine

From the cloud console and the job `train-model` submit the job. You can watch its progress from Logging in the drop down menu top right.

Then, check that the model file `mnist-model.h5` is in your COS bucket.

Next, you can launch the image recognition web app. Go to application `digit-image`. 


