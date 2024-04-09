# OCPDAS - Supporting Code

Supporting code for the paper "OCPDAS: Online Change-Point-Detection-Based Autoscaling for Stable Pod Scaling in Kubernetes."

## Environment

### Docker

We recommend installing Docker Engine following the [official documentation](https://docs.docker.com/engine/install/).

### Minikube

We recommend installing Minikube, a local Kubernetes cluster, following the [official documentation](https://minikube.sigs.k8s.io/docs/start/).
Then start Minikube by running:

```bash
minikube start --driver=docker --cni=bridge
minikube addons enable metrics-server
```

### Kubectl

We recommend install kubectl following the [official document](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/).
Then create an independent namespace named `ocpdas` for testing the repository by running:

```bash
kubectl create namespace ocpdas
```

all the benchmark tests are conducted in the `ocpdas` namespace and you can safely remove all content after reproduction the namespace following

```bash

kubectl delete namespace ocpdas
```

### Conda Environment

We recommend using conda to manage the running environment to reproduce the experiments.

``` bash
conda env create -f environment.yml
```

## Reproduction

Follow the instructions in the `README.md` file in every subfolder to run the code.
