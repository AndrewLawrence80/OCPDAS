from kubernetes import client, config

def get_pod_cpu_utilization(namespace):
    # Load Kubernetes configuration
    config.load_kube_config()

    # Create Kubernetes API client
    api_instance = client.CustomObjectsApi()

    # Make API request to fetch pod metrics
    api_response = api_instance.list_namespaced_custom_object(
        group="metrics.k8s.io",
        version="v1beta1",
        namespace=namespace,
        plural="pods",
    )

    # Extract CPU utilization metrics
    pod_metrics = api_response.get("items", [])
    cpu_utilizations = {}
    for metric in pod_metrics:
        metadata = metric.get("metadata", {})
        name = metadata.get("name", "Unknown")
        containers = metric.get("containers", [])
        for container in containers:
            cpu_usage = container.get("usage", {}).get("cpu", "0")
            cpu_utilizations[name] = cpu_usage

    return cpu_utilizations

# Example usage
namespace = "autoscaling"
cpu_utilizations = get_pod_cpu_utilization(namespace)
for pod, cpu_usage in cpu_utilizations.items():
    print(f"Pod: {pod}, CPU Usage: {cpu_usage}")