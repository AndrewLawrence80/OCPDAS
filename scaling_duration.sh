#!/bin/bash
# Capture start time in milliseconds
start_time=$(date +%s%3N)

# Send scaling request
kubectl scale deployment/apache-deployment --replicas=1

# Get the total desired replicas
total_replicas=$(kubectl get deployment/apache-deployment -o jsonpath='{.spec.replicas}')

# Monitor pod readiness
while true; do
    ready_replicas=$(kubectl get pods -l app=apache -o 'jsonpath={..status.conditions[?(@.type=="Ready")].status}' | tr -s ' ' '\n' | grep -c "True")
    if [ "$ready_replicas" -eq "$total_replicas" ]; then
        break
    fi
    sleep 0.01
done

# Capture end time in milliseconds
end_time=$(date +%s%3N)

# Calculate duration in milliseconds
duration=$((end_time - start_time))

# Display duration
echo "Scaling operation took $duration milliseconds for all pods to become ready."
