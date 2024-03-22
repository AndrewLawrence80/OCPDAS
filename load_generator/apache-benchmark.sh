URL="http://192.168.49.2:31080"
for ((requests = min_requests; requests <= max_requests; requests += step)); do
    echo "Running ab with $requests requests:"
    ab -n $requests -c 100 $URL
    echo ""
done