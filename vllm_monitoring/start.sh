#!/bin/bash
# Start monitoring stack for vLLM

cd "$(dirname "$0")"

# Get host IP for container networking
export HOST_IP=$(hostname -I | awk '{print $1}')

# Generate prometheus config with actual host IP
cat > prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'vllm'
    static_configs:
      # vLLM metrics - dynamically set host IP
      - targets: ['${HOST_IP}:8000']
    metrics_path: /metrics
EOF

echo "Generated prometheus config with HOST_IP=${HOST_IP}"
cat prometheus/prometheus.yml

# Start containers
docker compose up -d

echo ""
echo "Monitoring started!"
echo "  - Prometheus: http://localhost:9091"
echo "  - Grafana:    http://localhost:3000 (admin/admin)"
