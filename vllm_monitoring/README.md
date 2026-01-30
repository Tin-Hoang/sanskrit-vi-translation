# Monitoring Setup for vLLM

This directory contains a Docker Compose setup for monitoring vLLM with Prometheus and Grafana.

<p align="center">
  <img src="../docs/grafana_vllm_dashboard.png" alt="vLLM Dashboard on Grafana" width="100%" />
</p>

## Quick Start

### 1. Start vLLM with metrics enabled
```bash
./serve_vllm.sh
```

### 2. Start monitoring stack
```bash
cd vllm_monitoring
./start.sh   # Automatically detects host IP
```

### 3. Access Dashboards

Choose **one** of the following methods based on your server configuration:

#### Option A: SSH Tunneling (Recommended for restricted servers)

If your server only allows SSH (port 22), use SSH tunneling to securely access the dashboards.

**From your laptop**, create SSH tunnels:
```bash
# Basic tunnel
ssh -L 3000:localhost:3000 -L 9091:localhost:9091 user@server-ip

# With SSH key
ssh -L 3000:localhost:3000 -L 9091:localhost:9091 -i ~/.ssh/your_key user@server-ip

# Background tunnel (keep running)
ssh -fN -L 3000:localhost:3000 -L 9091:localhost:9091 user@server-ip
```

Then open in your browser:
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9091

#### Option B: Open Ports on Cloud Firewall

If you prefer direct access, open ports 3000 and 9091 on your cloud provider's firewall.

**AWS EC2:**
1. Go to **EC2 Console** → **Security Groups**
2. Select your instance's security group
3. Click **Edit inbound rules** → **Add rule**
4. Add rules:
   | Type | Port | Source |
   |------|------|--------|
   | Custom TCP | 3000 | Your IP (or company VPN) |
   | Custom TCP | 9091 | Your IP (or company VPN) |
5. Click **Save rules**

**Google Cloud (GCE):**
```bash
# Create firewall rule for Grafana
gcloud compute firewall-rules create allow-grafana \
    --allow tcp:3000 \
    --source-ranges=YOUR_IP/32 \
    --description="Allow Grafana access"

# Create firewall rule for Prometheus
gcloud compute firewall-rules create allow-prometheus \
    --allow tcp:9091 \
    --source-ranges=YOUR_IP/32 \
    --description="Allow Prometheus access"
```

Or via **GCP Console**:
1. Go to **VPC Network** → **Firewall**
2. Click **Create Firewall Rule**
3. Set: Direction=Ingress, Targets=All instances, Source=Your IP, Protocols=tcp:3000,9091

> ⚠️ **Security Warning**: Avoid using `0.0.0.0/0` (open to all) in production. Always restrict to your IP address.

---

## Services

| Service | Port | Purpose |
|---------|------|---------|
| vLLM | 8000 | LLM API + Metrics endpoint |
| Prometheus | 9091 | Metrics collection |
| Grafana | 3000 | Dashboards |

## Pre-configured Dashboard

The vLLM dashboard includes:
- **Requests**: Running/waiting request counts
- **Throughput**: Prompt and generation tokens/s
- **Cache Usage**: GPU KV cache utilization

## Stopping

```bash
cd vllm_monitoring
docker compose down
```
