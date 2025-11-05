#!/usr/bin/env bash
# Cloud-init / user-data for EC2 to install Docker and run the bot container
set -eux

apt-get update
apt-get install -y ca-certificates curl gnupg lsb-release

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu || true

# Pull and run container (user should build and push image beforehand)
# Example: docker run --restart unless-stopped -e CONFIG_FROM_ENV=1 -e ROOSTOO_API_KEY=... roostoo/bot:latest
echo "# After provisioning, build or pull your image and run it. Example:" > /root/README_ROOT
echo "docker run -d --name roostoo_bot --restart unless-stopped -e CONFIG_FROM_ENV=1 -e ROOSTOO_API_KEY=... roostoo/bot:latest" >> /root/README_ROOT
