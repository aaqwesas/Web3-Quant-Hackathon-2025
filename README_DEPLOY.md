# Deploying the Roostoo Bot to AWS EC2

This document contains a minimal, safe deployment path for running the trading bot on an EC2 instance.

Options provided:
- Docker container (recommended) — easy to run and update.
- systemd unit + Docker (for auto-start).
- cloud-init user-data script to install Docker on startup.

Security notes
- Never store API keys in the repository. Use environment variables, AWS Secrets Manager, or an instance profile to inject secrets at runtime.
- Run the bot in dry-run mode while testing: `python -m roostoo_bot_template.bot --dry-run`.

Quick steps (Docker)
1. Build and push the Docker image from your workstation:

```bash
docker build -t yourrepo/roostoo-bot:latest .
docker push yourrepo/roostoo-bot:latest
```

2. Launch an EC2 instance (Ubuntu). Use `deploy/ec2_user_data.sh` as user-data if you want Docker preinstalled.

3. On the EC2 instance, run the container with environment-injected config (recommended):

```bash
docker run -d --name roostoo_bot \
  --restart unless-stopped \
  -e CONFIG_FROM_ENV=1 \
  -e ROOSTOO_BASE_URL=https://api.roostoo.com \
  -e ROOSTOO_API_KEY=YOUR_API_KEY \
  -e ROOSTOO_API_SECRET=YOUR_SECRET \
  -e BOT_SYMBOLS='["BTC/USD"]' \
  yourrepo/roostoo-bot:latest
```

Systemd method
1. Copy `deploy/roostoo_bot.service` to `/etc/systemd/system/` and edit the ExecStart line to include your image and env vars or wrap a small shell script.
2. Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now roostoo_bot.service
```

Operational notes
- Monitor logs with `docker logs -f roostoo_bot` or `journalctl -u roostoo_bot.service`.
- Start in `--dry-run` to validate signals without placing live orders.
- Consider adding Prometheus metrics + alerts for production.
