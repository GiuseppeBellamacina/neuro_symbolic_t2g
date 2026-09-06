# External Cluster Queue Driver

The FastAPI service in `remote/` advances the SLURM queue by invoking the cluster's idempotent one-shot tick over SSH. Render may host the service and cron-job.org may call `POST /tick` every five minutes. The cluster `.chain_state/` directory is authoritative; Render's SQLite database is only an ephemeral cache and event journal.

## Current experiment registry

The driver accepts these semantic config names:

```text
baseline-zero       baseline-few
sft
grpo-zero           grpo-few
sft-grpo-zero       sft-grpo-few
sft-grpo-zero-pda   sft-grpo-zero-hot
grpo-few-reward-edit        grpo-few-reward-token-f1
grpo-few-reward-chrfpp      grpo-few-reward-rouge-l
grpo-few-reward-sbleu2
```

The default queue is **2 eval-only baselines + 5 train/eval cells = 12 entries**. PDA, hot, and all reward configs are manual. Markov and rollout probes are non-training analyses and are not queue configs. Zero-shot/few-shot name prompt conditioning; the baseline is the untrained base method. See `docs/EXPERIMENT_DESIGN.md`.

## Local service

Prerequisites: repository environment installed and non-interactive SSH access to the cluster.

```powershell
$env:T2G_AUTH_TOKEN="local-token"
$env:T2G_SSH_USER="<cluster-user>"
uv run --extra dev uvicorn app:app --port 8000 --app-dir remote
```

Without `T2G_SSH_KEY_CONTENT` or `T2G_SSH_KEY_FILE`, local execution uses the normal SSH agent/config. Test the read-only status endpoint:

```powershell
curl.exe -H "X-Auth-Token: local-token" http://localhost:8000/status
```

Run the optional TUI with:

```powershell
uv run --extra tui python remote/tui.py --url http://localhost:8000 --token local-token
```

## Render deployment

Create a Python web service with:

```text
Build: pip install -r remote/requirements.txt
Start: uvicorn remote.app:app --host 0.0.0.0 --port $PORT
```

Required environment variables:

| Variable | Purpose |
|---|---|
| `T2G_AUTH_TOKEN` | Secret required by all mutating/status API calls |
| `T2G_SSH_USER` | Cluster account |
| `T2G_SSH_KEY_CONTENT` | Dedicated private key on ephemeral Render hosts |

Common optional variables are `T2G_SSH_HOST` (default `gcluster.dmi.unict.it`), `T2G_SSH_PORT`, `T2G_SSH_TIMEOUT`, `T2G_DB_PATH`, and `T2G_HELPER_AUTO_INSTALL`. Use a dedicated revocable SSH key. Never place a token or private key in the repository or logs.

Configure cron-job.org to send authenticated `POST /tick` requests every five minutes with a 90-120 second timeout. The tick is safe when a job is active or the queue is empty.

## API essentials

All routes except `GET /` require `X-Auth-Token`.

| Route | Action |
|---|---|
| `GET /status` | Refresh and return cluster/queue state |
| `GET /jobs` | Return queued entries |
| `POST /jobs` | Append one train or eval entry |
| `POST /queue` | Replace queue with `{"ablation": true}` (current default campaign) or explicit jobs |
| `DELETE /jobs/{tag}` | Remove queued entries with the tag |
| `POST /pause` / `POST /resume` | Pause or resume submissions |
| `POST /tick` | Run one immediate tick |

Examples:

```bash
TOKEN="..."; BASE="https://<service>.onrender.com"
curl -H "X-Auth-Token: $TOKEN" "$BASE/status"
curl -X POST -H "X-Auth-Token: $TOKEN" -H "Content-Type: application/json" -d '{"ablation":true}' "$BASE/queue"
curl -X POST -H "X-Auth-Token: $TOKEN" -H "Content-Type: application/json" -d '{"type":"train","config":"sft-grpo-zero-pda","tag":"pda-manual"}' "$BASE/jobs"
```

The API field `ablation` is retained as the queue-replacement command name; it expands the current 12-entry default campaign, not the manual PDA/hot variants.

Reward training (`grpo-few-reward-*`) requires an exact `mode` such as `--reward-qualification-report experiments/analysis/qwen25-05b/rewards/report.json`. The path must be repository-relative, remain below `experiments/analysis/`, and contain no traversal or shell metacharacters. Eval entries have no qualification mode. In the TUI custom queue editor, use:

```text
train:grpo-few-reward-edit:reward-edit:--reward-qualification-report=experiments/analysis/qwen25-05b/rewards/report.json
```

Reward train rows missing the explicit fourth field are rejected; other custom modes are not accepted by the editor.

## Operational rules

- Run `docs/TRAINING.md` preflight before enqueuing compute work. The driver does not prepare missing HF artifacts or permit online compute-node fallback.
- W&B and Hugging Face remain offline during compute jobs.
- A 401 indicates token mismatch; a 502 generally indicates SSH/cluster failure. `/status` may still return cached state with `cluster_reachable: false`.
- Pause is soft: the active SLURM job continues, but no new job is submitted.
- Queue and run outputs use the canonical hierarchy in `docs/EXPERIMENT_DESIGN.md`.
