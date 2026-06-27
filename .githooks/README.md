# Git Hooks

Hook personalizzati per questa repo. Vengono eseguiti automaticamente da git.

## Hook disponibili

### `pre-commit`

Prima di ogni `git commit`, esegue automaticamente il formattatore di codice:

- **Windows** → `powershell -File format.ps1`
- **Linux/macOS** → `bash format.sh`

I file modificati dal formatter vengono automaticamente ri-aggiunti allo stage (`git add -u`).
Se il formatter fallisce, il commit viene abortito.

Per saltare il formatting:
```bash
git commit --no-verify
```

### `pre-push`

Ogni volta che fai `git push`, sincronizza automaticamente sul cluster DMI:

- `src/` — tutto il codice sorgente (training, rewards, grammar, data, utils)
- `cluster/` — script di gestione cluster (setup, train, eval, pipeline)
- `experiments/configs/` — configurazioni YAML degli esperimenti
- `grammarllm/` — libreria grammarllm
- `docs/` — documentazione
- `pyproject.toml`, `README.md`, `CLUSTER.md`, `TRAINING.md`, `main.py`, `sync_cluster.ps1`

Usa `scp` per sincronizzare solo i file modificati nel push. Se il cluster non è raggiungibile o il sync fallisce, il push su GitHub prosegue comunque (l'hook non blocca mai il push).

## Configurazione

Dopo aver clonato la repo su un nuovo PC, esegui:

```bash
git config core.hooksPath .githooks
```

Questo dice a git di usare `.githooks/` invece della cartella default `.git/hooks/`. L'impostazione è locale alla repo (non viene pushato su GitHub).

> **Nota**: su Linux/macOS potrebbe servire rendere eseguibile l'hook:
> ```bash
> chmod +x .githooks/pre-push
> ```

## Verificare che funzioni

Dopo la configurazione, al prossimo `git push` vedrai nel terminale:

```
[pre-push] Syncing to cluster: src/ pyproject.toml
[pre-push]   syncing src/ ...
[pre-push]   syncing pyproject.toml ...
[pre-push] ✅ Cluster sync done
```

Sul cluster, i file sincronizzati appariranno in `~/neuro_symbolic_t2g/`.

## Disabilitare temporaneamente

Per fare un push senza triggerare la sincronizzazione:

```bash
git push --no-verify
```

## Risoluzione problemi

### "Permission denied" su Linux/macOS

```bash
chmod +x .githooks/pre-push
```

### "Connection refused" o "Host not found"

Il cluster non è raggiungibile dalla rete corrente (es. sei a casa senza VPN). Il push su GitHub prosegue comunque — il sync verrà fatto al prossimo push da una rete che raggiunge il cluster.

### Sync manuale se il pre-push non ha funzionato

```powershell
# Da Windows PowerShell:
.\sync_cluster.ps1 -Action upload
```
