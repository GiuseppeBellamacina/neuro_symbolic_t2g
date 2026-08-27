# Driver esterno della catena T2G — Render + cronjob.org

Micro-servizio **esterno** che orchestra la catena di job SLURM sul cluster
`gcluster` (DMI UniCT) via **tick**. Deployato su **Render** (free tier),
tickato da **cronjob.org** ogni 5 minuti. È il successore moderno di
`cluster/remote_tick.sh`: stesso principio (ssh + `chain_tick.sh` one-shot
idempotente), ma con **API REST** per pilotare la coda, un **DB SQLite** di
diario/cache e un helper lato cluster che riduce ogni tick a **1 sola
connessione ssh**.

```
cronjob.org ──POST /tick (X-Auth-Token)──▶ Render (uvicorn)
    (ogni 5 min)                              │
                                              ├─ ssh ─▶ gcluster: bash cluster_helper.sh tick
                                              │              └─▶ chain_tick.sh --quiet (avanza la coda)
                                              ├─ riceve snapshot KEY=VALUE (1 riga per chiave)
                                              └─ sincronizza SQLite (cache + diario eventi)
```

La **fonte di verità è sempre lo stato sul cluster** (`.chain_state/`):
il filesystem di Render free tier è **effimero** (si resetta a ogni
redeploy/restart), quindi il DB SQLite è solo una **cache + diario** e viene
riletto da zero a ogni tick. Se il cluster è irraggiungibile, `GET /status`
risponde comunque con l'ultimo stato noto e `cluster_reachable: false`.

---

## 1. Componenti

| File                         | Ruolo                                                                                                                                                 |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `remote/app.py`            | Servizio FastAPI completo (un file, zero librerie SSH esotiche: subprocess +`ssh` nativo)                                                           |
| `remote/cluster_helper.sh` | Helper**lato cluster**: un'unica shell che in una sola connessione restituisce TUTTO lo stato (o esegue una mutazione) in formato `KEY=VALUE` |
| `remote/requirements.txt`  | Dipendenze Python: solo`fastapi` + `uvicorn[standard]`                                                                                            |

> `app.py` fa **auto-install** del helper sul cluster (via `scp`) al primo
> tick se `T2G_HELPER_AUTO_INSTALL=1` (default) e il file non è già presente.
> Puoi comunque copiarlo a mano (sezione 3.2).

### Protocollo `cluster_helper.sh`

Invocato da `app.py` come `bash ~/neuro_symbolic_t2g/cluster/cluster_helper.sh <subcomando>`.

**Output** (machine-readable, niente python sul login node → `KEY=VALUE` per
riga; la coda è serializzata con il separatore sicuro `\x1f` — unit
separator, mai usato nelle entry):

```
STATUS_OK=1
ACTIVE_JOB=<id>|<name>|<state>          # vuoto se nessun job attivo
QUEUE=<e1>\x1f<e2>\x1f...               # coda completa (una riga), vuota = ""
QUEUE_COUNT=<n>
LAST_JOB=<id>:<type>:<cfg>:<tag>:<retries>   # vuoto se nessuno
STOPPED=0|1                             # chain_stopped presente = pausa
WATCHER_ALIVE=0|1                       # watcher fallback vivo
ERRORS_COUNT=<n>                        # righe totali di chain_errors
ERRORS_TAIL=[...]                       # ultime 5 righe raw come JSON array
```

**Subcomandi**:

| Comando                     | Effetto                                                                     |
| --------------------------- | --------------------------------------------------------------------------- |
| `status` (default)        | stampa lo snapshot completo                                                 |
| `enqueue <entry>`         | appende`type:cfg:tag[:extra]` a `job_chain`                             |
| `rewrite_queue <content>` | rimpiazza l'intera coda (entry separate da`\x1f`; stringa vuota = svuota) |
| `pause`                   | crea`.chain_state/chain_stopped` (soft stop: nessuna nuova sottomissione) |
| `resume`                  | rimuove`.chain_state/chain_stopped`                                       |
| `tick`                    | esegue`chain_tick.sh --quiet`                                             |

Dopo ogni mutazione il helper stampa comunque lo snapshot fresco: il driver
fa **una sola** connessione e riceve esito + stato insieme.

---

## 2. Deploy LOCALE (test) — prima di Render

Prima di sbarcare su Render puoi far girare il servizio **in locale su
Windows** e testarlo con una vera autenticazione SSH verso `gcluster`: quella
di default dell'utente (ssh-agent / identità default / `~/.ssh/config`), la
stessa che usa `sync_cluster.ps1`. La chiave dedicata del driver è
**opzionale**: se non imposti né `T2G_SSH_KEY_CONTENT` né `T2G_SSH_KEY_FILE`,
`app.py` NON passa `-i` a ssh/scp e lascia che usino la configurazione SSH di
default. È anche il modo più comodo per provare il servizio prima di creare la
chiave dedicata per Render.

### 2.1. Prerequisiti

- `uv` installato e ambiente del repo sincronizzato: `uv sync --extra dev`
  (`fastapi`/`uvicorn`/`httpx` vivono nell'extra `dev`; la TUI aggiunge l'extra
  `tui`).
- **Autenticazione a chiave già funzionante** verso il cluster dalla macchina
  locale: `ssh <user>@gcluster.dmi.unict.it 'echo ok'` deve rispondere senza
  chiedere password. Se funziona `sync_cluster.ps1 -Action upload`, funziona
  anche questo. Il servizio usa `-o BatchMode=yes`: senza chiave caricata la
  connessione fallisce subito con un 502 — **mai** un prompt password che blocca
  il server.

### 2.2. Avvio (pwsh)

```powershell
$env:T2G_AUTH_TOKEN="test-token-locale"
$env:T2G_SSH_USER="<codice-fiscale>"          # utente su gcluster
uv run --extra dev uvicorn app:app --port 8000 --app-dir remote
```

`--app-dir remote` fa importare `app` direttamente da `remote/` (stessa
struttura del repo). Il servizio ascolta su `http://localhost:8000`; i log
vanno su stderr (compreso l'evento di startup in `data/t2g_driver.db`).

Env vars minime per il test locale:

| Variabile                                      | Obbligatoria | Note                                                |
| ---------------------------------------------- | ------------ | --------------------------------------------------- |
| `T2G_AUTH_TOKEN`                             | sì          | token richiesto dall'header`X-Auth-Token`         |
| `T2G_SSH_USER`                               | sì          | codice fiscale su gcluster                          |
| `T2G_SSH_HOST`                               | no           | default`gcluster.dmi.unict.it`                    |
| `T2G_SSH_KEY_FILE` / `T2G_SSH_KEY_CONTENT` | **no** | assenti = chiave opzionale → ssh/config di default |
| `T2G_SSH_PORT` / `T2G_SSH_TIMEOUT` / ...   | no           | vedi tabella completa in sezione 4                  |

### 2.3. Test con curl (pwsh)

```powershell
# health — nessuna auth
curl.exe http://localhost:8000/

# 401 senza token
curl.exe http://localhost:8000/status

# /status CON token → fa una ssh VERA verso il cluster (BatchMode, key default):
#   200 + cluster_reachable:true  se la chiave di default funziona
#   502 + cluster_reachable:false se il cluster è irraggiungibile
# il servizio NON crasha mai: GET /status continuerà a rispondere dall'ultimo
# stato noto (cache DB) con cluster_reachable:false.
curl.exe -H "X-Auth-Token: test-token-locale" http://localhost:8000/status
```

> ⚠️ Quest'ultima chiamata esegue una **vera connessione ssh** verso `gcluster`:
> è il test che vogliamo fare (e `status` è read-only, non tocca nulla). Il
> comando è scritto per te: eseguilo tu dalla macchina dove la chiave verso il
> cluster è già attiva.

Per accodare davvero un job (modifica la coda sul cluster):

```powershell
curl.exe -X POST -H "X-Auth-Token: test-token-locale" -H "Content-Type: application/json" `
  -d '{"type":"train","config":"grpo_optimal","tag":"test-locale"}' http://localhost:8000/jobs
```

### 2.4. Puntare la TUI al servizio locale

```powershell
uv run --extra tui python remote/tui.py --url http://localhost:8000 --token test-token-locale
```

### 2.5. Fermare il servizio

`Ctrl+C` nel terminale che ospita uvicorn. Se avviato in background:

```powershell
Get-Process | Where-Object { $_.ProcessName -match "uvicorn|python" } |
    Where-Object { $_.Path -like "*neuro_symbolic_t2g*" -or $_.CommandLine -like "*remote*" } |
    Stop-Process -Force
# oppure, con Start-Process: Stop-Process -Id <PID>
```

---

## 3. Deploy su Render (passo-passo)

### 3.1. Crea la chiave SSH dedicata su gcluster

Su una macchina da cui puoi già accedere al cluster (o dal cluster stesso):

```bash
# 1. Genera una chiave DEDICATA senza passphrase, SOLO per il driver
ssh-keygen -t ed25519 -f ~/.ssh/t2g_driver -N "" -C "t2g-driver-render"

# 2. Autorizza la chiave sul cluster (se la generi localmente)
ssh-copy-id -i ~/.ssh/t2g_driver.pub <codice-fiscale>@gcluster.dmi.unict.it
# oppure manualmente: aggiungi la riga `ssh-ed25519 AAAA... t2g-driver-render`
# a ~/.ssh/authorized_keys sul cluster

# 3. Verifica l'accesso non interattivo
ssh -i ~/.ssh/t2g_driver -o BatchMode=yes <codice-fiscale>@gcluster.dmi.unict.it 'echo OK'
```

> **Sicurezza**: usa una chiave **dedicata** (non la tua personale): se
> trapela da Render la revochi rimuovendo una sola riga da
> `authorized_keys`, senza toccare il tuo accesso. L'host viene registrato
> con `StrictHostKeyChecking=accept-new` nel `known_hosts` locale del
> servizio.

### 3.2. Copia il helper sul cluster (una volta)

```bash
# dal repo locale (o lascia che il servizio lo installi da solo)
scp remote/cluster_helper.sh <codice-fiscale>@gcluster.dmi.unict.it:~/neuro_symbolic_t2g/cluster/

# verifica che risponda
ssh <codice-fiscale>@gcluster.dmi.unict.it \
  'bash ~/neuro_symbolic_t2g/cluster/cluster_helper.sh status'
```

> Assicurati che il resto del progetto sia aggiornato sul cluster
> (`.\neuro_symbolic_t2g\sync_cluster.ps1 -Action upload` da Windows) così
> `chain_tick.sh` e `_lib.sh` sono presenti.

### 3.3. Crea il Web Service su Render

1. **Dashboard Render → New → Web Service**; collega il repo (o usa
   "Public repository").
2. **Name**: `t2g-cluster-driver` (o simile).
3. **Runtime**: Python 3.
4. **Build Command**:
   ```
   pip install -r remote/requirements.txt
   ```
5. **Start Command**:
   ```
   cd remote && uvicorn app:app --host 0.0.0.0 --port $PORT
   ```

   (alternativa senza `cd`: `uvicorn remote.app:app --host 0.0.0.0 --port $PORT`)
6. **Instance Type**: Free.
7. **Environment Variables** (sezione 4).
8. **Create Web Service**. Su free tier l'istanza va in sleep dopo ~15 min
   di inattività: cronjob.org la riattiva con la POST successiva (primo
   tick "freddo" un po' più lento; usa un timeout generoso, es. 90 s).

### 3.4. Varie (opzionali)

- **Health check**: aggiungi `GET /` (o `/docs`) alla "Health Check Path"
  di Render con percorso `/` — utile per vedere il servizio come Healthy.
- Il servizio ascolta su `0.0.0.0:$PORT` (standard Render).

---

## 4. Env vars (Render)

| Variabile                   | Obbligatoria  | Default                   | Descrizione                                                                                                        |
| --------------------------- | ------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `T2G_AUTH_TOKEN`          | **sì** | —                        | Token API richiesto da ogni chiamata (header`X-Auth-Token`). Genera: `openssl rand -hex 32`                    |
| `T2G_SSH_USER`            | sì           | —                        | Utente sul cluster (codice fiscale)                                                                                |
| `T2G_SSH_HOST`            | no            | `gcluster.dmi.unict.it` | Host del login node                                                                                                |
| `T2G_SSH_PORT`            | no            | `22`                    | Porta ssh                                                                                                          |
| `T2G_SSH_KEY_CONTENT`     | (1)           | —                        | Chiave privata**intera** (righe separate da `\n` letterali) — scritta su file all'avvio con permessi 0600 |
| `T2G_SSH_KEY_FILE`        | (1)           | `data/ssh_key`          | Path della chiave (alternativa: chiave montata / già presente)                                                    |
| `T2G_SSH_TIMEOUT`         | no            | `30`                    | Timeout (s) per ogni comando ssh                                                                                   |
| `T2G_SSH_KNOWN_HOSTS`     | no            | `data/known_hosts`      | File known_hosts locale (scrivibile)                                                                               |
| `T2G_DB_PATH`             | no            | `data/t2g_driver.db`    | Path del DB SQLite (cache/diario)                                                                                  |
| `T2G_DATA_DIR`            | no            | `data/`                 | Directory dati del servizio                                                                                        |
| `T2G_HELPER_AUTO_INSTALL` | no            | `1`                     | `1` = copia `cluster_helper.sh` sul cluster via scp se manca                                                   |

(1) **Opzionale** (per il deploy locale NON serve alcuna chiave): imposta
`T2G_SSH_KEY_CONTENT` (chiave incollata nella UI di Render — su Render free
tier non esistono volumi/mount) *oppure* `T2G_SSH_KEY_FILE` (file montato /
già presente). Se NON imposti nessuna delle due, `app.py` non passa `-i` a
ssh/scp e usa l'autenticazione di default dell'utente (ssh-agent / identità
default / `~/.ssh/config`) — il caso del test locale (sezione 2).

**Per incollare la chiave privata su Render** (con `\n` letterali):

```bash
# genera la stringa da incollare
awk '{printf "%s\\n", $0}' ~/.ssh/t2g_driver
```

**Mai** mettere la chiave privata né il token in file del repo: entrambi
vivono solo nelle env vars di Render.

---

## 5. Setup cronjob.org

1. Vai su [cron-job.org](https://cron-job.org) → **Create cronjob**.
2. **URL**:
   ```
   https://t2g-cluster-driver.onrender.com/tick
   ```

   (sostituisci con l'URL reale del tuo servizio Render).
3. **Method**: `POST`.
4. **Custom HTTP Headers**: aggiungi
   `X-Auth-Token: <il tuo T2G_AUTH_TOKEN>`.
5. **Schedule**: ogni 5 minuti (`*/5 * * * *`).
6. **Request Timeout**: generoso (90–120 s), per assorbire il cold-start
   free tier.
7. Salva e attiva. Il job log di cron-job.org ti mostra lo stato della POST
   (201/200 = tick riuscito, 502 = cluster irraggiungibile, 401 = token
   errato).

---

## 6. API REST

Tutte le route richiedono l'header `X-Auth-Token`. Base URL: il tuo servizio
Render (`...onrender.com`).

| Metodo & path          | Descrizione                                                                                                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `GET /`              | info servizio (niente auth)                                                                                                                                        |
| `GET /status`        | `{active_job, queue, last_job, stopped, watcher_alive, errors_recent, last_tick_at, cluster_reachable, events}` — dalla cache DB, funziona anche a cluster giù |
| `GET /jobs`          | lista job in coda (dal DB sincronizzato)                                                                                                                           |
| `POST /jobs`         | accoda`{type: "train"\|"eval", config: "<nome o path>", tag?, mode?}` → 201                                                                                      |
| `POST /queue`        | rimpiazza la coda:`{jobs: [...]}` oppure `{ablation: true}` (12 config nell'ordine esatto di `run_all.sh`)                                                   |
| `DELETE /jobs/{tag}` | rimuove tutti i job col tag dato (riscrive`job_chain` filtrato)                                                                                                  |
| `POST /pause`        | crea`chain_stopped` sul cluster                                                                                                                                  |
| `POST /resume`       | rimuove`chain_stopped` + tick immediato                                                                                                                          |
| `POST /tick`         | tick manuale (comodo per testare senza cronjob.org)                                                                                                                |

Esempi:

```bash
TOKEN="..."  BASE="https://t2g-cluster-driver.onrender.com"

# stato
curl -H "X-Auth-Token: $TOKEN" $BASE/status

# accoda un training
curl -X POST -H "X-Auth-Token: $TOKEN" -H "Content-Type: application/json" \
     -d '{"type":"train","config":"grpo_qwen05","tag":"my-run"}' $BASE/jobs

# accoda con retry-esplicito (extra = --resume)
curl -X POST -H "X-Auth-Token: $TOKEN" -H "Content-Type: application/json" \
     -d '{"type":"train","config":"grpo_optimal","mode":"--resume"}' $BASE/jobs

# ablation completa (stesso ordine di run_all.sh --ablation)
curl -X POST -H "X-Auth-Token: $TOKEN" -H "Content-Type: application/json" \
     -d '{"ablation": true}' $BASE/queue

# pausa / riprendi
curl -X POST -H "X-Auth-Token: $TOKEN" $BASE/pause
curl -X POST -H "X-Auth-Token: $TOKEN" $BASE/resume

# tick manuale
curl -X POST -H "X-Auth-Token: $TOKEN" $BASE/tick
```

### Config validi per la coda

I 12 nomi noti (l'API accetta anche il path o il solo nome file):

```
grpo_optimal, grpo_qwen05, sft, grpo_experimental_all,
zero_shot, zero_shot_grammar, grpo_no_grammar, grpo_no_sft,
grpo_pda, grpo_pda_lookahead, grpo_soft_viterbi, grpo_verifier_scaled
```

Il tag è obbligatorio per distinguere i run e per `DELETE /jobs/{tag}`: se
non lo passi, viene derivato dal nome del config (`_` → `-`, come `run_all.sh`).

### Note su pause/resume

`POST /pause` è la versione **soft** (`touch chain_stopped`): il job SLURM
attivo continua, semplicemente non vengono sottomessi nuovi job. Il
`chain-stop` completo (che cancella il job attivo e salva lo stato per
`chain-start`) resta un'operazione da fare sul cluster con gli alias.

---

## 7. Filesystem effimero e DB

Render free tier **non offre volumi persistenti**: ogni deploy/restart azzera
il filesystem. Conseguenze (documentate per design):

- Il DB SQLite (`t2g_driver.db`) è una **cache + diario**: la fonte di verità
  è sempre `.chain_state/` sul cluster, riletto a ogni tick.
- Il puntatore `errors_offset` evita di riscrivere gli errori già visti nei
  tick precedenti; dopo un restart riparte da zero (gli errori del cluster
  vengono comunque riletti → al massimo qualche evento duplicato, innocuo).
- La chiave SSH scritta da `T2G_SSH_KEY_CONTENT` viene rigenerata a ogni
  avvio (se il file non esiste già).

---

## 8. Troubleshooting

| Sintomo                                                              | Causa                                                                                                                                                                                                                       | Soluzione                                                                                                                                                                                                                                                                                                                                                       |
| -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET /status` → `cluster_reachable: false`                      | ssh fallita all'ultimo tick                                                                                                                                                                                                 | verifica env vars, chiave, e`BatchMode`; guarda gli `events` recenti in `/status`                                                                                                                                                                                                                                                                         |
| `POST /tick` → **502** con dettaglio                        | cluster giù / timeout / protocollo                                                                                                                                                                                         | il dettaglio non espone segreti; controlla che il cluster sia raggiungibile                                                                                                                                                                                                                                                                                     |
| **401** su ogni chiamata                                       | token errato                                                                                                                                                                                                                | confronta`X-Auth-Token` con `T2G_AUTH_TOKEN` su Render                                                                                                                                                                                                                                                                                                      |
| **503** su ogni chiamata                                       | `T2G_AUTH_TOKEN` non impostato                                                                                                                                                                                            | imposta la env var (sicurezza by-default)                                                                                                                                                                                                                                                                                                                       |
| **502** "cluster_helper.sh non presente"                       | helper non copiato e auto-install disattivato                                                                                                                                                                               | `scp remote/cluster_helper.sh <user>@gcluster:~/neuro_symbolic_t2g/cluster/` oppure imposta `T2G_HELPER_AUTO_INSTALL=1`                                                                                                                                                                                                                                     |
| "Host key verification failed"                                       | known_hosts non scrivibile                                                                                                                                                                                                  | assicurati che`T2G_SSH_KNOWN_HOSTS` punti sotto `T2G_DATA_DIR` (scrivibile)                                                                                                                                                                                                                                                                                 |
| Primo tick dopo sleep molto lento                                    | cold-start free tier                                                                                                                                                                                                        | timeout generoso (90–120 s) su cron-job.org                                                                                                                                                                                                                                                                                                                    |
| La catena non avanza                                                 | QoS (1 job attivo), coda vuota, o`chain_stopped`                                                                                                                                                                          | `GET /status` mostra `stopped` e `queue`; se `stopped: true` fai `POST /resume`                                                                                                                                                                                                                                                                       |
| Tick log di cron-job.org mostra**201/200** ma nessun job parte | coda vuota (catena completata)                                                                                                                                                                                              | è il comportamento corretto: il tick è idempotente e innocuo                                                                                                                                                                                                                                                                                                  |
| Doppio submit dello stesso job                                       | watcher fallback (`chain_next.sh`) attivo CONTEMPORANEAMENTE al tick del servizio: il watcher NON usa il flock di `chain_tick.sh` (usa solo il proprio PID guard) → in race possono sottomettere 2 job o perdere entry | quando il servizio esterno è il driver PRIMARIO, ferma il watcher sul cluster:`watcher-kill` (o `chain-stop`) e controlla `GET /status` → `watcher_alive: false`. Se vuoi lasciarlo attivo, il rischio è un doppio submit (QoS permette comunque 1 solo job: il secondo resta pending o viene rifiutato da sbatch, ma la coda può saltare un'entry) |

---

## 9. Client TUI locale

Client **TUI** (app Textual, nessun REPL) per pilotare il driver da remoto
direttamente dal terminale, Windows/pwsh incluso: dashboard di stato, coda
job, accodamento, rimpiazzo dell'intera coda (incl. ablation completa),
pause/resume e tick manuale.

### Requisiti

```powershell
uv run --extra tui python remote/tui.py [--url URL] [--token TOKEN]
```

L'extra `tui` (`textual` + `httpx`) è **autosufficiente**: non serve attivare
`dev`. I flag `--url`/`--token` hanno la precedenza più alta e non scrivono
nulla su disco (utili per lanci temporanei).

### Prima configurazione

Ordine di risoluzione di URL e token:

1. flag CLI `--url` / `--token`
2. env vars `T2G_SERVICE_URL` / `T2G_AUTH_TOKEN`
3. file `.env` (cwd o repo root)
4. se mancano entrambi → **schermata di configurazione** all'avvio, che li
   salva nel `.env` locale dentro la sezione marcata

   ```
   # >>> t2g-tui >>>
   T2G_SERVICE_URL="https://t2g-cluster-driver.onrender.com"
   T2G_AUTH_TOKEN="..."
   ```

   La sezione è idempotente: al rilancio viene riscritta per intero, il resto
   del file `.env` resta intatto.

Il token **non viene mai** stampato né loggato: nel form di configurazione è
inserito in un campo mascherato (`Input(password=True)`) e nel `.env` esiste
solo come riga `T2G_AUTH_TOKEN=...`.

### Mappa tasti

| Tasto   | Schermata         | Azione                                                        |
| ------- | ----------------- | ------------------------------------------------------------- |
| `r`   | Dashboard / Queue | refresh manuale                                               |
| `g`   | Dashboard         | apri la coda                                                  |
| `a`   | Dashboard / Queue | apri il form "aggiungi job"                                   |
| `w`   | Dashboard         | apri "rimpiazza coda"                                         |
| `p`   | Dashboard         | `POST /pause` (soft stop: nessuna nuova sottomissione)      |
| `R`   | Dashboard         | `POST /resume` (rimuove `chain_stopped` + tick immediato) |
| `t`   | Dashboard         | `POST /tick` manuale (spinner durante la chiamata)          |
| `d`   | Queue             | cancella per tag (con conferma)                               |
| `Esc` | Queue / form      | torna alla dashboard                                          |
| `q`   | ovunque           | esci                                                          |

### Schermate

- **Dashboard** (principale, auto-refresh ogni 10s): header col nome del
  servizio; pannello stato con cluster **raggiungibile** (verde) o
  **irraggiungibile** (rosso + banner giallo con l'ultimo stato noto dalla
  cache del servizio), stato catena (attivo/in pausa), watcher, ultimo tick,
  ultimo job, job attivo e contatore coda; pannello **errori recenti** (ultimi
  5, in rosso) ed **eventi recenti** (ultimi 8, colorati per tipo).
- **Queue**: `DataTable` con posizione, tipo, config (basename) e tag di ogni
  job; `d` cancella per tag (con conferma), `a` apre il form.
- **Add job**: form con Select tipo (`train`/`eval`), Select config (i 12 nomi
  noti) e Input tag opzionale, precompilato col default derivato dal config
  (`_` → `-`, stessa regola del driver).
- **Replace queue**: due azioni — "Ablation completa (12 config → 22 job)" e
  "custom" (una `tipo:config[:tag]` per riga; le righe che iniziano con `#`
  sono ignorate). Entrambe chiedono conferma e avvisano che la coda esistente
  viene **SOSTITUITA**.
- **Config** (solo al primo avvio senza env/.env): URL + token, salvati nel
  `.env` locale.

Gli esiti arrivano come toast Textual: **verde** per le operazioni riuscite,
**rosso** con dettaglio per gli errori.

### Troubleshooting TUI

| Sintomo                                           | Causa                                                   | Soluzione                                                                                 |
| ------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| toast rosso "Token non valido (401)"              | `T2G_AUTH_TOKEN` errato                               | aggiornalo su Render e nel`.env` locale (o via `--token`)                             |
| toast rosso "Impossibile connettersi"             | URL sbagliato o servizio spento                         | verifica`T2G_SERVICE_URL`; su free tier Render va in sleep dopo ~15 min di inattività  |
| prima chiamata lenta (30–50s)                    | cold start free tier dopo lo sleep                      | è normale: la prima GET /status riflette il ritardo; il POST /tick usa un timeout di 90s |
| `cluster_reachable: false` senza toast d'errore | il cluster non risponde ma il**servizio** è vivo | guarda gli eventi recenti in`/status` per il dettaglio ssh                              |

---

## 10. Sicurezza — riepilogo

- **Nessun log** di token o chiave: l'app non stampa mai `T2G_AUTH_TOKEN`,
  `T2G_SSH_KEY_CONTENT` né contenuti della chiave.
- Chiave **dedicata** senza passphrase su gcluster, revocabile da
  `authorized_keys` senza impattare l'accesso personale.
- Ogni route (tranne `GET /`) richiede `X-Auth-Token`, confrontato con
  `secrets.compare_digest` (constant-time).
- Binding `0.0.0.0:$PORT` (standard Render); il TLS lo termina Render.
- Se una chiave/token trapela: ruota `T2G_AUTH_TOKEN` su Render e rimuovi la
  riga del driver da `~/.ssh/authorized_keys` sul cluster.
