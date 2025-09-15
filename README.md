# Monitoraggio della reputazione online di un’azienda

---

## 📖 Presentazione e Navigazione del Software

Questa repository contiene una soluzione completa per il monitoraggio della reputazione aziendale sui social media tramite analisi automatica del sentiment, pipeline MLOps e monitoraggio delle performance. Il progetto è strutturato in moduli separati per API, training, inferenza, monitoraggio e automazione.

### Struttura delle cartelle principali

- `api/` — Backend FastAPI per esporre le API REST di inferenza e gestione.
- `model/` — Script per training, gestione e caricamento del modello di sentiment analysis.
- `docker/` — Dockerfile per l’applicazione e per Prometheus.
- `prometheus/` — Configurazioni per il monitoraggio delle metriche.
- `results/` — Output di training, metriche e risultati.
- `tests/` — Test di integrazione automatizzati.

---


## 🔄 Pipeline CI/CD e Automazione

Il progetto integra una pipeline CI/CD che automatizza le fasi di:

- **Training e retraining** del modello di sentiment analysis
- **Esecuzione dei test di integrazione** per garantire la qualità del codice
- **Deploy** dell’applicazione e del modello su HuggingFace
- **Monitoraggio continuo** delle performance tramite Prometheus

La pipeline consente di:
- Ridurre errori manuali e velocizzare il ciclo di sviluppo
- Garantire ripetibilità e tracciabilità delle operazioni
- Facilitare l’integrazione e la scalabilità del sistema

> **Nota:** La configurazione della pipeline (file YAML per GitHub Actions o altri strumenti CI/CD) è fornita nella repository.
---

## 🚀 Avvio dei Servizi

### 1. Avvio API Backend (FastAPI)

Per avviare il backend API in locale:

```powershell
uvicorn api.backend:app
```

L’API sarà disponibile su `http://127.0.0.1:8000`.

### 2. Avvio tramite Docker

Per eseguire l’applicazione in container:

```powershell
docker build -f .\docker\app_dockerfile -t machineinnovators:latest .
docker run -p 8000:8000 machineinnovators:latest
```

### 3. Avvio di Prometheus per il Monitoraggio

Costruisci e avvia Prometheus con la configurazione custom:

```powershell
docker build -f docker/prometheus_dockerfile -t prometheus-custom .
docker run -p 9090:9090 -v ${PWD}/prometheus:/etc/prometheus prometheus-custom
```

Prometheus sarà accessibile su `http://localhost:9090`.

---


## 🛠️ Uso delle API

Le API sono documentate automaticamente da FastAPI su `/docs` (Swagger UI) e `/redoc`.

### Endpoint Principali

- `POST /inference` — Predice il sentiment di un testo fornendo la stringa e il modello.
- `POST /performance` — Calcola le metriche di performance del modello sul dataset di test.
- `GET /metrics` — Espone le metriche per Prometheus.
- `GET /health` — Verifica lo stato di salute dell’API.

### Esempi di chiamata API

#### 1. Predizione del Sentiment

**Richiesta:**
```bash
curl -X POST "http://127.0.0.1:8000/inference" \
     -H "Content-Type: application/json" \
     -d '{"input_string": "Ottimo servizio!", "model": "infinitydreams/roberta-base-sentiment-finetuned"}'
```

**Risposta:**
```json
{
  "label": "positive",
  "score": 0.98
}
```

#### 2. Calcolo delle metriche di performance

**Richiesta:**
```bash
curl -X POST "http://127.0.0.1:8000/performance"
```

**Risposta:**
```json
{
  "accuracy": 0.92,
  "precision": 0.91,
  "recall": 0.92,
  "f1": 0.91
}
```

#### 3. Esposizione metriche Prometheus

**Richiesta:**
```bash
curl http://127.0.0.1:8000/metrics
```

**Risposta:**
Output in formato Prometheus (testuale, per sistemi di monitoraggio).

#### 4. Health check

**Richiesta:**
```bash
curl http://127.0.0.1:8000/health
```

**Risposta:**
```json
{
  "status": "ok"
}
```

---

## 🔁 Pipeline di Training e Retraining

Per addestrare o riaddestrare il modello:

```powershell
python model/train.py
```

I risultati e le metriche saranno salvati nella cartella `results/`.

---

## 📊 Monitoraggio con Prometheus

Le metriche esposte dall’API (`/metrics`) possono essere raccolte da Prometheus per monitorare:
- Numero di richieste
- Tempi di risposta
- Performance del modello

La configurazione di Prometheus si trova in `prometheus/prometheus.yml`.

---

## 🧪 Prove in Autonomia

1. **Avvia il backend API** come descritto sopra.
2. **Testa le API** tramite Swagger UI (`http://127.0.0.1:8000/docs`) oppure con strumenti come `curl` o Postman.
3. **Avvia Prometheus** e verifica la raccolta delle metriche su `http://localhost:9090`.
4. **Esegui i test di integrazione**:
   ```powershell
   python tests/integration.py
   ```

---

## 📚 Risorse Utili

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Prometheus Docs](https://prometheus.io/docs/)
- [Swagger UI](http://127.0.0.1:8000/docs)

Per ulteriori dettagli, consultare i commenti nei singoli file e la documentazione inline.

**MachineInnovators Inc.** è leader nello sviluppo di applicazioni di machine learning scalabili e pronte per la produzione. Il focus principale del progetto è integrare metodologie **MLOps** per facilitare lo sviluppo, l'implementazione, il monitoraggio continuo e il retraining dei modelli di analisi del sentiment. L'obiettivo è abilitare l'azienda a migliorare e monitorare la reputazione sui social media attraverso l'analisi automatica dei sentiment.

Le aziende si trovano spesso a fronteggiare la sfida di gestire e migliorare la propria reputazione sui social media in modo efficace e tempestivo. Monitorare manualmente i sentiment degli utenti può essere inefficiente e soggetto a errori umani, mentre la necessità di rispondere rapidamente ai cambiamenti nel sentiment degli utenti è cruciale per mantenere un'immagine positiva dell'azienda.

---

## Benefici della Soluzione

- **Automazione dell'Analisi del Sentiment:**
  - Implementando un modello di analisi del sentiment basato su FastText, MachineInnovators Inc. automatizzerà l'elaborazione dei dati dai social media per identificare sentiment positivi, neutrali e negativi. Ciò permetterà una risposta rapida e mirata ai feedback degli utenti.
- **Monitoraggio Continuo della Reputazione:**
  - Utilizzando metodologie MLOps, l'azienda implementerà un sistema di monitoraggio continuo per valutare l'andamento del sentiment degli utenti nel tempo. Questo consentirà di rilevare rapidamente cambiamenti nella percezione dell'azienda e di intervenire prontamente se necessario.
- **Retraining del Modello:**
  - Introdurre un sistema di retraining automatico per il modello di analisi del sentiment assicurerà che l'algoritmo si adatti dinamicamente ai nuovi dati e alle variazioni nel linguaggio e nei comportamenti degli utenti sui social media. Mantenere alta l'accuratezza predittiva del modello è essenziale per una valutazione corretta del sentiment.

---

## Dettagli del Progetto

### Fase 1: Implementazione del Modello di Analisi del Sentiment con FastText
- **Modello:** Utilizzare un modello pre-addestrato FastText per un’analisi del sentiment in grado di classificare testi dai social media in sentiment positivo, neutro o negativo. Servirsi di questo modello: [twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- **Dataset:** Utilizzare dataset pubblici contenenti testi e le rispettive etichette di sentiment.

### Fase 2: Creazione della Pipeline CI/CD
- **Pipeline CI/CD:** Sviluppare una pipeline automatizzata per il training del modello, i test di integrazione e il deploy dell'applicazione su HuggingFace.

### Fase 3: Deploy e Monitoraggio Continuo
- **Deploy su HuggingFace (facoltativo):** Implementare il modello di analisi del sentiment, inclusi dati e applicazione, su HuggingFace per facilitare l'integrazione e la scalabilità.
- **Sistema di Monitoraggio:** Configurare un sistema di monitoraggio per valutare continuamente le performance del modello e il sentiment rilevato.

---

## Consegna

- **Codice Sorgente:** Repository pubblica su GitHub con codice ben documentato per la pipeline CI/CD e l'implementazione del modello. La consegna vera e propria dovrà avvenire mediante un notebook Google Colab con al suo interno il link al repository GitHub.
- **Documentazione:** Descrizione delle scelte progettuali, delle implementazioni e dei risultati ottenuti durante il progetto.

---

## Motivazione del Progetto

L'implementazione di FastText per l'analisi del sentiment consente a MachineInnovators Inc. di migliorare significativamente la gestione della reputazione sui social media. Automatizzando l'analisi del sentiment, l'azienda potrà rispondere più rapidamente alle esigenze degli utenti, migliorando la soddisfazione e rafforzando l'immagine dell'azienda sul mercato.

Con questo progetto, MachineInnovators Inc. promuove l'innovazione nel campo delle tecnologie AI, offrendo soluzioni avanzate e scalabili per le sfide moderne di gestione della reputazione aziendale.
