# SupportIQ: AI Customer Support Ticket Triage

Hybrid AI system that triages incoming customer support tickets. A small trained model handles
classification; Gemini handles summarisation, sentiment, and grounded response drafting via RAG.
An agent reviews every AI output, and every correction is recorded so the system can be measured.

```
customer message
   │
   ├─► TF-IDF + Logistic Regression ──► category (+ confidence) ──► team routing
   ├─► rules (keywords, repeat contact, wait time, sentiment) ──► priority (+ reasons)
   ├─► Gemini ──► summary + sentiment
   └─► Gemini embeddings + cosine search over policy docs ──► Gemini ──► draft reply with [sources] + groundedness score
                                                            │
                                               agent reviews / corrects / approves
                                                            │
                                                   feedback + audit trail
```

## Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + SQLAlchemy | typed, async-ready, auto docs |
| DB | SQLite now, PostgreSQL later | zero setup; same models migrate in an afternoon |
| Classifier | scikit-learn TF-IDF + LogReg | fast, cheap, reproducible, explainable |
| LLM | Gemini 3.5 Flash-Lite (free tier) | summary, sentiment, response drafts |
| RAG | Gemini embeddings + NumPy cosine search in the main DB | multilingual, no PyTorch, no vector DB service, fits 512 MB hosts |
| Dashboard | Streamlit | agent queue, review, metrics |

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-dev.txt
copy .env.example .env        # then paste your Gemini key from https://aistudio.google.com/apikey
```

## Data

1. (Optional) Download the Kaggle dataset: https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset
   and place `customer_support_tickets.csv` in `data/raw/`. It is excluded from training by default; see Results.
2. Generate synthetic tickets to cover `delivery` and `account` (the Kaggle set has neither) and
   add realistic variety:
   ```powershell
   python ml/generate_synthetic.py --per-category 120
   ```
3. Merge and split:
   ```powershell
   python ml/prepare_data.py
   ```

## Train and evaluate the classifier

```powershell
python ml/train_classifier.py     # saves ml/artifacts/classifier.joblib + train_metrics.json
python ml/evaluate.py             # accuracy, macro F1, routing accuracy, latency, confusion matrix
```

Until a classifier artifact exists the API falls back to a keyword heuristic, so you can run the
app before training.

## Run

```powershell
uvicorn app.main:app --reload --port 8010  # API at http://127.0.0.1:8010, docs at /docs
$env:SUPPORTIQ_API="http://127.0.0.1:8010"  # tell the dashboard where the API is
streamlit run dashboard/app.py --server.port 8501   # agent dashboard (internal)
streamlit run portal/app.py --server.port 8502      # customer Help Center (public)
```

On first start the API embeds the knowledge base (one Gemini call) and backfills ticket
embeddings in the background. A ticket takes roughly 5 to 15 seconds end to end: one Gemini
embedding call plus one Gemini triage call.

Hindi or Hinglish tickets: submit them through the dashboard or a Python client. `curl` from
Git Bash or cmd on Windows mangles Devanagari into question marks before it reaches the API.

Try the example from the spec:

```
POST /tickets
{"subject": "Order not delivered",
 "body": "I ordered a product five days ago, but it still hasn't been delivered. I have contacted support twice and received no update. Please resolve this."}
```

## Two front ends, one API

| App | Who uses it | What it can do |
|---|---|---|
| `portal/app.py` (port 8502) | Customers | Raise a ticket, get a ticket number and a promised reply time, check status with ticket number + email, read the agent's approved reply. Talks only to `/public/*`. |
| `dashboard/app.py` (port 8501) | Support agents | Queue with filters, AI analysis, similar tickets, review and approve, analytics, CSV import, logging tickets that came by phone. Talks to `/tickets/*`. |

The public endpoints return only customer-safe fields (no AI confidence, priority, team or audit
trail), require the matching email for any lookup, and answer 404 rather than 403 so ticket
numbers cannot be enumerated. An agent's reply becomes visible to the customer only after
"Approve & send". For a demo, open both apps side by side: raise a ticket on the left and watch
it appear, triaged, in the queue on the right.

## Stage 3 features

**Similar-ticket detection.** Every ticket's embedding is stored in the `embeddings` table when it is
triaged. `GET /tickets/{id}/similar` returns the closest earlier tickets with their status and the
agent's final response, so the detail page can show "5 similar, 2 already resolved" and how.
Older tickets are indexed automatically at startup.

**SLA monitoring.** Deadlines come from `data/knowledge_base/escalation_sla.md`: first response
in 2 / 8 / 24 business hours and resolution in 1 / 3 / 5 business days for high / medium / low,
with business hours 09:00 to 21:00 IST, Monday to Saturday (`app/services/sla.py`). Approving a
reply stops the first-response clock; setting status to resolved stops the resolution clock.
Correcting the priority recomputes both. Each ticket exposes `response_sla` and `resolution_sla`
(on_track / at_risk / breached / met); the queue can filter `?sla=at_risk` or `?sla=breached`.

**Analytics.** `GET /tickets/analytics?days=30`: volume per day, category by status, language mix,
average resolution hours by priority, average first-response hours, SLA compliance counts, the
most frequent complaint phrases per category (mined from the AI summaries), correction rate and
average classifier confidence. The dashboard's Analytics page charts all of it.

**Hindi and Hinglish.** Gemini detects the language, writes the summary in English and the reply in
the customer's language and script. The embedding model is multilingual so Hindi tickets retrieve
the right English policy sections. The classifier is trained on English, Hindi and Hinglish
synthetic tickets (`ml/generate_synthetic.py --language hi|hinglish`); char n-grams in the
TF-IDF handle Devanagari. When Gemini is unavailable a script-based guess sets the language.

**Schema upgrades without data loss.** `app/db.py` adds any new model column to the live SQLite
table at startup (forward-only, ADD COLUMN), and `app/services/maintenance.py` backfills SLA clocks
and the similarity index for existing tickets. Move to Alembic when you need renames or drops.

## Free-tier quota and bulk uploads

Observed on 2026-11-12 with a new key: `gemini-3.5-flash` allows about 20 requests per day on the
free tier, while `gemini-3.5-flash-lite` allows far more. The defaults are therefore Lite as the
primary model and Flash as the fallback, and the pipeline makes exactly one Gemini call per ticket
(summary, sentiment and draft reply in one JSON response). If both models are over quota the
ticket is still created with the classifier, priority rules and routing; only the summary falls
back to the truncated message and the draft reply is left empty for the agent.

CSV uploads insert all rows immediately and return at once. Triage then runs in the background,
one ticket at a time, and tickets fill in as they finish. `GET /tickets/metrics` reports
`ai_pending`, and the dashboard shows it on the Metrics page and marks untriaged rows as pending.

## Deployment

Three free services, one repository. Total cost: zero.

| Piece | Host | How |
|---|---|---|
| API | Render, free web service | New -> Blueprint -> this repo (`render.yaml`). Set `GEMINI_API_KEY` and `DATABASE_URL` when prompted. |
| PostgreSQL | Neon | Create a project, copy the connection string into `DATABASE_URL`. `postgres://` URLs are normalised. |
| Agent dashboard | Streamlit Community Cloud, main file `dashboard/app.py` | Secrets: `SUPPORTIQ_API = "https://supportiq-api.onrender.com"`, `DASHBOARD_PASSWORD = "..."` |
| Customer portal | Streamlit Community Cloud, main file `portal/app.py` | Secrets: `SUPPORTIQ_API = "https://supportiq-api.onrender.com"` |

Why it fits free tiers: the API has no PyTorch and no vector database. Embeddings come from
Gemini's embedding API and are stored as rows in PostgreSQL, searched with NumPy. The process
uses a few hundred MB. The knowledge base is embedded once at first start and the ticket
index is backfilled automatically, so nothing needs a persistent disk.

Behaviour on free tiers: Render sleeps the API after 15 idle minutes and takes 30 to 60 seconds
to wake; both Streamlit apps show a "connecting" note and retry. `ci.yml` runs the tests on push.

## Tests

```powershell
pytest
```

Tests run offline: no Gemini key, RAG disabled, temporary SQLite database.

## Project layout

```
app/
  main.py              FastAPI app
  config.py            settings (.env), category/priority/status enums
  db.py, models.py     SQLAlchemy: Ticket, Feedback (audit trail)
  schemas.py           Pydantic request/response models
  routers/tickets.py   internal API: create, list/filter, detail, agent review (PATCH), CSV upload, metrics, analytics, similar
  routers/public.py    customer API: raise ticket, status by ticket number + email, my tickets
  services/
    classifier.py      trained model (or keyword fallback)
    priority.py        explainable rules -> low/medium/high + reasons
    routing.py         category -> team
    llm.py             Gemini: analyse (summary+sentiment), draft_response
    embeddings.py      Gemini embedding API (or local Sentence Transformers)
    vectors.py         tiny vector store in the main DB, NumPy cosine search
    rag.py             knowledge-base indexing, retrieval, similar tickets, groundedness
    pipeline.py        runs everything for one ticket
ml/
  generate_synthetic.py  Gemini-generated labelled tickets
  prepare_data.py        Kaggle + synthetic -> train/test split
  train_classifier.py    TF-IDF + LogReg with grid search
  evaluate.py            held-out metrics + confusion matrix
data/
  raw/                 Kaggle CSV (not committed)
  processed/           synthetic.csv, train.csv, test.csv (not committed)
  knowledge_base/      policy markdown docs used for RAG
dashboard/app.py       Streamlit agent UI (internal)
portal/app.py          Streamlit customer Help Center (public)
tests/                 pytest (offline)
```

## Results so far (2026-11-12)

Classifier: TF-IDF (word 1-2 grams + char 3-5 grams) + logistic regression, trained on 960
synthetic tickets in English, Hindi and Hinglish, tested on 240 held out (48 per class).

| Metric | Value |
|---|---|
| Accuracy | 0.933 |
| Macro F1 | 0.934 |
| Macro F1 by language | English 0.96 · Hinglish 0.95 · Hindi 0.86 |
| Routing accuracy | 0.933 |
| Predict latency | ~6 ms |
| Weakest classes | billing (0.88) and returns (0.90), confused with each other on refund wording |

Hindi is the weakest language because 60 rows per class is thin for Devanagari character n-grams;
generating more Hindi rows is the obvious next lever. English-only training scored 0.95.

**Why the Kaggle data is excluded from training.** The first run mixed 1,600 Kaggle rows with
600 synthetic rows and scored macro F1 0.52 overall: 0.98 on synthetic test rows, 0.23 on Kaggle
test rows, where chance is 0.25. Inspection showed why: the Kaggle descriptions are generic
templates ("I'm having an issue with the {product}. Please assist.") padded with unrelated scraped
text, and the subjects do not match the labels either ("Billing inquiry" tickets titled "Software
bug"). The labels carry no signal, so `ml/prepare_data.py` now excludes that set by default.
Pass `--kaggle-per-class 400` to reproduce the noisy result.

Caveat: train and test both come from the same generator, so 0.95 is optimistic. The honest
number will come from the human correction rate once agents review real tickets.

## Evaluation metrics

| Metric | Where |
|---|---|
| Classification accuracy, macro F1, per-class F1 | `ml/evaluate.py` -> `ml/artifacts/eval_report.json` |
| Routing accuracy | `ml/evaluate.py` |
| Response groundedness | per ticket (`ai_groundedness`), averaged in `GET /tickets/metrics` |
| Latency | per ticket (`ai_latency_ms`), classifier-only in `evaluate.py` |
| Human correction rate | `GET /tickets/metrics` from the feedback table |

## Roadmap

- [x] Stage 1: ticket creation (form + CSV), classification, priority, summary, dashboard
- [x] Sentiment, response drafts, RAG over policies, routing, human feedback with audit trail
- [x] Similar-ticket detection, SLA monitoring, analytics, Hindi/Hinglish
- [ ] Retrain the classifier from agent corrections (feedback table -> training rows)
- [ ] LLM-as-judge groundedness alongside the lexical score
- [ ] Compare the 10 ms classifier against Gemini on the same test set (accuracy, cost, latency)
- [ ] Docker Compose with PostgreSQL + Alembic; deploy (Render/Railway + Streamlit Cloud)
