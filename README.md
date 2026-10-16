# AI Goods Receipt Assistant V7

This version adds conditional document requirements:
- MRC / Recurring Service workflow
- First vs Subsequent payment stage
- PO + II + Invoice as the core MRC set
- Optional DO / Acceptance document
- AI decides whether supplied/missing evidence is applicable, with policy kept separate in the RAG knowledge base
- II annual budget and cumulative utilisation display
- SQLite transaction database
- Finance Approve, Process GR and Return to Requester workflow
- RAG Finance assistant

## Run
1. Extract ZIP.
2. Rename `.env.example` to `.env` and add your OpenAI API key.
3. Open PowerShell in this folder.
4. `python -m venv .venv`
5. `.venv\Scripts\Activate.ps1`
6. `python -m pip install -r requirements.txt`
7. `streamlit run app.py`

Or double-click START_WINDOWS.bat after creating the `.env`.

IMPORTANT: The knowledge base policy is synthetic demo content. Replace with approved organisational policy for a real deployment. No real Finance/ERP/NFS connection is included.
