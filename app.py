
import os, json, sqlite3, uuid, base64, re
from datetime import datetime
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
DB_PATH = "gr_transactions.db"
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

st.set_page_config(page_title="AI Goods Receipt Assistant", page_icon="🤖", layout="wide")

# ---------------- DB ----------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id TEXT PRIMARY KEY,
        reference TEXT,
        gr_type TEXT,
        payment_stage TEXT,
        supplier TEXT,
        po_number TEXT,
        ii_number TEXT,
        invoice_number TEXT,
        invoice_amount REAL,
        ii_annual_budget REAL,
        prior_utilisation REAL,
        projected_utilisation REAL,
        ai_status TEXT,
        finance_status TEXT,
        summary TEXT,
        exception_text TEXT,
        budget_status TEXT,
        created_at TEXT,
        updated_at TEXT,
        gr_reference TEXT
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT,
        document_type TEXT,
        filename TEXT,
        uploaded_at TEXT
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT,
        action TEXT,
        comments TEXT,
        actor TEXT,
        acted_at TEXT,
        gr_reference TEXT
    )""")
    conn.commit()
    conn.close()

init_db()

def money(v):
    if v is None or v == "":
        return "—"
    try: return f"${float(v):,.2f}"
    except Exception: return str(v)

# ---------------- AI helpers ----------------
def require_key():
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not configured in .env.")
        st.stop()

def clean_json(txt):
    txt = txt.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.I)
        txt = re.sub(r"\s*```$", "", txt)
    return json.loads(txt)

def extract_document(uploaded_file, requested_type):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    b64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    schema = {
        "document_type": "string",
        "supplier_name": "string|null",
        "po_number": "string|null",
        "invoice_number": "string|null",
        "ii_number": "string|null",
        "document_date": "string|null",
        "financial_year": "string|null",
        "billing_period": "string|null",
        "currency": "string|null",
        "net_amount": "number|null",
        "gst_amount": "number|null",
        "gross_amount": "number|null",
        "annual_ii_amount": "number|null",
        "service_description": "string|null",
        "line_items": [
            {"description":"string","quantity":"number|null","unit_price":"number|null","amount":"number|null"}
        ],
        "acceptance_evidence": "string|null",
        "notes": "string|null"
    }
    prompt = f"""
You extract structured facts for a Goods Receipt system.
The requested role of this file is {requested_type}. Read the full PDF, including scanned pages.
Return ONLY JSON matching this schema:
{json.dumps(schema, indent=2)}
Rules:
- Never invent information. Use null if not found.
- Preserve document/reference numbers exactly.
- For an Invoice Instruction (II), only populate annual_ii_amount if an annual approved/allocation/budget amount is explicitly stated.
- For Delivery/Acceptance, describe visible signature/acknowledgement evidence in acceptance_evidence.
- Capture billing period where shown on an invoice.
"""
    response = client.responses.create(
        model=MODEL,
        input=[{
            "role":"user",
            "content":[
                {"type":"input_text","text":prompt},
                {"type":"input_file","filename":uploaded_file.name,
                 "file_data":f"data:application/pdf;base64,{b64}"}
            ]
        }]
    )
    return clean_json(response.output_text)

def analyse_docs(gr_type, payment_stage, po, ii, inv, do):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    docs = {"PO":po, "Invoice Instruction":ii, "Invoice":inv, "Delivery/Acceptance":do}
    prompt = f"""
You are the validation engine for an AI-assisted Goods Receipt workflow.

Transaction type: {gr_type}
Payment stage: {payment_stage}

The business goal is to avoid unnecessarily requiring documents while ensuring applicable evidence is checked.
For MRC/Recurring Service:
- PO + II + Invoice form the core financial validation set.
- Delivery/Acceptance is OPTIONAL unless the available policy/evidence indicates it is applicable, especially for first payment.
For Physical Goods:
- DO/acceptance is usually relevant, but still distinguish missing evidence from a document that is not required by the selected workflow.
Never invent a requirement. Use the supplied policy context separately if available.

Return ONLY JSON:
{{
 "overall_status":"READY|REVIEW|BLOCKED",
 "summary":"string",
 "exceptions":["string"],
 "required_documents":[
   {{"document":"PO|Invoice Instruction|Invoice|Delivery/Acceptance","status":"REQUIRED|OPTIONAL|NOT_REQUIRED|PRESENT|MISSING","reason":"string"}}
 ],
 "checks":[
   {{"check":"string","status":"PASS|REVIEW|FAIL|NOT_AVAILABLE","finding":"string"}}
 ],
 "budget_check": {{
   "status":"WITHIN_BUDGET|VARIANCE|NOT_AVAILABLE",
   "ii_amount":"number|null",
   "current_invoice_amount":"number|null",
   "prior_utilisation":"number|null",
   "projected_after":"number|null",
   "variance":"number|null",
   "explanation":"string"
 }}
}}

Important:
1. Compare PO/II/Invoice identifiers, supplier, service description, invoice amounts and billing period where available.
2. If an II annual amount is present and prior utilisation is not supplied, set prior_utilisation to 0 only as a clearly labelled first-invoice/demo calculation.
3. Do NOT treat the absence of a DO as an exception for MRC unless the transaction or policy makes acceptance evidence applicable.
4. If the MRC is a first payment and no acceptance evidence exists, flag this for review only when the supplied policy context or user-selected option says it is required.
5. Finance remains the final decision maker.

DOCUMENTS:
{json.dumps(docs, indent=2)}
"""
    response = client.responses.create(model=MODEL, input=prompt)
    return clean_json(response.output_text)

def rag_answer(question):
    from openai import OpenAI
    kb = []
    for p in Path("knowledge_base").glob("*.txt"):
        kb.append({"source":p.name, "content":p.read_text(encoding="utf-8")})
    if not kb:
        return {"answer":"No knowledge-base documents have been added.","sources":[]}
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = f"""
Answer the user's question using ONLY the supplied knowledge base.
Do not invent a policy.
If the policy is silent, say it is silent.
Cite the source filename.

Question:
{question}

Knowledge base:
{json.dumps(kb, indent=2)}
"""
    response = client.responses.create(model=MODEL, input=prompt)
    return {"answer":response.output_text, "sources":[x["source"] for x in kb]}

# ---------------- persistence ----------------
def save_tx(gr_type, stage, po, ii, inv, result):
    b = result.get("budget_check", {})
    txid = str(uuid.uuid4())
    reference = po.get("po_number") or inv.get("invoice_number") or ii.get("ii_number") or txid[:8]
    conn = get_db()
    conn.execute("""INSERT INTO transactions
    (id,reference,gr_type,payment_stage,supplier,po_number,ii_number,invoice_number,
     invoice_amount,ii_annual_budget,prior_utilisation,projected_utilisation,ai_status,
     finance_status,summary,exception_text,budget_status,created_at,updated_at,gr_reference)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
        txid, reference, gr_type, stage,
        po.get("supplier_name") or inv.get("supplier_name") or "Not found",
        po.get("po_number") or inv.get("po_number"),
        ii.get("ii_number"),
        inv.get("invoice_number"),
        inv.get("gross_amount") or inv.get("net_amount"),
        b.get("ii_amount") or ii.get("annual_ii_amount"),
        b.get("prior_utilisation"),
        b.get("projected_after"),
        result.get("overall_status","REVIEW"),
        "PENDING",
        result.get("summary",""),
        "\n".join(result.get("exceptions",[])),
        b.get("status","NOT_AVAILABLE"),
        datetime.now().isoformat(timespec="seconds"),
        datetime.now().isoformat(timespec="seconds"),
        None))
    conn.commit(); conn.close()
    return txid

def save_docs(txid, files):
    conn = get_db()
    for f, dtype in files:
        if f:
            conn.execute("INSERT INTO documents(transaction_id,document_type,filename,uploaded_at) VALUES (?,?,?,?)",
                         (txid,dtype,f.name,datetime.now().isoformat(timespec="seconds")))
    conn.commit(); conn.close()

def add_action(txid, action, comments="", gr_reference=None):
    conn=get_db()
    conn.execute("INSERT INTO actions(transaction_id,action,comments,actor,acted_at,gr_reference) VALUES (?,?,?,?,?,?)",
                 (txid,action,comments,"Finance Demo User",datetime.now().isoformat(timespec="seconds"),gr_reference))
    conn.commit(); conn.close()

def approve(txid, comments=""):
    conn=get_db()
    conn.execute("UPDATE transactions SET finance_status='APPROVED', updated_at=? WHERE id=?",
                 (datetime.now().isoformat(timespec="seconds"),txid))
    conn.commit(); conn.close()
    add_action(txid,"APPROVE_GR",comments)

def process(txid, comments=""):
    conn=get_db()
    gr = "GR-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    conn.execute("UPDATE transactions SET finance_status='PROCESSED',gr_reference=?,updated_at=? WHERE id=?",
                 (gr,datetime.now().isoformat(timespec="seconds"),txid))
    conn.commit(); conn.close()
    add_action(txid,"PROCESS_GR",comments,gr)
    return gr

def return_tx(txid, reason):
    conn=get_db()
    conn.execute("UPDATE transactions SET finance_status='RETURNED_TO_REQUESTER',updated_at=? WHERE id=?",
                 (datetime.now().isoformat(timespec="seconds"),txid))
    conn.commit(); conn.close()
    add_action(txid,"RETURN_TO_REQUESTER",reason)

# ---------------- UI ----------------
st.title("🤖 AI Goods Receipt Assistant")
st.caption("Capstone prototype — document AI + validation + II/budget check + RAG + human Finance approval")

requester, finance = st.tabs(["Requester / Create GR","Finance Review"])

with requester:
    st.subheader("1. Start Goods Receipt")
    gr_type = st.radio("Goods Receipt Type", ["MRC / Recurring Service","Physical Goods","One-off Service"], horizontal=True)
    stage = st.radio("Payment Stage", ["First Payment","Subsequent / Recurring Payment"], horizontal=True)

    st.markdown("### Upload supporting documents")
    c1,c2,c3,c4 = st.columns(4)
    with c1: po_file = st.file_uploader("Purchase Order", type=["pdf"], key="po")
    with c2: ii_file = st.file_uploader("Invoice Instruction (II)", type=["pdf"], key="ii")
    with c3: inv_file = st.file_uploader("Invoice", type=["pdf"], key="inv")
    with c4: do_file = st.file_uploader("DO / Acceptance (optional)", type=["pdf"], key="do")

    if gr_type == "MRC / Recurring Service":
        st.info("For MRC, PO + II + Invoice are the core documents. DO / acceptance can be supplied when applicable, including where first-payment acceptance evidence is required.")
    elif gr_type == "Physical Goods":
        st.info("For physical goods, DO / acceptance evidence is normally more relevant. The AI will distinguish applicable requirements from missing documents.")
    else:
        st.info("For one-off services, acceptance evidence may be relevant depending on the applicable procedure.")

    if st.button("🔍 Analyse & Prepare GR", type="primary", use_container_width=True):
        require_key()
        if not any([po_file,ii_file,inv_file,do_file]):
            st.warning("Upload at least one document.")
        else:
            try:
                with st.spinner("AI is reading the documents and checking the transaction..."):
                    po = extract_document(po_file,"Purchase Order") if po_file else {}
                    ii = extract_document(ii_file,"Invoice Instruction") if ii_file else {}
                    inv = extract_document(inv_file,"Invoice") if inv_file else {}
                    do = extract_document(do_file,"Delivery / Acceptance") if do_file else {}
                    result = analyse_docs(gr_type, stage, po, ii, inv, do)
                    txid = save_tx(gr_type, stage, po, ii, inv, result)
                    save_docs(txid,[(po_file,"PO"),(ii_file,"Invoice Instruction"),(inv_file,"Invoice"),(do_file,"Delivery/Acceptance")])
                    st.session_state["latest"] = {"txid":txid,"po":po,"ii":ii,"inv":inv,"do":do,"result":result,"gr_type":gr_type,"stage":stage}
                st.success("Analysis complete. The transaction has been added to Finance Review.")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    latest = st.session_state.get("latest")
    if latest:
        r=latest["result"]; po=latest["po"]; ii=latest["ii"]; inv=latest["inv"]
        st.divider()
        status=r.get("overall_status","REVIEW")
        if status=="READY": st.success("🟢 READY FOR FINANCE REVIEW")
        elif status=="REVIEW": st.warning("🟠 REVIEW REQUIRED")
        else: st.error("🔴 BLOCKED / INCOMPLETE")
        st.write(r.get("summary",""))

        st.markdown("### Document applicability")
        import pandas as pd
        st.dataframe(pd.DataFrame(r.get("required_documents",[])), use_container_width=True, hide_index=True)

        st.markdown("### Extracted information")
        st.dataframe(pd.DataFrame([
            ["GR Type",latest["gr_type"],"Payment Stage",latest["stage"]],
            ["Supplier",po.get("supplier_name") or inv.get("supplier_name"),"PO",po.get("po_number") or "—"],
            ["II",ii.get("ii_number") or "—","Invoice",inv.get("invoice_number") or "—"],
            ["Invoice Net",money(inv.get("net_amount")),"Invoice Gross",money(inv.get("gross_amount"))],
            ["Service",inv.get("service_description") or "—","Billing Period",inv.get("billing_period") or "—"],
        ],columns=["Field","Value","Field 2","Value 2"]),use_container_width=True,hide_index=True)

        for x in r.get("exceptions",[]): st.warning("⚠️ "+x)

        st.markdown("### 📊 II / Annual Budget Check")
        b=r.get("budget_check",{})
        m1,m2,m3,m4=st.columns(4)
        m1.metric("II Annual Budget",money(b.get("ii_amount") or ii.get("annual_ii_amount")))
        m2.metric("Prior Utilisation",money(b.get("prior_utilisation")))
        m3.metric("Current Invoice",money(b.get("current_invoice_amount") or inv.get("gross_amount") or inv.get("net_amount")))
        m4.metric("Projected After",money(b.get("projected_after")))
        if b.get("status")=="WITHIN_BUDGET": st.success("🟢 Within approved II budget")
        elif b.get("status")=="VARIANCE": st.warning("🟠 II / budget review required")
        else: st.info("Budget could not be determined from the supplied documents.")

with finance:
    st.subheader("2. Finance Review")
    conn=get_db()
    txs=conn.execute("SELECT * FROM transactions ORDER BY created_at DESC").fetchall()
    conn.close()
    pending=sum(1 for t in txs if t["finance_status"]=="PENDING")
    approved=sum(1 for t in txs if t["finance_status"]=="APPROVED")
    processed=sum(1 for t in txs if t["finance_status"]=="PROCESSED")
    returned=sum(1 for t in txs if t["finance_status"]=="RETURNED_TO_REQUESTER")
    exceptions=sum(1 for t in txs if t["finance_status"]=="PENDING" and t["ai_status"]!="READY")
    incomplete=sum(1 for t in txs if t["finance_status"]=="PENDING" and t["ai_status"]=="BLOCKED")
    a,b,c,d,e,f=st.columns(6)
    a.metric("Pending",pending); b.metric("AI Ready",sum(1 for t in txs if t["finance_status"]=="PENDING" and t["ai_status"]=="READY"))
    c.metric("Exceptions",exceptions); d.metric("Incomplete",incomplete); e.metric("Approved",approved); f.metric("Processed",processed)

    if not txs:
        st.info("No analysed transactions yet.")
    else:
        import pandas as pd
        rows=[]
        for t in txs:
            s={"PENDING":"🟢 Pending" if t["ai_status"]=="READY" else "🟠 Review",
               "APPROVED":"✅ Approved","PROCESSED":"✅ Processed","RETURNED_TO_REQUESTER":"↩ Returned"}[t["finance_status"]]
            rows.append([t["reference"],t["gr_type"],t["payment_stage"],t["supplier"],money(t["invoice_amount"]),s,t["gr_reference"] or "—"])
        st.dataframe(pd.DataFrame(rows,columns=["Reference","GR Type","Stage","Supplier","Invoice","Status","GR Reference"]),
                     use_container_width=True,hide_index=True)

        options={f"{t['reference']} — {t['supplier']} — {t['finance_status']}":t["id"] for t in txs}
        selected=st.selectbox("Select transaction",list(options))
        txid=options[selected]
        tx=next(x for x in txs if x["id"]==txid)
        st.divider()
        st.markdown(f"### Review `{tx['reference']}`")
        q1,q2,q3,q4=st.columns(4)
        q1.metric("GR Type",tx["gr_type"]); q2.metric("Stage",tx["payment_stage"])
        q3.metric("Invoice",money(tx["invoice_amount"])); q4.metric("Status",tx["finance_status"])

        if tx["summary"]: st.write(tx["summary"])
        if tx["exception_text"]: st.warning(tx["exception_text"])

        st.markdown("### II / Budget")
        z1,z2,z3,z4=st.columns(4)
        z1.metric("II Annual",money(tx["ii_annual_budget"]))
        z2.metric("Prior",money(tx["prior_utilisation"]))
        z3.metric("Current",money(tx["invoice_amount"]))
        z4.metric("Projected",money(tx["projected_utilisation"]))
        if tx["budget_status"]=="WITHIN_BUDGET": st.success("🟢 Within approved II budget")
        elif tx["budget_status"]=="VARIANCE": st.warning("🟠 Budget variance requires review")
        else: st.info("II budget not established.")

        if tx["finance_status"]=="PENDING":
            st.markdown("### Finance decision")
            c1,c2=st.columns(2)
            with c1:
                comments=st.text_area("Approval / processing comment",placeholder="e.g. Budget acknowledged; supporting evidence verified.")
                can_approve=(tx["ai_status"]=="READY" and tx["budget_status"]!="VARIANCE")
                if st.button("✅ Approve GR",disabled=not can_approve,use_container_width=True):
                    approve(txid,comments); st.success("GR approved."); st.rerun()
                if st.button("🧾 Process GR",disabled=not (tx["finance_status"]=="APPROVED"),use_container_width=True):
                    gr=process(txid,comments); st.success(f"Processed successfully: **{gr}**"); st.rerun()
            with c2:
                reason=st.text_area("Return reason",placeholder="e.g. Please provide signed acceptance evidence.")
                if st.button("↩ Return to Requester",use_container_width=True):
                    if not reason.strip(): st.warning("Enter a return reason.")
                    else: return_tx(txid,reason); st.success("Returned to requester."); st.rerun()
        elif tx["finance_status"]=="APPROVED":
            st.success("✅ Finance approved this GR.")
            if st.button("🧾 Process GR",use_container_width=True):
                gr=process(txid); st.success(f"Processed successfully: **{gr}**"); st.rerun()
        elif tx["finance_status"]=="PROCESSED":
            st.success(f"✅ Processed — GR Reference: **{tx['gr_reference']}**")
        else:
            st.info("↩ Returned to requester.")

    st.divider()
    st.markdown("### 💬 RAG Finance Assistant")
    question=st.text_input("Ask about the applicable GR procedure")
    if st.button("Ask RAG Assistant"):
        require_key()
        if question.strip():
            try:
                ans=rag_answer(question)
                st.chat_message("assistant").write(ans["answer"])
                st.caption("Sources: "+", ".join(ans["sources"]))
            except Exception as e:
                st.error(f"RAG query failed: {e}")
        else: st.warning("Enter a question.")
