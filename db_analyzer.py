#!/usr/bin/env python3
"""
db_analyzer.py - Pro version for AI CODEFIX 2025 (Medium challenge)
Generates analysis, charts, PDF report and emails it.

Usage:
    python db_analyzer.py --db data.db --email recipient@example.com --output output/
"""

import os
import sys
import argparse
import sqlite3
from datetime import datetime
import tempfile
import logging
from pathlib import Path
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.enums import TA_CENTER
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email import encoders
import smtplib
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("db_analyzer")

# ---- Utilities ----

def safe_mkdir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def connect_sqlite(db_path: str):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def list_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    return [r[0] for r in cur.fetchall()]

def get_table_schema(conn, table_name):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table_name}');")
    rows = cur.fetchall()
    cols = [{"cid": r[0], "name": r[1], "type": r[2], "notnull": r[3], "dflt_value": r[4], "pk": r[5]} for r in rows]
    return cols

def is_probable_date_column(col_name, dtype_hint):
    lower = col_name.lower()
    if "date" in lower or "time" in lower or "month" in lower or "year" in lower:
        return True
    if dtype_hint:
        t = dtype_hint.lower()
        if "date" in t or "time" in t:
            return True
    return False

# ---- Analysis functions ----

def table_basic_stats(conn, table_name, max_preview=5):
    logger.info("Analyzing table: %s", table_name)
    df = pd.read_sql_query(f"SELECT * FROM '{table_name}' LIMIT 50000", conn)  # guard for huge tables
    stats = {}
    stats['rows'] = int(pd.read_sql_query(f"SELECT COUNT(*) AS cnt FROM '{table_name}'", conn).iloc[0]['cnt'])
    stats['columns'] = list(df.columns)
    stats['dtypes'] = df.dtypes.apply(lambda x: str(x)).to_dict()
    stats['null_counts'] = df.isnull().sum().to_dict()
    stats['duplicate_count'] = int(df.duplicated().sum())
    stats['sample'] = df.head(max_preview).to_dict(orient='records')
    # numeric stats
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        stats['numeric_summary'] = numeric_df.describe().to_dict()
    else:
        stats['numeric_summary'] = {}
    # detect possible date column
    date_candidates = []
    for c in stats['columns']:
        hint = stats['dtypes'].get(c, "")
        if is_probable_date_column(c, hint):
            date_candidates.append(c)
    # fallback: try to parse columns to datetime heuristically
    for c in df.columns:
        if c in date_candidates:
            continue
        if df[c].dtype == object and df[c].dropna().shape[0] > 0:
            sample = df[c].dropna().astype(str).head(20).tolist()
            # quick check for patterns YYYY- or dd/ or ISO
            join = " ".join(sample)
            if any(part.isdigit() and len(part) >= 4 for part in join.split()):
                date_candidates.append(c)
    stats['date_candidates'] = date_candidates
    # store df for later visualization decisions (but keep memory use bounded)
    stats['preview_df'] = df.head(2000).copy()  # safe preview
    return stats

# ---- Visualization functions ----

def save_fig(fig, path, dpi=200):
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    logger.info("Saved figure: %s", path)

def chart_table_row_counts(summary_tables, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    tables = [t['name'] for t in summary_tables]
    counts = [t['rows'] for t in summary_tables]
    sns.barplot(x=counts, y=tables, ax=ax)
    ax.set_title("Row Counts per Table")
    ax.set_xlabel("Row Count")
    ax.set_ylabel("Table")
    save_fig(fig, out_path / "chart_table_counts.png")
    return out_path / "chart_table_counts.png"

def chart_time_series_if_possible(table_stats, out_path: Path):
    # Try to find a table with a date column and numeric measure
    for t in table_stats:
        df = t.get('preview_df')
        if df is None or df.shape[0] == 0:
            continue
        # choose date column
        date_col = None
        for cand in t.get('date_candidates', []):
            try:
                df[cand] = pd.to_datetime(df[cand], errors='coerce')
                if df[cand].notnull().sum() > 0:
                    date_col = cand
                    break
            except Exception:
                continue
        if date_col is None:
            continue
        # pick numeric column to aggregate
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            # nothing numeric, aggregate counts per period
            series = df.set_index(date_col).resample('M').size()
            fig, ax = plt.subplots(figsize=(8, 4))
            series.plot(ax=ax, marker='o')
            ax.set_title(f"Records Over Time: {t['name']} (monthly)")
            ax.set_ylabel("Count")
            save_fig(fig, out_path / "chart_time_series.png")
            return out_path / "chart_time_series.png"
        else:
            measure = numeric_cols[0]
            series = df.set_index(date_col).resample('M')[measure].sum()
            if series.dropna().shape[0] == 0:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            series.plot(ax=ax, marker='o')
            ax.set_title(f"Time Series of {measure} ({t['name']})")
            ax.set_ylabel(f"Sum({measure})")
            save_fig(fig, out_path / "chart_time_series.png")
            return out_path / "chart_time_series.png"
    # If none found, fallback to categorical distribution of a largest table
    # choose table with max rows
    biggest = max(table_stats, key=lambda x: x['rows'])
    df = biggest.get('preview_df')
    if df is None or df.shape[0] == 0:
        return None
    # pick top non-numeric short-cardinality column
    candidates = [c for c in df.columns if df[c].nunique() < max(50, int(df.shape[0]*0.1))]
    if not candidates:
        return None
    col = candidates[0]
    counts = df[col].value_counts().nlargest(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_title(f"Top distribution of {col} in {biggest['name']}")
    save_fig(fig, out_path / "chart_distribution.png")
    return out_path / "chart_distribution.png"

def chart_correlation_or_scatter(table_stats, out_path: Path):
    # pick table with most numeric columns
    best = None
    best_count = 0
    for t in table_stats:
        df = t.get('preview_df')
        if df is None:
            continue
        numcols = df.select_dtypes(include=[np.number]).shape[1]
        if numcols > best_count:
            best_count = numcols
            best = t
    if best is None or best_count == 0:
        # no numeric columns available
        return None

    # get numeric dataframe and drop rows with NaNs (for correlation)
    df_num = best['preview_df'].select_dtypes(include=[np.number]).dropna(axis=0, how='any')
    if df_num.shape[1] >= 2 and df_num.shape[0] > 0:
        cols = df_num.columns.tolist()[:6]
        corr = df_num[cols].corr()
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title(f"Correlation matrix ({best['name']})")
        save_fig(fig, out_path / "chart_correlation.png")
        return out_path / "chart_correlation.png"
    elif df_num.shape[1] == 1:
        # only a single numeric column; create a histogram instead
        col = df_num.columns[0]
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(df_num[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col} ({best['name']})")
        save_fig(fig, out_path / "chart_distribution_single_numeric.png")
        return out_path / "chart_distribution_single_numeric.png"
    else:
        return None


# ---- Report generation ----

def build_pdf_report(summary, charts: list, output_pdf_path: Path, team_name="Team"):
    doc = SimpleDocTemplate(str(output_pdf_path), pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    flow = []
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], alignment=TA_CENTER)
    flow.append(Paragraph(f"Database Analysis Report - {team_name}", title_style))
    flow.append(Spacer(1, 0.3*cm))
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    flow.append(Paragraph(f"<b>Analysis Date:</b> {ts}", styles['Normal']))
    flow.append(Spacer(1, 0.5*cm))

    # Executive summary
    flow.append(Paragraph("<b>EXECUTIVE SUMMARY</b>", styles['Heading2']))
    total_tables = len(summary['tables'])
    total_records = sum(t['rows'] for t in summary['tables'])
    flow.append(Paragraph(f"- Total Tables: {total_tables}", styles['Normal']))
    flow.append(Paragraph(f"- Total Records (sum of counts): {total_records}", styles['Normal']))
    flow.append(Spacer(1, 0.2*cm))
    # top insights (simple heuristics)
    flow.append(Paragraph("<b>KEY INSIGHTS</b>", styles['Heading3']))
    insights = []
    # example heuristic insights:
    # top table by rows
    if summary['tables']:
        top_by_rows = max(summary['tables'], key=lambda x: x['rows'])
        insights.append(f"Table <b>{top_by_rows['name']}</b> contains the most rows: {top_by_rows['rows']}.")
    # any table with many nulls
    null_warnings = []
    for t in summary['tables']:
        nc = t.get('null_counts', {})
        for col, cnt in nc.items():
            if cnt > 0 and cnt / max(1, t['rows']) > 0.3:
                null_warnings.append((t['name'], col, cnt))
    if null_warnings:
        sample = null_warnings[:3]
        for tbl, col, cnt in sample:
            insights.append(f"Column <b>{col}</b> in table <b>{tbl}</b> has many nulls ({cnt} nulls).")
    if not insights:
        insights.append("No major data quality red flags found in quick scan.")
    for i, ins in enumerate(insights, start=1):
        flow.append(Paragraph(f"{i}. {ins}", styles['Normal']))
    flow.append(Spacer(1, 0.5*cm))

    # Per-table details
    flow.append(Paragraph("<b>DATA OVERVIEW</b>", styles['Heading2']))
    for t in summary['tables']:
        flow.append(Paragraph(f"<b>Table: {t['name']}</b>", styles['Heading3']))
        flow.append(Paragraph(f"Rows: {t['rows']} &nbsp;&nbsp; Columns: {len(t['columns'])}", styles['Normal']))
        # small table: top 5 null counts
        nulls = t.get('null_counts', {})
        if nulls:
            rows = [["Column", "Nulls"]]
            for col, cnt in sorted(nulls.items(), key=lambda x: -x[1])[:8]:
                rows.append([col, str(cnt)])
            tbl = Table(rows, colWidths=[9*cm, 4*cm])
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('GRID', (0,0), (-1,-1), 0.2, colors.black),
            ]))
            flow.append(tbl)
            flow.append(Spacer(1, 0.2*cm))
        # sample rows
        # sample rows (limit columns + truncate long text)
        sample = t.get('sample', [])
        if sample:
            try:
                df_sample = pd.DataFrame(sample)
            except Exception:
                df_sample = None

    if df_sample is not None and not df_sample.empty:
        # Limit to first 8 columns
        df_sample = df_sample.iloc[:, :8]

        # Truncate long text (>40 chars)
        df_sample = df_sample.applymap(
            lambda x: (str(x)[:40] + "...") if (x is not None and len(str(x)) > 40) else ("" if x is None else x)
        )

        headers = list(df_sample.columns)
        rows = [headers]

        for _, row in df_sample.head(3).iterrows():
            rows.append([str(row[h]) for h in headers])

        col_width = max(2.0 * cm, (16 * cm) / max(1, len(headers)))
        tbl = Table(rows, colWidths=[col_width] * len(headers))
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f2f2f2")),
            ('GRID', (0,0), (-1,-1), 0.2, colors.black),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        flow.append(tbl)
        flow.append(Spacer(1, 0.3*cm))
    else:
            # SAFE CLEAN FALLBACK for inconsistent sample rows
            clean_rows = []
            for r in sample[:3]:
                if isinstance(r, dict):
                    clean_rows.append(r)
                else:
                    # Convert non-dict rows into indexed dict
                    try:
                        clean_rows.append({f"col_{i}": v for i, v in enumerate(r)})
                    except:
                        clean_rows.append({"col_0": str(r)})

            df_fallback = pd.DataFrame(clean_rows)

            # limit to 8 columns
            df_fallback = df_fallback.iloc[:, :8]

            # truncate long v


            flow.append(Spacer(1, 0.1*cm))

    # Visualizations section
    flow.append(PageBreak())
    flow.append(Paragraph("<b>VISUALIZATIONS</b>", styles['Heading2']))
    for ch in charts:
        flow.append(Spacer(1, 0.2*cm))
        flow.append(Paragraph(Path(ch).name, styles['Heading3']))
        try:
            img = Image(str(ch), width=16*cm, height=9*cm)
            flow.append(img)
        except Exception as e:
            flow.append(Paragraph(f"Could not embed image {ch}: {e}", styles['Normal']))
        flow.append(Spacer(1, 0.3*cm))

    # Recommendations (placeholder)
    flow.append(PageBreak())
    flow.append(Paragraph("<b>RECOMMENDATIONS</b>", styles['Heading2']))
    flow.append(Paragraph("- Consider cleaning columns with many nulls or imputing values.", styles['Normal']))
    flow.append(Paragraph("- Add indexes to frequently joined key columns to improve query performance.", styles['Normal']))
    flow.append(Paragraph("- If time-series analysis is needed, ensure a canonical date column is present and standardized.", styles['Normal']))

    doc.build(flow)
    logger.info("PDF report built: %s", output_pdf_path)

# ---- Email sender ----

def send_email_with_attachments(smtp_server, smtp_port, sender_email, sender_password, recipient_email, subject, body_text, attachments):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body_text, 'plain'))
    for path in attachments:
        with open(path, 'rb') as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(path))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(path)}"'
            msg.attach(part)
    # send
    try:
        server = smtplib.SMTP(smtp_server, int(smtp_port))
        server.ehlo()
        if int(smtp_port) == 587:
            server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, [recipient_email], msg.as_string())
        server.quit()
        logger.info("Email sent to %s", recipient_email)
    except Exception as e:
        logger.error("Failed to send email: %s", e)
        raise

# ---- Main orchestration ----

def analyze_and_report(db_path, recipient_email, output_dir, smtp_config):
    output_dir = Path(output_dir)
    safe_mkdir(output_dir)
    conn = connect_sqlite(db_path)
    tables = list_tables(conn)
    logger.info("Found tables: %s", tables)
    table_summaries = []
    for t in tables:
        try:
            s = table_basic_stats(conn, t)
            s['name'] = t
            table_summaries.append(s)
        except Exception as e:
            logger.warning("Failed analyzing table %s: %s", t, e)

    summary = {
        'db_path': db_path,
        'analyzed_at': datetime.utcnow().isoformat(),
        'tables': [{'name': t['name'], 'rows': t['rows'], 'columns': t['columns'], 'null_counts': t['null_counts'], 'sample': t['sample']} for t in table_summaries],
    }

    # Generate charts
    charts = []
    try:
        c1 = chart_table_row_counts(summary['tables'], output_dir)
        charts.append(str(c1))
    except Exception as e:
        logger.exception("Chart 1 failed: %s", e)
    try:
        c2 = chart_time_series_if_possible(table_summaries, output_dir)
        if c2:
            charts.append(str(c2))
    except Exception as e:
        logger.exception("Chart 2 failed: %s", e)
    try:
        c3 = chart_correlation_or_scatter(table_summaries, output_dir)
        if c3:
            charts.append(str(c3))
    except Exception as e:
        logger.exception("Chart 3 failed: %s", e)

    # Ensure we have at least 3 charts; if not, create simple fallbacks (top-cols)
    while len(charts) < 3:
        # fallback generate a pie of top table columns count
        fallback = output_dir / f"chart_fallback_{len(charts)+1}.png"
        # build generic histogram of column counts
        try:
            # pick biggest table and plot unique counts of first categorical col
            biggest = max(table_summaries, key=lambda x: x['rows'])
            df = biggest.get('preview_df')
            if df is not None and df.shape[0] > 0:
                col = None
                for c in df.columns:
                    if df[c].nunique() > 1 and df[c].nunique() < 50 and df[c].dtype == object:
                        col = c
                        break
                if col is None:
                    col = df.columns[0]
                counts = df[col].value_counts().nlargest(10)
                fig, ax = plt.subplots(figsize=(8,5))
                sns.barplot(x=counts.values, y=counts.index, ax=ax)
                ax.set_title(f"Fallback distribution of {col}")
                save_fig(fig, fallback)
                charts.append(str(fallback))
                continue
        except Exception:
            pass
        # final fallback: simple empty placeholder chart
        fig, ax = plt.subplots(figsize=(6,4))
        ax.text(0.5,0.5,"No data for chart", ha='center', va='center')
        save_fig(fig, fallback)
        charts.append(str(fallback))

    # Build PDF
    team_name = os.environ.get("TEAM_NAME", smtp_config.get('team_name', 'Team'))
    pdf_path = output_dir / f"report_{team_name.replace(' ','_')}.pdf"
    build_pdf_report({'tables': summary['tables']}, charts, pdf_path, team_name=team_name)

    # Compose email body
    total_tables = len(summary['tables'])
    total_records = sum(t['rows'] for t in summary['tables'])
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    subj = f"Database Analysis Report - {team_name}"
    body = f"""Dear Recipient,

Please find the automated database analysis report below.

=== DATABASE SUMMARY ===
- Total Tables: {total_tables}
- Total Records: {total_records}
- Analysis Date: {ts}

=== KEY INSIGHTS ===
1. See attached PDF for executive summary and visualizations.

Attached: {pdf_path.name} + {', '.join([Path(c).name for c in charts])}

Best regards,
{team_name}
AI CODEFIX 2025
"""

    attachments = [str(pdf_path)] + charts
    # Send email
    send_email_with_attachments(
        smtp_config['server'],
        smtp_config['port'],
        smtp_config['sender'],
        smtp_config['password'],
        recipient_email,
        subj,
        body,
        attachments
    )

# ---- CLI ----

def parse_args():
    p = argparse.ArgumentParser(description="Database Analyzer - Pro")
    p.add_argument("--db", required=True, help="Path to SQLite database (data.db)")
    p.add_argument("--email", required=True, help="Recipient email to send report to")
    p.add_argument("--output", default="output", help="Output folder for artifacts")
    p.add_argument("--smtp-server", default=os.getenv("SMTP_SERVER"), help="SMTP server (or set SMTP_SERVER env)")
    p.add_argument("--smtp-port", default=os.getenv("SMTP_PORT", 587), help="SMTP port (or set SMTP_PORT env)")
    p.add_argument("--sender-email", default=os.getenv("SENDER_EMAIL"), help="Sender email (or set SENDER_EMAIL env)")
    p.add_argument("--sender-password", default=os.getenv("SENDER_PASSWORD"), help="Sender password (or set SENDER_PASSWORD env)")
    p.add_argument("--team-name", default=os.getenv("TEAM_NAME", "Team"), help="Team name shown in report")
    return p.parse_args()

def main():
    args = parse_args()
    # valid args check
    smtp_server = args.smtp_server
    smtp_port = args.smtp_port
    sender = args.sender_email
    sender_pwd = args.sender_password
    if not (smtp_server and smtp_port and sender and sender_pwd):
        logger.error("SMTP configuration incomplete. Set via CLI or environment vars: SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PASSWORD")
        sys.exit(1)
    smtp_config = {
        "server": smtp_server,
        "port": smtp_port,
        "sender": sender,
        "password": sender_pwd,
        "team_name": args.team_name
    }
    analyze_and_report(args.db, args.email, args.output, smtp_config)

if __name__ == "__main__":
    main()
