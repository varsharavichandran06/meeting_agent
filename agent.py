import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.process import Process
from crewai_tools import SerperDevTool
import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

load_dotenv(override=True)

# Belt-and-suspenders: manually read any lines dotenv skipped
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path, encoding="utf-8", errors="ignore") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            try:
                _k, _v = _line.split("=", 1)
                _k = _k.strip()
                _v = _v.strip().strip('"').strip("'")
                if _k and _k not in os.environ:
                    os.environ[_k] = _v
            except Exception:
                pass
os.environ["CREWAI_TELEMETRY"] = "false"

# ── Config from .env ──────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY   = os.getenv("SERPER_API_KEY")
NEWS_API_KEY     = os.getenv("NEWS_API_KEY")          # https://newsapi.org — free, 100 req/day
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o")
JOB_FIELD        = os.getenv("JOB_FIELD", "Software Engineering")
EXPERIENCE_LEVEL = os.getenv("EXPERIENCE_LEVEL", "Entry Level")
JOB_TYPE         = os.getenv("JOB_TYPE", "Full-time, Remote")
LOCATIONS        = os.getenv("LOCATIONS", "USA, India")
NUM_RESULTS      = int(os.getenv("NUM_RESULTS", "10"))

# Google Calendar embed — set GOOGLE_CALENDAR_EMBED_URL in .env
# Get it: Google Calendar → Settings → your calendar → "Integrate calendar" → copy Embed src URL
GCAL_EMBED_URL = os.getenv("GOOGLE_CALENDAR_EMBED_URL", "PLACEHOLDER")

now       = datetime.now(timezone.utc).replace(tzinfo=None)
cutoff    = now - timedelta(hours=24)
DATE_FROM = cutoff.strftime("%B %d, %Y %H:%M UTC")
DATE_TO   = now.strftime("%B %d, %Y %H:%M UTC")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Varsha's Job Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css(filepath):
    with open(filepath, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(Path(__file__).parent / "styles.css")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">Job<span>Radar</span></div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#6b6b85;font-size:12px;margin-bottom:24px;'>Varsha's personal job discovery engine</p>", unsafe_allow_html=True)

    st.markdown("#### 📋 Active Configuration")
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;gap:10px;margin-top:8px">
        <div class="config-row"><span class="config-key">Role</span><span class="config-val">{JOB_FIELD}</span></div>
        <div class="config-row"><span class="config-key">Level</span><span class="config-val">{EXPERIENCE_LEVEL}</span></div>
        <div class="config-row"><span class="config-key">Type</span><span class="config-val">{JOB_TYPE}</span></div>
        <div class="config-row"><span class="config-key">Locations</span><span class="config-val">{LOCATIONS}</span></div>
        <div class="config-row"><span class="config-key">Results</span><span class="config-val">{NUM_RESULTS}</span></div>
        <div class="config-row"><span class="config-key">Window</span><span class="config-val">Last 24 hours</span></div>
        <div class="config-row"><span class="config-key">Model</span><span class="config-val">{OPENAI_MODEL}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='color:#6b6b85;font-size:11px;line-height:1.8;'>To change any setting,<br>edit your <code style='color:#00e5a0'>.env</code> file and restart the app.</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### 🤖 Agent Pipeline")
    st.markdown("""
    <div style="display:flex;flex-direction:column;gap:10px;margin-top:8px">
        <div style="display:flex;align-items:center;gap:10px">
            <span style="width:8px;height:8px;background:#00e5a0;border-radius:50%;flex-shrink:0;box-shadow:0 0 6px #00e5a0"></span>
            <span style="font-size:12px;color:#e8e8f0;font-family:'DM Mono',monospace">1 · Job Search Agent <span style="color:#00e5a0">● active</span></span>
        </div>
        <div style="display:flex;align-items:center;gap:10px">
            <span style="width:8px;height:8px;background:#1e1e2e;border:1px solid #2e2e4e;border-radius:50%;flex-shrink:0"></span>
            <span style="font-size:12px;color:#6b6b85;font-family:'DM Mono',monospace">2 · Rerank Agent</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px">
            <span style="width:8px;height:8px;background:#1e1e2e;border:1px solid #2e2e4e;border-radius:50%;flex-shrink:0"></span>
            <span style="font-size:12px;color:#6b6b85;font-family:'DM Mono',monospace">3 · Apply Agent</span>
        </div>
    </div>
    <p style="color:#6b6b85;font-size:11px;margin-top:12px;font-family:'DM Mono',monospace">More agents coming soon…</p>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  AI NEWS — NewsAPI.org
# ════════════════════════════════════════════════════════════
TAG_COLORS = {
    "LLM":      ("#7c3aed", "rgba(124,58,237,0.15)"),
    "Vision":   ("#00b4d8", "rgba(0,180,216,0.15)"),
    "Infra":    ("#ff9f43", "rgba(255,159,67,0.15)"),
    "Agent":    ("#00e5a0", "rgba(0,229,160,0.15)"),
    "Hardware": ("#f72585", "rgba(247,37,133,0.15)"),
    "Research": ("#ffd60a", "rgba(255,214,10,0.15)"),
}
COMPANY_ICONS = {
    "Google": "🔵", "Meta": "🟣", "Microsoft": "🔷",
    "OpenAI": "⚪", "Apple": "🍎", "Amazon": "🟠", "NVIDIA": "🟩",
}
_COMPANY_KW = {
    "openai": "OpenAI",  "chatgpt": "OpenAI",  "gpt-": "OpenAI",
    "google": "Google",  "gemini": "Google",    "deepmind": "Google",
    "meta":   "Meta",    "llama":  "Meta",
    "microsoft": "Microsoft", "copilot": "Microsoft", "azure": "Microsoft",
    "apple":  "Apple",   "siri": "Apple",
    "amazon": "Amazon",  "aws":  "Amazon",
    "nvidia": "NVIDIA",  "blackwell": "NVIDIA",
}
_TAG_KW = {
    "language model": "LLM",    "llm": "LLM",   "gpt": "LLM",  "llama": "LLM",  "gemini": "LLM",
    "vision": "Vision",          "image": "Vision", "video": "Vision", "multimodal": "Vision",
    "agent": "Agent",            "copilot": "Agent", "autonomous": "Agent",
    "chip": "Hardware",          "gpu": "Hardware",  "blackwell": "Hardware", "hardware": "Hardware",
    "cloud": "Infra",            "infrastructure": "Infra", "aws": "Infra", "azure": "Infra",
    "research": "Research",      "paper": "Research", "benchmark": "Research",
}

FALLBACK_NEWS = [
    {"company":"OpenAI",    "headline":"o3 model tops new reasoning benchmarks",       "summary":"Scores highest ever on GPQA and AIME evaluations.",          "tag":"LLM",      "time":"today","url":"#"},
    {"company":"Google",    "headline":"Gemini 2.5 Pro multimodal update ships",       "summary":"Handles 2M-token context natively including video.",          "tag":"Vision",   "time":"today","url":"#"},
    {"company":"Meta",      "headline":"Llama 4 Scout open-weights model released",    "summary":"10M context window, MoE architecture, Apache 2.0 license.",  "tag":"LLM",      "time":"today","url":"#"},
    {"company":"Microsoft", "headline":"Copilot gets autonomous multi-step execution", "summary":"AI now handles complex workflows across Microsoft 365 apps.", "tag":"Agent",    "time":"today","url":"#"},
    {"company":"NVIDIA",    "headline":"Blackwell Ultra B300 specs confirmed at GTC",  "summary":"Delivers 4× inference throughput over prior H100 generation.","tag":"Hardware", "time":"today","url":"#"},
]


def _guess_company(text):
    lower = text.lower()
    for kw, co in _COMPANY_KW.items():
        if kw in lower:
            return co
    return "Tech"


def _guess_tag(text):
    lower = text.lower()
    for kw, tag in _TAG_KW.items():
        if kw in lower:
            return tag
    return "LLM"


def _rel_time(iso):
    try:
        dt   = datetime.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S")
        diff = datetime.now(timezone.utc).replace(tzinfo=None) - dt
        h    = int(diff.total_seconds() // 3600)
        if h < 1:
            return f"{int(diff.total_seconds()//60)}m ago"
        return f"{h}h ago" if h < 24 else f"{h//24}d ago"
    except Exception:
        return "recently"


@st.cache_data(ttl=1800)
def fetch_ai_news():
    """Pull latest AI big-tech headlines from NewsAPI.org (free, 100 req/day)."""
    if not NEWS_API_KEY:
        return FALLBACK_NEWS
    try:
        since = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%S")
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": (
                    "(artificial intelligence OR AI OR LLM OR machine learning) AND "
                    "(OpenAI OR Google OR Meta OR Microsoft OR Apple OR Amazon OR NVIDIA)"
                ),
                "from":     since,
                "sortBy":   "publishedAt",
                "language": "en",
                "pageSize": 30,
                "apiKey":   NEWS_API_KEY,
            },
            timeout=8,
        )
        r.raise_for_status()
        articles = r.json().get("articles", [])

        seen, results = set(), []
        for a in articles:
            title = (a.get("title") or "").strip()
            desc  = (a.get("description") or "").strip()
            if not title or title == "[Removed]":
                continue
            company = _guess_company(title + " " + desc)
            if company in seen:
                continue
            seen.add(company)
            words    = title.split()
            headline = " ".join(words[:12]) + ("…" if len(words) > 12 else "")
            dwords   = desc.split()
            summary  = " ".join(dwords[:20]) + ("…" if len(dwords) > 20 else "") or "Click to read more."
            results.append({
                "company":  company,
                "headline": headline,
                "summary":  summary,
                "tag":      _guess_tag(title + " " + desc),
                "time":     _rel_time(a.get("publishedAt", "")),
                "url":      a.get("url", "#"),
            })
            if len(results) == 5:
                break

        return results or FALLBACK_NEWS
    except Exception:
        return FALLBACK_NEWS


# ════════════════════════════════════════════════════════════
#  PAGE LAYOUT  —  left ¼ | right ¾
# ════════════════════════════════════════════════════════════
left_col, right_col = st.columns([1, 3], gap="large")

# ────────────────────────────────────────────────────────────
#  LEFT  ─  Calendar tile  +  AI News tile
# ────────────────────────────────────────────────────────────
with left_col:

    # ── Calendar (top) ────────────────────────────────────────
    st.markdown("""
    <div class="tile-header">
        <span class="tile-dot pulse-blue"></span>
        <span class="tile-label">MY CALENDAR</span>
    </div>
    """, unsafe_allow_html=True)

    if GCAL_EMBED_URL == "PLACEHOLDER":
        st.markdown("""
        <div class="cal-placeholder">
            <div style="font-size:30px;margin-bottom:10px">📅</div>
            <div style="color:#e8e8f0;font-size:12px;font-weight:700;margin-bottom:10px">Connect Google Calendar</div>
            <div style="color:#6b6b85;font-size:10px;line-height:2;text-align:left">
                1. Open <strong style="color:#4285F4">Google Calendar</strong><br>
                2. Settings → your calendar<br>
                3. Scroll to <em>Integrate calendar</em><br>
                4. Copy the <strong style="color:#00e5a0">Embed URL</strong><br>
                5. Add to <code style="color:#ffd60a">.env</code>:<br>
                <code style="color:#ffd60a;font-size:9px;word-break:break-all">GOOGLE_CALENDAR_EMBED_URL=&lt;url&gt;</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="cal-embed-wrap">
            <iframe src="{GCAL_EMBED_URL}&showTitle=0&showPrint=0&showCalendars=0&mode=WEEK"
                style="border:0;width:100%;height:430px;border-radius:10px"
                frameborder="0" scrolling="no">
            </iframe>
        </div>
        """, unsafe_allow_html=True)

    # ── AI News (bottom) ──────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="tile-header">
        <span class="tile-dot pulse-pink"></span>
        <span class="tile-label">AI PULSE · Big Tech</span>
        <span class="live-badge">● LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    news = fetch_ai_news()
    cards_html = ""
    for item in news:
        tag         = item.get("tag", "LLM")
        color, bg   = TAG_COLORS.get(tag, ("#00e5a0", "rgba(0,229,160,0.15)"))
        icon        = COMPANY_ICONS.get(item.get("company", ""), "·")
        url         = item.get("url", "#")
        cards_html += f"""
        <a class="news-card" href="{url}" target="_blank" rel="noopener">
            <div class="news-card-top">
                <span class="news-company">{icon} {item.get('company','')}</span>
                <span class="news-tag" style="color:{color};background:{bg};border:1px solid {color}44">{tag}</span>
            </div>
            <div class="news-headline">{item.get('headline','')}</div>
            <div class="news-summary">{item.get('summary','')}</div>
            <div class="news-time">{item.get('time','')}</div>
        </a>"""

    no_key_banner = "" if NEWS_API_KEY else """
    <div class="news-setup-banner">
        <strong>Add NEWS_API_KEY to .env</strong><br>
        <span>Free at <a href="https://newsapi.org" target="_blank">newsapi.org</a> — showing cached headlines.</span>
    </div>"""

    st.markdown(f'<div class="news-tile">{no_key_banner}{cards_html}</div>', unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
#  RIGHT  ─  Job Dashboard (original content)
# ────────────────────────────────────────────────────────────
with right_col:

    st.markdown(f"""
    <div class="hero">
        <div class="hero-tag">⚡ Varsha's Personal Dashboard · Live Search</div>
        <div class="hero-title">Hey Varsha, let's find<br>your <span>Dream Role</span> 🎯</div>
        <div class="hero-sub">Your AI agent scans the web for the freshest {JOB_FIELD} openings posted in the last 24 hours across {LOCATIONS}. No stale listings, ever.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="stat-card"><div class="stat-label">Results Target</div><div class="stat-value">{NUM_RESULTS}</div></div>""", unsafe_allow_html=True)
    with col2:
        fd = JOB_FIELD[:16] + "…" if len(JOB_FIELD) > 16 else JOB_FIELD
        st.markdown(f"""<div class="stat-card" style="--accent:#ff9f43"><div class="stat-label">Field</div><div class="stat-value" style="color:#ff9f43;font-size:18px;padding-top:4px">{fd}</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="stat-card" style="--accent:#7c3aed"><div class="stat-label">Model</div><div class="stat-value" style="color:#7c3aed;font-size:18px;padding-top:4px">{OPENAI_MODEL}</div></div>""", unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown(f"""<div class="stat-card" style="--accent:#00b4d8"><div class="stat-label">Locations</div><div class="stat-value" style="color:#00b4d8;font-size:15px;padding-top:6px">{LOCATIONS}</div></div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="stat-card" style="--accent:#f72585"><div class="stat-label">Experience</div><div class="stat-value" style="color:#f72585;font-size:15px;padding-top:6px">{EXPERIENCE_LEVEL}</div></div>""", unsafe_allow_html=True)
    with col6:
        st.markdown(f"""<div class="stat-card" style="--accent:#ffd60a"><div class="stat-label">Posted Within</div><div class="stat-value" style="color:#ffd60a;font-size:14px;padding-top:6px">Last 24 hours</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    launch = st.button("🎯  Launch Job Search Agent")

    if launch:
        if not OPENAI_API_KEY or not SERPER_API_KEY:
            st.error("⚠️  API keys missing. Add OPENAI_API_KEY and SERPER_API_KEY to your .env file.")
            st.stop()

        description = f"""
You are searching for job openings posted STRICTLY between {DATE_FROM} and {DATE_TO}.
Any listing posted before {DATE_FROM} must be DISCARDED.

Search criteria:
- Role / Field   : {JOB_FIELD}
- Experience     : {EXPERIENCE_LEVEL}
- Job type       : {JOB_TYPE}
- Locations      : {LOCATIONS}
- Posted between : {DATE_FROM} → {DATE_TO}

Instructions:
1. Run multiple targeted searches:
   - '{JOB_FIELD} {EXPERIENCE_LEVEL} jobs {LOCATIONS} posted today site:linkedin.com'
   - '{JOB_FIELD} entry level jobs India USA last 24 hours'
   - 'site:indeed.com {JOB_FIELD} {EXPERIENCE_LEVEL} {LOCATIONS} today'
2. Verify each post date is after {DATE_FROM}. Skip anything older.
3. Collect exactly {NUM_RESULTS} verified listings.
4. For each: Company · Title · Location · Job type · Date posted · Application URL · 2-3 sentence requirements summary.
5. Format as a clean numbered list.
"""
        with st.spinner(f"🤖  Scanning for {JOB_FIELD} jobs since {DATE_FROM}… (30–90 sec)"):
            try:
                model_llm   = LLM(OPENAI_MODEL, api_key=OPENAI_API_KEY)
                search_tool = SerperDevTool()
                agent = Agent(
                    role="Real-Time Job Search Specialist",
                    goal=(f"Find exactly {NUM_RESULTS} verified {EXPERIENCE_LEVEL} {JOB_FIELD} job postings "
                          f"published after {DATE_FROM} in {LOCATIONS}. Reject anything older."),
                    backstory=("Elite technical recruiter obsessed with real-time data. "
                               "Never surfaces stale listings. Always provides verified, working application URLs."),
                    llm=model_llm, tools=[search_tool], verbose=True,
                )
                task = Task(
                    description=description, agent=agent,
                    expected_output=(f"Numbered list of {NUM_RESULTS} {EXPERIENCE_LEVEL} {JOB_FIELD} jobs "
                                     f"posted after {DATE_FROM} in {LOCATIONS}. "
                                     "Each entry: company · title · location · job type · date posted · URL · summary."),
                )
                crew = Crew(agents=[agent], tasks=[task], verbose=True, process=Process.sequential)
                result = crew.kickoff()
            except Exception as e:
                st.error(f"❌  Agent error: {e}")
                st.stop()

        st.markdown("""
        <div class="result-header">
            <div class="result-dot"></div>
            <div class="result-title">Search Complete · Results Ready</div>
            <span class="status-pill status-running">● LIVE · last 24h</span>
        </div>
        """, unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown(str(result))
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="⬇  Download Results as .txt",
            data=str(result),
            file_name=f"varsha_jobs_{now.strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
        )

    else:
        st.markdown(f"""
        <div style="border:1px dashed #1e1e2e;border-radius:12px;padding:60px 40px;text-align:center">
            <div style="font-size:48px;margin-bottom:16px">🎯</div>
            <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:700;color:#e8e8f0;margin-bottom:12px">Ready, Varsha!</div>
            <div style="font-family:'DM Mono',monospace;font-size:13px;line-height:2;color:#6b6b85">
                Searching for <strong style="color:#00e5a0">{EXPERIENCE_LEVEL} {JOB_FIELD}</strong> jobs<br>
                in <strong style="color:#00b4d8">{LOCATIONS}</strong><br>
                posted after <strong style="color:#ffd60a">{DATE_FROM}</strong><br><br>
                Hit <strong style="color:#00e5a0">Launch Job Search Agent</strong> to begin.
            </div>
        </div>
        """, unsafe_allow_html=True)