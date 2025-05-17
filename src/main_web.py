import os
# Attempt to import pandas; fall back to None if unavailable
try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas may not be installed
    pd = None

# NumPy is optional for running unit tests
try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy may not be installed
    np = None
# Set matplotlib to non-interactive 'Agg' backend before importing pyplot
try:
    import matplotlib
    matplotlib.use('Agg')  # Use the Agg backend for non-interactive image generation
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - matplotlib may not be installed
    matplotlib = None
    plt = None
from datetime import datetime, timedelta
import base64
from io import BytesIO
import re
import time
# Optional imports used for the web application. They are not required for unit
# tests and may be missing in minimal environments.
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv may not be installed
    def load_dotenv(*args, **kwargs):
        return False
try:
    import requests
    from requests.exceptions import ConnectionError, Timeout, RequestException
except ImportError:  # pragma: no cover - requests may not be installed
    requests = None
    ConnectionError = Timeout = RequestException = Exception
from zoneinfo import ZoneInfo
import sys
try:
    from fastapi import FastAPI, Form, BackgroundTasks
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
except ImportError:  # pragma: no cover - fastapi/uvicorn may not be installed
    FastAPI = Form = BackgroundTasks = HTMLResponse = JSONResponse = None
    uvicorn = None
try:
    import openai
    from openai import OpenAI
except ImportError:  # pragma: no cover - openai may not be installed
    openai = None
    OpenAI = None
import json
try:
    import httpx  # Add this import
except ImportError:  # pragma: no cover - httpx may not be installed
    httpx = None
try:
    from markdown import markdown as md_to_html
except ImportError:  # pragma: no cover - markdown may not be installed
    md_to_html = None
import ast  # for safer literal evaluation
import uuid
# Import web search functionality
try:
    from .search import search_web_mentions, generate_ai_summary
except Exception:  # pragma: no cover - search module may not be available
    search_web_mentions = generate_ai_summary = None

# Always load .env first so env vars are set
load_dotenv(override=True)
print("Environment variables loaded")

# Print OpenAI version to verify if available
if openai is not None:
    print("OpenAI version:", openai.__version__)

# Check if proxy environment variables are set (which can cause issues)
for var in ("HTTP_PROXY", "HTTPS_PROXY"):
    proxy_value = os.environ.get(var)
    if proxy_value:
        print(f"[WARNING] {var} is set to '{proxy_value}'. This may cause issues with OpenAI API.")

# Load API credentials
USERNAME = os.getenv("BRANDWATCH_USERNAME")
PASSWORD = os.getenv("BRANDWATCH_PASSWORD")
PROJECT_ID = os.getenv("BRANDWATCH_PROJECT_ID")
API_TOKEN = os.getenv("BRANDWATCH_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Load Perplexity API key
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not PERPLEXITY_API_KEY:
    print("[WARNING] PERPLEXITY_API_KEY not found in .env file. Web search will be disabled.")
else:
    print("[INFO] Perplexity API key found")

if not OPENAI_API_KEY:
    print("[ERROR] OPENAI_API_KEY is not set in your .env file")
else:
    print("[INFO] OpenAI API key found")

print(f"API credentials found: {bool(API_TOKEN)} and OpenAI: {bool(OPENAI_API_KEY)}")

# Correct way to initialize the OpenAI client without proxies
if openai is not None and httpx is not None and OPENAI_API_KEY:
    http_client = httpx.Client(trust_env=False)
    client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
else:  # pragma: no cover - in unit test environment
    http_client = None
    client = None

# --- CONFIGURATION ---
# List of topic display names
TOPIC_LIST = [
    "university_cityu",
    "university_cuhk",
    "university_hku",
    "university_hkust",
    "university_polyu"
]
print("Topic configuration loaded (Hong Kong universities)")

# The actual tag names in Brandwatch
TAG_NAMES = [
    "university_cityu",
    "university_cuhk",
    "university_hku",
    "university_hkust",
    "university_polyu"
]

# Mapping from tag names to display names (same for now)
TAG_TO_TOPIC = {tag: tag for tag in TAG_NAMES}

# Optional colour palette for each university topic
TAG_COLORS = {
    "university_cityu": "#1f77b4",  # Strong blue
    "university_cuhk": "#ff7f0e",   # Orange
    "university_hku": "#2ca02c",    # Green
    "university_hkust": "#d62728",  # Red
    "university_polyu": "#9467bd"   # Purple
}

# Brandwatch account configuration (optional – pulled from .env if provided)
# If you have a specific Brandwatch account ID, set BRANDWATCH_ACCOUNT_ID in your .env.
BRANDWATCH_ACCOUNT_ID = os.getenv("BRANDWATCH_ACCOUNT_ID")

# Cache for accountId -> first projectId lookup to avoid repeated API calls
_ACCOUNT_PROJECT_CACHE = {}

def get_first_project_id(account_id: str):
    """Return first projectId for a given Brandwatch accountId, with simple in-process cache."""
    if not account_id:
        raise ValueError("account_id required")
    if account_id in _ACCOUNT_PROJECT_CACHE:
        return _ACCOUNT_PROJECT_CACHE[account_id]

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    url = f"https://api.brandwatch.com/projects?accountId={account_id}"
    print(f"[INFO] Resolving account {account_id} to projectId …")
    data = make_request(url, headers)
    if not data.get("results"):
        raise RuntimeError(f"No projects found for account {account_id}")
    project_id = str(data["results"][0]["id"])
    print(f"[INFO] Account {account_id} -> project {project_id}")
    _ACCOUNT_PROJECT_CACHE[account_id] = project_id
    return project_id

# Analyst guidance per university with expanded information
TOPIC_CONTEXT = {
    "CityU": "City University of Hong Kong (香港城市大學) – A dynamic, forward-looking university that emphasizes professional education and research. Established in 1984 and located in Kowloon Tong. Also known as 城大 (Shing Dai).",
    "CUHK": "Chinese University of Hong Kong (香港中文大學) – A comprehensive research university with a collegiate system established in 1963 in Sha Tin. Features a bilingual (Chinese and English) policy. Commonly called 中大 (Zhongda).",
    "HKU": "The University of Hong Kong (香港大學) – The territory's oldest tertiary institution founded in 1911, located in Pok Fu Lam. Known for its international outlook and English-medium instruction. Often referred to as 港大 (Gang Dai).",
    "HKUST": "Hong Kong University of Science and Technology (香港科技大學) – A research-intensive university focusing on science, technology, engineering, and business. Founded in 1991 and located in Clear Water Bay. Nicknamed 科大 (Ke Dai).",
    "PolyU": "The Hong Kong Polytechnic University (香港理工大學) – A public university with strong emphasis on applied science, engineering, business, and design. Established in 1937 as a trade school and gained university status in 1994. Known as 理大 (Lei Dai)."
}

# Create DataFrame for topic names if pandas is available; otherwise store a list
if pd is not None:
    topics_df = pd.DataFrame({
        'topic_name': TOPIC_LIST,
        'name_variations': [t for t in TOPIC_LIST]
    })
else:  # pragma: no cover - used only when pandas is missing
    topics_df = [
        {'topic_name': t, 'name_variations': t}
        for t in TOPIC_LIST
    ]

# ---------------------------------------------------------------------------
# In-memory job store for background report generation (Option B polling).
# Structure: {job_id: {status: 'pending'|'done'|'error', 'html': str|None, 'error': str|None}}
# NOTE: This resets when the process restarts. For production use Redis or DB.
JOB_STORE = {}

# --- DEFAULT DATE RANGE IF NOT PROVIDED ---
def get_default_dates():
    """Get default date range (last 7 days including today)"""
    today = datetime.now()
    # Make sure we're using past dates, not future dates
    default_end = today.strftime('%Y-%m-%d')
    default_start = (today - timedelta(days=6)).strftime('%Y-%m-%d')
    print(f"[INFO] Default date range: {default_start} to {default_end}")
    return default_start, default_end

# --- BRANDWATCH API HELPERS ---
def get_oauth_token():
    """Get OAuth token from Brandwatch API"""
    token_url = "https://api.brandwatch.com/oauth/token"
    params = {
        "username": USERNAME,
        "grant_type": "api-password",
        "client_id": "brandwatch-api-client"
    }
    response = requests.post(token_url, params=params, data={"password": PASSWORD})
    if response.status_code == 200:
        return response.json()["access_token"]
    raise Exception(f"Failed to get OAuth token: {response.text}")

def make_request(url, headers, retries=5):
    """Make request to Brandwatch API with retry logic"""
    attempt = 0
    base_wait_time = 10  # shorter retry backoff base for quicker failure during debugging
    while attempt < retries:
        try:
            print(f"[DEBUG] Request attempt {attempt+1}/{retries}: {url}")
            response = requests.get(url, headers=headers, timeout=15)  # shorter per-request timeout
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_time = int(retry_after) if retry_after else base_wait_time * (2 ** attempt)
                time.sleep(wait_time)
            elif 500 <= response.status_code < 600:
                delay = min(base_wait_time * (2 ** attempt), 300)
                time.sleep(delay)
            else:
                print(f"[WARNING] Unexpected status {response.status_code}: {response.text[:200]}")
                delay = min(base_wait_time * (2 ** attempt), 300)
                time.sleep(delay)
        except (ConnectionError, Timeout, RequestException) as ex:
            print(f"[ERROR] Connection error on attempt {attempt+1}: {ex}")
            delay = min(base_wait_time * (2 ** attempt), 300)
            time.sleep(delay)
        attempt += 1
        if attempt < retries:
            time.sleep(5)
    raise Exception(f"[ERROR] Failed to retrieve data after {retries} attempts. URL: {url}")

def get_brandwatch_data(start_date, end_date, project_id=None, account_id=None):
    """Get data from Brandwatch API for a SINGLE account/project defined in .env (no region looping)."""
    # Use provided parameters or fall back to environment values
    account_id_use = account_id or BRANDWATCH_ACCOUNT_ID
    project_id_use = project_id or PROJECT_ID

    print(f"[INFO] Querying Brandwatch project {project_id_use} under account {account_id_use}")
    
    # Validate required credentials
    if not API_TOKEN:
        raise Exception("[ERROR] Missing required credentials. Please check your environment variables.")
    
    try:
        headers = {
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if account_id_use:
            headers["X-Account-Id"] = account_id_use
            print(f"[INFO] Added account ID {account_id_use} to request headers")
        
        # Get query ID
        queries_url = f"https://api.brandwatch.com/projects/{project_id_use}/queries"
        queries_response = make_request(queries_url, headers)
        
        if not queries_response.get('results'):
            raise Exception("[ERROR] No queries found in the project. Please check your PROJECT_ID.")
        
        query_id = queries_response['results'][0]['id']
        print(f"[INFO] Using query ID: {query_id}")
        
        # Build API URL with specific parameters for university tags
        base_url = (
            f"https://api.brandwatch.com/projects/{project_id_use}/data/mentions/fulltext"
            f"?startDate={start_date}T00:00:00"
            f"&endDate={end_date}T23:59:59"
            f"&pageSize=5000"
            f"&queryId={query_id}"
            f"&includeTags=true"
            f"&includeTagCategories=true"
            f"&includeCategories=true"
        )
        
        if account_id_use:
            base_url += f"&accountId={account_id_use}"
            print(f"[INFO] Added account ID {account_id_use} to request URL")
        
        # Fetch all data with pagination
        cursor = None
        all_data = []
        while True:
            url = base_url + (f"&cursor={cursor}" if cursor else "")
            data = make_request(url, headers)
            results = data.get("results", [])
            if not results:
                break
            all_data.extend(results)
            cursor = data.get("nextCursor")
            print(f"[INFO] Fetched {len(results)} mentions, total so far: {len(all_data)}")
            if len(results) < 5000 or not cursor:
                break
        
        return all_data
        
    except Exception as e:
        print(f"[ERROR] Failed to get data from Brandwatch API: {e}")
        raise

def convert_to_hk_time(date_str):
    """Convert a date string to Hong Kong time."""
    try:
        if date_str is None:
            return None

        if pd is not None:
            dt = pd.to_datetime(date_str)
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC")
        else:
            # Basic ISO 8601 parsing without pandas
            if date_str.endswith("Z"):
                date_str = date_str[:-1] + "+00:00"
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("UTC"))

        hk_tz = ZoneInfo("Asia/Hong_Kong")
        return dt.astimezone(hk_tz)
    except Exception as e:  # pragma: no cover - defensive programming
        print(f"Error processing date {date_str}: {e}")
        return None

# --- Topic mention classification and summary functions ---

def classify_mention(text):
    """Classify a social media post for sentiment (positive / neutral / negative) using AI, with fallback."""
    prompt = f"""
Given this social media post about Hong Kong universities:
{text}

Sentiment? Only choose: positive, neutral, or negative.

Return your answer in this JSON format, using only the allowed values:
{{
  "sentiment": "positive" | "negative" | "neutral",
  "explanation": "reason for the label, quoting key words"
}}
""".strip()
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",  # Using the mini model as specified
            messages=[
                {"role": "system", "content": "You are a strict sentiment classifier. Reply ONLY with valid JSON as in user's instruction."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=60,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        raw_content = response.choices[0].message.content.strip()
        # Some models wrap JSON in markdown fences ```json ... ``` ; strip them if present
        if raw_content.startswith("```"):
            # Remove first and last fence
            raw_content = re.sub(r"```(?:json)?", "", raw_content, flags=re.IGNORECASE).strip()
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3].strip()

        # Try ast.literal_eval for single-quoted dicts
        try:
            classification = ast.literal_eval(raw_content)
            if isinstance(classification, dict):
                return classification
        except Exception:
            pass

        # Try to extract the first JSON-like object substring
        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if match:
            block = match.group(0)
            try:
                classification = json.loads(block)
                return classification
            except Exception:
                try:
                    classification = ast.literal_eval(block)
                    if isinstance(classification, dict):
                        return classification
                except Exception:
                    pass

        # If the model returned just a word like "positive"
        simple = raw_content.strip().lower()
        if simple in ("positive", "negative", "neutral"):
            return {"sentiment": simple, "explanation": "[model simple output]"}

        # If still failing, raise to trigger fallback
        raise ValueError("Could not parse JSON from model response")
    except Exception as e:
        print(f"[ERROR] AI sentiment classification failed: {e}")
        # Simple rule-based fallback: detect sentiment keywords
        text_lower = text.lower() if isinstance(text, str) else ""
        pos = any(kw in text_lower for kw in ['great','excellent','good','amazing','love','recommend','best','perfect'])
        neg = any(kw in text_lower for kw in ['bad','terrible','disappointing','dirty','avoid','complaint','worst'])
        sent = "positive" if pos else "negative" if neg else "neutral"
        return {
            "sentiment": sent, "explanation": "[fallback rule-based]"
        }

def classify_mentions(_):
    """Deprecated helper kept for backward compatibility (no longer used)."""
    return {}

def generate_ai_summary(topic_mentions):
    """Generate sentiment & thematic analysis for a university topic using OpenAI."""
    if not topic_mentions or not OPENAI_API_KEY:
        return {
            "overall": "No data available for summary.",
            "analysis": "No content found.",
            "insights": "No insights available.",
            "urls": []
        }

    try:
        topic_name = topic_mentions[0].get('topic', 'this university').replace('university_', '').upper()

        # Gather mention texts and urls
        unique_urls = {m.get('originalUrl', '') for m in topic_mentions if m.get('originalUrl')}

        texts = []
        for m in topic_mentions:
            title_part = m.get('title', '')
            body_part = m.get('fullText', '') or m.get('content', '')
            combined = f"{title_part} {body_part}".strip()
            texts.append(combined)

        posts_blob = "\n---\n".join(texts)

        context_note = TOPIC_CONTEXT.get(topic_name, "")

        analysis_prompt = f"""
# Social Media Analysis for {topic_name}

You are an experienced higher-education analyst.  Analyse {len(topic_mentions)} social-media posts that mention {topic_name}.  The full text of every post is between <posts> tags.

<posts>
{posts_blob}
</posts>

---

Return the report in Markdown using ONLY the sections described below and in the same order.  Follow every formatting rule exactly.

## 1. Sentiment Analysis
Create a two-column markdown table that shows the share of posts that are Positive, Neutral and Negative.  Percentages MUST be written in numerals with the % sign (e.g. 65%).  The table must look like:

| Sentiment | Percentage |
|-----------|------------|
| Positive  | 00% |
| Neutral   | 00% |
| Negative  | 00% |

After the table add 2-3 sentences that briefly explain the sentiment distribution.

## 2. Notable Mentions
Group the posts into 3-6 topical clusters (e.g. "Research & Innovation", "Student Life", "International Partnerships", etc.).  For EACH cluster, provide:
* A short bold heading for the cluster.
* A concise summary paragraph (8-15 sentences) that synthesises the key points, citing concrete details (names, numbers, initiatives) drawn from the posts.

When deciding clusters, prioritise these commonly observed buckets if relevant (you may merge or omit if not present, and you may add new clusters):
- Academic productivity & assessment
- Commercial services for students & academics
- Academic-integrity issues & scandals
- Housing, real-estate & student accommodation
- Study-abroad, career & professional advice
- Gov't policy, funding & international ties
- Student life / on-campus events
- Every-day campus life & opportunities
- Scientific research breakthroughs
- Innovation & entrepreneurship
- Conferences, forums & academic events

List the clusters in order of volume (largest first).  Ensure this section is substantially informative (≈150-200 words total).

## 3. Strategic Insights
Provide 3–4 numbered, actionable recommendations informed by the analysis.  Keep each recommendation to ≤40 words.

Important style rules:
- Use numerals for all percentages and figures.
- Do NOT include any other sections (e.g. remove the previous "Key Themes & Opportunities" section).
- Use clear professional tone.
- Avoid raw URLs inside the narrative; they will be listed separately.
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[
                {
                    "role": "system",
                    "content": """You are a higher education analyst. Format your response with:
1. Clear markdown tables with proper column alignment
2. No special characters in regular text that could break table formatting
3. Consistent spacing between sections
4. Proper markdown heading levels (##)"""
                },
                {"role": "user", "content": analysis_prompt}
            ],
            max_tokens=800,
            temperature=0.2
        )
        
        analysis_md = response.choices[0].message.content.strip()
        
        # Post-process the response to ensure proper table formatting
        def fix_table_formatting(text):
            lines = text.split('\n')
            fixed_lines = []
            in_table = False
            
            for line in lines:
                if '|' in line:  # Potential table row
                    if not in_table:
                        in_table = True
                    # Ensure proper spacing around | characters
                    fixed_line = '|' + '|'.join(cell.strip() for cell in line.split('|')[1:-1]) + '|'
                    fixed_lines.append(fixed_line)
                else:
                    if in_table:
                        in_table = False
                    fixed_lines.append(line)
            
            return '\n'.join(fixed_lines)
        
        analysis_md = fix_table_formatting(analysis_md)

        # Format URLs as a compact, wrapped list
        url_list = list(unique_urls)
        formatted_urls = '<div class="source-links">Sources: ' + \
            ' '.join(f'<a href="{url}" target="_blank">link{i+1}</a>{", " if i < len(url_list)-1 else ""}' 
                    for i, url in enumerate(url_list)) + '</div>'

        return {
            "overall": f"Analysis based on {len(topic_mentions)} mentions of {topic_name}",
            "analysis": analysis_md,
            "insights": "",  # Insights are now part of the main analysis
            "urls": formatted_urls,
        }

    except Exception as e:
        print(f"[ERROR] Failed to generate AI summary: {e}")
        return {
            "overall": f"Unable to generate summary: {str(e)}",
            "analysis": "No content found.",
            "insights": "No insights available.",
            "urls": []
        }

def generate_topic_summaries(df, end_date):
    """Generate AI summaries for each topic's mentions on the end date"""
    if df.empty:
        print("[WARNING] No data available for summaries")
        return {}
    
    print(f"[DEBUG] Generating summaries for end date: {end_date}")
    print(f"[DEBUG] DataFrame date range: {df['date_day'].min()} to {df['date_day'].max()}")
    print(f"[DEBUG] Date values in DataFrame: {sorted(df['date_day'].unique())}")
    
    # Filter data for end date
    end_date_df = df[df['date_day'] == end_date]
    print(f"[DEBUG] Found {len(end_date_df)} mentions for end date {end_date}")
    
    if end_date_df.empty:
        print(f"[WARNING] No mentions found for end date {end_date}")
        return {}
    
    # Group by topic
    topic_groups = end_date_df.groupby('topic')
    
    # Generate summaries
    summaries = {}
    for topic, topic_df in topic_groups:
        topic_mentions = topic_df.to_dict('records')
        if topic_mentions:
            print(f"[DEBUG] Generating summary for {topic} with {len(topic_mentions)} mentions")
            summaries[topic] = generate_ai_summary(topic_mentions)
        else:
            print(f"[WARNING] No mentions found for {topic} on {end_date}")
    
    # Make sure all topics have a summary
    for topic in TOPIC_LIST:
        if topic not in summaries:
            print(f"[DEBUG] Adding default summary for {topic} (no mentions found)")
            summaries[topic] = {
                "overall": "No mentions found for this topic on the end date.",
                "analysis": "No content found.",
                "insights": "No insights available.",
                "urls": []
            }
    
    print(f"[INFO] Generated summaries for {len(summaries)} topics")
    return summaries

def process_brandwatch_data(data, start_dt, end_dt, topics_df):
    """Process Brandwatch data"""
    if not data:
        print("[ERROR] No data received from API")
        return pd.DataFrame()
    
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    
    # Print detailed debug info
    print(f"[DEBUG] Raw data count: {len(df)}")
    print(f"[DEBUG] Available columns in Brandwatch data: {df.columns.tolist()}")
    
    # Check if tags field exists
    if 'tags' not in df.columns:
        print("[ERROR] 'tags' field not found in data. Cannot identify topics.")
        print("[DEBUG] Sample data (first record):")
        if len(df) > 0:
            print(df.iloc[0].to_dict())
        return pd.DataFrame()
    
    # Check necessary columns
    required_columns = ['originalUrl', 'date', 'tags']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"[ERROR] Missing critical columns in Brandwatch data: {missing_columns}")
        return pd.DataFrame()
    
    # Map the actual API columns to expected columns
    print("[INFO] Mapping API columns to expected structure...")
    
    # Handle date column (already checked it exists)
    df['date_crawled'] = df['date']
    
    # Use fullText for content if available
    if 'fullText' in df.columns:
        print("[INFO] Using 'fullText' as content field")
        df['content'] = df['fullText'].fillna('')
    else:
        print("[WARNING] 'fullText' column not found, checking alternatives")
        content_fields = ['snippet', 'text']
        found_content = False
        for field in content_fields:
            if field in df.columns:
                print(f"[INFO] Using '{field}' as content field")
                df['content'] = df[field].fillna('')
                found_content = True
                break
        
        if not found_content:
            print("[WARNING] No content field found, using empty values")
            df['content'] = ''
    
    # Ensure all required columns exist for deduplication
    subset_cols = [col for col in ['originalUrl', 'date_crawled'] if col in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols)
        print(f"[DEBUG] After deduplication: {len(df)} mentions")
    
    # Convert crawl date to Hong Kong time
    df['date_hk'] = df['date_crawled'].apply(convert_to_hk_time)
    print(f"[DEBUG] After date conversion: {len(df.dropna(subset=['date_hk']))} valid dates")
    
    # Drop rows with failed date conversions
    df = df.dropna(subset=['date_hk'])
    
    # Create date_day column (date only, no time)
    df['date_day'] = df['date_hk'].dt.strftime('%Y-%m-%d')
    
    # Filter by date range
    df['date_naive'] = df['date_hk'].dt.tz_localize(None)
    end_dt_inclusive = (end_dt + timedelta(days=1)) - timedelta(seconds=1)
    df = df[(df['date_naive'] >= start_dt) & (df['date_naive'] <= end_dt_inclusive)]
    print(f"[INFO] After date filtering: {len(df)} mentions")
    df = df.drop(columns=['date_naive'])
    
    # Print the tags we're looking for
    print(f"[INFO] Looking for these tags: {TAG_NAMES}")
    
    # Process mentions and assign tags
    processed_data = []
    tag_counts = {tag: 0 for tag in TAG_NAMES}
    
    for idx, row in df.iterrows():
        # Get tags from the tags field
        raw_tags = row.get('tags', [])
        if isinstance(raw_tags, str):
            try:
                # Try parsing as JSON if it's a string
                tags_list = json.loads(raw_tags)
            except:
                # If not JSON, treat as single tag
                tags_list = [raw_tags]
        elif isinstance(raw_tags, list):
            tags_list = raw_tags
        else:
            tags_list = [str(raw_tags)]
        
        # Find exact matches with our tag names
        topic_tags = [t for t in tags_list if t in TAG_NAMES]
        
        # Debug the first few rows
        if idx < 3:
            print(f"\n[DEBUG] Row {idx}:")
            print(f"Raw tags: {raw_tags}")
            print(f"Parsed tags: {tags_list}")
            print(f"Matched tags: {topic_tags}")
        
        # Skip if no matching tags found
        if not topic_tags:
            continue
        
        # Update tag counts
        for tag in topic_tags:
            tag_counts[tag] += 1
        
        # Get basic fields
        originalUrl = row['originalUrl']
        date_hk = row['date_hk']
        date_day = row['date_day']
        content = row.get('content', '')
        
        # Add a row for each matched tag
        for tag in topic_tags:
            topic_display = TAG_TO_TOPIC.get(tag)
            if not topic_display:
                print(f"[WARNING] Tag {tag} not found in TAG_TO_TOPIC mapping, skipping")
                continue
            
            processed_data.append({
                'originalUrl': originalUrl,
                'date_hk': date_hk,
                'date_day': date_day,
                'topic': topic_display,
                'tag': tag,
                'fullText': content
            })
    
    print(f"\n[INFO] Processing Summary:")
    print(f"Total mentions processed: {len(df)}")
    print(f"Final mentions after processing: {len(processed_data)}")
    print("\n[INFO] Tag counts:")
    for tag, count in tag_counts.items():
        topic_name = TAG_TO_TOPIC.get(tag, "Unknown")
        print(f"  - {tag} ({topic_name}): {count}")
    
    # Convert to DataFrame
    result_df = pd.DataFrame(processed_data)
    print(f"[INFO] Final processed data: {len(result_df)} topic mentions")
    
    return result_df

def calculate_buzz_volume(df):
    """Calculate buzz volume as unique URL count per topic and date."""
    if pd is not None and hasattr(df, "empty"):
        if df.empty:
            print("[WARNING] No data to calculate buzz volume")
            return pd.DataFrame()

        print(f"[DEBUG] Buzz calculation input: {df.shape}")
        buzz = df.groupby(['topic', 'date_day']).originalUrl.nunique().reset_index(name='buzz_volume')
        print(f"[INFO] Calculated buzz volume using unique originalUrls")
        print("[DEBUG] Raw buzz counts:")
        print(buzz)

        pivot = buzz.pivot(index='topic', columns='date_day', values='buzz_volume')
        try:
            dr = date_range
        except NameError:
            dr = sorted(pivot.columns)
        pivot = pivot.reindex(columns=dr)

        date_format_map = {d: datetime.strptime(d, '%Y-%m-%d').strftime('%d %b') for d in pivot.columns}
        pivot = pivot.rename(columns=date_format_map)
        pivot = pivot.reindex(TOPIC_LIST)
        pivot = pivot.fillna(0).astype(int)
        print("[DEBUG] Final pivot table shape:", pivot.shape)
        return pivot

    # Fallback implementation without pandas
    records = list(df)
    if not records:
        print("[WARNING] No data to calculate buzz volume")
        return {}

    # Count unique URLs by topic/date
    counts = {}
    unique_dates = set()
    for row in records:
        topic = row['topic']
        date = row['date_day']
        unique_dates.add(date)
        counts.setdefault(topic, {}).setdefault(date, set()).add(row['originalUrl'])

    try:
        dr = date_range
    except NameError:
        dr = sorted(unique_dates)

    result = {}
    for topic in TOPIC_LIST:
        row = {}
        for d in dr:
            row[d] = len(counts.get(topic, {}).get(d, set()))
        result[topic] = row

    date_map = {d: datetime.strptime(d, '%Y-%m-%d').strftime('%d %b') for d in dr}
    formatted = {
        topic: {date_map[d]: v for d, v in vals.items()}
        for topic, vals in result.items()
    }
    return formatted

def create_charts(pivot_data):
    """
    Create visualization charts from the buzz volume data
    """
    if pivot_data is None or pivot_data.empty:
        print("No data available for charts")
        return None
    
    # Get topics and dates
    topics = pivot_data.index.tolist()
    dates = pivot_data.columns.tolist()
    
    if not dates:
        print("No date columns available for charts")
        return None
    
    # Sort topics by total volume for better visual
    topic_totals = pivot_data.sum(axis=1).sort_values(ascending=False)
    sorted_topics = topic_totals.index.tolist()
    
    # Create stacked bar chart – larger size for better readability
    plt.figure(figsize=(24, 12))
    
    bottom = np.zeros(len(dates))
    for topic in sorted_topics:
        if topic in pivot_data.index:
            values = pivot_data.loc[topic, dates].values
            # Get the color from TAG_COLORS, fallback to gray if not found
            color = TAG_COLORS.get(topic, "#cccccc")
            plt.bar(dates, values, label=topic.replace('university_', '').upper(), 
                   color=color, bottom=bottom)
            bottom += values
    
    plt.ylabel('Buzz Volume', fontsize=16, fontweight='bold')
    plt.title('Daily Social Media Mentions by University', fontsize=18, fontweight='bold', pad=20)
    plt.xticks(rotation=45, fontsize=14, ha='right')
    plt.yticks(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=14, 
              title="Universities", title_fontsize=16)
    
    # Add some padding to prevent label cutoff
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    return chart_base64

def generate_html_report(pivot_data, chart_base64, summaries=None):
    """Generate HTML report with buzz volume table and chart"""
    # Make a copy to avoid modifying the original
    pivot_for_html = pivot_data.copy()
    
    # Drop the date_formatted column if it exists
    if 'date_formatted' in pivot_for_html.columns:
        pivot_for_html = pivot_for_html.drop(columns=['date_formatted'])
    
    # Convert pivot table to HTML without the index name
    pivot_reset = pivot_for_html.reset_index()
    
    # Get the topic column name (first column after reset_index)
    topic_col = pivot_reset.columns[0]
    
    # Generate the buzz volume table HTML
    table_html = """
    <table class="buzz-table">
        <tr>
            <th>Topic</th>
    """
    
    # Add column headers (dates)
    for col in pivot_reset.columns[1:]:  # Skip the first column which is the topic name
        table_html += f"<th>{col}</th>\n"
    table_html += "</tr>\n"
    
    # Add rows
    for _, row in pivot_reset.iterrows():
        table_html += "<tr>\n"
        # Add topic name - using the actual column name
        topic_display = row[topic_col].replace('university_', '').upper()
        table_html += f"<td>{topic_display}</td>\n"
        # Add values - Use column names instead of positions
        for col in pivot_reset.columns[1:]:
            table_html += f"<td>{row[col]}</td>\n"
        table_html += "</tr>\n"
    
    table_html += "</table>"
    
    # Create topic summaries section with analysis
    summaries_section = """
    <div class="analysis-sections">
    """
    
    if summaries:
        for topic in TOPIC_LIST:
            summary = summaries.get(topic, {})
            topic_display = topic.replace('university_', '').upper()
            
            combined_md = summary.get("analysis", "No analysis available.")
            combined_html = md_to_html(combined_md, extensions=['tables'])
            
            summaries_section += f"""
            <div class="analysis-section">
                <h3>{topic_display}</h3>
                <div class="analysis-content">
                    {combined_html}
                </div>
                {summary.get("urls", "")}
            </div>
            """
    
    summaries_section += "</div>"

    html = f"""
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        /* Headers */
        h1, h2, h3, h4 {{ 
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.8em;
        }}
        h1 {{ font-size: 2em; }}
        h2 {{ font-size: 1.5em; }}
        h3 {{ font-size: 1.3em; }}
        h4 {{ font-size: 1.1em; }}
        
        /* Tables */
        table {{
            border-collapse: collapse;
            margin: 1em 0;
            width: 100%;
            background: white;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
            vertical-align: top;
        }}
        .buzz-table td {{
            text-align: center;
        }}
        th {{
            background: #f5f7fa;
            font-weight: 600;
            color: #2c3e50;
        }}
        tr:nth-child(even) {{
            background: #f9fafb;
        }}
        
        /* Analysis Sections */
        .analysis-section {{
            background: white;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 20px;
            margin: 20px 0;
        }}
        .analysis-section h2 {{
            margin-top: 0;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }}
        .analysis-content {{
            margin-top: 15px;
        }}
        
        /* Charts */
        .chart-container {{
            background: white;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 20px;
            margin: 20px 0;
        }}
        
        /* Links */
        .source-links {{
            margin: 15px 0;
            padding: 10px 15px;
            background: #f8f9fa;
            border-radius: 4px;
            line-height: 1.6;
            font-size: 0.9em;
            white-space: normal;
            word-wrap: break-word;
        }}
        .source-links a {{
            color: #0366d6;
            text-decoration: none;
            margin-right: 2px;
            display: inline;
        }}
        .source-links a:hover {{
            text-decoration: underline;
        }}
        
        /* Content Sections */
        .content-section {{
            margin-bottom: 2em;
        }}
        
        /* Markdown Content */
        .markdown-content {{
            line-height: 1.6;
        }}
        .markdown-content p {{
            margin: 1em 0;
        }}
        .markdown-content ul {{
            margin: 1em 0;
            padding-left: 2em;
        }}
        .markdown-content li {{
            margin: 0.5em 0;
        }}
    </style>
</head>
<body>
    <h1>CUHKPRO Report</h1>
    
    <div class="content-section">
        <h2>Daily Mention Volume</h2>
        {table_html}
    </div>
    
    <div class="chart-container">
        <h2>Trend Analysis</h2>
        <img src="data:image/png;base64,{chart_base64}" style="max-width: 100%; height: auto;"/>
    </div>
    
    <div class="content-section">
        <h2>Detailed Analysis by University</h2>
        {summaries_section}
    </div>
</body>
</html>
"""
    return html

def process_report(start_date, end_date, project_id=None, account_id=None, region_label=None):
    """Process the report with given date range and return path to HTML report"""
    try:
        # Create output directory if it doesn't exist
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate date range (for sorting)
        global date_range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        date_range = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_range.append(current_dt.strftime('%Y-%m-%d'))
            current_dt += timedelta(days=1)
        
        # Fetch raw data from Brandwatch API
        print(f"[INFO] Fetching data from Brandwatch API for region {region_label or 'default'}...")
        raw_data = get_brandwatch_data(start_date, end_date, project_id=project_id, account_id=account_id)
        
        # Process data - explode tags and calculate unique URL counts
        # Pass the date range to ensure proper filtering
        df = process_brandwatch_data(raw_data, start_dt, end_dt, topics_df)
        
        # Check if we have data to process
        if df.empty:
            print("[WARNING] No data available for processing.")
            # Create a simple report saying no data available
            simple_html = """
<html>
<body>
<p>Hello,</p>
<p>We are pleased to inform you that your scheduled export for the "Hong Kong Universities Overview" dashboard is now ready, but no data was found for the specified period.</p>
</body>
</html>
"""
            filename_base = "hk_uni_buzz_report"
            if region_label:
                filename_base += f"_{region_label.lower()}"
            html_path = os.path.join(output_dir, f"{filename_base}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(simple_html)
            print(f"[INFO] Empty report generated: {html_path}")
            return html_path
        
        # Calculate buzz volume
        buzz_pivot = calculate_buzz_volume(df)
        
        # Create chart
        chart_base64 = create_charts(buzz_pivot)
        
        # Generate AI summaries for each topic on the end date
        print(f"[INFO] Generating AI summaries for topics on {end_date}...")
        summaries = generate_topic_summaries(df, end_date)
        print(f"[DEBUG] Summaries generated: {len(summaries)}")
        
        # Generate HTML report
        html_report = generate_html_report(buzz_pivot, chart_base64, summaries)
        
        # Save HTML report – include region label in filename if provided
        filename_base = "hk_uni_buzz_report"
        if region_label:
            filename_base += f"_{region_label.lower()}"
        html_path = os.path.join(output_dir, f"{filename_base}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_report)
        
        print(f"[INFO] HTML report generated: {html_path}")
        
        return html_path
    
    except Exception as e:
        print(f"[ERROR] Error in process_report function: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

def run_reports(job_id: str, start_date: str, end_date: str):
    """Background task that builds a single Brandwatch report plus a Perplexity web-search section."""
    try:
        JOB_STORE[job_id]["status"] = "running"
        JOB_STORE[job_id]["step"] = "Fetching Brandwatch data"

        # 1) Brandwatch report (single region)
        print(f"[JOB {job_id}] Generating Brandwatch report")
        path = process_report(start_date, end_date, account_id=BRANDWATCH_ACCOUNT_ID)
        with open(path, "r", encoding="utf-8") as fp:
            html_segment = fp.read()
        import re as _re
        body_match = _re.search(r"<body[^>]*>([\s\S]*?)</body>", html_segment, _re.IGNORECASE)
        brandwatch_inner = body_match.group(1) if body_match else html_segment
        combined_sections = "<h2 style='margin-top:40px;'>Brandwatch Report</h2>\n" + brandwatch_inner + "<hr/>"
        JOB_STORE[job_id]["step"] = "Brandwatch report done"

        # 2) Perplexity search section
        if PERPLEXITY_API_KEY:
            JOB_STORE[job_id]["step"] = "Performing web search"
            print(f"[JOB {job_id}] Searching web for HK university mentions (last 24h)")

            uni_queries = {
                "CUHK": "CUHK OR 'Chinese University of Hong Kong' OR 香港中文大學",
                "HKU": "HKU OR 'The University of Hong Kong' OR 香港大學",
                "CITYU": "CityU OR 'City University of Hong Kong' OR 香港城市大學",
                "HKUST": "HKUST OR 'Hong Kong University of Science and Technology' OR 香港科技大學",
                "POLYU": "PolyU OR 'The Hong Kong Polytechnic University' OR 香港理工大學"
            }

            web_sections = ""
            for uni, q in uni_queries.items():
                mentions = search_web_mentions(q + " site:news OR social", hours=24)

                web_sections += f"<h4>{uni}</h4>"
                if mentions:
                    web_sections += f"<p>Found {len(mentions)} relevant web mentions in the last 24 hours.</p>"

                    # Build bullet list of title + link
                    web_sections += "<ul class='web-sources'>"
                    for m in mentions:
                        title = (m.get('title') or 'link').strip()
                        url = m.get('url')
                        snippet = m.get('snippet', '').strip()
                        web_sections += f"<li><a href='{url}' target='_blank' rel='noopener'>{title}</a> – {snippet}</li>"
                    web_sections += "</ul>"
                else:
                    web_sections += "<p>No public web mentions found in the last 24 hours.</p>"

            if web_sections:
                combined_sections += "<h2 style='margin-top:40px;'>Recent Web Mentions (Last 24 h)</h2>" + web_sections + "<hr/>"
            else:
                combined_sections += "<h2 style='margin-top:40px;'>Recent Web Mentions (Last 24 h)</h2><p>No relevant web mentions found.</p><hr/>"

            JOB_STORE[job_id]["step"] = "web search done"

        if not combined_sections:
            raise RuntimeError("No sections generated")

        JOB_STORE[job_id]["step"] = "assembling final HTML"
        final_html = f"""
        <html>
        <head>
            <title>CUHKPRO – Combined Reports</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                hr {{ border: 1px solid #ccc; margin: 40px 0; }}
                .web-analysis {{ margin-top: 15px; }}
                .analysis-content {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px; }}
                .web-sources {{ max-height: 200px; overflow-y: auto; }}
            </style>
        </head>
        <body>
            <h1>CUHKPRO – Combined Reports</h1>
            {combined_sections}
        </body>
        </html>
        """
        JOB_STORE[job_id].update({"status": "done", "html": final_html, "step": "done"})
    except Exception as exc:
        JOB_STORE[job_id].update({"status": "error", "error": str(exc), "step": "error"})

def gradio_interface(start_date, end_date):
    """Gradio interface for report generation"""
    try:
        # No date validation - use exactly what user provided
        print(f"[INFO] Processing report for date range: {start_date} to {end_date}")
        
        # Process the report with the given date range
        result = process_report(start_date, end_date)
        
        # Check if there was an error
        if isinstance(result, str) and result.startswith("Error:"):
            return result
        
        # Read the HTML file content to display directly in the interface
        with open(result, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return html_content
    except Exception as e:
        return f"Error generating report: {str(e)}"

def main():
    """
    Start a FastAPI server that renders a very small HTML form to collect a
    date range and then returns the generated Hong Kong universities report.

    After running the script, open http://localhost:8081 in your browser.

    To expose the service on your own (sub-)domain through Cloudflare Tunnel
    you can run **outside of Python** (after `cloudflared` is installed &
    authenticated):

        cloudflared tunnel --url http://localhost:8081 \
            --hostname charles.tocanan.ai

    Replace `YOUR_SUBDOMAIN.YOURDOMAIN.com` with the hostname you have added to
    Cloudflare. The tunnel will forward traffic from that hostname to the
    locally running FastAPI server.
    """

    # Default dates for the form
    default_start, default_end = get_default_dates()

    app = FastAPI(title="CUHKPRO Report")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Return a form with spinner to generate CUHKPRO reports."""
        return HTMLResponse(content=f"""
        <html>
        <head>
            <title>CUHKPRO Report</title>
            <style>
                .loader {{
                    border: 8px solid #f3f3f3;
                    CUHKPRO reports will be generated. You will be automatically redirected when they are ready. border-top: 8px solid #3498db;
                    border-radius: 50%;
                    width: 60px;
                    height: 60px;
                    animation: spin 1s linear infinite;
                    margin: 40px auto;
                }}
                @keyframes spin {{
                    0%   {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
            </style>
            <script>
                function submitReport(event) {{
                    event.preventDefault();
                    document.getElementById('spinner').style.display = 'block';
                    const formData = new FormData(event.target);
                    fetch('/generate', {{ method: 'POST', body: formData }})
                        .then(resp => resp.text())
                        .then(html => {{
                            // Replace entire document with the generated report
                            document.open();
                            document.write(html);
                            document.close();
                        }})
                        .catch(err => {{
                            console.error(err);
                            document.getElementById('spinner').innerHTML = '<p style=\"color:red;\">Error generating report.</p>';
                        }});
                }}
            </script>
        </head>
        <body style='font-family: Arial, sans-serif; max-width: 800px; margin: auto;'>
            <h2>CUHKPRO Report</h2>
            <form id='reportForm' onsubmit='submitReport(event)'>
                <label>Start Date (YYYY-MM-DD):</label><br/>
                <input type='text' name='start_date' value='{default_start}' style='width: 200px;' required/><br/><br/>
                <label>End Date (YYYY-MM-DD):</label><br/>
                <input type='text' name='end_date' value='{default_end}' style='width: 200px;' required/><br/><br/>
                <button type='submit' style='padding: 6px 12px;'>Generate CUHKPRO Reports</button>
            </form>
            <div id='spinner' style='display:none;'>
                <div class='loader'></div>
                <p style='text-align:center;'>Generating reports, please wait...</p>
            </div>
            <p style='margin-top:40px;font-size:0.9em;color:#555;'></p>
        </body>
        </html>
        """)

    @app.post("/generate", response_class=HTMLResponse)
    async def generate(background_tasks: BackgroundTasks, start_date: str = Form(...), end_date: str = Form(...)):
        """Enqueue report generation job and return a spinner page that polls status."""
        job_id = str(uuid.uuid4())
        JOB_STORE[job_id] = {"status": "pending", "html": None, "error": None}

        # Schedule background job
        background_tasks.add_task(run_reports, job_id, start_date, end_date)

        spinner_html = f"""
        <html>
        <head>
            <title>Generating Report…</title>
            <style>
                .loader {{
                    border: 8px solid #f3f3f3;
                    border-top: 8px solid #3498db;
                    border-radius: 50%;
                    width: 60px;
                    height: 60px;
                    animation: spin 1s linear infinite;
                    margin: 40px auto;
                }}
                @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
            </style>
            <script>
                const jobId = "{job_id}";
                async function poll() {{
                    try {{
                        const resp = await fetch(`/status/${{jobId}}`);
                        const data = await resp.json();
                        if (data.step) {{
                            document.getElementById('msg').innerText = data.step;
                        }}
                        if (data.status === 'done') {{
                            const htmlResp = await fetch(`/view/${{jobId}}`);
                            const html = await htmlResp.text();
                            document.open();
                            document.write(html);
                            document.close();
                        }} else if (data.status === 'error') {{
                            document.getElementById('msg').innerText = 'Error: ' + (data.error || 'unknown');
                        }} else {{
                            setTimeout(poll, 4000);
                        }}
                    }} catch (e) {{
                        console.error(e);
                        setTimeout(poll, 5000);
                    }}
                }}
                window.onload = poll;
            </script>
        </head>
        <body style='font-family: Arial, sans-serif; max-width: 800px; margin: auto; text-align:center;'>
            <h2>Report is being generated…</h2>
            <div class='loader'></div>
            <p id='msg'>Job ID: {job_id}</p>
            <p>This may take a few minutes – stay on this page.</p>
        </body>
        </html>
        """

        return HTMLResponse(content=spinner_html)

    @app.get("/status/{job_id}")
    async def job_status(job_id: str):
        """Return JSON status for a job."""
        job = JOB_STORE.get(job_id)
        if not job:
            return JSONResponse({"status": "not_found"}, status_code=404)
        return JSONResponse({"status": job["status"], "step": job.get("step"), "error": job.get("error")})

    @app.get("/view/{job_id}", response_class=HTMLResponse)
    async def job_view(job_id: str):
        """Return combined HTML report if done."""
        job = JOB_STORE.get(job_id)
        if not job:
            return HTMLResponse(content="<p>Job not found.</p>", status_code=404)
        if job["status"] == "done" and job.get("html"):
            return HTMLResponse(content=job["html"])
        elif job["status"] == "error":
            return HTMLResponse(content=f"<p>Error: {job.get('error')}</p>", status_code=500)
        else:
            return HTMLResponse(content="<p>Job still in progress…</p>", status_code=202)

    # Honour the PORT env variable if supplied (useful on PaaS)
    port = int(os.getenv("PORT", "8081"))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()

def top_sources_by_platform(df, platform=None):
    """
    Get top sources by platform
    """
    if df.empty:
        return pd.DataFrame()
    
    if platform:
        df = df[df['platform'] == platform]
    
    # Group by source and date
    sources = df.groupby(['source', 'date_day']).size().reset_index(name='count')
    
    # Create pivot table with sources as index and dates as columns
    pivot = sources.pivot(index='source', columns='date_day', values='count')
    
    # Sort columns chronologically
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    
    # Format date columns for display
    date_format_map = {}
    for date_day in pivot.columns:
        date_obj = datetime.strptime(date_day, '%Y-%m-%d')
        date_format_map[date_day] = date_obj.strftime('%d %b')
    
    # Rename columns using the mapping
    pivot = pivot.rename(columns=date_format_map)
    
    # Fill NaN values with 0
    pivot = pivot.fillna(0)
    
    # Calculate row totals
    pivot['Total'] = pivot.sum(axis=1)
    
    # Sort by total mentions in descending order
    pivot = pivot.sort_values('Total', ascending=False)
    
    # Return top 10 sources
    return pivot.head(10)