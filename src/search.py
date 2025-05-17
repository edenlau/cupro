#!/usr/bin/env python3
import os
import sys
import time
import json
import re
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
from openai import OpenAI
import httpx

# Load environment variables
load_dotenv(override=True)
print("Environment variables loaded")

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[ERROR] OPENAI_API_KEY is not set in your .env file")
    sys.exit(1)
print("[INFO] OpenAI API key found")

# Load Perplexity API key
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not PERPLEXITY_API_KEY:
    print("[WARNING] PERPLEXITY_API_KEY is not set in your .env file. Add PERPLEXITY_API_KEY=YOUR_KEY")

# Initialize OpenAI client without proxies
http_client = httpx.Client(trust_env=False)
client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)

# Perplexity API endpoint & model
PERPLEXITY_ENDPOINT = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_MODEL = "sonar-pro"

def search_web_mentions(query, hours=24, max_items=50):
    """
    Use Perplexity 'sonar-pro' model to search the web.

    Parameters
    ----------
    query : str
        Free-form search query (e.g. university keywords).
    hours : int, default 24
        Look-back window in hours. 24 = last day.
    max_items : int, default 50
        Maximum number of result objects to return.

    Returns
    -------
    list[dict]
        Each dict has 'title', 'snippet', 'url'.
    """
    if not PERPLEXITY_API_KEY:
        print("[ERROR] Cannot perform Perplexity search – PERPLEXITY_API_KEY missing.")
        return []

    # Build prompt – instruct model to respect recency window & JSON structure
    prompt = (
        f"Search the public web for pages from the last {hours} hours that match this query (case-insensitive):\n"
        f"{query}\n\n"
        f"Respond ONLY with minified JSON containing an array named 'results' (max {max_items} items). "
        "Each item must have 'title', 'snippet', and 'url'."
    )

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1200
    }

    print(f"[INFO] Calling Perplexity API for query (last {hours}h): {query[:80]}…")
    try:
        r = requests.post(PERPLEXITY_ENDPOINT, headers=headers, json=payload, timeout=45)
        if r.status_code != 200:
            print(f"[ERROR] Perplexity API error {r.status_code}: {r.text[:200]}")
            return []
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        # Parse JSON safely
        try:
            parsed = json.loads(content)
            results = parsed.get("results", [])
        except json.JSONDecodeError:
            # Fallback: extract first JSON object substring
            m = re.search(r"\{[\s\S]*\}", content)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    results = parsed.get("results", [])
                except Exception:
                    results = []
            else:
                results = []
        print(f"[INFO] Retrieved {len(results)} results from Perplexity")
        return results[:max_items]
    except Exception as exc:
        print(f"[ERROR] Perplexity request failed: {exc}")
        return []

def generate_ai_summary(mentions):
    """Generate analysis for LCMS & HKIS mentions using OpenAI."""
    if not mentions:
        return {
            "overall": "No data available for summary.",
            "lcms_analysis": "No content found.",
            "hkis_analysis": "No content found.",
            "insights": "No insights available.",
            "urls": []
        }

    try:
        # Gather mention texts and urls
        unique_urls = {m.get('url', '') for m in mentions if m.get('url')}

        texts = []
        for m in mentions:
            title_part = m.get('title', '')
            body_part = m.get('snippet', '')
            combined = f"{title_part} {body_part}".strip()
            texts.append(combined)

        # Combine all mention texts into a single block separated by lines
        posts_blob = "\n---\n".join(texts)

        analysis_prompt = f"""
# HKIS & LCMS Web Mentions

There are {len(mentions)} web mentions where HKIS (Hong Kong International School) and LCMS (Lutheran Church-Missouri Synod) are mentioned together. The full text of each mention is provided between <posts> tags.

<posts>
{posts_blob}
</posts>

---

**Instructions:** You are an educational and religious institutions analyst. First, briefly estimate the overall sentiment distribution (positive / neutral / negative) for these mentions. Then provide:

1. LCMS (Lutheran Church-Missouri Synod) Analysis: Summarize the mentions related to LCMS. Include key themes, concerns, and sentiment.

2. HKIS (Hong Kong International School) Analysis: Summarize the mentions related to HKIS. Include key themes, concerns, and sentiment.

3. Insights: Provide 2-3 key insights about the relationship between LCMS and HKIS based on these mentions.

Write in markdown format with clear section headings.
"""

        try:
            analysis_response = client.responses.create(
                model="gpt-4.1-mini-2025-04-14",
                input=analysis_prompt,
                temperature=0.2
            )
            analysis_content = analysis_response.output_text.strip()

            # Parse sections using flexible regex for headings
            lcms_analysis = "Analysis unavailable."
            hkis_analysis = "Analysis unavailable."
            insights = "No insights available."

            def extract_section(text, header):
                """Extract section content following a heading that may be markdown (#) or numbered list (e.g., 1. LCMS)."""
                pattern = rf"(?is)(?:^|\n)(?:#+\s*|\d+\.\s*){header}[^\n]*\n([\s\S]*?)(?=\n(?:#+\s*|\d+\.\s*)(?:LCMS|HKIS|Insights)|$)"
                m = re.search(pattern, text)
                return m.group(1).strip() if m else None

            lcms_text = extract_section(analysis_content, "LCMS")
            if lcms_text:
                lcms_analysis = lcms_text

            hkis_text = extract_section(analysis_content, "HKIS")
            if hkis_text:
                hkis_analysis = hkis_text

            insights_text = extract_section(analysis_content, "Insights")
            if insights_text:
                insights = insights_text

            # Fallback: if analyses still unavailable but we have content, assign whole content
            if lcms_analysis.startswith("Analysis unavailable") and hkis_analysis.startswith("Analysis unavailable") and analysis_content:
                lcms_analysis = analysis_content
                hkis_analysis = ""
                insights = ""

        except Exception as e:
            print(f"[ERROR] Analysis generation failed: {e}")
            lcms_analysis = "Analysis generation failed."
            hkis_analysis = "Analysis generation failed."
            insights = "Insights generation failed."

        overall_summary = f"Found {len(mentions)} web mentions where HKIS and LCMS are mentioned together in the last 48 hours (unique URLs: {len(unique_urls)})."

        return {
            "overall": overall_summary,
            "lcms_analysis": lcms_analysis,
            "hkis_analysis": hkis_analysis,
            "insights": insights,
            "urls": list(unique_urls),
        }

    except Exception as e:
        print(f"[ERROR] Failed to generate AI summary: {e}")
        return {
            "overall": f"Unable to generate summary: {str(e)}",
            "lcms_analysis": "No content found.",
            "hkis_analysis": "No content found.",
            "insights": "No insights available.",
            "urls": []
        }

def main():
    """Main function to run the script"""
    print("\n=== HKIS & LCMS Web Mention Finder ===\n")
    
    # Search for mentions in the last 2 days (48 hours)
    days = 2
    query = "HKIS Hong Kong International School AND LCMS Lutheran Church-Missouri Synod"
    mentions = search_web_mentions(query)
    
    if not mentions:
        print(f"[INFO] No mentions found in the last {days} days.")
        return
    
    print(f"[INFO] Found {len(mentions)} web mentions of HKIS and LCMS")
    
    # Generate summary
    summary = generate_ai_summary(mentions)
    
    # Print summary
    print("\n=== Summary of HKIS & LCMS Web Mentions ===\n")
    print(summary["overall"])
    print("\n--- LCMS Analysis ---")
    print(summary["lcms_analysis"])
    print("\n--- HKIS Analysis ---")
    print(summary["hkis_analysis"])
    print("\n--- Insights ---")
    print(summary["insights"])
    print("\n--- Source URLs ---")
    for url in summary["urls"]:
        print(f"- {url}")
    
    # Save results to a file
    output_file = f"hkis_lcms_web_mentions_{datetime.now().strftime('%Y%m%d')}.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"=== Summary of HKIS & LCMS Web Mentions ({datetime.now().strftime('%Y-%m-%d')}) ===\n\n")
            f.write(f"{summary['overall']}\n\n")
            f.write("--- LCMS Analysis ---\n")
            f.write(f"{summary['lcms_analysis']}\n\n")
            f.write("--- HKIS Analysis ---\n")
            f.write(f"{summary['hkis_analysis']}\n\n")
            f.write("--- Insights ---\n")
            f.write(f"{summary['insights']}\n\n")
            f.write("--- Source URLs ---\n")
            for url in summary["urls"]:
                f.write(f"- {url}\n")
        print(f"\n[INFO] Results saved to {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save results to file: {e}")

if __name__ == "__main__":
    main() 