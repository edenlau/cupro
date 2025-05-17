#!/usr/bin/env python3
import os
import sys
import time
import json
import re
import logging
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
from openai import OpenAI
import httpx

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)
logger.info("Environment variables loaded")

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in your .env file")
    sys.exit(1)
logger.info("OpenAI API key found")

# Load Perplexity API key
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not PERPLEXITY_API_KEY:
    logger.warning("PERPLEXITY_API_KEY is not set in your .env file. Add PERPLEXITY_API_KEY=YOUR_KEY")

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
        logger.error("Cannot perform Perplexity search – PERPLEXITY_API_KEY missing.")
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

    logger.info("Calling Perplexity API for query (last %s h): %s…", hours, query[:80])
    try:
        r = requests.post(PERPLEXITY_ENDPOINT, headers=headers, json=payload, timeout=45)
        if r.status_code != 200:
            logger.error("Perplexity API error %s: %s", r.status_code, r.text[:200])
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
        logger.info("Retrieved %s results from Perplexity", len(results))
        return results[:max_items]
    except Exception as exc:
        logger.error("Perplexity request failed: %s", exc)
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
            logger.error("Analysis generation failed: %s", e)
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
        logger.error("Failed to generate AI summary: %s", e)
        return {
            "overall": f"Unable to generate summary: {str(e)}",
            "lcms_analysis": "No content found.",
            "hkis_analysis": "No content found.",
            "insights": "No insights available.",
            "urls": []
        }

def main():
    """Main function to run the script"""
    logger.info("=== HKIS & LCMS Web Mention Finder ===")
    
    # Search for mentions in the last 2 days (48 hours)
    days = 2
    query = "HKIS Hong Kong International School AND LCMS Lutheran Church-Missouri Synod"
    mentions = search_web_mentions(query)
    
    if not mentions:
        logger.info("No mentions found in the last %s days.", days)
        return

    logger.info("Found %s web mentions of HKIS and LCMS", len(mentions))
    
    # Generate summary
    summary = generate_ai_summary(mentions)
    
    # Print summary
    logger.info("=== Summary of HKIS & LCMS Web Mentions ===")
    logger.info(summary["overall"])
    logger.info("--- LCMS Analysis ---")
    logger.info(summary["lcms_analysis"])
    logger.info("--- HKIS Analysis ---")
    logger.info(summary["hkis_analysis"])
    logger.info("--- Insights ---")
    logger.info(summary["insights"])
    logger.info("--- Source URLs ---")
    for url in summary["urls"]:
        logger.info("- %s", url)
    
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
        logger.info("Results saved to %s", output_file)
    except Exception as e:
        logger.error("Failed to save results to file: %s", e)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
