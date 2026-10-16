#!/usr/bin/env python3
"""Daily news brief: last-24h articles from one feed -> 4 summaries + synopsis -> Markdown / Facebook text.

Dry run by default (prints + saves files). Add --post to publish to a Facebook Page.
Sources are added by this script from the real article URLs, never written by the model.
"""
import argparse, json, os, re, sys, time
from datetime import datetime, timezone
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERY = ('Give me a summary of the last 24 hours about gas prices '
                 'from the feed "Google Alert - gas price"')


def load_env(path=ROOT / ".env"):
    """Read .env as KEY=VALUE text (never sourced). Real environment wins."""
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    env.update({k: v for k, v in os.environ.items() if v})
    return env


def strip_thinking(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    return re.sub(r"^.*?</think>", "", t, flags=re.S).strip() if "</think>" in t else t.strip()


def fetch_articles(q, collection, vec, feed, hours, k):
    since = int(time.time()) - hours * 3600
    flt = Filter(must=[FieldCondition(key="feed", match=MatchValue(value=feed)),
                       FieldCondition(key="published_at", range=Range(gte=since))])
    pts = q.query_points(collection_name=collection, query=vec, query_filter=flt, limit=k).points
    seen, arts = set(), []
    for p in pts:
        pl = p.payload or {}
        url = pl.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        arts.append({"title": pl.get("title", "Untitled"), "url": url, "feed": pl.get("feed", feed),
                     "ts": pl.get("published_at", 0), "text": (pl.get("text") or "")[:700]})
    arts.sort(key=lambda a: a["ts"], reverse=True)
    return arts


def build_prompt(query, arts):
    ctx = "\n\n".join(
        f"[{i}] {a['title']} ({datetime.fromtimestamp(a['ts'], timezone.utc):%b %d %H:%M} UTC)\n{a['text']}"
        for i, a in enumerate(arts, 1))
    return (
        f"Request: {query}\n\nArticles from the last 24 hours (numbered):\n{ctx}\n\n"
        "Write the brief using ONLY these articles. Neutral, factual tone; include figures (prices, percentages, dates) when given.\n"
        "Format EXACTLY:\n"
        "1. <short headline>: <2-3 sentence summary> [n]\n2. ...\n3. ...\n4. ...\n"
        "Synopsis: <2-3 sentences tying the four together>\n"
        "Rules: exactly 4 numbered summaries, each citing the article numbers it used like [2] or [1][4]; "
        "do not invent facts or URLs; do not write a sources list; if the articles lack information, say so briefly."
    )


def render(body, arts, hours, topic, markdown):
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", body) if 0 < int(n) <= len(arts)})
    if markdown:  # clickable [n](url) in the text
        body = re.sub(r"\[(\d+)\]", lambda m: f"[[{m.group(1)}]]({arts[int(m.group(1))-1]['url']})"
                      if 0 < int(m.group(1)) <= len(arts) else m.group(0), body)
    head = f"{topic} - last {hours} hours ({datetime.now().astimezone():%b %d, %Y})"
    out = [f"**{head}**\n" if markdown else f"{head}\n", body, ""]
    out.append("**Sources**" if markdown else "Sources:")
    for n in cited or range(1, len(arts) + 1):
        a = arts[n - 1]
        out.append(f"{n}. [{a['title']}]({a['url']})" if markdown else f"{n}. {a['title']}\n   {a['url']}")
    return "\n".join(out)


def post_facebook(env, text, schedule_at):
    pid, tok = env.get("FB_PAGE_ID"), env.get("FB_PAGE_TOKEN")
    if not pid or not tok:
        sys.exit("FB_PAGE_ID / FB_PAGE_TOKEN missing in .env")
    data = {"message": text, "access_token": tok}
    if schedule_at:
        data.update(published="false", scheduled_publish_time=int(schedule_at.timestamp()))
    r = requests.post(f"https://graph.facebook.com/v21.0/{pid}/feed", data=data, timeout=30)
    print("Facebook:", r.status_code, r.text)
    r.raise_for_status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--feed", default="Google Alert - gas price")
    ap.add_argument("--topic", default="Gas Price Brief")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--out-dir", default=str(ROOT / "brief" / "output"))
    ap.add_argument("--post", action="store_true", help="publish to the Facebook Page")
    ap.add_argument("--schedule", metavar="YYYY-MM-DDTHH:MM", help="local time to publish (with --post)")
    a = ap.parse_args()

    env = load_env()
    ov = env.get("OPENVINO_URL", f"http://127.0.0.1:{env.get('OV_PORT', '11434')}")
    q = QdrantClient(url=env.get("QDRANT_URL", f"http://127.0.0.1:{env.get('QDRANT_PORT', '6333')}"))
    coll = env.get("QDRANT_COLLECTION", "rss_articles")

    emb = requests.post(f"{ov}/api/embed", json={"model": env.get("EMBED_MODEL", "BAAI/bge-base-en-v1.5"),
                                                "input": a.query}, timeout=120).json()["embeddings"][0]
    arts = fetch_articles(q, coll, emb, a.feed, a.hours, a.top_k)
    if not arts:
        sys.exit(f'No articles from feed "{a.feed}" in the last {a.hours}h. Nothing to post.')
    print(f"Found {len(arts)} articles from {a.feed!r}")

    t0 = time.time()
    r = requests.post(f"{ov}/api/chat", json={"model": env.get("DEFAULT_MODEL", ""), "stream": False,
        "messages": [{"role": "system", "content": "You are an objective news editor writing a short public brief."},
                     {"role": "user", "content": build_prompt(a.query, arts)}]}, timeout=900)
    r.raise_for_status()
    body = strip_thinking(r.json()["message"]["content"])
    print(f"Generated in {time.time() - t0:.0f}s")

    md, fb = render(body, arts, a.hours, a.topic, True), render(body, arts, a.hours, a.topic, False)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    (out / f"brief_{stamp}.md").write_text(md); (out / f"brief_{stamp}_facebook.txt").write_text(fb)
    print("\n===== FACEBOOK TEXT =====\n" + fb + f"\n\nSaved to {out}/brief_{stamp}*")

    if a.post:
        when = datetime.fromisoformat(a.schedule).astimezone() if a.schedule else None
        post_facebook(env, fb, when)
    else:
        print("(dry run - add --post to publish)")


if __name__ == "__main__":
    main()
