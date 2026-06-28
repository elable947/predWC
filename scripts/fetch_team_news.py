import json
import os
import time

import feedparser
import requests
import trafilatura

KNOCKOUT_MATCHES = "data/knockout_matches.json"
RAW_NEWS_DIR = "data/raw_news"

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "https://www.espn.com/espn/rss/soccer/news",
]


def get_teams():
    with open(KNOCKOUT_MATCHES) as f:
        matches = json.load(f)
    teams = set()
    for m in matches:
        teams.add(m["local"])
        teams.add(m["visitante"])
    return sorted(teams)


def wiki_page_name(team):
    exceptions = {
        "United States": "United_States_men%27s_national_soccer_team",
        "South Korea": "South_Korea_national_football_team",
        "Ivory Coast": "Ivory_Coast_national_football_team",
        "Bosnia-Herzegovina": "Bosnia_and_Herzegovina_national_football_team",
        "Canada": "Canada_men%27s_national_soccer_team",
    }
    name = exceptions.get(team, f"{team.replace(' ', '_')}_national_football_team")
    return name.replace("'", "%27")


def fetch_wikipedia_summary(team):
    page = wiki_page_name(team)
    url = f"https://en.wikipedia.org/wiki/{page}"
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            body = trafilatura.extract(downloaded)
            if body and len(body) > 100:
                return body[:5000]
        return ""
    except Exception:
        return ""


def extract_article_body(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            body = trafilatura.extract(downloaded)
            if body and len(body) > 100:
                return body[:3000]
    except Exception:
        pass
    return ""


def fetch_team_news(team):
    articles = []

    summary = fetch_wikipedia_summary(team)
    if summary:
        articles.append({
            "title": f"{team} - Wikipedia summary",
            "source": "Wikipedia",
            "link": f"https://en.wikipedia.org/wiki/{wiki_page_name(team)}",
            "published": "",
            "body": summary[:3000],
        })

    return articles


def fetch_general_feeds():
    articles = []
    seen_links = set()
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:60]:
            link = entry.get("link", "")
            if not link or link in seen_links:
                continue
            seen_links.add(link)
            body = extract_article_body(link)
            articles.append({
                "title": entry.get("title", ""),
                "source": feed_url.split("/")[2],
                "link": link,
                "published": entry.get("published", ""),
                "body": body if body else "",
            })
            time.sleep(0.5)
    return articles


def main():
    os.makedirs(RAW_NEWS_DIR, exist_ok=True)
    teams = get_teams()

    existing = set(f.replace(".json", "") for f in os.listdir(RAW_NEWS_DIR)
                   if f != "_general_feeds.json")
    to_fetch = [t for t in teams if t.lower().replace(" ", "_") not in existing]

    for team in to_fetch:
        print(f"  FETCH {team}...")
        articles = fetch_team_news(team)
        fname = os.path.join(RAW_NEWS_DIR, f"{team.lower().replace(' ', '_')}.json")
        with open(fname, "w") as f:
            json.dump({"team": team, "articles": articles}, f, indent=2)
        print(f"    -> {len(articles)} articles")
        time.sleep(0.5)

    general_file = os.path.join(RAW_NEWS_DIR, "_general_feeds.json")
    if not os.path.exists(general_file):
        print("\n  FETCHING general feeds (BBC + ESPN)...")
        general = fetch_general_feeds()
        with open(general_file, "w") as f:
            json.dump({"articles": general}, f, indent=2)
        print(f"    -> {len(general)} articles")
    else:
        with open(general_file) as f:
            general = json.load(f)["articles"]
        print(f"\n  General feeds already cached ({len(general)} articles)")

    with open(general_file) as f:
        general_articles = json.load(f)["articles"]

    for team in [t for t in teams if t.lower().replace(" ", "_") not in existing]:
        fname = os.path.join(RAW_NEWS_DIR, f"{team.lower().replace(' ', '_')}.json")
        data = json.load(open(fname))
        existing_links = {a["link"] for a in data.get("articles", [])}
        relevant = [
            a for a in general_articles
            if team.lower() in a.get("title", "").lower()
            and a["link"] not in existing_links
            and a.get("body")
        ]
        if relevant:
            data["articles"].extend(relevant[:5])
            with open(fname, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  Added {len(relevant[:5])} general articles to {team}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
