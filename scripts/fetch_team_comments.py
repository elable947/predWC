import json
import os
import sys
import time

from googleapiclient.discovery import build

KNOCKOUT_MATCHES = "data/knockout_matches.json"
RAW_COMMENTS_DIR = "data/raw_comments"

APIS_FILE = "apis.txt"

def load_youtube_api_key():
    if not os.path.exists(APIS_FILE):
        print(f"  ERROR: {APIS_FILE} not found. Create it with:")
        print("    YOUTUBE:")
        print("     - <your_youtube_api_key>")
        sys.exit(1)
    with open(APIS_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith("- "):
                return line[2:].strip()
    print(f"  ERROR: No API key found in {APIS_FILE}. Format:")
    print("    YOUTUBE:")
    print("     - <your_youtube_api_key>")
    sys.exit(1)

YOUTUBE_API_KEY = load_youtube_api_key()


def get_teams():
    with open(KNOCKOUT_MATCHES) as f:
        matches = json.load(f)
    teams = set()
    for m in matches:
        teams.add(m["local"])
        teams.add(m["visitante"])
    return sorted(teams)


def fetch_youtube_comments(youtube, team, max_comments=120):
    query = f'"{team}" World Cup 2026 football'
    comments = []
    seen = set()

    try:
        search_resp = youtube.search().list(
            q=query, part="id", type="video",
            maxResults=10, relevanceLanguage="en"
        ).execute()
        video_ids = [
            item["id"]["videoId"] for item in search_resp.get("items", [])
            if "videoId" in item.get("id", {})
        ]
    except Exception as e:
        print(f"    [WARN] YouTube search error for {team}: {e}")
        return comments

    for vid in video_ids:
        if len(comments) >= max_comments:
            break
        try:
            next_page = None
            for _ in range(3):
                resp = youtube.commentThreads().list(
                    part="snippet", videoId=vid,
                    maxResults=min(100, max_comments - len(comments)),
                    pageToken=next_page, order="relevance"
                ).execute()
                for item in resp.get("items", []):
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    text = snippet.get("textDisplay", "")
                    text_clean = text[:2000]
                    if len(text_clean) > 10 and text_clean not in seen:
                        seen.add(text_clean)
                        comments.append({
                            "text": text_clean,
                            "likes": snippet.get("likeCount", 0),
                            "published": snippet.get("publishedAt", ""),
                        })
                        if len(comments) >= max_comments:
                            break
                next_page = resp.get("nextPageToken")
                if not next_page:
                    break
                time.sleep(0.3)
        except Exception:
            pass
        time.sleep(0.3)

    return comments


def main():
    os.makedirs(RAW_COMMENTS_DIR, exist_ok=True)
    teams = get_teams()
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    for team in teams:
        fname = os.path.join(RAW_COMMENTS_DIR, f"{team.lower().replace(' ', '_')}.json")
        if os.path.exists(fname):
            print(f"  SKIP {team} (already exists)")
            continue
        print(f"  FETCH {team}...")
        yt_comments = fetch_youtube_comments(youtube, team)
        with open(fname, "w") as f:
            json.dump({"team": team, "youtube": yt_comments}, f, indent=2)
        print(f"    -> YouTube: {len(yt_comments)}")

    print(f"\nDone. {len(teams)} teams processed.")


if __name__ == "__main__":
    main()
