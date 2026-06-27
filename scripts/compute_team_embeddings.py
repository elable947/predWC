import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

KNOCKOUT_MATCHES = "data/knockout_matches.json"
RAW_NEWS_DIR = "data/raw_news"
RAW_COMMENTS_DIR = "data/raw_comments"
OUTPUT_FILE = "data/team_nlp_features.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_teams():
    with open(KNOCKOUT_MATCHES) as f:
        matches = json.load(f)
    teams = set()
    for m in matches:
        teams.add(m["local"])
        teams.add(m["visitante"])
    return sorted(teams)


def load_news_text(team):
    fname = os.path.join(RAW_NEWS_DIR, f"{team.lower().replace(' ', '_')}.json")
    if not os.path.exists(fname):
        return ""
    with open(fname) as f:
        data = json.load(f)
    texts = []
    for art in data.get("articles", []):
        txt = (art.get("body") or art.get("title") or "").strip()
        if txt and len(txt) > 50:
            texts.append(txt[:3000])
    return " ".join(texts[:10])


def load_yt_comments(team):
    fname = os.path.join(RAW_COMMENTS_DIR, f"{team.lower().replace(' ', '_')}.json")
    if not os.path.exists(fname):
        return []
    with open(fname) as f:
        data = json.load(f)
    return [c["text"][:2000] for c in data.get("youtube", []) if c.get("text")]


def compute_sentiment_batch(texts):
    if not texts:
        return []
    from textblob import TextBlob
    scores = []
    for t in texts:
        try:
            scores.append(TextBlob(t[:2000]).sentiment.polarity)
        except Exception:
            scores.append(0.0)
    return scores


def main():
    teams = get_teams()
    model = SentenceTransformer(EMBEDDING_MODEL)

    team_news_text = {}
    for team in teams:
        text = load_news_text(team)
        team_news_text[team] = text

    print(f"News texts loaded: {sum(1 for t in team_news_text.values() if t)} teams with content")

    teams_with_news = [t for t in teams if team_news_text[t]]
    if teams_with_news:
        print("Computing news embeddings...")
        news_list = [team_news_text[t][:3000] for t in teams_with_news]
        embeddings = model.encode(news_list, show_progress_bar=True, batch_size=32)
        print(f"Embeddings shape: {embeddings.shape}")

        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(embeddings)

        pca = PCA(n_components=11, random_state=42)
        team_pcs = pca.fit_transform(emb_scaled)
        print(f"PCA explained variance: {pca.explained_variance_ratio_}")
        print(f"Cumulative: {pca.explained_variance_ratio_.sum():.3f}")

        n_pc = team_pcs.shape[1]
        team_news_pc = {}
        for i, team in enumerate(teams_with_news):
            team_news_pc[team] = [float(team_pcs[i, j]) for j in range(n_pc)]
        for team in teams:
            if team not in team_news_pc:
                team_news_pc[team] = [0.0] * n_pc
    else:
        print("No news content found.")
        team_news_pc = {team: [0.0, 0.0, 0.0] for team in teams}

    print("\nComputing YouTube sentiment (batched)...")
    all_yt_texts = []
    yt_team_indices = []
    for team in teams:
        texts = load_yt_comments(team)
        if texts:
            all_yt_texts.extend(texts)
            yt_team_indices.extend([team] * len(texts))

    team_yt_sent = {t: 0.0 for t in teams}
    if all_yt_texts:
        all_scores = compute_sentiment_batch(all_yt_texts)
        from collections import defaultdict
        team_scores = defaultdict(list)
        for team, score in zip(yt_team_indices, all_scores):
            team_scores[team].append(score)
        for team, scores in team_scores.items():
            team_yt_sent[team] = float(np.mean(scores))

    result = {}
    for team in teams:
        yt_sent = team_yt_sent.get(team, 0.0)
        entry = {"youtube_sentiment": yt_sent}
        for j, val in enumerate(team_news_pc[team]):
            entry[f"news_pc{j+1}"] = val
        result[team] = entry
        pc_str = "  ".join(f"PC{j+1}={val:+.3f}" for j, val in enumerate(team_news_pc[team][:3]))
        print(f"  {team:25s}  {pc_str}  ...  YT={yt_sent:+.3f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
