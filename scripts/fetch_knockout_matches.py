import asyncio
import json
from playwright.async_api import async_playwright

OUTPUT = "data/knockout_matches.json"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://www.bbc.com/sport/football/world-cup/schedule#KnockoutStage")
        await page.wait_for_timeout(8000)

        all_teams = await page.evaluate("""() => {
            const container = document.getElementById('KnockoutStage');
            if (!container) return [];
            const nicknames = container.querySelectorAll('.ssrcss-bwsqmy-Nickname');
            return Array.from(nicknames).map(el => el.textContent.trim()).filter(t => t);
        }""")

        relevant_teams = all_teams[:32]

        pairings = [
            {"local": relevant_teams[i], "visitante": relevant_teams[i + 1]}
            for i in range(0, len(relevant_teams), 2)
            if i + 1 < len(relevant_teams)
        ]

        if pairings:
            with open(OUTPUT, "w", encoding="utf-8") as f:
                json.dump(pairings, f, indent=2, ensure_ascii=False)
            print(f"Knockout matches updated: {len(pairings)} matches -> {OUTPUT}")
            for m in pairings:
                print(f"   {m['local']} vs {m['visitante']}")
        else:
            print("Warning: no knockout matches found on BBC page.")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
