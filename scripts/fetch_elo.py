import asyncio
import json
from playwright.async_api import async_playwright

OUTPUT = "data/elo_rankings.json"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://eloratings.net")
        await page.wait_for_timeout(8000)

        data = await page.evaluate("""() => {
            const rows = document.querySelectorAll('.slick-row');
            const result = [];
            rows.forEach(row => {
                const cells = row.querySelectorAll('.slick-cell');
                let rank = '', team = '', rating = '';
                cells.forEach((cell, idx) => {
                    const text = cell.textContent.trim();
                    const cls = cell.className;
                    if (idx === 0) rank = text;
                    else if (cls.includes('team-cell')) team = text;
                    else if (text && !isNaN(parseFloat(text.replace(',',''))) && text.length >= 3) {
                        if (!rating) rating = text;
                    }
                });
                if (team) result.push({rank, team, rating});
            });
            return result;
        }""")

        with open(OUTPUT, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"ELO rankings updated: {len(data)} teams -> {OUTPUT}")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
