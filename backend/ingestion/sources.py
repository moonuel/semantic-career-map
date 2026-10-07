"""Source registry for scheduled scraping.

The single source of truth for which boards each scheduled scrape run
ingests, so cron config stays frozen: adding a board is adding one entry
here. ``--boards`` on ``scripts/scrape.py`` overrides per invocation; this
module supplies the defaults.

Boards are public, key-free, and verified alive at registration time (an
empty or vanished board returns zero usable postings and is reported, not
scraped silently — see ``scripts/scrape.py`` summary).
"""

SOURCE_BOARDS: dict[str, tuple[str, ...]] = {
    "ashby": ("supabase", "elevenlabs"),
    "greenhouse": ("anthropic", "vercel"),
    "lever": ("palantir", "moonpay"),
}
