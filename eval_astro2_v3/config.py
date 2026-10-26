"""Central configuration for the NLA 5/90 research pipeline."""

N_BALLS = 90
N_DRAWN = int(__import__("os").environ.get("ASTRO_K", "5"))
NUMBER_COLS = [f"Number{i}" for i in range(1, N_DRAWN + 1)]
REQUIRED_COLS = ["Date", "Promotionday"] + NUMBER_COLS
DEFAULT_SEED = 42

# menu key -> (display name, csv filename)
GAMES = {
    "1": ("Main", "main_draw_results.csv"),
    "2": ("Noonrush", "noonrush_results.csv"),
    "3": ("DayWA", "daywa_results.csv"),
    "4": ("VAG", "vag_results.csv"),
    "5": ("Other", "other_games_results.csv"),
}
