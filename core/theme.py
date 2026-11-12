"""
FactorFolio light/dark switch and theme-aware chart colors.

Streamlit has no Python command that changes the theme, but it remembers each visitor's choice
from its own menu (Settings > Theme) in the browser's localStorage and reads it when the page
loads. The switch writes that same choice and reloads the page, using
st.html(..., unsafe_allow_javascript=True) from the Streamlit docs.

Reloading starts a new Streamlit session, which would wipe the portfolio. So, just before the
reload, the inputs are saved in a short-lived browser cookie (2 minutes) and read back with
st.context.cookies when the new session starts.

Charts use Streamlit's Plotly theme, which already follows light or dark; colors() only swaps the
few colors that would disappear on a dark background (near-black needles and markers).

Text size (professor's check-in feedback: text may be too small for some users, such as older people).
The -/+ buttons change the size in 10% steps. Streamlit sizes its text, buttons and spacing in rem
(multiples of the page's root font size), so changing the root font size scales the whole page the way
Streamlit's own base font size setting does. Plotly charts are sized in pixels, so scaled() makes their
fonts and heights bigger to match. (CSS zoom was tested first and rejected: it pushed popovers out of
place and blanked the charts.)
"""
import json
from urllib.parse import unquote

import pandas as pd
import streamlit as st

from core import data as fdata

COOKIE = "ff_state"
COOKIE_SECONDS = 120


def is_dark():
    """True when the visitor's browser is showing the dark theme (st.context.theme)."""
    try:
        return st.context.theme.type == "dark"
    except Exception:            # e.g. no browser attached (tests)
        return False


def colors():
    """Chart colors that read well on the current theme's background."""
    dark = is_dark()
    return {
        "ink": "#f3f4f6" if dark else "#111827",          # gauge bar, stars, "today's price" line
        "portfolio": "#60a5fa" if dark else "#2563eb",    # your portfolio's line and SPY marker
        "spy": "#9ca3af",
        "muted": "#9ca3af" if dark else "#6b7280",        # reference lines and notes
        # band colors are see-through, so they look light on white and muted on dark
        "low": "rgba(34, 197, 94, 0.35)",
        "mid": "rgba(234, 179, 8, 0.38)",
        "high": "rgba(239, 68, 68, 0.35)",
    }


def _flip(key):
    st.session_state.theme_request = "Dark" if st.session_state[key] else "Light"


def switch(where=None, key="theme_switch"):
    """The light/dark toggle. Flipping it saves the inputs and reloads the page in the other theme."""
    where = where or st
    where.toggle("Dark mode", value=is_dark(), key=key, on_change=_flip, args=(key,),
                 help="Switch between a light and a dark color theme. The page reloads and keeps your portfolio.")


def _saved_state():
    """The inputs worth keeping across the reload: every saved_* value and the stock table."""
    state = {k: v for k, v in st.session_state.items()
             if isinstance(k, str) and k.startswith("saved_") and isinstance(v, (bool, int, float, str))}
    holdings = st.session_state.get("holdings")
    if isinstance(holdings, pd.DataFrame) and {"Ticker", "Weight (%)"} <= set(holdings.columns):
        rows = []
        for t, w in zip(holdings["Ticker"], holdings["Weight (%)"]):
            w = pd.to_numeric(w, errors="coerce")
            if pd.notna(t) and str(t).strip() and pd.notna(w):
                rows.append([str(t).strip()[:12], float(w)])
        state["holdings"] = rows[:30]
    return state


def apply_request(page_paths):
    """Run before the pages: if the switch was flipped, save the inputs, store the theme and reload.

    page_paths: every page's url_path (the homepage is ""), so the choice sticks on every page.
    """
    choice = st.session_state.pop("theme_request", None)
    if not choice:
        return
    # json.dumps twice gives a safe JavaScript string; "<" is escaped so the text can't end the script
    payload = json.dumps(json.dumps(_saved_state())).replace("<", "\\u003c")
    paths = json.dumps([p for p in page_paths if p])
    st.html(f"""<script>
(function () {{
  var choice = JSON.stringify("{choice}");
  var pages = {paths};
  var path = window.location.pathname;
  var base = path;
  pages.forEach(function (p) {{
    if (base.endsWith("/" + p) || base.endsWith("/" + p + "/")) {{ base = base.slice(0, base.lastIndexOf("/" + p) + 1); }}
  }});
  if (!base.endsWith("/")) {{ base += "/"; }}
  [""].concat(pages).forEach(function (p) {{
    try {{ window.localStorage.setItem("stActiveTheme-" + base + p + "-v2", choice); }} catch (e) {{}}
  }});
  document.cookie = "{COOKIE}=" + encodeURIComponent({payload}) + "; path=/; max-age={COOKIE_SECONDS}; SameSite=Lax";
  window.location.reload();
}})();
</script>""", unsafe_allow_javascript=True)
    st.info(f"Switching to {choice.lower()} mode...")
    st.stop()


def restore_after_reload():
    """Once per session: bring back the inputs saved just before a theme switch."""
    if st.session_state.get("theme_restore_checked"):
        return
    st.session_state.theme_restore_checked = True
    try:
        raw = st.context.cookies.get(COOKIE)
        state = json.loads(unquote(raw)) if raw else None
    except Exception:
        state = None
    if not isinstance(state, dict):
        return
    for k, v in state.items():
        if k.startswith("saved_") and isinstance(v, (bool, int, float, str)):
            st.session_state[k] = v
    rows = state.get("holdings")
    if isinstance(rows, list):
        table = pd.DataFrame([r for r in rows if isinstance(r, list) and len(r) == 2],
                             columns=["Ticker", "Weight (%)"])
        table = table.astype({"Ticker": "object", "Weight (%)": "float64"})
        st.session_state.holdings = table
        st.session_state.editor_base = table.copy()
        st.session_state.pop("holdings_editor", None)
        weights = {}
        for t, w in zip(table["Ticker"], table["Weight (%)"]):
            clean = fdata.clean_tickers(str(t))
            if clean and pd.notna(w) and w > 0:
                weights[clean[0]] = weights.get(clean[0], 0) + float(w)
        if weights:
            st.session_state.portfolio = weights      # the other pages read this


CASE_COLORS = {"(worst)": "rgba(239, 68, 68, 0.22)", "(most likely)": "rgba(59, 130, 246, 0.22)",
               "(best)": "rgba(34, 197, 94, 0.22)"}


def highlight_cases(table):
    """Shade the worst, most likely and best cells of the beta range table (see-through, so it works
    in light and dark mode). The words in the cells say the same thing for anyone who can't see color."""
    def shade(value):
        for mark, color in CASE_COLORS.items():
            if isinstance(value, str) and value.endswith(mark):
                return f"background-color: {color}"
        return ""
    return table.style.map(shade)


# ---------- Text size ----------
SCALE_KEY = "saved_text_scale"          # saved_ so it survives page changes and the dark mode reload
SCALE_MIN, SCALE_MAX, SCALE_STEP = 70, 200, 10
BASE_FONT_PX = 16                       # Streamlit's root font size at 100%


def scale():
    """The chosen text size as a fraction (1.0 = 100%)."""
    return st.session_state.get(SCALE_KEY, 100) / 100


def _step(delta):
    now = st.session_state.get(SCALE_KEY, 100)
    st.session_state[SCALE_KEY] = 100 if delta == 0 else min(SCALE_MAX, max(SCALE_MIN, now + delta))


def size_buttons(where=None, key="size"):
    """The text size control: smaller, the current size (click to reset) and bigger."""
    where = where or st
    now = st.session_state.get(SCALE_KEY, 100)
    with where.container(horizontal=True, vertical_alignment="center", gap="small", width="content"):
        st.button("−", key=f"{key}_minus", on_click=_step, args=(-SCALE_STEP,), disabled=now <= SCALE_MIN,
                  help="Smaller text and charts")
        st.button(f"{now}%", key=f"{key}_reset", on_click=_step, args=(0,), type="tertiary",
                  help="Text size. Click to go back to 100%.")
        st.button("+", key=f"{key}_plus", on_click=_step, args=(SCALE_STEP,), disabled=now >= SCALE_MAX,
                  help="Bigger text and charts")


# Extra rules while the text size isn't 100%:
# - Streamlit's centered pages are 736px wide (46rem at 100%); in rem the column widens with the text
# - on a phone, titles and big card numbers are capped by the screen width (vw) so a long word like
#   "FactorFolio" or "Underpriced" fits instead of breaking or being cut off
# - card labels and the colored notes under card numbers wrap instead of ending in "..."
SCALE_CSS = """
[data-testid='stMainBlockContainer'] { max-width: max(46rem, 736px); }
h1 { font-size: min(2.75rem, 12vw) !important; }
[data-testid='stMetricValue'] { font-size: min(2.25rem, 11vw) !important; }
[data-testid='stMetricLabel'] *, [data-testid='stMetricDelta'] * { white-space: normal !important; }
[data-testid='stButtonGroup'] div:has(> button) { flex-wrap: wrap; }
"""


def apply_scale():
    """Run on every page: set the root font size so all of Streamlit's text, buttons and spacing scale."""
    s = scale()
    if s != 1:
        # Streamlit's centered pages are 736px wide (46rem at 100%); in rem the column widens with the text
        st.html(f"<style>html {{ font-size: {BASE_FONT_PX * s:.1f}px !important; }}" + SCALE_CSS + "</style>")


def scaled(fig):
    """Make a Plotly chart's text and height match the text size (Streamlit's chart theme uses 12px for
    labels and legends and 14px for axis titles). At 100% the chart is left exactly as it was."""
    s = scale()
    if s == 1:
        return fig
    small, title = round(12 * s, 1), round(14 * s, 1)
    fig.update_layout(font_size=small, legend_font_size=small, legend_title_font_size=small,
                      hoverlabel_font_size=small)
    fig.update_xaxes(tickfont_size=small, title_font_size=title)
    fig.update_yaxes(tickfont_size=small, title_font_size=title)
    gauge = any(trace.type == "indicator" for trace in fig.data)
    if fig.layout.height:            # taller charts; a gauge grows less, since a phone limits its width
        fig.update_layout(height=round(fig.layout.height * (min(s, 1.3) if gauge else s)))
    # Bigger margins make room for the bigger labels. A gauge's size is set by the card's width, so on a
    # phone its margins grow less and its big number stays the same size.
    grow = min(s, 1.3) if gauge else s
    m = fig.layout.margin
    fig.update_layout(margin={k: round(getattr(m, k) * grow) for k in ("l", "r", "t", "b")
                              if getattr(m, k) is not None})
    for note in fig.layout.annotations or []:
        note.font.size = (note.font.size or 12) * s
    return fig
