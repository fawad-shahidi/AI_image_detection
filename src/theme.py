"""Theming for app.py, matching the Pashto fake-news project's visual style
(cream background, orange gradient primary actions, teal checkmarks, card
layout, dark-mode toggle) -- same token values as the approved design
mockup.

First attempt used only a hand-written global stylesheet targeting Gradio's
native components via elem_classes-tagged wrappers + generic descendant
selectors (input[type=checkbox], button). That mostly failed in practice --
a user screenshot showed dark mode rendering with unstyled buttons,
unstyled checkboxes, and a plain white (not cream) upload dropzone. Only
elements this file fully controls as raw HTML (the header badge, the
results panel) picked up the custom CSS; Gradio's own Button/Checkbox/Image
components resisted the overrides. This version uses Gradio's native Theme
API instead (gr.themes.Base().set(...)) -- the documented, version-stable
variables Gradio's own Svelte components actually read internally
(button_primary_background_fill, checkbox_label_background_fill_selected,
input_background_fill, etc.), confirmed present via introspection on the
installed gradio==6.26.0 rather than guessed. CSS is now only used for
elements this file renders directly (badge, results HTML, section/tier
label typography via Markdown), which the first attempt already proved
does work.

A second bug surfaced after that: the dark-mode toggle button used its own
data-theme attribute, independent of Gradio's own .dark class -- so
clicking it flipped this file's --bg/--text CSS variables to dark while
Gradio's native components (which read :root.dark, confirmed via the
served /theme.css) stayed light, producing pale, barely-legible text over
an unchanged light background in the results panel. Fixed by keying this
file's dark-mode CSS off :root.dark too and having the toggle button flip
that same class, so both systems move together.
"""
import gradio as gr

_LIGHT = dict(
    bg="#fbf3dc", bg_alt="#fcefc7", card="#ffffff", card_border="#f0e0ad",
    text="#2a2420", text_muted="#8a7d68",
    accent_1="#f7ad3d", accent_2="#e8590c",
    teal="#0f9c8d", teal_soft="#e0f5f1",
    warn="#d94f36", warn_soft="#fbe6e0",
)
_DARK = dict(
    bg="#1c1712", bg_alt="#272019", card="#241d16", card_border="#3a2f22",
    text="#f4ead6", text_muted="#a89880",
    accent_1="#f7ad3d", accent_2="#e8590c",
    teal="#33cabb", teal_soft="#1a3733",
    warn="#f07a63", warn_soft="#3a241d",
)

THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Karla"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill=_LIGHT["bg"], body_background_fill_dark=_DARK["bg"],
    background_fill_primary=_LIGHT["card"], background_fill_primary_dark=_DARK["card"],
    background_fill_secondary=_LIGHT["bg_alt"], background_fill_secondary_dark=_DARK["bg_alt"],
    block_background_fill=_LIGHT["card"], block_background_fill_dark=_DARK["card"],
    block_border_color=_LIGHT["card_border"], block_border_color_dark=_DARK["card_border"],
    block_radius="20px",
    block_shadow="0 1px 2px rgba(60,45,10,.04), 0 12px 28px -14px rgba(60,45,10,.22)",
    block_shadow_dark="0 1px 2px rgba(0,0,0,.3), 0 12px 28px -14px rgba(0,0,0,.5)",
    border_color_primary=_LIGHT["card_border"], border_color_primary_dark=_DARK["card_border"],
    body_text_color=_LIGHT["text"], body_text_color_dark=_DARK["text"],
    body_text_color_subdued=_LIGHT["text_muted"], body_text_color_subdued_dark=_DARK["text_muted"],

    input_background_fill=_LIGHT["bg_alt"], input_background_fill_dark=_DARK["bg_alt"],
    input_radius="14px",

    button_primary_background_fill=f"linear-gradient(135deg,{_LIGHT['accent_1']},{_LIGHT['accent_2']})",
    button_primary_background_fill_dark=f"linear-gradient(135deg,{_DARK['accent_1']},{_DARK['accent_2']})",
    button_primary_background_fill_hover=f"linear-gradient(135deg,{_LIGHT['accent_1']},{_LIGHT['accent_1']})",
    button_primary_background_fill_hover_dark=f"linear-gradient(135deg,{_DARK['accent_1']},{_DARK['accent_1']})",
    button_primary_border_color="transparent", button_primary_border_color_dark="transparent",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",

    button_secondary_background_fill=_LIGHT["bg_alt"], button_secondary_background_fill_dark=_DARK["bg_alt"],
    button_secondary_background_fill_hover="#f7e6ad", button_secondary_background_fill_hover_dark="#332a1d",
    button_secondary_border_color=_LIGHT["card_border"], button_secondary_border_color_dark=_DARK["card_border"],
    button_secondary_text_color=_LIGHT["text"], button_secondary_text_color_dark=_DARK["text"],

    button_large_radius="999px", button_small_radius="999px", button_medium_radius="999px",

    checkbox_background_color=_LIGHT["card"], checkbox_background_color_dark=_DARK["card"],
    checkbox_background_color_selected=_LIGHT["teal"], checkbox_background_color_selected_dark=_DARK["teal"],
    checkbox_border_color=_LIGHT["card_border"], checkbox_border_color_dark=_DARK["card_border"],
    checkbox_border_color_selected=_LIGHT["teal"], checkbox_border_color_selected_dark=_DARK["teal"],
    checkbox_label_background_fill=_LIGHT["bg_alt"], checkbox_label_background_fill_dark=_DARK["bg_alt"],
    checkbox_label_background_fill_selected=_LIGHT["teal_soft"],
    checkbox_label_background_fill_selected_dark=_DARK["teal_soft"],
    checkbox_label_border_color=_LIGHT["card_border"], checkbox_label_border_color_dark=_DARK["card_border"],
    checkbox_label_border_color_selected=_LIGHT["teal"], checkbox_label_border_color_selected_dark=_DARK["teal"],
    checkbox_label_text_color=_LIGHT["text"], checkbox_label_text_color_dark=_DARK["text"],
    checkbox_label_text_color_selected=_LIGHT["teal"], checkbox_label_text_color_selected_dark=_DARK["teal"],

    color_accent=_LIGHT["teal"], color_accent_soft=_LIGHT["teal_soft"],
)

HEAD_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
"""

# CSS custom properties (--bg, --warn, etc.) power the badge + results HTML
# below, which this file renders directly as raw markup -- confirmed working
# in the first pass, unlike the native-component overrides that were dropped.
CSS = f"""
:root{{
  --bg:{_LIGHT['bg']}; --bg-alt:{_LIGHT['bg_alt']}; --card:{_LIGHT['card']}; --card-border:{_LIGHT['card_border']};
  --text:{_LIGHT['text']}; --text-muted:{_LIGHT['text_muted']};
  --accent-1:{_LIGHT['accent_1']}; --accent-2:{_LIGHT['accent_2']};
  --teal:{_LIGHT['teal']}; --good:{_LIGHT['teal']}; --good-soft:{_LIGHT['teal_soft']};
  --warn:{_LIGHT['warn']}; --warn-soft:{_LIGHT['warn_soft']};
}}
/* :root.dark is the exact selector Gradio's own generated theme.css uses
   (confirmed by inspecting the served /theme.css -- "--body-background-fill"
   etc. are redefined under ":root.dark, :root .dark", not a data-theme
   attribute or a prefers-color-scheme media query). A first version used a
   separate data-theme attribute toggled independently of Gradio's own
   .dark class, so clicking the theme button flipped these --bg/--text
   variables to dark while Gradio's native components (checkboxes, buttons,
   block backgrounds) stayed light -- pale text on an unchanged light
   background, confirmed unreadable in a user screenshot. Keying off the
   same class Gradio itself uses keeps both in lockstep. */
:root.dark{{
  --bg:{_DARK['bg']}; --bg-alt:{_DARK['bg_alt']}; --card:{_DARK['card']}; --card-border:{_DARK['card_border']};
  --text:{_DARK['text']}; --text-muted:{_DARK['text_muted']};
  --teal:{_DARK['teal']}; --good:{_DARK['teal']}; --good-soft:{_DARK['teal_soft']};
  --warn:{_DARK['warn']}; --warn-soft:{_DARK['warn_soft']};
}}

.badge{{ width:46px; height:46px; border-radius:13px; display:flex; align-items:center; justify-content:center;
  font-size:22px; background:linear-gradient(135deg,var(--accent-1),var(--teal) 130%);
  box-shadow:0 1px 2px rgba(60,45,10,.04), 0 12px 28px -14px rgba(60,45,10,.22); }}
.brand-title{{ font-family:'Manrope',sans-serif !important; font-weight:800 !important; font-size:21px !important;
  margin:0 !important; letter-spacing:-.01em; }}
.brand-sub{{ font-size:13.5px !important; color:var(--text-muted) !important; margin:2px 0 0 !important; }}
.theme-toggle-btn{{ max-width:52px !important; }}

.section-label{{ font-family:'Manrope',sans-serif !important; font-weight:800 !important; font-size:15px !important;
  margin:0 0 10px !important; }}
.tier-label{{ font-size:11.5px !important; font-weight:700 !important; text-transform:uppercase; letter-spacing:.07em;
  color:var(--text-muted) !important; margin:14px 0 6px 2px !important; }}

.results-html{{ font-family:'Karla',system-ui,sans-serif; }}
.verdict{{ border-radius:14px; padding:18px 20px; display:flex; align-items:center; justify-content:space-between;
  gap:14px; flex-wrap:wrap; margin-bottom:14px; border:1px solid var(--card-border); }}
.verdict.fake{{ background:var(--warn-soft); border-color:var(--warn); }}
.verdict.real{{ background:var(--good-soft); border-color:var(--good); }}
.verdict .vlabel{{ font-size:11.5px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }}
.verdict.fake .vlabel{{ color:var(--warn); }}
.verdict.real .vlabel{{ color:var(--good); }}
.verdict .vtitle{{ font-family:'Manrope',sans-serif; font-weight:800; font-size:20px; margin-top:2px; color:var(--text); }}
.verdict .vstat{{ text-align:right; }}
.verdict .vstat .big{{ font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:22px;
  font-variant-numeric:tabular-nums; color:var(--text); }}
.verdict .vstat .small{{ font-size:12px; color:var(--text-muted); }}

.result-row{{ display:flex; align-items:center; gap:12px; padding:10px 4px; border-bottom:1px solid var(--card-border); }}
.result-row:last-child{{ border-bottom:none; }}
.rmeta{{ flex:1; min-width:0; }}
.rmeta .rname{{ font-weight:700; font-size:14px; color:var(--text); }}
.rmeta .rtier{{ font-size:11.5px; color:var(--text-muted); }}
.chip{{ font-family:'Manrope',sans-serif; font-weight:700; font-size:11.5px; padding:5px 11px; border-radius:999px; flex:none; }}
.chip.fake{{ background:var(--warn-soft); color:var(--warn); }}
.chip.real{{ background:var(--good-soft); color:var(--good); }}
.rconf{{ font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:14px; width:52px; text-align:right;
  flex:none; font-variant-numeric:tabular-nums; color:var(--text); }}

.note-text{{ font-size:12.5px !important; color:var(--text-muted) !important; text-align:center !important;
  margin-top:8px !important; line-height:1.5 !important; }}
"""

TOGGLE_THEME_JS = """
() => {
  document.documentElement.classList.toggle('dark');
}
"""
