"""
Channapatna Toys - Exact Design Replication & Complete UI Redesign
AI-Based Fair Pricing & Market Intelligence Platform

Features:
- Exact reproduction of Homepage reference design:
  - Dark forest green header (#073F2D / #0B4B35)
  - Warm sunlit tabletop backdrop with bokeh and botanical leaves
  - Centered welcome heading (44px) and subtitle (20px)
  - Two side-by-side feature cards:
    * Market Price Prediction (crimson red accents, horse artwork, pill button)
    * Fair Price Prediction (deep forest green accents, peg doll artwork, pill button)
  - Bottom beige trust bar: Supporting Artisans | Sustainable Livelihoods | Data-Driven Market Intelligence
- Completely redesigned Fair Price Prediction page:
  - Substantially enlarged typography matching desktop targets
  - Unified form card (NO empty boxes or broken layout)
  - Green sliders & accent controls
  - Large price figure (44px) & three range comparison boxes
  - Itemized transparent cost sheet & proportion chart
- Completely redesigned Market Price Prediction page with sales demand seasonality graph
"""

import os
import sys
import base64
import re
import altair as alt
import pandas as pd
import numpy as np
import streamlit as st

# Setup python path to include project root for modular imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from frontend.api_client import api_client

# -------------------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Channapatna Toys | AI-Based Fair Pricing Platform",
    page_icon="🪵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Function to load and base64 encode local images
def get_base64_img(rel_path):
    full_path = os.path.join(CURRENT_DIR, rel_path)
    if os.path.exists(full_path):
        with open(full_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

LOGO_IMG_B64 = get_base64_img(os.path.join("assets", "logo_channapatna.png"))
BG_WARM_B64 = get_base64_img(os.path.join("assets", "bg_channapatna_warm.jpg"))
CARD_STACK_B64 = get_base64_img(os.path.join("assets", "card_stacking_ring.png"))
CARD_ELEPHANT_B64 = get_base64_img(os.path.join("assets", "card_painted_elephant.png"))
# Fallbacks for existing workflows
HERO_BG_B64 = get_base64_img(os.path.join("assets", "hero_bg.jpg"))
CARD_HORSE_B64 = get_base64_img(os.path.join("assets", "card_horse.jpg"))
CARD_DOLL_B64 = get_base64_img(os.path.join("assets", "card_doll.jpg"))

# Load CSS Styles
CSS_PATH = os.path.join(CURRENT_DIR, "styles.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Navigation & Session State
# -------------------------------------------------------------
if "current_view" not in st.session_state:
    st.session_state.current_view = "home"
if "market_result" not in st.session_state:
    st.session_state.market_result = None
if "fair_result" not in st.session_state:
    st.session_state.fair_result = None
if "fair_result_inputs" not in st.session_state:
    st.session_state.fair_result_inputs = None

# Query params navigation check
nav_param = st.query_params.get("nav")
if nav_param and nav_param in ["home", "about", "insights", "help", "market", "fair"]:
    st.session_state.current_view = nav_param


# =============================================================
# 1. NAVIGATION HEADER (EXACT REFERENCE RECREATION)
# =============================================================
st.markdown(
    f"""
    <header class="ch-navbar">
        <div class="ch-navbar-inner">
            <div class="ch-nav-brand-row">
                <a href="?nav=home" target="_self" class="ch-brand">
                    <img src="data:image/png;base64,{LOGO_IMG_B64}" class="ch-brand-logo" alt="Channapatna Toys Logo" />
                    <span class="ch-brand-name">Channapatna Toys</span>
                </a>
                <div class="ch-brand-tag" aria-label="Channapatna Studio">
                    <span class="ch-status-dot"></span>
                    <span>Channapatna Studio</span>
                </div>
                <div class="ch-hamburger ch-hamburger-mobile" title="Menu" aria-label="Menu">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
            <div class="ch-nav-right">
                <div class="ch-hamburger ch-hamburger-desktop" title="Menu" aria-label="Menu">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
    </header>
    """,
    unsafe_allow_html=True
)


# =============================================================
# 2. VIEW: HOMEPAGE (EXACT REFERENCE RECREATION)
# =============================================================
if st.session_state.current_view == "home":
    # Safely inject image styles via CSS
    st.markdown(
        f"""
        <style>
        .ch-home-scene {{
            background-image: linear-gradient(180deg, rgba(250, 245, 236, 0.6), rgba(250, 245, 236, 0.24)), url('data:image/jpeg;base64,{BG_WARM_B64}') !important;
            background-size: cover !important;
            background-position: center bottom !important;
            background-repeat: no-repeat !important;
            isolation: isolate;
        }}
        .ch-home-scene::before {{
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 12% 18%, rgba(115, 169, 120, 0.18), transparent 28%), radial-gradient(circle at 80% 12%, rgba(212, 169, 96, 0.14), transparent 26%), linear-gradient(180deg, rgba(255, 255, 255, 0.08), rgba(93, 65, 38, 0.06));
            z-index: 0;
            pointer-events: none;
        }}
        .ch-card-img-stack {{
            background-image: url('data:image/png;base64,{CARD_STACK_B64}') !important;
            background-size: cover !important;
            background-position: center !important;
            width: 100%;
            height: 100%;
        }}
        .ch-card-img-elephant {{
            background-image: url('data:image/png;base64,{CARD_ELEPHANT_B64}') !important;
            background-size: cover !important;
            background-position: center !important;
            width: 100%;
            height: 100%;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """<div class="ch-home-scene">
<div class="ch-cards-row">

<!-- CARD 1: Get Market Price -->
<div class="ch-card-item" id="card-get-market-price">
  <a href="?nav=market" target="_self" class="ch-card-anchor" aria-label="Get Market Price">
    <div class="ch-card-top">
      <div class="ch-card-img-stack" role="img" aria-label="Channapatna Stacking Toy"></div>
    </div>
    <div class="ch-card-bottom ch-bottom-green">
      <h2 class="ch-card-title">Get Market<br/>Price</h2>
      <div class="ch-circle-arrow" aria-hidden="true">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="5" y1="12" x2="19" y2="12"></line>
          <polyline points="12 5 19 12 12 19"></polyline>
        </svg>
      </div>
    </div>
  </a>
  <div class="ch-help-anchor" tabindex="0" aria-label="Market Price explanation">
    <div class="ch-help-circle" id="help-btn-market" role="button" tabindex="0" aria-label="Market Price explanation">?</div>
    <div class="ch-tooltip-card" id="tooltip-market">
      <p class="ch-tooltip-text">See how much your toy could sell for in the marketplace, based on the ML model's price estimate.</p>
      <div class="ch-tooltip-tail"></div>
    </div>
  </div>
</div>

<!-- CARD 2: Know Your Fair Price -->
<div class="ch-card-item" id="card-know-fair-price">
  <a href="?nav=fair" target="_self" class="ch-card-anchor" aria-label="Know Your Fair Price">
    <div class="ch-card-top">
      <div class="ch-card-img-elephant" role="img" aria-label="Channapatna Painted Elephant"></div>
    </div>
    <div class="ch-card-bottom ch-bottom-terracotta">
      <h2 class="ch-card-title">Know Your<br/>Fair Price</h2>
      <div class="ch-circle-arrow" aria-hidden="true">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="5" y1="12" x2="19" y2="12"></line>
          <polyline points="12 5 19 12 12 19"></polyline>
        </svg>
      </div>
    </div>
  </a>
  <div class="ch-help-anchor" tabindex="0" aria-label="Fair Price explanation">
    <div class="ch-help-circle" id="help-btn-fair" role="button" tabindex="0" aria-label="Fair Price explanation">?</div>
    <div class="ch-tooltip-card" id="tooltip-fair">
      <p class="ch-tooltip-text">Find a fair price that covers your materials, making time, and a reasonable profit.</p>
      <div class="ch-tooltip-tail"></div>
    </div>
  </div>
</div>

</div>
</div>""",
        unsafe_allow_html=True
    )

# =============================================================
# 3. VIEW: FAIR PRICE PREDICTION (REDESIGNED TO HIGH FIDELITY)
# =============================================================
elif st.session_state.current_view == "fair":
    st.markdown('<div class="workspace-page-container">', unsafe_allow_html=True)
    
    # Back to Home Button
    if st.button("← Back to Home", key="back_from_fair"):
        st.session_state.current_view = "home"
        st.session_state.market_step = 1
        st.session_state.fair_step = 1
        st.session_state.fair_result = None
        st.session_state.fair_result_inputs = None
        st.session_state.fair_error = None
        st.session_state.should_scroll_to_fair_result = False
        st.session_state.market_result = None
        st.session_state.market_error = None
        st.session_state.should_scroll_to_result = False
        st.rerun()

    # Prominent Header Banner Card
    st.markdown(
        """
        <div class="page-header-card fair">
            <div class="page-header-badge green">
                <svg width="34" height="34" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M17 8C8 10 5.9 16.17 3.82 21.34l1.89.66.95-2.3c.48.17.98.3 1.34.3C19 20 22 3 22 3c-1 2-8 2.25-13 3.25S2 11.5 2 13.5c0 2.21 1.79 4 4 4 2.27 0 4.19-1.28 6.5-2.5 2.12-1.12 4.18-1.5 5.5-2.5-1 0-2 .25-3 .25z"/>
                </svg>
            </div>
            <div>
                <h1 class="page-title">Know Your Fair Price</h1>
                <p class="page-subtitle">
                    Calculate a fair and sustainable price for artisans based on material cost, labour, and overheads.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Initialize Fair Price session state keys if not already set (all empty/None)
    if "f_form_name" not in st.session_state:
        st.session_state.f_form_name = ""
    if "f_form_mat" not in st.session_state:
        st.session_state.f_form_mat = None
    if "f_form_len" not in st.session_state:
        st.session_state.f_form_len = None
    if "f_form_hgt" not in st.session_state:
        st.session_state.f_form_hgt = None
    if "f_form_wid" not in st.session_state:
        st.session_state.f_form_wid = None
    if "f_form_wt" not in st.session_state:
        st.session_state.f_form_wt = None
    if "f_form_qty" not in st.session_state:
        st.session_state.f_form_qty = None
    if "f_form_hours" not in st.session_state:
        st.session_state.f_form_hours = None
    if "f_form_wage" not in st.session_state:
        st.session_state.f_form_wage = None
    if "f_form_margin" not in st.session_state or st.session_state.f_form_margin is None:
        st.session_state.f_form_margin = 25.0
    if "fair_result" not in st.session_state:
        st.session_state.fair_result = None
    if "fair_result_inputs" not in st.session_state:
        st.session_state.fair_result_inputs = None
    if "fair_error" not in st.session_state:
        st.session_state.fair_error = None
    if "fair_step" not in st.session_state:
        st.session_state.fair_step = 1
    if "should_scroll_to_fair_result" not in st.session_state:
        st.session_state.should_scroll_to_fair_result = False

    # Validation / Error Banner (Near Top)
    if st.session_state.get("fair_error"):
        st.markdown(
            f"""
            <div style="background: #FFF4F2; border-left: 5px solid #0B4B35; padding: 18px 24px; border-radius: 12px; margin-bottom: 24px; font-size: 18px; color: #073F2D; font-weight: 600; border: 1.5px solid rgba(195, 225, 205, 0.85); box-shadow: 0 4px 12px rgba(11,75,53,0.08);">
                ⚠️ {st.session_state.fair_error}
            </div>
            """,
            unsafe_allow_html=True
        )

    # =========================================================
    # PREDICTION RESULT AT THE TOP (Directly below page heading)
    # =========================================================
    if st.session_state.get("fair_result"):
        f_data = st.session_state.fair_result
        fair_val = f_data["fair_artisan_price_inr"]
        base_cost = f_data["base_production_cost_inr"]
        breakdown = f_data["cost_breakdown"]
        parameters = f_data["parameters"]
        result_inputs = st.session_state.get("fair_result_inputs") or {}
        input_material = result_inputs.get("display_material", parameters["wood_type"])
        input_quantity = parameters["set_quantity"]

        below_range = round(fair_val * 0.90, 2)
        above_range = round(fair_val * 1.15, 2)

        st.markdown('<div id="fair-result-anchor"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="results-card-container fair" style="margin-bottom: 26px;">
                <div class="result-tagline" style="color: #0B4B35;">
                    🛡️ Recommended Fair Artisan Selling Price
                </div>
                <div class="price-headline-figure" style="color: #073F2D;">
                    ₹{fair_val:,.2f}
                </div>
                <div class="result-meta-text">
                    Base Production Cost: <strong>₹{base_cost:,.2f}</strong> &bull; Living Wage Rate: <strong>₹{f_data['parameters']['hourly_wage_inr']}/hr</strong> &bull; Artisan Cluster Profit Margin: <strong>{f_data['parameters']['artisan_margin_percent']}%</strong>
                </div>
                <div class="three-range-grid">
                    <div class="range-box-item" style="border-left: 5px solid #C07028;">
                        <div class="range-box-label">Below Range (Minimum Floor)</div>
                        <div class="range-box-val" style="color: #C07028;">₹{below_range:,.2f}</div>
                        <div class="range-box-sub">Below this causes artisan wage loss</div>
                    </div>
                    <div class="range-box-item" style="border-left: 5px solid #0B4B35; background: #F8FAF7;">
                        <div class="range-box-label" style="color: #0B4B35;">Recommended Fair Price</div>
                        <div class="range-box-val" style="color: #0B4B35;">₹{fair_val:,.2f}</div>
                        <div class="range-box-sub">Balanced sustainable living price</div>
                    </div>
                    <div class="range-box-item" style="border-left: 5px solid #073F2D;">
                        <div class="range-box-label">Above Range (Premium Craft)</div>
                        <div class="range-box-val" style="color: #073F2D;">₹{above_range:,.2f}</div>
                        <div class="range-box-sub">For master artisan collector grade toys</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption(
            f"Estimate for {result_inputs.get('product_name') or 'your toy'}: "
            f"{result_inputs.get('length_cm', '—')} × {result_inputs.get('width_cm', '—')} × "
            f"{result_inputs.get('height_cm', '—')} cm, {parameters['weight_g']:,.0f} g, "
            f"{input_material}, {input_quantity} piece(s)."
        )

        # Transparent Itemized Cost Sheet
        st.markdown(
            """
            <div class="cost-sheet-card" style="margin-bottom: 30px;">
                <h3>🧾 Transparent Itemized Cost Sheet</h3>
                <p>Complete cost accounting verifying materials, lathe turning hours, tools, and living margin.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        wood_rate = parameters.get("wood_cost_rate_inr_per_cm3")
        if wood_rate is None:
            wood_cost_basis = (
                f"{parameters['volume_cm3']:,.2f} cm³ volume × {input_quantity} piece(s); "
                "cost calculated by the pricing engine"
            )
        else:
            wood_cost_basis = (
                f"{parameters['volume_cm3']:,.2f} cm³ × ₹{wood_rate:.3f}/cm³ × "
                f"{1 + parameters.get('wood_wastage_allowance_percent', 25) / 100:.2f} turning "
                f"allowance × {input_quantity} piece(s), with ₹35/piece minimum"
            )
        cost_rows = [
            {
                "Cost Component": f"🌲 {input_material} raw material",
                "Calculation": wood_cost_basis,
                "Amount (₹)": breakdown["wood_raw_material_inr"],
            },
            {
                "Cost Component": "🎨 Natural lacquer & dyes",
                "Calculation": (
                    f"{parameters['weight_g']:,.0f} g × ₹"
                    f"{parameters.get('natural_lacquer_rate_inr_per_g', 0):.2f}/g × "
                    f"{input_quantity} piece(s), with ₹25/piece minimum"
                ),
                "Amount (₹)": breakdown["natural_lacquer_and_dyes_inr"],
            },
            {
                "Cost Component": "⏱️ Artisan labor",
                "Calculation": (
                    f"{parameters['crafting_hours']:,.2f} hours × "
                    f"₹{parameters['hourly_wage_inr']:,.2f}/hour"
                ),
                "Amount (₹)": breakdown["artisan_labor_inr"],
            },
            {
                "Cost Component": "🛠️ Workshop overhead",
                "Calculation": (
                    f"{parameters.get('workshop_overhead_percent', 15):.0f}% of wood, lacquer, and labor costs"
                ),
                "Amount (₹)": breakdown["workshop_overhead_tools_inr"],
            },
            {
                "Cost Component": "✨ Artisan margin",
                "Calculation": (
                    f"{parameters['artisan_margin_percent']:,.0f}% of base production cost "
                    f"(₹{base_cost:,.2f})"
                ),
                "Amount (₹)": breakdown["fair_artisan_margin_inr"],
            },
        ]
        cost_df = pd.DataFrame(cost_rows)

        col_t1, col_t2 = st.columns([1.1, 0.9], gap="large")
        with col_t1:
            st.table(cost_df.set_index("Cost Component"))
        with col_t2:
            st.caption(f"Cost breakdown for {result_inputs.get('product_name') or 'your toy'} (₹)")
            st.bar_chart(cost_df.set_index("Cost Component")["Amount (₹)"], color="#0B4B35")

        # Auto-scroll to fair result anchor
        if st.session_state.get("should_scroll_to_fair_result"):
            st.html(
                """
                <script>
                setTimeout(function() {
                    const anchor = window.parent.document.getElementById('fair-result-anchor') || document.getElementById('fair-result-anchor');
                    if (anchor) {
                        anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                }, 150);
                </script>
                """
            )
            st.session_state.should_scroll_to_fair_result = False

    # =========================================================
    # STEPWISE FAIR PRICE FORM
    # =========================================================
    fair_step = int(st.session_state.get("fair_step", 1))
    fair_step_titles = ["Product & Dimensions", "Artisan Details"]
    fair_step_markup = "".join(
        f'<div class="{ "market-step-item active" if i == fair_step - 1 else "market-step-item" }"><span class="market-step-number">{i + 1}</span><span>{title}</span></div>'
        for i, title in enumerate(fair_step_titles)
    )
    st.markdown(f'<div class="market-stepper">{fair_step_markup}</div>', unsafe_allow_html=True)

    with st.container(border=True):
        if fair_step == 1:
            st.markdown('<div class="form-section-title">Step 1 — Product Details & Dimensions</div>', unsafe_allow_html=True)
            mat_options = ["hale wood (Wrightia tinctoria)", "solid wood", "pine", "mango wood"]
            if st.session_state.get("f_form_mat") and st.session_state["f_form_mat"] not in mat_options:
                mat_options.append(st.session_state["f_form_mat"])

            for widget_key, value_key in [
                ("f_widget_name", "f_form_name"),
                ("f_widget_mat", "f_form_mat"),
                ("f_widget_len", "f_form_len"),
                ("f_widget_wid", "f_form_wid"),
                ("f_widget_hgt", "f_form_hgt"),
                ("f_widget_wt", "f_form_wt"),
            ]:
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = st.session_state[value_key]

            f_name = st.text_input("Toy Name or Description", key="f_widget_name", placeholder="e.g., 7-Ring Stacking Pyramid")
            st.session_state.f_form_name = f_name
            f_mat = st.selectbox("Primary Wood Species *", options=mat_options, key="f_widget_mat", placeholder="Select wood species")
            st.session_state.f_form_mat = f_mat
            f_len = st.number_input("Toy Length (cm) *", min_value=0.5, max_value=200.0, step=0.5, key="f_widget_len")
            st.session_state.f_form_len = f_len
            f_wid = st.number_input("Toy Width (cm) *", min_value=0.5, max_value=200.0, step=0.5, key="f_widget_wid")
            st.session_state.f_form_wid = f_wid
            f_hgt = st.number_input("Toy Height (cm) *", min_value=0.5, max_value=200.0, step=0.5, key="f_widget_hgt")
            st.session_state.f_form_hgt = f_hgt
            f_wt = st.number_input("Weight (grams) *", min_value=1.0, max_value=10000.0, step=10.0, key="f_widget_wt")
            st.session_state.f_form_wt = f_wt

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Next", key="fair_step_1_next", type="primary", use_container_width=True):
                missing_fields = []
                if not str(f_name or "").strip():
                    missing_fields.append("Toy Name or Description")
                if not f_mat:
                    missing_fields.append("Primary Wood Species")
                if f_len is None or float(f_len) <= 0:
                    missing_fields.append("Toy Length (cm)")
                if f_wid is None or float(f_wid) <= 0:
                    missing_fields.append("Toy Width (cm)")
                if f_hgt is None or float(f_hgt) <= 0:
                    missing_fields.append("Toy Height (cm)")
                if f_wt is None or float(f_wt) <= 0:
                    missing_fields.append("Weight (grams)")
                if missing_fields:
                    st.session_state.fair_error = f"Please enter the required product details: {', '.join(missing_fields)}."
                    st.rerun()
                else:
                    st.session_state.fair_error = None
                    st.session_state.fair_step = 2
                    st.rerun()

        elif fair_step == 2:
            st.markdown('<div class="form-section-title">Step 2 — Artisan Labour & Additional Details</div>', unsafe_allow_html=True)
            for widget_key, value_key in [
                ("f_widget_hours", "f_form_hours"),
                ("f_widget_wage", "f_form_wage"),
                ("f_widget_margin", "f_form_margin"),
                ("f_widget_qty", "f_form_qty"),
            ]:
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = st.session_state[value_key]

            f_hours = st.number_input("Manual Lathe & Carving Time (Hours) *", min_value=0.1, max_value=50.0, step=0.25, key="f_widget_hours", help="Time required for turning, lacquer application, and screw-pine leaf buffing.")
            st.session_state.f_form_hours = f_hours
            f_wage = st.number_input("Artisan Living Wage Rate (₹ / Hour) *", min_value=50.0, max_value=2000.0, step=10.0, key="f_widget_wage", help="Karnataka standard fair living wage baseline: ₹140/hour.")
            st.session_state.f_form_wage = f_wage
            f_margin = st.slider("Fair Cluster Profit Margin (%) *", min_value=5.0, max_value=60.0, step=5.0, key="f_widget_margin")
            st.session_state.f_form_margin = f_margin

            with st.expander("Additional Details (Optional)", expanded=False):
                f_qty = st.number_input("Pieces in Toy Set", min_value=1, max_value=50, step=1, key="f_widget_qty")
                st.session_state.f_form_qty = f_qty

            col_back, col_calc = st.columns([1, 1], gap="small")
            with col_back:
                if st.button("Back", key="fair_step_2_back", use_container_width=True):
                    st.session_state.fair_step = 1
                    st.session_state.fair_error = None
                    st.rerun()
            with col_calc:
                if st.button("Calculate Fair Price", key="submit_fair_calc", type="primary", use_container_width=True):
                    required_step1_dimensions = {
                        "Toy Length (cm)": st.session_state.get("f_form_len"),
                        "Toy Width (cm)": st.session_state.get("f_form_wid"),
                        "Toy Height (cm)": st.session_state.get("f_form_hgt"),
                        "Weight (grams)": st.session_state.get("f_form_wt"),
                    }
                    missing_fields = []
                    for label, value in required_step1_dimensions.items():
                        if value is None or float(value) <= 0:
                            missing_fields.append(label)
                    material = st.session_state.get("f_form_mat")
                    if material is None or not str(material).strip():
                        missing_fields.append("Primary Wood Species")
                    if f_hours is None or float(f_hours) <= 0:
                        missing_fields.append("Manual Lathe & Carving Time (Hours)")
                    if f_wage is None or float(f_wage) <= 0:
                        missing_fields.append("Artisan Living Wage Rate (₹ / Hour)")
                    if missing_fields:
                        st.session_state.fair_error = f"Please complete the required fields before calculating: {', '.join(sorted(set(missing_fields)))}."
                        st.rerun()
                    else:
                        st.session_state.fair_error = None
                        selected_material = str(st.session_state.f_form_mat).strip()
                        material_for_calculation = selected_material.split(" (", 1)[0].strip().lower()
                        payload = {
                            "length_cm": float(st.session_state.f_form_len),
                            "width_cm": float(st.session_state.f_form_wid),
                            "height_cm": float(st.session_state.f_form_hgt),
                            "weight_g": float(st.session_state.f_form_wt),
                            "material": material_for_calculation,
                            "set_quantity": int(st.session_state.f_form_qty) if st.session_state.f_form_qty is not None and int(st.session_state.f_form_qty) > 0 else 1,
                            "crafting_hours": float(f_hours),
                            "hourly_wage": float(f_wage),
                            "artisan_margin_percent": float(f_margin) if f_margin is not None else 25.0
                        }
                        with st.spinner("Calculating fair artisan living wage breakdown..."):
                            art_res = api_client.calculate_artisan_price(payload)
                            if art_res.get("success"):
                                st.session_state.fair_result = art_res.get("data")
                                st.session_state.fair_result_inputs = {
                                    **payload,
                                    "product_name": str(st.session_state.f_form_name).strip(),
                                    "display_material": selected_material,
                                }
                                st.session_state.fair_error = None
                                st.session_state.should_scroll_to_fair_result = True
                                st.session_state.fair_step = 2
                                st.rerun()
                            else:
                                st.session_state.fair_error = f"Calculation failed: {art_res.get('error')}"
                                st.session_state.fair_result = None
                                st.session_state.fair_result_inputs = None
                                st.session_state.should_scroll_to_fair_result = False
                                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================
# 4. VIEW: MARKET PRICE PREDICTION (REDESIGNED TO HIGH FIDELITY)
# =============================================================
elif st.session_state.current_view == "market":
    st.markdown('<div class="workspace-page-container">', unsafe_allow_html=True)


    if st.button("← Back to Home", key="back_from_market"):
        st.session_state.current_view = "home"
        st.session_state.market_step = 1
        st.session_state.fair_step = 1
        st.session_state.fair_result = None
        st.session_state.fair_result_inputs = None
        st.session_state.fair_error = None
        st.session_state.should_scroll_to_fair_result = False
        st.session_state.market_result = None
        st.session_state.market_error = None
        st.session_state.should_scroll_to_result = False
        st.rerun()

    st.markdown(
        """
        <div class="page-header-card market">
            <div class="page-header-badge red">
                <svg width="34" height="34" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/>
                </svg>
            </div>
            <div>
                <h1 class="page-title">Know Your Market Price</h1>
                <p class="page-subtitle">
                    Estimate the expected marketplace selling price of your Channapatna toy using AI.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Initialize Market Price session state keys if not already set (all empty/None)
    if "m_form_name" not in st.session_state:
        st.session_state.m_form_name = ""
    if "m_form_cat" not in st.session_state:
        st.session_state.m_form_cat = None
    if "m_form_mat" not in st.session_state:
        st.session_state.m_form_mat = None
    if "m_form_mrp" not in st.session_state:
        st.session_state.m_form_mrp = None
    if "m_form_len" not in st.session_state:
        st.session_state.m_form_len = None
    if "m_form_wid" not in st.session_state:
        st.session_state.m_form_wid = None
    if "m_form_hgt" not in st.session_state:
        st.session_state.m_form_hgt = None
    if "m_form_wt" not in st.session_state:
        st.session_state.m_form_wt = None
    if "m_form_qty" not in st.session_state:
        st.session_state.m_form_qty = None
    if "m_form_age" not in st.session_state or st.session_state.m_form_age is None:
        st.session_state.m_form_age = 12
    if "market_step" not in st.session_state:
        st.session_state.market_step = 1
    if "market_result" not in st.session_state:
        st.session_state.market_result = None
    if "market_result_source" not in st.session_state:
        st.session_state.market_result_source = None
    if "market_result_mrp" not in st.session_state:
        st.session_state.market_result_mrp = None
    if "market_error" not in st.session_state:
        st.session_state.market_error = None
    if "should_scroll_to_result" not in st.session_state:
        st.session_state.should_scroll_to_result = False

    def validate_step_1(name_value, mrp_value, category_value, material_value):
        errors = {}
        if not str(name_value or "").strip():
            errors["name"] = "Please enter a toy name or description."
        if mrp_value is None or float(mrp_value) <= 0:
            errors["mrp"] = "Please enter a valid MRP greater than 0."
        if category_value in [None, "", "Select a toy category"]:
            errors["category"] = "Please select a toy category."
        if material_value in [None, "", "Select a material"]:
            errors["material"] = "Please select a material."
        return errors

    def validate_step_2(len_value, wid_value, hgt_value, wt_value):
        errors = {}
        if len_value is None or float(len_value) <= 0:
            errors["length"] = "Length must be a positive number."
        if wid_value is None or float(wid_value) <= 0:
            errors["width"] = "Width must be a positive number."
        if hgt_value is None or float(hgt_value) <= 0:
            errors["height"] = "Height must be a positive number."
        if wt_value is None or float(wt_value) <= 0:
            errors["weight"] = "Weight must be a positive number."
        return errors

    def get_market_comparables(product_type, material, length_cm, width_cm, height_cm):
        master_path = os.path.join(PROJECT_ROOT, "master_channapatna_toys_cleaned.csv")
        marketplace_path = os.path.join(PROJECT_ROOT, "all_my_scraped_marketplace_data (1).xlsx")
        required_columns = {"product_name", "product_type", "material", "price", "collection_date"}
        sources = []
        source_errors = []

        if os.path.exists(marketplace_path):
            try:
                marketplace_records = pd.read_excel(marketplace_path).rename(columns={"category": "product_type"})

                def parse_dimensions(value):
                    if pd.isna(value) or "free" in str(value).lower():
                        return [np.nan, np.nan, np.nan]
                    measurements = re.findall(r"\d+(?:\.\d+)?", str(value))
                    if len(measurements) < 3:
                        return [np.nan, np.nan, np.nan]
                    dimensions = [float(value) for value in measurements[:3]]
                    if "inch" in str(value).lower():
                        dimensions = [dimension * 2.54 for dimension in dimensions]
                    return dimensions

                if "size" in marketplace_records.columns:
                    marketplace_records[["length_cm", "width_cm", "height_cm"]] = marketplace_records["size"].apply(
                        lambda value: pd.Series(parse_dimensions(value))
                    )
                marketplace_records["source_file"] = "all_my_scraped_marketplace_data (1).xlsx"
                sources.append(marketplace_records)
            except (OSError, ImportError, ValueError, pd.errors.ParserError) as exc:
                source_errors.append(f"Marketplace workbook could not be read: {exc}")

        if os.path.exists(master_path):
            try:
                master_records = pd.read_csv(master_path)
                master_records["source_file"] = "master_channapatna_toys_cleaned.csv"
                sources.append(master_records)
            except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                source_errors.append(f"Dimension-rich marketplace CSV could not be read: {exc}")

        if not sources:
            details = "; ".join(source_errors) or "No local marketplace snapshot files were found."
            return pd.DataFrame(), [], details

        records = pd.concat(sources, ignore_index=True, sort=False)
        if not required_columns.issubset(records.columns):
            return pd.DataFrame(), [], "Local marketplace snapshots are missing required comparison fields."

        records["price"] = pd.to_numeric(records["price"], errors="coerce")
        records["collection_date"] = pd.to_datetime(
            records["collection_date"], format="mixed", dayfirst=True, errors="coerce"
        )
        records = records.dropna(subset=["price", "collection_date", "product_name"])
        records = records[records["price"] > 0].copy()
        source_periods = sorted(
            records["collection_date"].dt.strftime("%d %b %Y").dropna().unique(),
            key=lambda value: pd.to_datetime(value, format="%d %b %Y")
        )

        def normalize_material(value):
            text = str(value or "").strip().lower()
            if any(term in text for term in ["engineered", "mdf", "plywood", "cardboard", "paper"]):
                return "composite"
            if "hale wood" in text:
                return "hale wood"
            if "solid wood" in text:
                return "solid wood"
            if "pine" in text:
                return "pine"
            if "wood" in text:
                return "wood"
            return ""

        def normalize_category(value):
            category = str(value or "").strip().lower()
            if category == "baby rattle":
                return "rattle"
            if category == "channapatna puzzle":
                return "puzzle"
            return category

        selected_category = normalize_category(product_type)
        records["normalized_category"] = records["product_type"].map(normalize_category)
        records["normalized_material"] = records["material"].map(normalize_material)
        selected_material = normalize_material(material)

        material_matches = records["normalized_material"].eq(selected_material)
        if selected_material != "wood":
            material_matches |= records["normalized_material"].eq("wood")
        else:
            material_matches = records["normalized_material"].isin(["wood", "hale wood", "solid wood", "pine"])

        records = records[
            records["normalized_category"].eq(selected_category) & material_matches
        ].copy()

        generic_description_terms = {"channapatna", "handcrafted", "handmade", "kid", "kids", "toy", "toys", "wood", "wooden", "set"}

        def description_terms(value):
            return {
                term for term in re.findall(r"[a-z0-9]+", str(value or "").lower())
                if len(term) > 2 and term not in generic_description_terms
            }

        selected_terms = description_terms(st.session_state.get("m_form_name", ""))
        records["description_overlap"] = records["product_name"].map(
            lambda value: len(selected_terms.intersection(description_terms(value)))
        )

        dimension_columns = ["length_cm", "width_cm", "height_cm"]
        for column in dimension_columns:
            if column in records.columns:
                records[column] = pd.to_numeric(records[column], errors="coerce")
        if set(dimension_columns).issubset(records.columns) and all(
            value is not None and float(value) > 0 for value in (length_cm, width_cm, height_cm)
        ):
            input_volume = float(length_cm) * float(width_cm) * float(height_cm)
            records["comparison_volume"] = records[dimension_columns].prod(axis=1, min_count=3)
            known_dimensions = records["comparison_volume"].notna() & (records["comparison_volume"] > 0)
            volume_ratio = records["comparison_volume"] / input_volume
            records = records[known_dimensions & volume_ratio.between(0.2, 5.0)].copy()
            records["volume_distance"] = np.abs(np.log(volume_ratio.loc[records.index]))

        if records.empty:
            return records, source_periods, "; ".join(source_errors) or None

        records["collection_period"] = records["collection_date"].dt.strftime("%d %b %Y")
        records["listing_label"] = records["product_name"].astype(str).str.slice(0, 22).str.rstrip() + "..."
        duplicate_labels = records.groupby("listing_label")["listing_label"].transform("size") > 1
        duplicate_numbers = records.groupby("listing_label").cumcount().add(1).astype(str)
        records.loc[duplicate_labels, "listing_label"] = (
            records.loc[duplicate_labels, "listing_label"] + " " + duplicate_numbers[duplicate_labels]
        )
        repeated_urls = pd.Series(False, index=records.index)
        if "product_url" in records.columns:
            repeated_url_values = records.groupby("product_url")["collection_date"].transform("nunique")
            repeated_urls = records["product_url"].notna() & (repeated_url_values > 1)
        records["has_repeated_history"] = repeated_urls
        return records.sort_values(["collection_date", "price"]), source_periods, "; ".join(source_errors) or None

    # Validation / Error Banner (Near Top)
    if st.session_state.get("market_error"):
        st.markdown(
            f"""
            <div style="background: #FFF4F2; border-left: 5px solid #982323; padding: 18px 24px; border-radius: 12px; margin-bottom: 24px; font-size: 18px; color: #982323; font-weight: 600; border: 1.5px solid rgba(225, 195, 190, 0.85); box-shadow: 0 4px 12px rgba(152,35,35,0.08);">
                ⚠️ {st.session_state.market_error}
            </div>
            """,
            unsafe_allow_html=True
        )

    current_step = int(st.session_state.market_step)
    step_classes = ["market-step-item active" if i == current_step - 1 else "market-step-item" for i in range(3)]
    step_titles = ["Basic Information", "Toy Dimensions", "Additional Details"]
    step_markup = "".join(
        f'<div class="{cls}"><span class="market-step-number">{i + 1}</span><span>{title}</span></div>'
        for i, (cls, title) in enumerate(zip(step_classes, step_titles))
    )
    st.markdown(f'<div class="market-stepper">{step_markup}</div>', unsafe_allow_html=True)

    category_list = ["stacking toy", "baby rattle", "wooden toy", "educational toy", "spinning top", "rocking toy", "puzzle", "peg dolls"]
    material_list = ["hale wood", "solid wood", "wood", "pine"]
    if st.session_state.get("m_form_cat") and st.session_state["m_form_cat"] not in category_list:
        category_list.append(st.session_state["m_form_cat"])
    if st.session_state.get("m_form_mat") and st.session_state["m_form_mat"] not in material_list:
        material_list.append(st.session_state["m_form_mat"])
    category_options = ["Select a toy category"] + category_list
    material_options = ["Select a material"] + material_list

    with st.container(border=True):
        if current_step == 1:
            st.markdown('<div class="form-section-title">Step 1 — Basic Information</div>', unsafe_allow_html=True)
            for widget_key, value_key in [
                ("m_widget_name", "m_form_name"),
                ("m_widget_mrp", "m_form_mrp")
            ]:
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = st.session_state[value_key]
            m_prod_name = st.text_input(
                "Toy Name or Description",
                key="m_widget_name",
                placeholder="e.g., 7-Ring Stacking Pyramid"
            )
            st.session_state.m_form_name = m_prod_name
            if st.session_state.get("market_error") and "Toy Name or Description" in st.session_state.get("market_error", ""):
                st.caption("Please enter a toy name or description.")

            m_mrp = st.number_input(
                "Reference Catalog MRP (₹)",
                min_value=1.0,
                max_value=50000.0,
                step=10.0,
                key="m_widget_mrp",
                placeholder="e.g., 550"
            )
            st.session_state.m_form_mrp = m_mrp
            if st.session_state.get("market_error") and "Reference Catalog MRP" in st.session_state.get("market_error", ""):
                st.caption("Please enter a valid MRP greater than 0.")

            m_category = st.selectbox(
                "Toy Category",
                options=category_options,
                index=(0 if st.session_state.get("m_form_cat") in [None, "", "Select a toy category"] else category_options.index(st.session_state.m_form_cat)) if st.session_state.get("m_form_cat") in category_options or st.session_state.get("m_form_cat") in [None, "", "Select a toy category"] else 0,
                key="m_widget_cat"
            )
            st.session_state.m_form_cat = m_category
            if st.session_state.get("market_error") and "Toy Category" in st.session_state.get("market_error", ""):
                st.caption("Please select a toy category.")

            m_material = st.selectbox(
                "Primary Material",
                options=material_options,
                index=(0 if st.session_state.get("m_form_mat") in [None, "", "Select a material"] else material_options.index(st.session_state.m_form_mat)) if st.session_state.get("m_form_mat") in material_options or st.session_state.get("m_form_mat") in [None, "", "Select a material"] else 0,
                key="m_widget_mat"
            )
            st.session_state.m_form_mat = m_material
            if st.session_state.get("market_error") and "Primary Material" in st.session_state.get("market_error", ""):
                st.caption("Please select a material.")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Next", key="market_step_1_next", type="primary", use_container_width=True):
                errors = validate_step_1(st.session_state.m_form_name, st.session_state.m_form_mrp, st.session_state.m_form_cat, st.session_state.m_form_mat)
                if errors:
                    st.session_state.market_error = "Please complete all required fields in Step 1."
                    st.rerun()
                else:
                    st.session_state.market_error = None
                    st.session_state.market_step = 2
                    st.rerun()

        elif current_step == 2:
            st.markdown('<div class="form-section-title">Step 2 — Toy Dimensions</div>', unsafe_allow_html=True)
            for widget_key, value_key in [
                ("m_widget_len", "m_form_len"),
                ("m_widget_wid", "m_form_wid"),
                ("m_widget_hgt", "m_form_hgt"),
                ("m_widget_wt", "m_form_wt")
            ]:
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = st.session_state[value_key]
            m_len = st.number_input(
                "Toy Length (cm)",
                min_value=0.5,
                max_value=200.0,
                step=0.5,
                key="m_widget_len",
                placeholder="e.g., 10 cm"
            )
            st.session_state.m_form_len = m_len
            if st.session_state.get("market_error") and "Toy Length" in st.session_state.get("market_error", ""):
                st.caption("Length must be a positive number.")

            m_wid = st.number_input(
                "Toy Width (cm)",
                min_value=0.5,
                max_value=200.0,
                step=0.5,
                key="m_widget_wid",
                placeholder="e.g., 10 cm"
            )
            st.session_state.m_form_wid = m_wid
            if st.session_state.get("market_error") and "Toy Width" in st.session_state.get("market_error", ""):
                st.caption("Width must be a positive number.")

            m_hgt = st.number_input(
                "Toy Height (cm)",
                min_value=0.5,
                max_value=200.0,
                step=0.5,
                key="m_widget_hgt",
                placeholder="e.g., 16 cm"
            )
            st.session_state.m_form_hgt = m_hgt
            if st.session_state.get("market_error") and "Toy Height" in st.session_state.get("market_error", ""):
                st.caption("Height must be a positive number.")

            m_wt = st.number_input(
                "Weight (grams)",
                min_value=1.0,
                max_value=10000.0,
                step=10.0,
                key="m_widget_wt",
                placeholder="e.g., 220 grams"
            )
            st.session_state.m_form_wt = m_wt
            if st.session_state.get("market_error") and "Weight" in st.session_state.get("market_error", ""):
                st.caption("Weight must be a positive number.")

            col_back, col_next = st.columns([1, 1], gap="small")
            with col_back:
                if st.button("Back", key="market_step_2_back", use_container_width=True):
                    st.session_state.market_step = 1
                    st.session_state.market_error = None
                    st.rerun()
            with col_next:
                if st.button("Next", key="market_step_2_next", type="primary", use_container_width=True):
                    errors = validate_step_2(st.session_state.m_form_len, st.session_state.m_form_wid, st.session_state.m_form_hgt, st.session_state.m_form_wt)
                    if errors:
                        st.session_state.market_error = "Please complete all required dimensions in Step 2."
                        st.rerun()
                    else:
                        st.session_state.market_error = None
                        st.session_state.market_step = 3
                        st.rerun()

        elif current_step == 3:
            st.markdown('<div class="form-section-title">Step 3 — Additional Details</div>', unsafe_allow_html=True)
            if "m_widget_qty" not in st.session_state:
                st.session_state.m_widget_qty = st.session_state.m_form_qty
            m_qty = st.number_input(
                "Pieces in Toy Set",
                min_value=1,
                max_value=50,
                step=1,
                key="m_widget_qty",
                placeholder="e.g., 1"
            )
            st.session_state.m_form_qty = m_qty
            m_age = st.slider(
                "Target Child Age (Months)",
                min_value=3,
                max_value=96,
                step=3,
                key="m_widget_age",
                value=st.session_state.m_form_age
            )
            st.session_state.m_form_age = m_age

            col_back, col_submit = st.columns([1, 1], gap="small")
            with col_back:
                if st.button("Back", key="market_step_3_back", use_container_width=True):
                    st.session_state.market_step = 2
                    st.session_state.market_error = None
                    st.rerun()
            with col_submit:
                if st.button("Get Market Price", key="submit_market_calc", type="primary", use_container_width=True):
                    step_1_errors = validate_step_1(
                        st.session_state.get("m_form_name"),
                        st.session_state.get("m_form_mrp"),
                        st.session_state.get("m_form_cat"),
                        st.session_state.get("m_form_mat")
                    )
                    step_2_errors = validate_step_2(
                        st.session_state.get("m_form_len"),
                        st.session_state.get("m_form_wid"),
                        st.session_state.get("m_form_hgt"),
                        st.session_state.get("m_form_wt")
                    )

                    if step_1_errors or step_2_errors:
                        st.session_state.market_step = 1 if step_1_errors else 2
                        st.session_state.market_error = "Please complete the required fields in Steps 1 and 2 before predicting."
                        st.rerun()

                    payload = {
                        "product_name": str(st.session_state.m_form_name).strip(),
                        "product_type": str(st.session_state.m_form_cat),
                        "material": str(st.session_state.m_form_mat),
                        "mrp": float(st.session_state.m_form_mrp),
                        "length_cm": float(st.session_state.m_form_len),
                        "width_cm": float(st.session_state.m_form_wid),
                        "height_cm": float(st.session_state.m_form_hgt),
                        "weight_g": float(st.session_state.m_form_wt),
                        "set_quantity": int(st.session_state.m_form_qty) if st.session_state.m_form_qty is not None and int(st.session_state.m_form_qty) > 0 else 1,
                        "age_group_months": float(st.session_state.m_form_age) if st.session_state.m_form_age is not None else 12.0
                    }
                    st.session_state.market_result = None
                    st.session_state.market_result_source = None
                    st.session_state.market_result_mrp = None
                    try:
                        with st.spinner("Running the trained marketplace pricing model..."):
                            pred_res = api_client.predict_price(payload)
                        prediction = pred_res.get("data") if pred_res.get("success") else None
                        if isinstance(prediction, dict) and prediction.get("predicted_price_inr") is not None:
                            st.session_state.market_result = prediction
                            st.session_state.market_result_source = pred_res.get("source")
                            st.session_state.market_result_mrp = payload["mrp"]
                            st.session_state.market_error = None
                            st.session_state.should_scroll_to_result = True
                        else:
                            st.session_state.market_error = f"Prediction failed: {pred_res.get('error', 'The pricing service returned no prediction.')}"
                            st.session_state.should_scroll_to_result = False
                    except Exception as exc:
                        st.session_state.market_error = f"Prediction failed: {exc}"
                        st.session_state.should_scroll_to_result = False
                    st.rerun()

    if st.session_state.get("market_result"):
        result = st.session_state.market_result
        result_mrp = st.session_state.get("market_result_mrp")
        if result_mrp is None:
            result_mrp = result.get("reference_mrp")

        st.markdown('<div id="market-result-anchor"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("### Predicted Market Price")
            st.markdown(
                f'<div class="market-price-result">₹{float(result["predicted_price_inr"]):,.2f}</div>',
                unsafe_allow_html=True
            )

        st.markdown("### Price Range")
        range_low = result.get("price_range_low")
        range_high = result.get("price_range_high")
        predicted_price = float(result["predicted_price_inr"])
        below_price = float(range_low) if range_low is not None else round(predicted_price * 0.90, 2)
        above_price = float(range_high) if range_high is not None else round(predicted_price * 1.15, 2)

        st.markdown(
            f"""
            <div class="three-range-grid">
                <div class="range-box-item" style="border-left: 5px solid #C07028;">
                    <div class="range-box-label">Below Range</div>
                    <div class="range-box-val" style="color: #C07028;">₹{below_price:,.2f}</div>
                    <div class="range-box-sub">Lower confidence floor</div>
                </div>
                <div class="range-box-item" style="border-left: 5px solid #0B4B35; background: #F8FAF7;">
                    <div class="range-box-label" style="color: #0B4B35;">Predicted Market Price</div>
                    <div class="range-box-val" style="color: #0B4B35;">₹{predicted_price:,.2f}</div>
                    <div class="range-box-sub">Expected sell value</div>
                </div>
                <div class="range-box-item" style="border-left: 5px solid #073F2D;">
                    <div class="range-box-label">Above Range</div>
                    <div class="range-box-val" style="color: #073F2D;">₹{above_price:,.2f}</div>
                    <div class="range-box-sub">Premium market band</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.session_state.get("should_scroll_to_result"):
            st.html(
                """
                <script>
                setTimeout(function() {
                    const anchor = window.parent.document.getElementById('market-result-anchor') || document.getElementById('market-result-anchor');
                    if (anchor) anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 150);
                </script>
                """
            )
            st.session_state.should_scroll_to_result = False

        if st.button("Make Another Prediction", key="market_make_another", use_container_width=True):
            st.session_state.market_result = None
            st.session_state.market_result_source = None
            st.session_state.market_result_mrp = None
            st.session_state.market_error = None
            st.session_state.market_step = 1
            st.rerun()

# =============================================================
# 5. VIEW: ABOUT / INSIGHTS / HELP MODALS
# =============================================================
elif st.session_state.current_view in ["about", "insights", "help"]:
    st.markdown('<div class="workspace-page-container">', unsafe_allow_html=True)
    if st.button("← Back to Home", key="back_from_info"):
        st.session_state.current_view = "home"
        st.rerun()

    if st.session_state.current_view == "about":
        st.markdown(
            """
            <div class="unified-form-card">
                <h2 class="form-section-title">About Channapatna Toys</h2>
                <p style="font-size:19px; line-height:1.7; color:#3A4740;">
                    Known as <em>Gombegala Ooru</em> (Toy City) of Karnataka, Channapatna has a 200-year-old heritage of wooden toy making. 
                    Awarded Geographical Indication (GI Tag No. 18), these toys are crafted exclusively from sustainably harvested 
                    <strong>Hale Mara</strong> (<em>Wrightia tinctoria</em>) and colored using 100% natural vegetable lacquers (turmeric, kumkum, indigo).
                </p>
                <p style="font-size:19px; line-height:1.7; color:#3A4740;">
                    This AI-Powered Platform was created to bridge traditional artisan craft economics with modern marketplace intelligence, 
                    preventing intermediary wage depreciation and protecting 3,000+ craft families across Karnataka.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif st.session_state.current_view == "insights":
        metrics_res = api_client.get_metrics()
        m_val = metrics_res.get("production_model", {})
        st.markdown(
            f"""
            <div class="unified-form-card">
                <h2 class="form-section-title">Machine Learning Model Insights & Benchmarks</h2>
                <p style="font-size:18px; color:#4A5851;">Our production Price-to-MRP Random Forest Regressor leverages physical geometry and TF-IDF text features.</p>
                <div class="three-range-grid" style="grid-template-columns: repeat(4, 1fr); margin-top:24px;">
                    <div class="range-box-item" style="text-align:center;">
                        <div class="range-box-label">R² CORRELATION</div>
                        <div class="range-box-val" style="color:#0B4B35; font-size:32px;">{m_val.get('r2_score', 0.9225):.4f}</div>
                    </div>
                    <div class="range-box-item" style="text-align:center;">
                        <div class="range-box-label">MEAN ABSOLUTE ERROR</div>
                        <div class="range-box-val" style="color:#982323; font-size:32px;">₹{m_val.get('mae', 194.43):.2f}</div>
                    </div>
                    <div class="range-box-item" style="text-align:center;">
                        <div class="range-box-label">MEAN % ERROR (MAPE)</div>
                        <div class="range-box-val" style="color:#C07028; font-size:32px;">{m_val.get('mape_percent', 15.30):.2f}%</div>
                    </div>
                    <div class="range-box-item" style="text-align:center;">
                        <div class="range-box-label">FEATURES USED</div>
                        <div class="range-box-val" style="color:#202820; font-size:32px;">{m_val.get('features_count', 100)}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif st.session_state.current_view == "help":
        st.markdown(
            """
            <div class="unified-form-card">
                <h2 class="form-section-title">User Guide & Help</h2>
                <div style="font-size:19px; line-height:1.7; color:#3A4740;">
                    <strong>1. Market Price Prediction:</strong> Enter your toy's dimensions and printed list MRP to calculate fair market realization and review seasonal demand peaks.<br><br>
                    <strong>2. Fair Price Prediction:</strong> Calculate ethical floor pricing based on turning hours and ₹140/hr living wages to ensure artisan sustenance.<br><br>
                    <strong>3. Fast Presets:</strong> Click on any of the preloaded presets (Stacking Rings, Rocking Horse, Spinning Tops) to instantly test the calculators.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)
