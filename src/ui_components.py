"""
UI components and CSS design system for AQI Missing Value Imputer.
Provides modern, responsive styling, metric cards, badges, and academic banners.
"""

from typing import Any
import streamlit as st


CUSTOM_CSS = """
<style>
/* Main Background and Typography */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Glassmorphism Metric Cards */
.aqi-metric-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, border-color 0.2s ease;
    margin-bottom: 1rem;
}

.aqi-metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.4);
}

.aqi-metric-label {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-bottom: 0.35rem;
}

.aqi-metric-val {
    font-size: 2rem;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.1;
    font-feature-settings: "cv02", "cv03", "cv04", "cv11";
}

.aqi-metric-sub {
    font-size: 0.8rem;
    color: #64748B;
    margin-top: 0.4rem;
}

/* Hero Banner */
.aqi-hero-banner {
    background: linear-gradient(135deg, #0284C7 0%, #0369A1 50%, #0F172A 100%);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
    box-shadow: 0 20px 25px -5px rgba(2, 132, 199, 0.25);
    border: 1px solid rgba(56, 189, 248, 0.3);
}

.aqi-hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.6rem;
    color: #FFFFFF;
}

.aqi-hero-subtitle {
    font-size: 1.1rem;
    font-weight: 400;
    color: #E0F2FE;
    max-width: 800px;
    line-height: 1.5;
}

/* Feature Badge */
.aqi-badge {
    display: inline-block;
    padding: 0.3rem 0.85rem;
    background: rgba(56, 189, 248, 0.15);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #38BDF8;
    margin-bottom: 0.8rem;
}

/* Callout Info Box */
.aqi-callout {
    background: rgba(30, 41, 59, 0.6);
    border-left: 4px solid #38BDF8;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem;
    margin: 1.25rem 0;
    color: #E2E8F0;
}

.aqi-callout-warning {
    border-left-color: #F59E0B;
    background: rgba(245, 158, 11, 0.08);
}

.aqi-callout-success {
    border-left-color: #10B981;
    background: rgba(16, 185, 129, 0.08);
}

/* Code span */
code {
    font-family: 'JetBrains Mono', monospace !important;
    background-color: rgba(15, 23, 42, 0.6) !important;
    border-radius: 4px !important;
    padding: 0.15rem 0.35rem !important;
}

/* Button polishing */
.stButton>button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stDownloadButton>button {
    border-radius: 10px;
    font-weight: 600;
    background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
    color: white;
    border: none;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
}

.stDownloadButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(2, 132, 199, 0.4);
}

/* Custom footer */
.aqi-footer {
    text-align: center;
    padding: 2.5rem 1rem 1rem 1rem;
    color: #64748B;
    font-size: 0.85rem;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 3rem;
}
</style>
"""


def apply_custom_css():
    """Injects high-end modern CSS into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str, badge: str = "Academic Machine Learning Tool"):
    """Renders the top hero banner."""
    html = f"""
    <div class="aqi-hero-banner">
        <div class="aqi-badge">{badge}</div>
        <div class="aqi-hero-title">{title}</div>
        <div class="aqi-hero-subtitle">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(label: str, value: Any, subtext: str = "", border_color: str = "#38BDF8"):
    """Renders a responsive glassmorphism KPI card."""
    html = f"""
    <div class="aqi-metric-card" style="border-left: 3px solid {border_color};">
        <div class="aqi-metric-label">{label}</div>
        <div class="aqi-metric-val">{value}</div>
        <div class="aqi-metric-sub">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_callout(message: str, title: str = "Important Notice", alert_type: str = "info"):
    """Renders a styled alert callout box."""
    css_class = "aqi-callout"
    if alert_type == "warning":
        css_class += " aqi-callout-warning"
    elif alert_type == "success":
        css_class += " aqi-callout-success"

    icon_map = {
        "info": "💡",
        "warning": "⚠️",
        "success": "✅",
        "danger": "🚨"
    }
    icon = icon_map.get(alert_type, "ℹ️")

    html = f"""
    <div class="{css_class}">
        <div style="font-weight: 700; margin-bottom: 0.35rem; color: #F8FAFC;">{icon} {title}</div>
        <div style="font-size: 0.92rem; line-height: 1.5;">{message}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_footer():
    """Renders project footer with academic metadata."""
    html = """
    <div class="aqi-footer">
        <div><strong>AQI Missing Value Imputer</strong> — Air Quality Data Preprocessing & Imputation System</div>
        <div style="margin-top: 0.4rem; font-size: 0.78rem;">
            Designed for Machine Learning & Environmental Data Science Research • Powered by Streamlit, Pandas & Scikit-learn
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
