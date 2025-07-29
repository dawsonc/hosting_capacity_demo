"""streamlit_app.py

Interactive Hosting Capacity Analysis (HCA) demo built with **pandapower** and **Streamlit**.

Features
--------
* Loads a fixed low‑voltage distribution feeder from the pandapower library (CIGRE LV network).
* Lets the user size a rooftop‑PV plant (export capacity slider) connected to a chosen bus.
* Runs a hosting‑capacity power‑flow study on demand ("Run study" button).
* Visualises results with a Plotly network map:
    * Line & transformer loading colour‑mapped to utilisation (0‒100 %) and highlighted **red** if overloaded.
    * Bus voltage colour‑mapped to 0.95–1.05 p.u. range and highlighted **red** if outside limits.
* Flags any thermal or voltage violations in the UI.

Run locally with:
```bash
streamlit run streamlit_app.py
```

Dependencies
------------
```bash
pip install streamlit pandapower plotly pandas numpy
```
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import pandapower as pp  # type: ignore
import pandapower.networks as pn  # type: ignore

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def build_network() -> pp.pandapowerNet:
    """Return the fixed demonstration feeder (CIGRE LV network)."""
    net = pn.create_cigre_network_lv()
    # Ensure geodata are present; otherwise generate a simple layout.
    if net.bus.geodata.isna().any():
        pp.plotting.create_generic_coordinates(net)
    return net


def add_or_update_pv(net: pp.pandapowerNet, *, bus: int, p_kw: float) -> None:
    """Add a PV sgen (or update if it already exists) at *bus* exporting *p_kw* (kW).

    pandapower convention: Negative ``p_mw`` means generation.
    """
    name = "PV"
    p_mw = -p_kw / 1_000  # convert to MW and make negative for generation
    existing = net.sgen[net.sgen.name == name]
    if not existing.empty:
        net.sgen.loc[existing.index, "p_mw"] = p_mw
    else:
        pp.create_sgen(net, bus, p_mw=p_mw, name=name)


def run_study(net: pp.pandapowerNet) -> dict[str, pd.Series]:
    """Execute a power‐flow and return boolean masks for violations."""
    pp.runpp(net, init="auto")

    # Thermal limits: >100 % loading
    line_over = net.res_line.loading_percent > 100
    trafo_over = net.res_trafo.loading_percent > 100

    # Voltage limits: outside 0.95‑1.05 p.u.
    vm_pu = net.res_bus.vm_pu
    overvoltage = vm_pu > 1.05
    undervoltage = vm_pu < 0.95

    return {
        "line_over": line_over,
        "trafo_over": trafo_over,
        "overvoltage": overvoltage,
        "undervoltage": undervoltage,
    }


# -----------------------------------------------------------------------------
# Plotting utilities
# -----------------------------------------------------------------------------

BWR = [[0, "#2166ac"], [0.5, "#ffffbf"], [1, "#b2182b"]]  # blue‑white‑red colour‐scale


def utilisation_colour(util: float) -> str:
    """Map utilisation (0‑100 %) to a hex colour (blue→yellow→red)."""
    x = np.clip(util / 100, 0, 1)
    if x <= 0.5:
        # Interpolate blue→white
        t = x / 0.5
        r = int((1 - t) * 33 + t * 255)
        g = int((1 - t) * 102 + t * 255)
        b = int((1 - t) * 172 + t * 255)
    else:
        # Interpolate white→red
        t = (x - 0.5) / 0.5
        r = int((1 - t) * 255 + t * 178)
        g = int((1 - t) * 255 + t * 24)
        b = int((1 - t) * 255 + t * 43)
    return f"#{r:02x}{g:02x}{b:02x}"


def voltage_colour(vm_pu: float) -> str:
    """Return colour for bus voltage (green inside band, red outside)."""
    if vm_pu < 0.95 or vm_pu > 1.05:
        return "#ff0000"  # red for violation
    # Map 0.95‑1.05 p.u. to green gradient
    t = (vm_pu - 0.95) / 0.10  # 0→1 across the band
    g = int(255 * t)
    return f"#00{g:02x}00"  # green gradient


def plot_network(
    net: pp.pandapowerNet,
    violations: dict[str, pd.Series],
) -> go.Figure:
    """Return a Plotly figure showing the network with utilisation/voltage colours."""
    fig = go.Figure()
    bus_x = net.bus.geodata.x
    bus_y = net.bus.geodata.y

    # Plot lines
    for l_idx, line in net.line.iterrows():
        fb, tb = line.from_bus, line.to_bus
        x0, y0 = bus_x[fb], bus_y[fb]
        x1, y1 = bus_x[tb], bus_y[tb]
        util = net.res_line.loading_percent.at[l_idx]
        col = "#ff0000" if violations["line_over"].at[l_idx] else utilisation_colour(util)
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=col, width=4),
                hovertemplate=f"Line {l_idx}<br>Loading: {util:.1f}%<extra></extra>",
                showlegend=False,
            )
        )

    # Plot transformers as dashed lines
    for t_idx, trafo in net.trafo.iterrows():
        fb, tb = trafo.hv_bus, trafo.lv_bus
        x0, y0 = bus_x[fb], bus_y[fb]
        x1, y1 = bus_x[tb], bus_y[tb]
        util = net.res_trafo.loading_percent.at[t_idx]
        col = "#ff0000" if violations["trafo_over"].at[t_idx] else utilisation_colour(util)
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=col, width=6, dash="dash"),
                hovertemplate=f"Trafo {t_idx}<br>Loading: {util:.1f}%<extra></extra>",
                showlegend=False,
            )
        )

    # Plot buses
    bus_trace = go.Scatter(
        x=bus_x,
        y=bus_y,
        mode="markers+text",
        marker=dict(
            size=14,
            color=[
                voltage_colour(v) for v in net.res_bus.vm_pu
            ],
            line=dict(width=1, color="#000000"),
        ),
        text=[f"{i}" for i in net.bus.index],
        textposition="top center",
        hovertemplate="Bus %{text}<br>V={vm:.3f} p.u.<extra></extra>".replace("{vm}", "%{customdata[0]:.3f}"),
        customdata=np.expand_dims(net.res_bus.vm_pu.values, axis=1),
        showlegend=False,
    )
    fig.add_trace(bus_trace)

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    return fig


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="Hosting Capacity Analysis", layout="wide")
    st.title("\U0001F4C8 Hosting Capacity Analysis (pandapower + Streamlit)")

    # Initialise / cache the base network
    @st.cache_resource
    def get_base_network() -> pp.pandapowerNet:  # noqa: D401
        return build_network()

    net = get_base_network().copy(deep=True)

    # Sidebar controls
    st.sidebar.header("Study parameters")
    pv_bus = st.sidebar.selectbox("PV connection bus", options=net.bus.index, index=5)
    pv_kw = st.sidebar.slider("PV export capacity [kW]", 0, 500, value=0, step=10)

    # Display selected parameters
    st.sidebar.write(f"**PV @ bus {pv_bus}:** {pv_kw} kW")

    # Add / update PV generator
    add_or_update_pv(net, bus=pv_bus, p_kw=pv_kw)

    # Run study on button press
    if st.sidebar.button("\u27A1\ufe0f Run study", use_container_width=True):
        violations = run_study(net)
        fig = plot_network(net, violations)
        st.plotly_chart(fig, use_container_width=True)

        # Report violations
        n_viol = sum(v.any() for v in violations.values())
        if n_viol == 0:
            st.success("No thermal or voltage violations detected.")
        else:
            st.error(
                f"Thermal / voltage violations detected in {n_viol} element"
                f"{'s' if n_viol > 1 else ''}."
            )


if __name__ == "__main__":
    main()
