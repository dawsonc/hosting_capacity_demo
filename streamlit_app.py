"""streamlit_app.py

Interactive Hosting Capacity Analysis (HCA) demo built with pandapower and Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import pandapower as pp
import pandapower.networks as pn


def build_network() -> pp.pandapowerNet:
    net = pn.create_cigre_network_lv()
    if net.bus_geodata.empty or net.bus_geodata.isna().any().any():
        pp.plotting.create_generic_coordinates(net)
    return net


def add_or_update_pv(net: pp.pandapowerNet, *, bus: int, p_kw: float) -> None:
    name = "PV"
    p_mw = -p_kw / 1000
    existing = net.sgen[net.sgen.name == name]
    if not existing.empty:
        net.sgen.loc[existing.index, "p_mw"] = p_mw
        net.sgen.loc[existing.index, "bus"] = bus
    else:
        pp.create_sgen(net, bus, p_mw=p_mw, name=name)


def run_study(net: pp.pandapowerNet) -> dict[str, pd.Series]:
    pp.runpp(net, init="auto")
    line_over = net.res_line.loading_percent > 100
    trafo_over = net.res_trafo.loading_percent > 100
    vm_pu = net.res_bus.vm_pu
    overvoltage = vm_pu > 1.05
    undervoltage = vm_pu < 0.95
    return {
        "line_over": line_over,
        "trafo_over": trafo_over,
        "overvoltage": overvoltage,
        "undervoltage": undervoltage,
    }


def utilisation_colour(util: float) -> str:
    x = np.clip(util / 100, 0, 1)
    if x <= 0.5:
        t = x / 0.5
        r = int((1 - t) * 33 + t * 255)
        g = int((1 - t) * 102 + t * 255)
        b = int((1 - t) * 172 + t * 255)
    else:
        t = (x - 0.5) / 0.5
        r = int((1 - t) * 255 + t * 178)
        g = int((1 - t) * 255 + t * 24)
        b = int((1 - t) * 255 + t * 43)
    return f"#{r:02x}{g:02x}{b:02x}"


def voltage_colour(vm_pu: float) -> str:
    if vm_pu < 0.95 or vm_pu > 1.05:
        return "#ff0000"
    t = (vm_pu - 0.95) / 0.10
    g = int(255 * t)
    return f"#00{g:02x}00"


def plot_network(net: pp.pandapowerNet, violations: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    bus_x = net.bus_geodata.x
    bus_y = net.bus_geodata.y
    for l_idx, line in net.line.iterrows():
        fb, tb = line.from_bus, line.to_bus
        util = net.res_line.loading_percent.at[l_idx]
        col = "#ff0000" if violations["line_over"].at[l_idx] else utilisation_colour(util)
        fig.add_trace(
            go.Scatter(
                x=[bus_x[fb], bus_x[tb]],
                y=[bus_y[fb], bus_y[tb]],
                mode="lines",
                line=dict(color=col, width=4),
                hovertemplate=f"Line {l_idx}<br>Loading: {util:.1f}%<extra></extra>",
                showlegend=False,
            )
        )
    for t_idx, trafo in net.trafo.iterrows():
        fb, tb = trafo.hv_bus, trafo.lv_bus
        util = net.res_trafo.loading_percent.at[t_idx]
        col = "#ff0000" if violations["trafo_over"].at[t_idx] else utilisation_colour(util)
        fig.add_trace(
            go.Scatter(
                x=[bus_x[fb], bus_x[tb]],
                y=[bus_y[fb], bus_y[tb]],
                mode="lines",
                line=dict(color=col, width=6, dash="dash"),
                hovertemplate=f"Trafo {t_idx}<br>Loading: {util:.1f}%<extra></extra>",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=bus_x,
            y=bus_y,
            mode="markers+text",
            marker=dict(
                size=14,
                color=[voltage_colour(v) for v in net.res_bus.vm_pu],
                line=dict(width=1, color="#000000"),
            ),
            text=[f"{i}" for i in net.bus.index],
            textposition="top center",
            hovertemplate="Bus %{text}<br>V=%{customdata[0]:.3f} p.u.<extra></extra>",
            customdata=np.expand_dims(net.res_bus.vm_pu.values, axis=1),
            showlegend=False,
        )
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Hosting Capacity Analysis", layout="wide")
    st.title("\U0001F4C8 Hosting Capacity Analysis (pandapower + Streamlit)")

    @st.cache_resource
    def get_base_network() -> pp.pandapowerNet:
        return build_network()

    net = get_base_network().copy(deep=True)
    st.sidebar.header("Study parameters")
    pv_bus = st.sidebar.selectbox("PV connection bus", options=net.bus.index, index=5)
    pv_kw = st.sidebar.slider("PV export capacity [kW]", 0, 500, value=0, step=10)
    st.sidebar.write(f"**PV @ bus {pv_bus}:** {pv_kw} kW")
    add_or_update_pv(net, bus=pv_bus, p_kw=pv_kw)

    if st.sidebar.button("\u27A1\ufe0f Run study", use_container_width=True):
        violations = run_study(net)
        fig = plot_network(net, violations)
        st.plotly_chart(fig, use_container_width=True)
        if any(v.any() for v in violations.values()):
            st.error("Thermal or voltage violations detected.")
        else:
            st.success("No thermal or voltage violations detected.")


if __name__ == "__main__":
    main()
