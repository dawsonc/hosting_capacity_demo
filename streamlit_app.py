"""streamlit_app.py

Interactive Hosting Capacity Analysis (HCA) demo built with pandapower and Streamlit.
"""

from __future__ import annotations
import copy
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import pandapower as pp
import pandapower.networks as pn


def build_network() -> pp.pandapowerNet:
    """Build the base electrical network for analysis."""
    net = pn.mv_oberrhein()
    return net


def add_or_update_pv(net: pp.pandapowerNet, *, bus: int, p_kw: float) -> None:
    """Add or update PV generation at specified bus.

    Args:
        net: pandapower network
        bus: Bus index for PV connection
        p_kw: PV capacity in kW (positive = generation)
    """
    name = "PV"
    p_mw = -p_kw / 1000  # Negative for generation in pandapower
    existing = net.sgen[net.sgen.name == name]
    if not existing.empty:
        net.sgen.loc[existing.index, "p_mw"] = p_mw
        net.sgen.loc[existing.index, "bus"] = bus
    else:
        pp.create_sgen(net, bus, p_mw=p_mw, name=name)


def run_study(net: pp.pandapowerNet) -> dict[str, pd.Series]:
    """Run power flow and identify violations.

    Args:
        net: pandapower network

    Returns:
        Dictionary of violation flags for lines, transformers, and buses
    """
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
    """Generate color based on utilization percentage.

    Args:
        util: Utilization percentage (0-100+)

    Returns:
        Hex color string
    """
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
    """Generate color based on voltage magnitude.

    Args:
        vm_pu: Voltage magnitude in per unit

    Returns:
        Hex color string (red for violations, green gradient for normal)
    """
    if vm_pu < 0.95 or vm_pu > 1.05:
        return "#ff0000"
    t = (vm_pu - 0.95) / 0.10
    g = int(255 * t)
    return f"#00{g:02x}00"


def get_bus_coordinates(net: pp.pandapowerNet) -> tuple[pd.Series, pd.Series]:
    """Extract bus coordinates from pandapower network.

    Args:
        net: pandapower network

    Returns:
        Tuple of (x_coords, y_coords) as pandas Series
    """
    # Check if bus_geodata exists (preferred method)
    if hasattr(net, "bus_geodata") and not net.bus_geodata.empty:
        print("Using bus_geodata")
        bus_x = net.bus_geodata["x"]
        bus_y = net.bus_geodata["y"]
    else:
        # Fallback: check if 'geo' column exists in bus table
        if "geo" in net.bus.columns and not net.bus["geo"].isna().all():
            geo_data = net.bus["geo"]
            # Parse JSON strings and extract coordinates
            def extract_coord(geo_str, coord_idx):
                """Extract coordinate from geo JSON string."""
                if pd.isna(geo_str) or geo_str is None:
                    return 0
                try:
                    geo_dict = json.loads(geo_str) if isinstance(geo_str, str) else geo_str
                    return geo_dict.get('coordinates', [0, 0])[coord_idx]
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    return 0
            
            bus_x = geo_data.apply(lambda g: extract_coord(g, 0))
            bus_y = geo_data.apply(lambda g: extract_coord(g, 1))
        else:
            # Generate simple grid layout as fallback
            bus_x = pd.Series([i % 5 for i in net.bus.index], index=net.bus.index)
            bus_y = pd.Series([i // 5 for i in net.bus.index], index=net.bus.index)

    return bus_x, bus_y

def plot_network(net: pp.pandapowerNet, violations: dict[str, pd.Series], pv_bus: int) -> go.Figure:
    """Create interactive network plot with violations highlighted.

    Args:
        net: pandapower network with results
        violations: Dictionary of violation flags

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Get bus coordinates
    bus_x, bus_y = get_bus_coordinates(net)

    # Plot lines with transparent center markers for hover
    for l_idx, line in net.line.iterrows():
        fb, tb = line.from_bus, line.to_bus
        util = net.res_line.loading_percent.at[l_idx]
        current_ka = net.res_line.i_ka.at[l_idx]
        p_from_mw = net.res_line.p_from_mw.at[l_idx]
        q_from_mvar = net.res_line.q_from_mvar.at[l_idx]
        
        col = (
            "#ff0000" if violations["line_over"].at[l_idx] else utilisation_colour(util)
        )
        
        # Line visualization (no hover)
        fig.add_trace(
            go.Scatter(
                x=[bus_x[fb], bus_x[tb]],
                y=[bus_y[fb], bus_y[tb]],
                mode="lines",
                line=dict(color=col, width=4),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        
        # Transparent marker at line center for hover
        center_x = (bus_x[fb] + bus_x[tb]) / 2
        center_y = (bus_y[fb] + bus_y[tb]) / 2
        
        hover_text = (
            f"<b>Line {l_idx}</b><br>"
            f"From Bus {fb} → To Bus {tb}<br>"
            f"Loading: {util:.1f}%<br>"
            f"Current: {current_ka:.2f} kA<br>"
            f"Power: {p_from_mw:.2f} MW<br>"
            f"Reactive: {q_from_mvar:.2f} MVAr<br>"
            f"Length: {line.length_km:.2f} km<br>"
            f"Type: {line.std_type}"
            "<extra></extra>"
        )
        
        fig.add_trace(
            go.Scatter(
                x=[center_x],
                y=[center_y],
                mode="markers",
                marker=dict(
                    size=12,
                    color="rgba(0,0,0,0)",  # Fully transparent
                    line=dict(width=0)
                ),
                hovertemplate=hover_text,
                showlegend=False,
            )
        )

    # Plot transformers with transparent center markers for hover
    for t_idx, trafo in net.trafo.iterrows():
        fb, tb = trafo.hv_bus, trafo.lv_bus
        util = net.res_trafo.loading_percent.at[t_idx]
        p_hv_mw = net.res_trafo.p_hv_mw.at[t_idx]
        q_hv_mvar = net.res_trafo.q_hv_mvar.at[t_idx]
        
        col = (
            "#ff0000"
            if violations["trafo_over"].at[t_idx]
            else utilisation_colour(util)
        )
        
        # Transformer line visualization (no hover)
        fig.add_trace(
            go.Scatter(
                x=[bus_x[fb], bus_x[tb]],
                y=[bus_y[fb], bus_y[tb]],
                mode="lines",
                line=dict(color=col, width=6, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        
        # Transparent marker at transformer center for hover
        center_x = (bus_x[fb] + bus_x[tb]) / 2
        center_y = (bus_y[fb] + bus_y[tb]) / 2
        
        hover_text = (
            f"<b>Transformer {t_idx}</b><br>"
            f"HV Bus {fb} → LV Bus {tb}<br>"
            f"Loading: {util:.1f}%<br>"
            f"Power: {p_hv_mw:.2f} MW<br>"
            f"Reactive: {q_hv_mvar:.2f} MVAr<br>"
            f"Capacity: {trafo.sn_mva:.1f} MVA<br>"
            f"Voltage: {trafo.vn_hv_kv:.1f}/{trafo.vn_lv_kv:.1f} kV"
            "<extra></extra>"
        )
        
        fig.add_trace(
            go.Scatter(
                x=[center_x],
                y=[center_y],
                mode="markers",
                marker=dict(
                    size=28,
                    color="rgba(0,0,0,1)",  # Fully transparent
                    line=dict(width=0),
                    symbol="square",
                ),
                hovertemplate=hover_text,
                showlegend=False,
            )
        )

    # Plot buses
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
            hovertemplate="<b>Bus %{text}</b><br>Voltage: %{customdata[0]:.3f} p.u.<br>Voltage: %{customdata[1]:.1f} kV<extra></extra>",
            customdata=np.column_stack([
                net.res_bus.vm_pu.values,
                net.res_bus.vm_pu.values * net.bus.vn_kv.values
            ]),
            showlegend=False,
        )
    )

    pv_x = [bus_x[bus] for bus in [pv_bus]]
    pv_y = [bus_y[bus] for bus in [pv_bus]]
    
    fig.add_trace(
        go.Scatter(
            x=pv_x,
            y=pv_y,
            mode="markers",
            marker=dict(
                symbol="star",
                size=20,
                color="#FFD700",  # Gold color
                line=dict(width=2, color="#FF8C00"),  # Orange border
            ),
            hovertemplate="PV Generator<br>Bus: %{customdata[0]}<br>Capacity: %{customdata[1]:.0f} kW<extra></extra>",
            name="PV Generator",
            showlegend=True,
        )
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    return fig


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(page_title="Hosting Capacity Analysis", layout="wide")
    st.title("⚡Hosting Capacity Analysis")

    @st.cache_resource
    def get_base_network() -> pp.pandapowerNet:
        """Cached function to build base network."""
        return build_network()

    net = copy.deepcopy(get_base_network())

    st.sidebar.header("Study parameters")

    # Debug info
    if st.sidebar.checkbox("Show debug info"):
        st.sidebar.write(f"Network has {len(net.bus)} buses")
        if hasattr(net, "bus_geodata"):
            st.sidebar.write(f"bus_geodata shape: {net.bus_geodata.shape}")
        st.sidebar.write(f"Bus geo column exists: {'geo' in net.bus.columns}")

    pv_bus = st.sidebar.selectbox(
        "PV connection bus", options=list(net.bus.index), index=5
    )
    pv_kw = st.sidebar.slider("PV export capacity [kW]", 0, 5000, value=0, step=10)
    st.sidebar.write(f"**PV @ bus {pv_bus}:** {pv_kw} kW")

    add_or_update_pv(net, bus=pv_bus, p_kw=pv_kw)

    if st.sidebar.button("➡️ Run study", use_container_width=True):
        violations = run_study(net)
        fig = plot_network(net, violations, pv_bus)
        st.plotly_chart(fig, use_container_width=True)

        # Check for violations and display detailed information
        has_violations = any(v.any() for v in violations.values())
        
        if has_violations:
            st.error("⚠️ **Network violations detected:**")
            
            # Line overloads
            if violations["line_over"].any():
                overloaded_lines = net.res_line[violations["line_over"]]
                st.write("**Line overloads:**")
                for idx in overloaded_lines.index:
                    loading = overloaded_lines.loc[idx, "loading_percent"]
                    from_bus = net.line.loc[idx, "from_bus"]
                    to_bus = net.line.loc[idx, "to_bus"]
                    st.write(f"• Line {idx} ({from_bus}→{to_bus}): {loading:.1f}% loading")
            
            # Transformer overloads
            if violations["trafo_over"].any():
                overloaded_trafos = net.res_trafo[violations["trafo_over"]]
                st.write("**Transformer overloads:**")
                for idx in overloaded_trafos.index:
                    loading = overloaded_trafos.loc[idx, "loading_percent"]
                    hv_bus = net.trafo.loc[idx, "hv_bus"]
                    lv_bus = net.trafo.loc[idx, "lv_bus"]
                    st.write(f"• Trafo {idx} ({hv_bus}→{lv_bus}): {loading:.1f}% loading")
            
            # Overvoltage violations
            if violations["overvoltage"].any():
                overvolt_buses = net.res_bus[violations["overvoltage"]]
                st.write("**Overvoltage violations (>1.05 p.u.):**")
                for idx in overvolt_buses.index:
                    voltage = overvolt_buses.loc[idx, "vm_pu"]
                    st.write(f"• Bus {idx}: {voltage:.3f} p.u. ({voltage*100:.1f}%)")
            
            # Undervoltage violations
            if violations["undervoltage"].any():
                undervolt_buses = net.res_bus[violations["undervoltage"]]
                st.write("**Undervoltage violations (<0.95 p.u.):**")
                for idx in undervolt_buses.index:
                    voltage = undervolt_buses.loc[idx, "vm_pu"]
                    st.write(f"• Bus {idx}: {voltage:.3f} p.u. ({voltage*100:.1f}%)")
                    
        else:
            st.success("✅ No thermal or voltage violations detected.")


if __name__ == "__main__":
    main()
