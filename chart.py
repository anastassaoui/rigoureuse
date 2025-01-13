import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import chart

def plot_stage_temperatures(stage_temperatures):
    df = pd.DataFrame({
        "Stage": range(1, len(stage_temperatures) + 1),
        "Temperature (°F)": stage_temperatures
    })
    fig = px.line(df, x="Stage", y="Temperature (°F)", 
                  title="Stage Temperatures", markers=True)
    return fig

def plot_normalized_compositions(x_normalized):
    components = list(x_normalized.keys())
    stages = range(1, len(x_normalized[components[0]]) + 1)
    df = pd.DataFrame({comp: x_normalized[comp] for comp in components})
    df["Stage"] = stages
    df_melted = df.melt(id_vars="Stage", var_name="Component", value_name="Normalized Composition")
    
    fig = px.bar(df_melted, x="Stage", y="Normalized Composition", color="Component",
                 title="Normalized Compositions per Stage", barmode="group")
    return fig

def plot_convergence(S_values):
    df = pd.DataFrame({
        "Stage": range(1, len(S_values) + 1),
        "S Value": S_values
    })
    fig = px.line(df, x="Stage", y="S Value", 
                  title="Convergence of S Values", markers=True)
    return fig

def plot_composition_totals(x_normalized):
    components = list(x_normalized.keys())
    totals = [sum(x_normalized[comp]) for comp in components]
    df = pd.DataFrame({"Component": components, "Total Composition": totals})
    fig = px.pie(df, names="Component", values="Total Composition", 
                 title="Total Composition Distribution")
    return fig

def plot_temperature_vs_composition(stage_temperatures, x_normalized):
    components = list(x_normalized.keys())
    df = pd.DataFrame({
        "Stage": range(1, len(stage_temperatures) + 1),
        "Temperature (°F)": stage_temperatures
    })
    for comp in components:
        df[comp] = x_normalized[comp]

    df_melted = df.melt(id_vars=["Stage", "Temperature (°F)"], 
                        var_name="Component", value_name="Normalized Composition")
    fig = px.scatter(df_melted, x="Temperature (°F)", y="Normalized Composition", 
                     color="Component", title="Temperature vs Normalized Composition")
    return fig

def plot_stage_contributions(x_normalized):
    components = list(x_normalized.keys())
    df = pd.DataFrame(x_normalized)
    df["Stage"] = range(1, len(next(iter(x_normalized.values()))) + 1)
    df_melted = df.melt(id_vars="Stage", var_name="Component", value_name="Contribution")

    fig = px.area(df_melted, x="Stage", y="Contribution", color="Component", 
                  title="Stage Contributions by Component", groupnorm="fraction")
    return fig
