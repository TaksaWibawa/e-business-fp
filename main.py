# Improved Dashboard Code

import streamlit as st
import pandas as pd
import altair as alt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Set page configuration
st.set_page_config(layout="wide", page_title="Gift Shop Transactional Dashboard", page_icon=":bar_chart:", initial_sidebar_state="expanded")


# Load data
@st.cache_data
def load_data():
    try:
        df_customers = pd.read_csv("data/customers.csv")
        df_products = pd.read_csv("data/products.csv")
        df_txs = pd.read_csv("data/transactions.csv")
        return df_customers, df_products, df_txs
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


df_customers, df_products, df_txs = load_data()

if df_customers is not None and df_products is not None and df_txs is not None:
    # Merge data
    merged_df = pd.merge(df_txs, df_customers, on="Customer ID")
    merged_df = pd.merge(merged_df, df_products, on="StockCode")

    # Preprocess data
    merged_df["InvoiceDate"] = pd.to_datetime(merged_df["InvoiceDate"])
    merged_df["Total Amount"] = merged_df["Quantity"] * merged_df["Price"]
    merged_df["Month"] = merged_df["InvoiceDate"].dt.to_period("M")
    merged_df["Year"] = merged_df["InvoiceDate"].dt.year
    merged_df = merged_df.dropna(subset=["Customer ID", "Invoice", "InvoiceDate", "Quantity", "Price", "StockCode", "Country"])
    merged_df = merged_df[(merged_df["Quantity"] > 0) & (merged_df["Price"] > 0)]

    # Remove year 2009 and unnecessary stock code
    stockcodes_for_removal = [
        "POST",
        "D",
        "DOT",
        "M",
        "C2",
        "BANK CHARGES",
        "TEST001",
        "gift_0001_80",
        "gift_0001_20",
        "TEST002",
        "gift_0001_10",
        "gift_0001_50",
        "gift_0001_30",
        "gift_0001_40",
        "gift_0001_60",
        "gift_0001_70",
        "gift_0001_90",
        "GIFT",
        "S",
        "B",
        "C3",
        "SP1002",
        "AMAZONFEE",
        "CRUK",
        "DCGS0006",
        "DCGS0016",
        "DCGS0027",
        "DCGS0036",
        "DCGS0039",
        "DCGS0060",
        "DCGS0056",
        "DCGS0059",
        "DCGSLBOY",
        "DCGS0053",
        "DCGSLGIRL",
        "DCGS0055",
        "DCGS0074",
        "DCGS0057",
        "DCGS0071",
        "DCGS0066P",
    ]
    merged_df = merged_df[merged_df["Year"] != 2009]
    merged_df = merged_df[~merged_df["StockCode"].isin(stockcodes_for_removal)]

    # Sidebar for year selection
    st.sidebar.header("Filter Options")
    years = merged_df["Year"].unique()
    selected_year = st.sidebar.selectbox("Select Year", years)

    # Filter data by selected year
    filtered_df = merged_df[merged_df["Year"] == selected_year]

    # Aggregate monthly data
    monthly_data = filtered_df.groupby("Month")["Total Amount"].sum().reset_index()
    monthly_data["Month"] = monthly_data["Month"].dt.to_timestamp()

    # Train an ARIMA model for prediction if there are enough samples
    def train_arima(data):
        if len(data) > 1:
            # Set the index to be the 'Month' column with a proper frequency
            data = data.set_index("Month")
            data.index = pd.DatetimeIndex(data.index, freq="MS")

            # Differencing the data to make it stationary
            data_diff = data["Total Amount"].diff().dropna()

            model = ARIMA(data_diff, order=(5, 1, 0))
            model_fit = model.fit()

            # Forecasting
            forecast_diff = model_fit.predict(start=0, end=len(data_diff) - 1)

            # Reverting the differencing
            forecast = forecast_diff.cumsum() + data["Total Amount"].iloc[0]

            # Adjust the forecast length to match the original data length
            forecast = np.insert(forecast.values, 0, data["Total Amount"].iloc[0])

            return forecast
        return np.array([])

    forecast = train_arima(monthly_data)

    # Calculate metrics for the selected year
    total_revenue = filtered_df["Total Amount"].sum()
    total_transactions = filtered_df["Invoice"].nunique()
    average_revenue_per_day = filtered_df.groupby(filtered_df["InvoiceDate"].dt.date)["Total Amount"].sum().mean()
    average_revenue_per_month = monthly_data["Total Amount"].mean()

    # Calculate metrics for the previous year
    previous_year = selected_year - 1
    previous_year_df = merged_df[merged_df["Year"] == previous_year]

    if not previous_year_df.empty:
        previous_total_revenue = previous_year_df["Total Amount"].sum()
        previous_total_transactions = previous_year_df["Invoice"].nunique()
        previous_average_revenue_per_day = previous_year_df.groupby(previous_year_df["InvoiceDate"].dt.date)["Total Amount"].sum().mean()
        previous_average_revenue_per_month = previous_year_df.groupby(previous_year_df["Month"])["Total Amount"].sum().mean()
    else:
        previous_total_revenue = previous_total_transactions = previous_average_revenue_per_day = previous_average_revenue_per_month = 0

    # Calculate differences
    revenue_diff = total_revenue - previous_total_revenue
    transactions_diff = total_transactions - previous_total_transactions
    average_revenue_per_day_diff = average_revenue_per_day - previous_average_revenue_per_day
    average_revenue_per_month_diff = average_revenue_per_month - previous_average_revenue_per_month

    st.title(f":bar_chart: Gift Shop Transactional Dashboard ({selected_year})")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${total_revenue:,.2f}", f"${revenue_diff:,.2f}", delta_color="inverse" if revenue_diff < 0 else "normal")
    col2.metric("Total Transactions", total_transactions, transactions_diff)
    col3.metric(
        "Average Revenue per Day",
        f"${average_revenue_per_day:,.2f}",
        f"${average_revenue_per_day_diff:,.2f}",
        delta_color="inverse" if average_revenue_per_day_diff < 0 else "normal",
    )
    col4.metric(
        "Average Revenue per Month",
        f"${average_revenue_per_month:,.2f}",
        f"${average_revenue_per_month_diff:,.2f}",
        delta_color="inverse" if average_revenue_per_month_diff < 0 else "normal",
    )

    # Second section: monthly transaction count and monthly revenue
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Monthly Transaction Count")
        monthly_transaction_count = filtered_df.groupby("Month")["Invoice"].nunique().reset_index()
        monthly_transaction_count["Month"] = monthly_transaction_count["Month"].dt.to_timestamp()

        bar_chart = (
            alt.Chart(monthly_transaction_count)
            .mark_bar(size=30, color="lightcoral")
            .encode(x=alt.X("Month:T", title="Month"), y=alt.Y("Invoice:Q", title="Number of Transactions"), tooltip=["Month", "Invoice"])
            .properties(width="container", height=400)
        )

        st.altair_chart(bar_chart, use_container_width=True)

    with col2:
        st.markdown("### Monthly Revenue")
        monthly_revenue = monthly_data

        bar_chart = (
            alt.Chart(monthly_revenue)
            .mark_bar(size=30, color="lightcoral")
            .encode(x=alt.X("Month:T", title="Month"), y=alt.Y("Total Amount:Q", title="Total Amount"), tooltip=["Month", "Total Amount"])
            .properties(width="container", height=400)
        )

        st.altair_chart(bar_chart, use_container_width=True)

    # Third section: Top 10 customers and top 10 products by transaction value
    col1, col2 = st.columns(2)

    with col1:
        top10_customers = filtered_df.groupby("Customer ID")["Total Amount"].sum().nlargest(10).reset_index()
        st.markdown("### Top 10 Customers by Transaction Value")
        bar_chart = (
            alt.Chart(top10_customers)
            .mark_bar(size=30, color="lightcoral")
            .encode(
                x=alt.X("Customer ID:N", title="Customer ID", sort="y"),
                y=alt.Y("Total Amount:Q", title="Total Amount"),
                tooltip=["Customer ID", "Total Amount"],
            )
            .properties(title=f"Top 10 Customers in {selected_year}", width="container", height=400)
        )

        st.altair_chart(bar_chart, use_container_width=True)

    with col2:
        top10_products = filtered_df.groupby("StockCode")["Total Amount"].sum().nlargest(10).reset_index()
        top10_products = pd.merge(top10_products, df_products[["StockCode", "Description"]], on="StockCode")
        st.markdown("### Top 10 Products by Transaction Value")
        bar_chart = (
            alt.Chart(top10_products)
            .mark_bar(size=30, color="lightcoral")
            .encode(
                x=alt.X("StockCode:N", title="Product Code", sort="y"),
                y=alt.Y("Total Amount:Q", title="Total Amount"),
                tooltip=["StockCode", "Description", "Total Amount"],
            )
            .properties(title=f"Top 10 Products in {selected_year}", width="container", height=400)
        )

        st.altair_chart(bar_chart, use_container_width=True)

    # Fourth section: Distribution of transactions by country and comparison of actual vs predicted monthly revenue
    col1, col2 = st.columns(2)

    with col1:
        country_data = filtered_df.groupby("Country")["Total Amount"].sum().reset_index()
        country_data["Percentage"] = (country_data["Total Amount"] / country_data["Total Amount"].sum()) * 100
        country_data["Country"] = country_data["Country"] + " (" + country_data["Percentage"].round(2).astype(str) + "%)"
        st.markdown("### Distribution of Transactions by Country")

        pie_chart = (
            alt.Chart(country_data)
            .mark_arc(color="lightcoral")
            .encode(
                theta=alt.Theta(field="Total Amount", type="quantitative"),
                color=alt.Color(field="Country", type="nominal"),
                tooltip=["Country", "Total Amount", "Percentage"],
            )
            .properties(title="Transactions by Country", width="container")
        )

        st.altair_chart(pie_chart, use_container_width=True)

    with col2:
        st.markdown("### Comparison of Actual vs Predicted Monthly Revenue")
        if len(forecast) > 0:
            actual_data = pd.DataFrame({"Month": monthly_data["Month"], "Revenue": monthly_data["Total Amount"], "Type": ["Actual"] * len(monthly_data)})

            predicted_data = pd.DataFrame({"Month": monthly_data["Month"], "Revenue": forecast, "Type": ["Predicted"] * len(forecast)})

            combined_data = pd.concat([actual_data, predicted_data])

            comparison_chart = (
                alt.Chart(combined_data)
                .mark_line(point=True)
                .encode(x="Month:T", y="Revenue:Q", color=alt.Color("Type:N", scale=alt.Scale(range=["lightcoral", "darkred"])), shape="Type:N")
                .properties(title="Comparison of Actual and Predicted Monthly Revenue", width="container", height=400)
            )

            st.altair_chart(comparison_chart, use_container_width=True)
        else:
            st.write("Not enough data to perform prediction.")

    # Fifth section: Heatmap of Transactions by Day of Week and Hour
    st.markdown("### Heatmap of Transactions by Day of Week and Hour")
    filtered_df = filtered_df.assign(Day_of_Week=filtered_df["InvoiceDate"].dt.day_name(), Hour=filtered_df["InvoiceDate"].dt.hour)

    heatmap_data = filtered_df.groupby(["Day_of_Week", "Hour"])["Invoice"].count().reset_index()
    heatmap_data_pivot = heatmap_data.pivot_table(index="Day_of_Week", columns="Hour", values="Invoice", fill_value=0)

    heatmap = (
        alt.Chart(heatmap_data_pivot.reset_index().melt(id_vars=["Day_of_Week"]))
        .mark_rect()
        .encode(
            x=alt.X("Hour:O", title="Hour of Day"),
            y=alt.Y("Day_of_Week:O", title="Day of Week"),
            color=alt.Color("value:Q", title="Number of Transactions", scale=alt.Scale(scheme="reds")),
            tooltip=["Day_of_Week", "Hour", "value"],
        )
        .properties(width="container", height=400)
        .configure_mark(color="lightcoral")
    )
    st.altair_chart(heatmap, use_container_width=True)

else:
    st.write("Data could not be loaded. Please check the data files and try again.")
