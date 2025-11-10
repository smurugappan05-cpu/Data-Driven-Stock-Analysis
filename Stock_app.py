
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import glob
import os

st.title('📊 Stock Analysis Dashboard')
st.markdown("---")  # Horizontal divider

stk_data_list = ['2023-10','2023-11','2023-12','2024-01','2024-02','2024-03','2024-04','2024-05','2024-06',
                 '2024-07','2024-08','2024-09','2024-10','2024-11']
all_dfs = []
# len_all_dfs = []

for i in stk_data_list:
    # Fix 1: Use f-string to insert the folder name into the path
    yaml_files = glob.glob(f"data/{i}/*.yaml")
    print(f"Processing month: {i}")
    print(f"Found YAML files: {len(yaml_files)}")

    
    file_dfs = []  # Store DataFrames for this folder
    # len_file_dfs = []
    
    for j in yaml_files:
        with open(j, 'r') as file:  # Fix 2: Remove quotes around j and use 'r' correctly
            yaml_data = yaml.safe_load(file)
            
        df = pd.DataFrame(yaml_data)  # Fix 3: No need for df(num)
        file_dfs.append(df)
    #     len_file_dfs.append(len(df))
    # print(sum(len_file_dfs))
    
    if file_dfs:
        # Fix 4: Append the concatenated result of this folder to the master list
        whole_df = pd.concat(file_dfs, ignore_index=True)
        all_dfs.append(whole_df)
#         len_all_dfs.append(len(whole_df))
# print (sum(len_all_dfs))
# Optional: Combine everything into one DataFrame
final_df = pd.concat(all_dfs, ignore_index=True)
# print(final_df.head())
# Save the combined and cleaned DataFrame
final_df.to_csv(r'C:\Users\z039692\OneDrive - Alliance\Desktop\Data_Driven_Stock_Analysis\data\final.csv', index=False)


Tickers = list(final_df['Ticker'].unique())
returns = []
start_prices = []
close_prices = []
avg_prices = []
avg_volumes = []
std_devs = []
cum_returns = []
clo_pct_df = pd.DataFrame()

for Tic in Tickers:
    data = final_df[final_df['Ticker']== Tic].reset_index(drop=True)
    clo_pct_df[Tic]= data['close'].pct_change()
    start_price = data.iloc[0]['open']
    close_price = data.iloc[-1]['close']
    yearly_return = (close_price/start_price)-1
    avg_price = data['close'].mean().round(2)
    avg_volume = data['volume'].mean()
    daily_rets = []
    for i in range(1,len(data.index)):
        daily_ret = (data.iloc[i]['close']-data.iloc[i-1]['close'])/data.iloc[i-1]['close']
        daily_rets.append(daily_ret)
    cum_returns.append(sum(daily_rets))
    mean = sum(daily_rets) / len(daily_rets)
    variance = sum((x - mean) ** 2 for x in daily_rets) / (len(daily_rets) - 1)
    std_dev = variance ** 0.5
 
    std_devs.append(std_dev)
    start_prices.append(start_price)
    close_prices.append(close_price)
    returns.append(yearly_return)
    avg_prices.append(avg_price)
    avg_volumes.append(avg_volume)

yearly_return_df = pd.DataFrame({
    'Tickers' : Tickers,
    'Annual_returns': returns,
    'start_prices': start_prices,
    'close_prices': close_prices,
    'avg_prices':avg_prices,
    'avg_volumes':avg_volumes,
    'std_devs':std_devs,
    'cum_returns':cum_returns
    })

r = st.sidebar.radio('Navigation',['1.Volatility Analysis','2.Cumulative Return Over Time', '3.Sector-wise Performance',
                                   '4.Stock Price Correlation','5.Top 5 Gainers and Losers'])

if r == '1.Volatility Analysis': 
    top_10_most_vol_df = (yearly_return_df.sort_values(['std_devs'],ascending=False)).reset_index(drop=True).head(10)

    st.dataframe(top_10_most_vol_df)

    plt.bar(top_10_most_vol_df['Tickers'],top_10_most_vol_df['std_devs'])
    plt.xlabel('Stock Tickers')
    plt.ylabel('Standard Deviation')
    plt.title('Top 10 Most Volatile Stocks')
    plt.xticks(rotation = 75)

    st.pyplot(plt)  

if r == '2.Cumulative Return Over Time':
    st.subheader("Cumulative Return for Top 5 Performing Stocks")

    # Ensure 'date' column is datetime
    final_df['date'] = pd.to_datetime(final_df['date'])

    # Dictionary to store each ticker's cumulative return
    cumulative_returns = {}

    for tic in Tickers:
        data = final_df[final_df['Ticker'] == tic].sort_values('date')
        data['daily_return'] = data['close'].pct_change()
        data['cumulative_return'] = (1 + data['daily_return']).cumprod() - 1
        cumulative_returns[tic] = data[['date', 'cumulative_return']]

    # Get final cumulative return values
    final_cum_returns = {tic: df['cumulative_return'].iloc[-1] for tic, df in cumulative_returns.items() if len(df) > 0}

    # Pick top 5 performing stocks
    top_5_cum = sorted(final_cum_returns.items(), key=lambda x: x[1], reverse=True)[:5]

    # Plot top 5 cumulative returns
    fig_cum, ax_cum = plt.subplots(figsize=(8, 4))
    for tic, _ in top_5_cum:
        ax_cum.plot(cumulative_returns[tic]['date'], cumulative_returns[tic]['cumulative_return'], label=tic)

    ax_cum.set_ylabel("Cumulative Return")
    ax_cum.set_xlabel("Date")
    ax_cum.set_title("Top 5 Performing Stocks - Cumulative Return Over Time")
    ax_cum.legend(fontsize=8)
    ax_cum.tick_params(axis='x', rotation=45)
    st.pyplot(fig_cum)
 

 
if r == '3.Sector-wise Performance':

    sectors_df = pd.read_csv(f"Sector_data - Sheet1.csv")

    sectors_df['tic_symbol']=sectors_df['Symbol'].str.split(':').str[1]

    yearly_return_df['Tickers'] = yearly_return_df['Tickers'].str.strip()
    sectors_df['tic_symbol'] = sectors_df['tic_symbol'].str.strip()

    sector_map = dict(zip(sectors_df['tic_symbol'],sectors_df['sector']))
    yearly_return_df['sector'] = yearly_return_df['Tickers'].map(sector_map)
    

    yearly_return_df.loc[yearly_return_df['Tickers']=='TATACONSUM', 'sector'] = 'FMCG'

    sector_map = ({
        'BHARTIARTL':'TELECOM',
        'ADANIENT':'MISCELLANEOUS'
    })

    yearly_return_df['sector'] = yearly_return_df['Tickers'].map(sector_map).fillna(yearly_return_df['sector'])

    yearly_return_df.loc[yearly_return_df['Tickers']=='BRITANNIA', 'sector'] = 'FMCG'

    sectors = list(yearly_return_df['sector'].unique())
    sec_annual_returns = []
    for sec in sectors:
        info = yearly_return_df[yearly_return_df['sector']==sec].reset_index(drop=True)
        sec_annual_return = info['Annual_returns'].mean()
        sec_annual_returns.append(sec_annual_return)

    secwise_returns = pd.DataFrame({
        'sectors':sectors,
        'Avg_yearly_return':sec_annual_returns
    })

    secwise_returns = secwise_returns.sort_values('Avg_yearly_return',ascending=False).reset_index(drop=True)
    st.dataframe(secwise_returns)

    plt.bar(secwise_returns['sectors'],secwise_returns['Avg_yearly_return'])
    plt.xlabel('Sectors')
    plt.ylabel('Avg_yearly_return')
    plt.title('Sectors Wise Avg Yearly Return')
    plt.xticks(rotation = 90)
    st.pyplot(plt)



if r == '4.Stock Price Correlation':
    correlation_matrix = clo_pct_df.corr()
    print(correlation_matrix)
    len(clo_pct_df.columns)

    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(56, 40))  # 👈 Increase width and height in inches

    sns.heatmap(
        clo_pct_df.corr(),
        annot=True,
        fmt=".2f",                   # Format values to 2 decimals
        cmap='coolwarm',
        linewidths=0.5,
        square=True,
        annot_kws={"size": 18},     # 👈 Increase annotation text size
        cbar_kws={"label": "Correlation", "shrink": 0.8}  # Optional: better colorbar
    )


    plt.title("Stock Correlation Matrix", fontsize=28)
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(rotation=0, fontsize=18)

    plt.tight_layout()                                  # Auto-fit everything
    st.pyplot(plt)


if r == '5.Top 5 Gainers and Losers':
    stocks = list(final_df['Ticker'].unique())
    tickers = []
    months = []
    months_open = []
    months_close = []
    for stock in stocks:
        infos = final_df[final_df['Ticker']==stock].reset_index(drop=True)
        grouped_data = infos.groupby(['month'])
        for a, b in grouped_data:
            months.append(b.iloc[0]['month'])

            tickers.append(stock)
            months_open.append(b.iloc[0]['open'])
            months_close.append(b.iloc[-1]['close'])

    monthly_return = pd.DataFrame({
        'tickers':tickers,
        'months':months,
        'months_open':months_open,
        'months_close':months_close
    })
    monthly_return['month_return'] = (
        (monthly_return['months_close'] - monthly_return['months_open'])
        / monthly_return['months_open']
    )

    periods = list(monthly_return['months'].unique())

    for period in periods:
        temp = monthly_return[monthly_return['months'] == period]
        top5 = temp.sort_values(by='month_return', ascending=False).head(5)
        bottom5 = temp[temp['month_return'] < 0].sort_values(by='month_return').head(5)

        # Combine both sets for plotting
        combined = pd.concat([top5, bottom5])

        # Ploty
        # thei plasit siteurgery id is forll
        plt.figure(figsize=(10, 6))
        bars = plt.bar(combined['tickers'], combined['month_return'], 
                        color=combined['month_return'].apply(lambda x: 'green' if x > 0 else 'red'))

        # Title and formatting

        plt.title(f'Top 5 Gainers and Losers – {pd.to_datetime(period).strftime("%B %Y")}')
        plt.ylabel('Monthly Return (%)')
        plt.xticks(rotation = 75)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.grid(axis='y', linestyle='--', linewidth=0.5)

        # Add return labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2%}', xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3 if height >= 0 else -15),
                            textcoords="offset points", ha='center', fontsize=8)

        plt.tight_layout()
        st.pyplot(plt)