import os
import sys
import psycopg
from psycopg import sql
from db_config import DB_CONFIG
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime, time
import mplfinance as mpf

MAX_CHARTS_PER_IMAGE = 10

def makeTradeAndPnlCharts(dateTimeDateStr):
    dateTimeDate = datetime.strptime(dateTimeDateStr, "%Y-%m-%d").date()
    dateTimeStart = datetime.combine(dateTimeDate, time(9, 30))  # 09:30:00
    dateTimeEnd = datetime.combine(dateTimeDate, time(16, 0))    # 16:00:00

    def getPostgresData(cur, dateTimeStart, dateTimeEnd, dateTimeDate):
        pnl_query = """
        SELECT *
        FROM pnl 
        WHERE date_time >= %s
        AND date_time < %s
        ORDER BY ticker, date_time ASC;
        """
        cur.execute(pnl_query, (dateTimeStart, dateTimeEnd))
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        
        df = pd.DataFrame(rows, columns=columns)
        df['date_time'] = pd.to_datetime(df['date_time'])

        vix_query = """
        SELECT date_time, open as vix_open
        FROM ticker_data
        WHERE date_time >= %s
        AND date_time < %s
        AND ticker = 'VIX.XO'
        ORDER BY date_time ASC;
        """
        cur.execute(vix_query, (dateTimeStart, dateTimeEnd))
        vix_data = pd.DataFrame(cur.fetchall(), columns=['date_time', 'vix_open'])
        vix_data['date_time'] = pd.to_datetime(vix_data['date_time'])
        
        df = pd.merge(df, vix_data, on='date_time', how='left')
        
        df['pnl_bps'] = np.where(
            df['open_price'].astype(float) != 0,
            df['pnl_cum'].astype(float) / df['open_price'].astype(float) * 100,
            0
        )

        macro_signals_query = """
        SELECT *
        FROM macro_signals
        WHERE date_time::date = %s
        ORDER BY ticker, date_time ASC;
        """
        cur.execute(macro_signals_query, (dateTimeDate,))
        macro_signals_columns = [desc[0] for desc in cur.description]
        macro_signals_rows = cur.fetchall()
        macro_signals_df = pd.DataFrame(macro_signals_rows, columns=macro_signals_columns)
        macro_signals_df['date_time'] = pd.to_datetime(macro_signals_df['date_time'])

        return df, df.groupby('ticker'), macro_signals_df.groupby('ticker')

    def makeSummaryChart(average_data, dateTimeDate, eod_pnl_df):

        fig, axs = plt.subplots(2, 1, figsize=(20, 20))
        ax_main = axs[0]

        x_values = average_data.index.to_numpy()

        average_data['vix_open'] = average_data['vix_open'].fillna(method='ffill').fillna(method='bfill')

        vix_values = average_data['vix_open'].astype(float).to_numpy()
        pnl_cum_values = average_data['pnl_cum'].astype(float).to_numpy()
        pnl_bps_values = average_data['pnl_bps'].astype(float).to_numpy()

        ax_pnl_bps = ax_main.twinx()
        ax_pnl_bps.spines['right'].set_position(('axes', 1.0))
        ax_pnl_bps.plot(x_values, pnl_bps_values, color='green', alpha=0.5, label='Average PnL BPS', zorder=2)
        ax_pnl_bps.set_ylabel('Average PnL BPS', color='green')
        ax_pnl_bps.tick_params(axis='y', labelcolor='green')

        ax_pnl_cum = ax_main.twinx()
        ax_pnl_cum.spines['right'].set_position(('axes', 1.05))
        ax_pnl_cum.bar(
            x_values,
            pnl_cum_values,
            alpha=0.2,
            color='purple',
            width=0.0005,
            label='Total Cumulative PnL',
            zorder=1
        )
        ax_pnl_cum.set_ylabel('Total Cumulative PnL ($)', color='purple')
        ax_pnl_cum.tick_params(axis='y', labelcolor='purple')

        ax_main.plot(x_values, vix_values, color='orange', label='VIX', zorder=3)
        ax_main.set_ylabel('VIX', color='orange')
        ax_main.tick_params(axis='y', labelcolor='orange')

        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax_main.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 30]))
        plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45)
        ax_main.grid(True, which='both', linestyle='--', alpha=0.7)

        dateTimeStart = datetime.combine(dateTimeDate, time(9, 30))
        dateTimeEnd = datetime.combine(dateTimeDate, time(15, 59))
        ax_main.set_xlim(dateTimeStart, dateTimeEnd)

        for hour in range(9, 16):
            for minute in [0, 30]:
                dt = datetime.combine(dateTimeDate, time(hour, minute))
                if dateTimeStart <= dt <= dateTimeEnd:
                    ax_main.axvline(dt, color='black', linestyle='--', linewidth=0.5)

        ax_main.set_title(f"Summary PnL and VIX - {dateTimeDate}")
        ax_main.set_xlabel("Time")

        ax_bar = axs[1]

        eod_pnl_df = eod_pnl_df.sort_values('ticker')
        tickers = eod_pnl_df['ticker'].tolist()
        pnl_cum_values = eod_pnl_df['pnl_cum'].astype(float).tolist()

        x = np.arange(len(tickers)) 
        width = 0.6  

        bars = ax_bar.bar(x, pnl_cum_values, width=width, color='blue', align='center', zorder=1)
        ax_bar.set_ylabel('PnL Cum ($)', color='blue')
        ax_bar.tick_params(axis='y', labelcolor='blue')

        max_pnl_cum = max(abs(value) for value in pnl_cum_values) if pnl_cum_values else 1
        min_pnl_cum = min(pnl_cum_values) if pnl_cum_values else 0
        ax_bar.set_ylim(min_pnl_cum * 1.1, max_pnl_cum * 1.1)

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(tickers, rotation=90)

        ax_bar.set_title('PnL Cum by Symbol at 15:59 PM')

        ax_bar.axhline(0, color='gray', linestyle='--', linewidth=1)

        ax_bar.yaxis.grid(True, linestyle='--', which='major', color='lightgray', alpha=0.7)

        plt.tight_layout()

        dir_name = f"subplots_{dateTimeDate}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        output_file = os.path.join(dir_name, f"summary_pnl_{dateTimeDate}.png")

        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Summary chart saved to {output_file}")
        plt.close(fig)

    def makeMatplotlibCharts(grouped_data):
        tickers = list(grouped_data.groups.keys())
        num_tickers = len(tickers)
        num_images = (num_tickers + MAX_CHARTS_PER_IMAGE - 1) // MAX_CHARTS_PER_IMAGE

        mc = mpf.make_marketcolors(up='g', down='r', volume='in', wick={'up':'g', 'down':'r'})
        s = mpf.make_mpf_style(marketcolors=mc)

        for img_num in range(num_images):
            start_idx = img_num * MAX_CHARTS_PER_IMAGE
            end_idx = min((img_num + 1) * MAX_CHARTS_PER_IMAGE, num_tickers)
            current_tickers = tickers[start_idx:end_idx]
            n_tickers = len(current_tickers)

            fig, axs = plt.subplots(n_tickers, 1, figsize=(20, 10*n_tickers), squeeze=False)

            fig.subplots_adjust(right=0.85, left=0.05) 

            for ticker, ax in zip(current_tickers, axs.flatten()):
                try:
                    data = grouped_data.get_group(ticker).sort_values('date_time')
                    data = data.set_index('date_time')

                    ohlc_data = data[['open', 'high', 'low', 'close', 'volume']]
                    ohlc_data = ohlc_data.astype(float)

                    volume_ax = ax.twinx()
                    pnl_ax = ax.twinx()
                    vix_ax = ax.twinx()
                    pnl_bps_ax = ax.twinx()

                    ax.set_zorder(2)
                    ax.patch.set_visible(False)
                    volume_ax.set_zorder(1)
                    pnl_ax.set_zorder(1)
                    vix_ax.set_zorder(1)
                    pnl_bps_ax.set_zorder(1)

                    pnl_bps_ax.spines['right'].set_position(('axes', 1.00))
                    pnl_ax.spines['right'].set_position(('axes', 1.05))
                    vix_ax.spines['right'].set_position(('axes', 1.10))
                    volume_ax.spines['right'].set_position(('axes', 1.15))

                    mpf.plot(ohlc_data, type='candle', ax=ax, volume=volume_ax, style=s, show_nontrading=True, volume_alpha=0.2)
                    
                    volume_ax.set_ylim(0, ohlc_data['volume'].max() * 2)

                    pnl_max = max(abs(data['pnl_cum'].min()), abs(data['pnl_cum'].max()))
                    pnl_limit = pnl_max * 2 

                    pnl_ax.bar(
                        data.index.to_numpy(),
                        data['pnl_cum'].astype(float).to_numpy(),
                        alpha=0.2,
                        color='purple',
                        width=0.001,
                        label='_nolegend_',
                        zorder=1
                    )
                    pnl_ax.set_ylim(-pnl_limit, pnl_limit) 
                    pnl_ax.set_ylabel('Cumulative PnL ($)')

                    pnl_ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

                    ax.set_ylabel("Price ($)")
                    volume_ax.set_ylabel("Volume")
                    vix_ax.set_ylabel("VIX")
                    pnl_bps_ax.set_ylabel("PnL BPS")

                    if 'vix_open' in data.columns:
                        vix_data = data['vix_open'].dropna()
                        if not vix_data.empty:
                            vix_dates = vix_data.index.to_numpy() 
                            vix_values = vix_data.values.astype(float)
                            vix_ax.plot(vix_dates, vix_values, color='orange', linewidth=1, alpha=0.6, label='_nolegend_', zorder=3)
                            vix_ax.set_ylim(vix_values.min() * 0.95, vix_values.max() * 1.05)

                    pnl_bps_data = data['pnl_bps'].dropna()
                    if not pnl_bps_data.empty:
                        pnl_bps_values = pnl_bps_data.values.astype(float)
                        pnl_bps_max = max(abs(pnl_bps_values.min()), abs(pnl_bps_values.max()))
                        pnl_bps_limit = pnl_bps_max * 2
                        pnl_bps_ax.set_ylim(-pnl_bps_limit, pnl_bps_limit)

                    for idx, row in data.iterrows():
                        if pd.notnull(row['pin_openclose']):
                            price = row['open_price'] if row['pin_openclose'] == 'O' else row['close_price']
                            direction = 'up' if row['pin_buysell'] == 'BUY' else 'down'
                            color = {
                                ('O', 'BUY'): 'blue',
                                ('O', 'SELL'): 'red',
                                ('C', 'SELL'): 'darkblue',
                                ('C', 'BUY'): '#800020'  
                            }[(row['pin_openclose'], row['pin_buysell'])]

                            marker = '^' if direction == 'up' else 'v'

                            ax.annotate(f'{float(price):.2f}', (idx, float(price)), 
                                        xytext=(0, 10 if direction == 'up' else -10), 
                                        textcoords='offset points',
                                        ha='center', va='center',
                                        fontsize=8, color=color)
                            ax.scatter(idx, float(price), marker=marker, s=100, color=color)

                    ax.set_title(f"{ticker} - {dateTimeDate}")
                    ax.set_xlabel("Time")
                    
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0,30]))
                    ax.grid(True, which='both', linestyle='--', alpha=0.7)
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                    
                    ax.set_xlim(data.index.min(), data.index.max())

                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = pnl_ax.get_legend_handles_labels()
                    lines3, labels3 = volume_ax.get_legend_handles_labels()
                    lines5, labels5 = pnl_bps_ax.get_legend_handles_labels()

                    all_lines = lines + lines2 + lines3 + lines5
                    all_labels = labels + labels2 + labels3 + labels5

                    all_lines_labels = [
                        (line, label) for line, label in zip(all_lines, all_labels) if label and label != '_nolegend_'
                    ]

                    if all_lines_labels:
                        all_lines_filtered, all_labels_filtered = zip(*all_lines_labels)
                        ax.legend(all_lines_filtered, all_labels_filtered, loc='upper left')
                    else:
                        pass 

                except Exception as e:
                    print(f"Error processing {ticker}: {str(e)}")
                    import traceback
                    traceback.print_exc()

            plt.tight_layout()
            
            dir_name = f"subplots_{dateTimeDate}"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            output_file = os.path.join(dir_name, f"bpsVix_{dateTimeDate}_pg{img_num+1}.png")

            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Charts saved to {output_file}")
            plt.close(fig)

    def makeIndividualCharts(grouped_data, macro_signals_grouped, dateTimeDate):
        dir_name = f"subplots_{dateTimeDate}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        for ticker in grouped_data.groups.keys():
            try:
                data = grouped_data.get_group(ticker).sort_values('date_time')
                data = data.set_index('date_time')

                if ticker in macro_signals_grouped.groups:
                    macro_data = macro_signals_grouped.get_group(ticker).sort_values('date_time')
                    macro_data = macro_data.set_index('date_time')
                    data = data.merge(macro_data, left_index=True, right_index=True, how='left')
                else:
                    print(f"No macro_signals data for ticker {ticker}")
                    data['long_prob1'] = np.nan
                    data['flat_prob1'] = np.nan
                    data['short_prob1'] = np.nan
                    data['up_prob2'] = np.nan
                    data['sedeways_prob2'] = np.nan
                    data['down_prob2'] = np.nan
                    data['high_prob3'] = np.nan
                    data['mid_prob3'] = np.nan
                    data['low_prob3'] = np.nan
                    data['kama_long'] = np.nan
                    data['kama_short'] = np.nan

                fig = plt.figure(figsize=(20, 30))
                gs = fig.add_gridspec(10, 1, height_ratios=[4,1,1,1,1,1,1,1,1,1])

                ax_main = fig.add_subplot(gs[0, 0])

                ohlc_data = data[['open', 'high', 'low', 'close', 'volume']]
                ohlc_data = ohlc_data.astype(float)

                volume_ax = ax_main.twinx()
                pnl_ax = ax_main.twinx()
                vix_ax = ax_main.twinx()
                pnl_bps_ax = ax_main.twinx()

                ax_main.set_zorder(2)
                ax_main.patch.set_visible(False)
                volume_ax.set_zorder(1)
                pnl_ax.set_zorder(1)
                vix_ax.set_zorder(1)
                pnl_bps_ax.set_zorder(1)

                pnl_bps_ax.spines['right'].set_position(('axes', 1.00))
                pnl_ax.spines['right'].set_position(('axes', 1.05))
                vix_ax.spines['right'].set_position(('axes', 1.10))
                volume_ax.spines['right'].set_position(('axes', 1.15))

                mc = mpf.make_marketcolors(up='g', down='r', volume='in', wick={'up':'g', 'down':'r'})
                s = mpf.make_mpf_style(marketcolors=mc)

                mpf.plot(ohlc_data, type='candle', ax=ax_main, volume=volume_ax, style=s, show_nontrading=True, volume_alpha=0.2)

                volume_ax.set_ylim(0, ohlc_data['volume'].max() * 2)

                pnl_max = max(abs(data['pnl_cum'].min()), abs(data['pnl_cum'].max()))
                pnl_limit = pnl_max * 2 

                pnl_ax.bar(
                    data.index.to_numpy(),
                    data['pnl_cum'].astype(float).to_numpy(),
                    alpha=0.2,
                    color='purple',
                    width=0.001,
                    label='_nolegend_',
                    zorder=1
                )
                pnl_ax.set_ylim(-pnl_limit, pnl_limit) 
                pnl_ax.set_ylabel('Cumulative PnL ($)')

                pnl_ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

                ax_main.set_ylabel("Price ($)")
                volume_ax.set_ylabel("Volume")
                vix_ax.set_ylabel("VIX")
                pnl_bps_ax.set_ylabel("PnL BPS")

                if 'vix_open' in data.columns:
                    vix_data = data['vix_open'].dropna()
                    if not vix_data.empty:
                        vix_dates = vix_data.index.to_numpy() 
                        vix_values = vix_data.values.astype(float)
                        vix_ax.plot(vix_dates, vix_values, color='orange', linewidth=1, alpha=0.6, label='_nolegend_', zorder=3)
                        vix_ax.set_ylim(vix_values.min() * 0.95, vix_values.max() * 1.05)

                pnl_bps_data = data['pnl_bps'].dropna()
                if not pnl_bps_data.empty:
                    pnl_bps_values = pnl_bps_data.values.astype(float)
                    pnl_bps_max = max(abs(pnl_bps_values.min()), abs(pnl_bps_values.max()))
                    pnl_bps_limit = pnl_bps_max * 2
                    pnl_bps_ax.set_ylim(-pnl_bps_limit, pnl_bps_limit)

                for idx, row in data.iterrows():
                    if pd.notnull(row['pin_openclose']):
                        price = row['open_price'] if row['pin_openclose'] == 'O' else row['close_price']
                        direction = 'up' if row['pin_buysell'] == 'BUY' else 'down'
                        color = {
                            ('O', 'BUY'): 'blue',
                            ('O', 'SELL'): 'red',
                            ('C', 'SELL'): 'darkblue',
                            ('C', 'BUY'): '#800020' 
                        }[(row['pin_openclose'], row['pin_buysell'])]

                        marker = '^' if direction == 'up' else 'v'

                        ax_main.annotate(f'{float(price):.2f}', (idx, float(price)), 
                                    xytext=(0, 10 if direction == 'up' else -10), 
                                    textcoords='offset points',
                                    ha='center', va='center',
                                    fontsize=8, color=color)
                        ax_main.scatter(idx, float(price), marker=marker, s=100, color=color)

                ax_main.set_title(f"{ticker} - {dateTimeDate}")

                ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                ax_main.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0,30]))
                ax_main.grid(True, which='both', linestyle='--', alpha=0.7)
                plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45)

                ax_main.set_xlim(data.index.min(), data.index.max())

                lines, labels = ax_main.get_legend_handles_labels()
                lines2, labels2 = pnl_ax.get_legend_handles_labels()
                lines3, labels3 = volume_ax.get_legend_handles_labels()
                lines5, labels5 = pnl_bps_ax.get_legend_handles_labels()

                all_lines = lines + lines2 + lines3 + lines5
                all_labels = labels + labels2 + labels3 + labels5

                all_lines_labels = [
                    (line, label) for line, label in zip(all_lines, all_labels) if label and label != '_nolegend_'
                ]

                if all_lines_labels:
                    all_lines_filtered, all_labels_filtered = zip(*all_lines_labels)
                    ax.legend(all_lines_filtered, all_labels_filtered, loc='upper left')
                else:
                    pass  

                ax_general1 = fig.add_subplot(gs[1, 0], sharex=ax_main)
                ax_general2 = fig.add_subplot(gs[2, 0], sharex=ax_main)
                ax_general3 = fig.add_subplot(gs[3, 0], sharex=ax_main)
                ax_general4 = fig.add_subplot(gs[4, 0], sharex=ax_main)

                if all(col in data.columns for col in ['long_prob1', 'flat_prob1', 'short_prob1']):
                    x_values = data.index.to_numpy()
                    long_prob = data['long_prob1'].astype(float).to_numpy()
                    flat_prob = data['flat_prob1'].astype(float).to_numpy()
                    short_prob = data['short_prob1'].astype(float).to_numpy()

                    ax_general1.plot(x_values, long_prob, color='green', label='long_prob1')
                    ax_general1.plot(x_values, flat_prob, color='gray', label='flat_prob1')
                    ax_general1.plot(x_values, short_prob, color='red', label='short_prob1')

                    ax_general1.set_ylabel('Probabilities')
                    ax_general1.set_yticks(np.arange(0, 1.1, 0.1))
                    ax_general1.set_ylim(0, 1.0)

                    ax_general1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax_general1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                    ax_general1.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0,30]))
                    plt.setp(ax_general1.xaxis.get_majorticklabels(), rotation=45)

                    box = ax_general1.get_position()
                    ax_general1.set_position([box.x0, box.y0, box.width * 0.85, box.height])
                    ax_general1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)

                    for hour in range(9, 17):
                        for minute in [0, 30]:
                            dt = datetime.combine(dateTimeDate, time(hour, minute))
                            if dateTimeStart <= dt <= dateTimeEnd:
                                ax_general1.axvline(dt, color='black', linestyle='--', linewidth=0.5)

                    ax_general1.axhline(0.5, color='black', linestyle='--', linewidth=0.5)

                if all(col in data.columns for col in ['up_prob2', 'sedeways_prob2', 'down_prob2']):
                    x_values = data.index.to_numpy()
                    up_prob = data['up_prob2'].astype(float).to_numpy()
                    sedeways_prob = data['sedeways_prob2'].astype(float).to_numpy()
                    down_prob = data['down_prob2'].astype(float).to_numpy()

                    ax_general2.plot(x_values, up_prob, color='green', label='up_prob2')
                    ax_general2.plot(x_values, sedeways_prob, color='gray', label='sedeways_prob2')
                    ax_general2.plot(x_values, down_prob, color='red', label='down_prob2')

                    ax_general2.set_ylabel('Probabilities')
                    ax_general2.set_yticks(np.arange(0, 1.1, 0.1))
                    ax_general2.set_ylim(0, 1.0)

                    ax_general2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax_general2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                    ax_general2.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0,30]))
                    plt.setp(ax_general2.xaxis.get_majorticklabels(), rotation=45)

                    box = ax_general2.get_position()
                    ax_general2.set_position([box.x0, box.y0, box.width * 0.85, box.height])
                    ax_general2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)

                    for hour in range(9, 17):
                        for minute in [0, 30]:
                            dt = datetime.combine(dateTimeDate, time(hour, minute))
                            if dateTimeStart <= dt <= dateTimeEnd:
                                ax_general2.axvline(dt, color='black', linestyle='--', linewidth=0.5)

                    ax_general2.axhline(0.5, color='black', linestyle='--', linewidth=0.5)

                if all(col in data.columns for col in ['high_prob3', 'mid_prob3', 'low_prob3']):
                    x_values = data.index.to_numpy()
                    high_prob = data['high_prob3'].astype(float).to_numpy()
                    mid_prob = data['mid_prob3'].astype(float).to_numpy()
                    low_prob = data['low_prob3'].astype(float).to_numpy()

                    ax_general3.plot(x_values, high_prob, color='green', label='high_prob3')
                    ax_general3.plot(x_values, mid_prob, color='gray', label='mid_prob3')
                    ax_general3.plot(x_values, low_prob, color='red', label='low_prob3')

                    ax_general3.set_ylabel('Probabilities')
                    ax_general3.set_yticks(np.arange(0, 1.1, 0.1))
                    ax_general3.set_ylim(0, 1.0)

                    ax_general3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax_general3.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                    ax_general3.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0,30]))
                    plt.setp(ax_general3.xaxis.get_majorticklabels(), rotation=45)

                    box = ax_general3.get_position()
                    ax_general3.set_position([box.x0, box.y0, box.width * 0.85, box.height])
                    ax_general3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)

                    for hour in range(9, 17):
                        for minute in [0, 30]:
                            dt = datetime.combine(dateTimeDate, time(hour, minute))
                            if dateTimeStart <= dt <= dateTimeEnd:
                                ax_general3.axvline(dt, color='black', linestyle='--', linewidth=0.5)

                    ax_general3.axhline(0.5, color='black', linestyle='--', linewidth=0.5)

                if all(col in data.columns for col in ['kama_long', 'kama_short']):
                    x_values = data.index.to_numpy()
                    kama_long = data['kama_long'].astype(float).to_numpy()
                    kama_short = data['kama_short'].astype(float).to_numpy()

                    ax_general4.plot(x_values, kama_long, color='green', label='kama_long', alpha=0.5)
                    ax_general4.plot(x_values, kama_short, color='red', label='kama_short', alpha=0.5)

                    ax_general4.set_ylabel('KAMA Signals')
                    ax_general4.set_ylim(-0.1, 1.1)
                    ax_general4.set_yticks([0, 1])

                    ax_general4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax_general4.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                    ax_general4.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0,30]))
                    plt.setp(ax_general4.xaxis.get_majorticklabels(), rotation=45)

                    box = ax_general4.get_position()
                    ax_general4.set_position([box.x0, box.y0, box.width * 0.85, box.height])
                    ax_general4.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)

                    for hour in range(9, 17):
                        for minute in [0, 30]:
                            dt = datetime.combine(dateTimeDate, time(hour, minute))
                            if dateTimeStart <= dt <= dateTimeEnd:
                                ax_general4.axvline(dt, color='black', linestyle='--', linewidth=0.5)

                plt.tight_layout()

                output_file = os.path.join(dir_name, f"{ticker}_{dateTimeDate}.png")
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Chart saved to {output_file}")
                plt.close(fig)

            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                import traceback
                traceback.print_exc()

    conn = None
    cur = None
    try:
        conn = psycopg.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        df, grouped_data, macro_signals_grouped = getPostgresData(cur, dateTimeStart, dateTimeEnd, dateTimeDate)
        
        average_data = df.groupby('date_time').agg({
            'pnl_bps': 'mean',
            'pnl_cum': 'sum', 
            'vix_open': 'mean'
        })
        
        dateTimeTarget = datetime.combine(dateTimeDate, time(15, 59))
        eod_pnl_df = df[df['date_time'] <= dateTimeTarget].groupby('ticker').apply(lambda x: x.sort_values('date_time').iloc[-1]).reset_index(drop=True)
        eod_pnl_df = eod_pnl_df[['ticker', 'pnl_cum']]

        total_pnl_cum = eod_pnl_df['pnl_cum'].sum()

        if total_pnl_cum != 0:
            eod_pnl_df['pnl_percentage'] = eod_pnl_df['pnl_cum'] / total_pnl_cum * 100
        else:
            eod_pnl_df['pnl_percentage'] = 0

        eod_pnl_df['pnl_percentage'] = eod_pnl_df['pnl_percentage'].fillna(0)

        makeSummaryChart(average_data, dateTimeDate, eod_pnl_df)
        
        makeMatplotlibCharts(grouped_data)
        
        makeIndividualCharts(grouped_data, macro_signals_grouped, dateTimeDate)
        
        for ticker, data in grouped_data:
            print(f"Ticker: {ticker}")
            print(f"Total PnL: {data['pnl_base'].astype(float).sum()}")
            print(f"Strategies used: {data['strategy'].unique()}")
            print("---")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        if conn is not None:
            conn.rollback()
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

dateTimeDateStr = "2025-04-11"
makeTradeAndPnlCharts(dateTimeDateStr)

