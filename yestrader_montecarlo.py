import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.font_manager as fm
from tkinter import ttk
import math
import seaborn as sns
import os
import sys

# Function to set matplotlib font to support Korean characters
def set_korean_font():
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 'Malgun Gothic' is commonly used in Windows for Korean
    plt.rcParams['axes.unicode_minus'] = False  # Ensure minus sign is shown correctly

# Function to calculate the maximum underwater period
def max_underwater_period(cumulative_returns):
    running_max = np.maximum.accumulate(cumulative_returns)
    underwater_periods = (running_max - cumulative_returns) > 0  # Boolean array indicating underwater periods
    underwater_durations = []
    current_duration = 0

    for period in underwater_periods:
        if period:
            current_duration += 1
        else:
            if current_duration > 0:
                underwater_durations.append(current_duration)
            current_duration = 0

    if current_duration > 0:
        underwater_durations.append(current_duration)

    return max(underwater_durations) if underwater_durations else 0

# Function to perform Monte Carlo Simulation and calculate benchmarks
def run_simulation():
    try:
        # Set font for Korean display
        set_korean_font()

        # Get inputs from the entries
        file_path = file_path_entry.get()
        sheet_name = sheet_name_entry.get()
        num_simulations = int(simulation_entry.get())
        initial_value = float(initial_value_entry.get())

        # Load the selected Excel file
        if sheet_name:
            data = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            # Load the first sheet by default and get its name
            xls = pd.ExcelFile(file_path)
            sheet_name = xls.sheet_names[0]
            data = pd.read_excel(file_path, sheet_name=sheet_name)

        # Select the second column
        if data.shape[1] < 2:
            raise ValueError("The sheet must contain at least two columns.")
        returns = data.iloc[:, 1].dropna()  # Select the second column and drop any missing values

        # Perform Monte Carlo simulation using daily returns
        num_days = len(returns)
        simulation_daily_returns = np.zeros((num_simulations, num_days))

        for i in range(num_simulations):
            random_daily_returns = np.random.choice(returns, size=num_days, replace=True)
            simulation_daily_returns[i] = random_daily_returns

        # Create a figure for plotting cumulative PnL of simulations with seaborn style
        sns.set_theme(style="whitegrid", font='Malgun Gothic')  # Apply seaborn whitegrid style

        fig, ax = plt.subplots(figsize=(12, 5))
        cumulative_simulations = np.cumsum(simulation_daily_returns, axis=1)  # Cumulative PnL for plotting

        # Calculate percentiles for shading
        percentiles = np.percentile(cumulative_simulations, [5, 25, 50, 75, 95], axis=0)
        days = np.arange(num_days)

        # Plot the median line
        ax.plot(days, percentiles[2], label='Median', color='blue', linewidth=2)

        # Fill between the percentiles for shading
        ax.fill_between(days, percentiles[0], percentiles[4], color='lightblue', alpha=0.3, label='5th-95th Percentile')
        ax.fill_between(days, percentiles[1], percentiles[3], color='blue', alpha=0.3, label='25th-75th Percentile')

        # Plot a few individual simulation paths for better visualization
        for i in range(min(num_simulations, 20)):  # Show up to 20 paths
            ax.plot(days, cumulative_simulations[i], color='gray', alpha=0.5, linewidth=0.8)

        # Highlight the first and last simulation paths
        # ax.plot(days, cumulative_simulations[0], color='red', linestyle='--', alpha=0.7, label='First Simulation')
        # ax.plot(days, cumulative_simulations[-1], color='green', linestyle='--', alpha=0.7, label='Last Simulation')

        # Set titles and labels
        # ax.set_title(f'Monte Carlo Simulation for {sheet_name}', fontsize=16)
        # ax.set_xlabel('Days', fontsize=12)
        # ax.set_ylabel('Cumulative PnL (Pt)', fontsize=12)
        ax.set_title(f'{sheet_name}에 대한 몬테카를로 시뮬레이션', fontsize=16)
        ax.set_xlabel('일수', fontsize=12)
        ax.set_ylabel('누적 PnL (Pt)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Add legend
        ax.legend()

        # Clear previous canvas if it exists
        if hasattr(run_simulation, 'canvas'):
            run_simulation.canvas.get_tk_widget().destroy()

        # Embed the plot in the Tkinter window
        run_simulation.canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        run_simulation.canvas.draw()
        run_simulation.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Calculate summary statistics based on final PnL
        final_pnl = cumulative_simulations[:, -1]
        mean_final_pnl = np.mean(final_pnl)
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        final_pnl_percentiles = np.percentile(final_pnl, percentiles)

        # 기본 전략에 대한 계산
        cumulative_returns_base = initial_value + np.cumsum(returns)  # 기본 전략의 누적 수익률
        final_value_base = cumulative_returns_base.iloc[-1]  # 기본 전략의 최종 누적 수익률

        # 기본 전략의 승률 계산 수정
        profitable_days = np.sum(returns > 0)
        non_zero_days = np.sum(returns != 0)
        winning_probability_base = profitable_days / non_zero_days if non_zero_days != 0 else 0

        # 기본 전략의 손익비
        positive_returns_base = returns[returns > 0]
        negative_returns_base = returns[returns < 0]
        average_profit_base = np.mean(positive_returns_base) if len(positive_returns_base) > 0 else 0
        average_loss_base = np.mean(negative_returns_base) if len(negative_returns_base) > 0 else 0
        profit_loss_ratio_base = average_profit_base / abs(average_loss_base) if average_loss_base != 0 else float('inf')

        # 기본 전략의 샤프 비율
        average_return_base = np.mean(returns)
        std_dev_return_base = np.std(returns)
        sharpe_ratio_base = average_return_base / std_dev_return_base * np.sqrt(252 / 12) if std_dev_return_base != 0 else float('inf')

        # 기본 전략의 CAGR
        years = num_days / 252  # Typical trading year assumption: 252 trading days in a year
        if final_value_base > initial_value:
            cagr_base = (final_value_base / initial_value)**(1/years) - 1
        else:
            cagr_base = 0

        # 기본 전략의 MDD
        running_max_base = np.maximum.accumulate(cumulative_returns_base)
        drawdown_base = (running_max_base - cumulative_returns_base) / running_max_base
        drawdown_base[running_max_base == 0] = 0  # Ensure no division by zero issues
        max_drawdown_base = np.max(drawdown_base)

        # 기본 전략의 최대 underwater 기간
        max_underwater_period_base = max_underwater_period(cumulative_returns_base)

        # 기본 전략의 보상비율
        reward_ratio_base = final_value_base / abs(np.max(running_max_base - cumulative_returns_base)) if max_drawdown_base != 0 else float('inf')

        # 몬테카를로 시뮬레이션에서 개별 지표의 중앙값 및 분위 계산
        win_probabilities_mc = []
        profit_loss_ratios_mc = []
        sharpe_ratios_mc = []
        cagr_mc = []
        max_drawdowns_mc = []
        max_underwater_periods_mc = []
        reward_ratios_mc = []

        for sim in range(num_simulations):
            sim_daily_returns = simulation_daily_returns[sim]
            sim_cumulative_returns = initial_value + np.cumsum(sim_daily_returns)
            final_pnl_sim = sim_cumulative_returns[-1]

            # 승률 계산
            sim_profitable_days = np.sum(sim_daily_returns > 0)
            sim_non_zero_days = np.sum(sim_daily_returns != 0)
            win_probabilities_mc.append(sim_profitable_days / sim_non_zero_days if sim_non_zero_days != 0 else 0)

            # 손익비 계산
            sim_positive_returns = sim_daily_returns[sim_daily_returns > 0]
            sim_negative_returns = sim_daily_returns[sim_daily_returns < 0]
            average_profit_mc = np.mean(sim_positive_returns) if len(sim_positive_returns) > 0 else 0
            average_loss_mc = np.mean(sim_negative_returns) if len(sim_negative_returns) > 0 else 0
            profit_loss_ratios_mc.append(average_profit_mc / abs(average_loss_mc) if average_loss_mc != 0 else float('inf'))

            # 샤프 비율 계산
            mean_return_mc = np.mean(sim_daily_returns)
            std_dev_return_mc = np.std(sim_daily_returns)
            sharpe_ratios_mc.append(mean_return_mc / std_dev_return_mc * np.sqrt(252 / 12) if std_dev_return_mc != 0 else float('inf'))

            # CAGR 계산
            if final_pnl_sim > initial_value:
                cagr_value = (final_pnl_sim / initial_value)**(1/years) - 1
            else:
                cagr_value = 0
            cagr_mc.append(cagr_value)

            # MDD 계산
            sim_running_max = np.maximum.accumulate(sim_cumulative_returns)
            sim_drawdown = (sim_running_max - sim_cumulative_returns) / sim_running_max
            sim_drawdown[sim_running_max == 0] = 0  # Ensure no division by zero issues
            max_drawdowns_mc.append(np.max(sim_drawdown))

            # 최대 underwater 기간 계산
            max_underwater_periods_mc.append(max_underwater_period(sim_cumulative_returns))

            # 보상비율 계산
            reward_ratios_mc.append(final_pnl_sim / abs(np.max(sim_running_max - sim_cumulative_returns)) if np.max(sim_drawdown) != 0 else float('inf'))

        # 각 지표의 중앙값 및 분위 계산
        win_probability_percentiles = np.percentile(win_probabilities_mc, percentiles)
        mean_win_probability = np.mean(win_probabilities_mc)

        profit_loss_ratio_percentiles = np.percentile(profit_loss_ratios_mc, percentiles)
        mean_profit_loss_ratio = np.mean(profit_loss_ratios_mc)

        sharpe_ratio_percentiles = np.percentile(sharpe_ratios_mc, percentiles)
        mean_sharpe_ratio = np.mean(sharpe_ratios_mc)

        cagr_percentiles = np.percentile(cagr_mc, percentiles)
        mean_cagr = np.mean(cagr_mc)

        max_drawdown_percentiles = np.flip(np.percentile(max_drawdowns_mc, percentiles)) # MDD는 내림차순
        mean_max_drawdown = np.mean(max_drawdowns_mc)

        max_underwater_period_percentiles = np.flip(np.percentile(max_underwater_periods_mc, percentiles)) # underwater는 내림차순
        mean_max_underwater_period = np.mean(max_underwater_periods_mc)

        reward_ratio_percentiles = np.percentile(reward_ratios_mc, percentiles)
        mean_reward_ratio = np.mean(reward_ratios_mc)

        # Clear the previous results in Treeview
        for item in results_table.get_children():
            results_table.delete(item)

        # Insert new results into the Treeview with the new format
        formatted_results = [
            ("최종 PnL", 
                f"{final_value_base:.2f}", 
                f"{mean_final_pnl:.2f}", 
                f"{final_pnl_percentiles[0]:.2f}",  # 1% 분위
                f"{final_pnl_percentiles[1]:.2f}",  # 5% 분위
                f"{final_pnl_percentiles[2]:.2f}",  # 10% 분위
                f"{final_pnl_percentiles[3]:.2f}",  # 25% 분위
                f"{final_pnl_percentiles[4]:.2f}",  # 50% 분위
                f"{final_pnl_percentiles[5]:.2f}",  # 75% 분위
                f"{final_pnl_percentiles[6]:.2f}",  # 90% 분위
                f"{final_pnl_percentiles[7]:.2f}",  # 95% 분위
                f"{final_pnl_percentiles[8]:.2f}"   # 99% 분위
            ),
            ("승률", 
                f"{winning_probability_base:.2%}", 
                f"{mean_win_probability:.2%}", 
                f"{win_probability_percentiles[0]:.2%}",  # 1% 분위
                f"{win_probability_percentiles[1]:.2%}",  # 5% 분위
                f"{win_probability_percentiles[2]:.2%}",  # 10% 분위
                f"{win_probability_percentiles[3]:.2%}",  # 25% 분위
                f"{win_probability_percentiles[4]:.2%}",  # 50% 분위
                f"{win_probability_percentiles[5]:.2%}",  # 75% 분위
                f"{win_probability_percentiles[6]:.2%}",  # 90% 분위
                f"{win_probability_percentiles[7]:.2%}",  # 95% 분위
                f"{win_probability_percentiles[8]:.2%}"   # 99% 분위
            ),
            ("평균 손익비", 
                f"{profit_loss_ratio_base:.2f}", 
                f"{mean_profit_loss_ratio:.2f}", 
                f"{profit_loss_ratio_percentiles[0]:.2f}",  # 1% 분위
                f"{profit_loss_ratio_percentiles[1]:.2f}",  # 5% 분위
                f"{profit_loss_ratio_percentiles[2]:.2f}",  # 10% 분위
                f"{profit_loss_ratio_percentiles[3]:.2f}",  # 25% 분위
                f"{profit_loss_ratio_percentiles[4]:.2f}",  # 50% 분위
                f"{profit_loss_ratio_percentiles[5]:.2f}",  # 75% 분위
                f"{profit_loss_ratio_percentiles[6]:.2f}",  # 90% 분위
                f"{profit_loss_ratio_percentiles[7]:.2f}",  # 95% 분위
                f"{profit_loss_ratio_percentiles[8]:.2f}"   # 99% 분위
            ),
            ("CAGR", 
                f"{cagr_base:.2%}", 
                f"{mean_cagr:.2%}", 
                f"{cagr_percentiles[0]:.2%}",  # 1% 분위
                f"{cagr_percentiles[1]:.2%}",  # 5% 분위
                f"{cagr_percentiles[2]:.2%}",  # 10% 분위
                f"{cagr_percentiles[3]:.2%}",  # 25% 분위
                f"{cagr_percentiles[4]:.2%}",  # 50% 분위
                f"{cagr_percentiles[5]:.2%}",  # 75% 분위
                f"{cagr_percentiles[6]:.2%}",  # 90% 분위
                f"{cagr_percentiles[7]:.2%}",  # 95% 분위
                f"{cagr_percentiles[8]:.2%}"   # 99% 분위
            ),
            ("최대 낙폭 (MDD)", 
                f"{max_drawdown_base:.2%}", 
                f"{mean_max_drawdown:.2%}", 
                f"{max_drawdown_percentiles[0]:.2%}",  # 1% 분위
                f"{max_drawdown_percentiles[1]:.2%}",  # 5% 분위
                f"{max_drawdown_percentiles[2]:.2%}",  # 10% 분위
                f"{max_drawdown_percentiles[3]:.2%}",  # 25% 분위
                f"{max_drawdown_percentiles[4]:.2%}",  # 50% 분위
                f"{max_drawdown_percentiles[5]:.2%}",  # 75% 분위
                f"{max_drawdown_percentiles[6]:.2%}",  # 90% 분위
                f"{max_drawdown_percentiles[7]:.2%}",  # 95% 분위
                f"{max_drawdown_percentiles[8]:.2%}"   # 99% 분위
            ),
            ("월 샤프 비율", 
                f"{sharpe_ratio_base:.2f}", 
                f"{mean_sharpe_ratio:.2f}", 
                f"{sharpe_ratio_percentiles[0]:.2f}",  # 1% 분위
                f"{sharpe_ratio_percentiles[1]:.2f}",  # 5% 분위
                f"{sharpe_ratio_percentiles[2]:.2f}",  # 10% 분위
                f"{sharpe_ratio_percentiles[3]:.2f}",  # 25% 분위
                f"{sharpe_ratio_percentiles[4]:.2f}",  # 50% 분위
                f"{sharpe_ratio_percentiles[5]:.2f}",  # 75% 분위
                f"{sharpe_ratio_percentiles[6]:.2f}",  # 90% 분위
                f"{sharpe_ratio_percentiles[7]:.2f}",  # 95% 분위
                f"{sharpe_ratio_percentiles[8]:.2f}"   # 99% 분위
            ),
            ("보상비율", 
                f"{reward_ratio_base:.2f}", 
                f"{mean_reward_ratio:.2f}", 
                f"{reward_ratio_percentiles[0]:.2f}",  # 1% 분위
                f"{reward_ratio_percentiles[1]:.2f}",  # 5% 분위
                f"{reward_ratio_percentiles[2]:.2f}",  # 10% 분위
                f"{reward_ratio_percentiles[3]:.2f}",  # 25% 분위
                f"{reward_ratio_percentiles[4]:.2f}",  # 50% 분위
                f"{reward_ratio_percentiles[5]:.2f}",  # 75% 분위
                f"{reward_ratio_percentiles[6]:.2f}",  # 90% 분위
                f"{reward_ratio_percentiles[7]:.2f}",  # 95% 분위
                f"{reward_ratio_percentiles[8]:.2f}"   # 99% 분위
            ),
            ("최대 underwater 기간", 
                f"{math.ceil(max_underwater_period_base)} 일", 
                f"{math.ceil(mean_max_underwater_period)} 일", 
                f"{math.ceil(max_underwater_period_percentiles[0])} 일",  # 1% 분위
                f"{math.ceil(max_underwater_period_percentiles[1])} 일",  # 5% 분위
                f"{math.ceil(max_underwater_period_percentiles[2])} 일",  # 10% 분위
                f"{math.ceil(max_underwater_period_percentiles[3])} 일",  # 25% 분위
                f"{math.ceil(max_underwater_period_percentiles[4])} 일",  # 50% 분위
                f"{math.ceil(max_underwater_period_percentiles[5])} 일",  # 75% 분위
                f"{math.ceil(max_underwater_period_percentiles[6])} 일",  # 90% 분위
                f"{math.ceil(max_underwater_period_percentiles[7])} 일",  # 95% 분위
                f"{math.ceil(max_underwater_period_percentiles[8])} 일"   # 99% 분위
            )
        ]

        # Set new headings
        for col in columns:
            results_table.heading(col, text=col)

        # Insert new formatted results into the Treeview
        for result in formatted_results:
            if result[0] == "기본 전략":
                # For basic strategy, only insert up to the '평균' value
                results_table.insert("", "end", values=(result[0], result[1], result[2], "", "", "", "", "", "", "", ""))
            else:
                # For Monte Carlo, insert all values
                results_table.insert("", "end", values=result)

    except Exception as e:
        messagebox.showerror("오류", str(e))

# Function to open file dialog
def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("엑셀 파일", "*.xlsx *.xls")])
    file_path_entry.delete(0, tk.END)
    file_path_entry.insert(0, file_path)

# Function to copy results to clipboard
def copy_results_to_clipboard():
    try:
        # Start with the column names
        result_text = "\t".join(columns) + "\n"

        # Add each row's data
        for child in results_table.get_children():
            row = results_table.item(child)["values"]
            result_text += "\t".join(str(value) for value in row) + "\n"

        # Copy to clipboard
        app.clipboard_clear()
        app.clipboard_append(result_text.strip())
        messagebox.showinfo("클립보드에 복사됨", "결과가 클립보드에 복사되었습니다.")
    except Exception as e:
        messagebox.showerror("오류", str(e))

# Define the function to handle window close event
def on_closing():
    if messagebox.askokcancel("종료", "프로그램을 종료하시겠습니까?"):
        app.destroy()  # Completely close the application

# Function to get the resource path
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS  # PyInstaller creates a temporary folder to unpack files
    except Exception:
        base_path = os.path.dirname(os.path.abspath(sys.argv[0]))  # Fallback for normal script execution
    return os.path.join(base_path, relative_path)

# Set up the main application window
app = tk.Tk()
app.title("몬테카를로 시뮬레이션 GUI")

# 아이콘 설정 (같은 디렉토리에 저장)
icon_path = resource_path("logo.ico")  # 수정된 부분
app.iconbitmap(icon_path)

# Create and place widgets
file_path_entry = tk.Entry(app, width=50)
file_path_entry.grid(row=0, column=1, padx=10, pady=3)

browse_button = tk.Button(app, text="파일 찾기", command=open_file_dialog)
browse_button.grid(row=0, column=2, padx=10, pady=3)

tk.Label(app, text="엑셀 파일 경로:").grid(row=0, column=0, padx=10, pady=3)

tk.Label(app, text="시트 이름 (선택사항):").grid(row=1, column=0, padx=10, pady=3)
sheet_name_entry = tk.Entry(app, width=50)
sheet_name_entry.grid(row=1, column=1, padx=10, pady=3)

tk.Label(app, text="초기 자본 (초기 자본금/거래승수, 단위: Pt):").grid(row=2, column=0, padx=10, pady=3)
initial_value_entry = tk.Entry(app, width=50)
initial_value_entry.insert(0, "4000")  # Set default value to 4000
initial_value_entry.grid(row=2, column=1, padx=10, pady=3)

tk.Label(app, text="시뮬레이션 횟수:").grid(row=3, column=0, padx=10, pady=3)
simulation_entry = tk.Entry(app, width=50)
simulation_entry.insert(0, "1000")  # Set default value to 1000
simulation_entry.grid(row=3, column=1, padx=10, pady=3)

run_button = tk.Button(app, text="시뮬레이션 실행", command=run_simulation)
run_button.grid(row=2, column=2, padx=10, pady=3)

# Add a button to copy results to clipboard
copy_button = tk.Button(app, text="결과를 클립보드에 복사", command=copy_results_to_clipboard)
copy_button.grid(row=3, column=2, padx=10, pady=3)

# Create Treeview widget to display results in a table format
columns = ("지표", "기본전략", "평균", "1% 분위", "5% 분위", "10% 분위", "25% 분위", "50% 분위", "75% 분위", "90% 분위", "95% 분위", "99% 분위")
results_table = ttk.Treeview(app, columns=columns, show='headings', height=8)
results_table.grid(row=4, column=0, columnspan=3, padx=10, pady=3)

# Define headings and set column width
column_widths = {
    "지표": 150,
    "기본전략": 100,
    "평균": 100,
    "1% 분위": 100,
    "5% 분위": 100,
    "10% 분위": 100,
    "25% 분위": 100,
    "50% 분위": 100,
    "75% 분위": 100,
    "90% 분위": 100,
    "95% 분위": 100,
    "99% 분위": 100,
}

# Define headings
for col in columns:
    results_table.heading(col, text=col)
    results_table.column(col, width=column_widths[col], anchor=tk.CENTER)

# Frame to hold the plot
plot_frame = tk.Frame(app)
plot_frame.grid(row=5, column=0, columnspan=3, padx=10, pady=3)

# Set the protocol for window close button to call the on_closing function
app.protocol("WM_DELETE_WINDOW", on_closing)

# Start the application
app.mainloop()
