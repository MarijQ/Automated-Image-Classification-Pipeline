import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

# ------------------- Data Loading Function -------------------

def load_data(file_path):
    """
    Load the data from the specified Excel file.

    :param file_path: Path to the Excel file.
    :return: Pandas DataFrame containing the data.
    """
    try:
        df = pd.read_excel(file_path)
        print(f"✅ Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        print(f"❌ Error loading data from {file_path}: {e}")
        return None

# ------------------- Plotting Functions -------------------

def plot_successful_downloads(df, title):
    """
    Plot percentage success rate against total images processed.

    :param df: DataFrame containing the data.
    :param title: Title of the plot.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Number of Images Processed')
    ax1.set_ylabel('Percentage Success Rate (%)')

    colors = {'bing': 'tab:blue', 'duckduckgo': 'tab:orange'}
    search_engines = df['Search Engine'].unique()

    for engine in search_engines:
        engine_data = df[df['Search Engine'] == engine]
        color = colors.get(engine, 'tab:gray')
        success_rate = (engine_data['Successful Downloads'] / engine_data['Total Images']) * 100
        ax1.plot(engine_data['Total Images'], success_rate,
                 label=f'{engine.capitalize()} Success Rate',
                 color=color, marker='o')

    ax1.set_ylim(0, 100)  # Set y-axis limit to 0-100%
    ax1.legend(loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.savefig("successful_downloads.png")


def plot_script_times(df, title):
    """
    Plot collection time and collection + download time for each search engine.

    :param df: DataFrame containing the data.
    :param title: Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlabel('Number of Images Processed')
    ax.set_ylabel('Time (seconds)')

    colors = {'bing': 'tab:blue', 'duckduckgo': 'tab:orange'}
    search_engines = df['Search Engine'].unique()

    for engine in search_engines:
        engine_data = df[df['Search Engine'] == engine]
        color = colors.get(engine, 'tab:gray')

        # Plot collection time
        ax.plot(engine_data['Total Images'], engine_data['Collection Time (s)'],
                label=f'{engine.capitalize()} Collection Time',
                color=color, linestyle='--', marker='x')

        # Plot collection + download time
        total_time = engine_data['Collection Time (s)'] + engine_data['Download Time (s)']
        ax.plot(engine_data['Total Images'], total_time,
                label=f'{engine.capitalize()} Collection + Download Time',
                color=color, linestyle='-', marker='o')

    ax.legend(loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.savefig("script_times.png")

def plot_new_vs_duplicate_urls(df, title):
    """
    Plot new vs duplicate URLs by search term number for both search engines in separate subplots.
    Also includes a total line (New Unique + Duplicates).

    :param df: DataFrame containing the data.
    :param title: Title of the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    search_engines = df['Search Engine'].unique()
    markers = {'Average New Unique Images': 'o', 'Average Duplicates': 'x', 'Total': 's'}
    colors = {'New Unique': 'green', 'Duplicates': 'red', 'Total': 'blue'}

    for i, engine in enumerate(search_engines):
        engine_data = df[df['Search Engine'] == engine]
        engine_data_sorted = engine_data.sort_values('Search Term Number')

        # Plot New Unique
        axes[i].plot(engine_data_sorted['Search Term Number'], engine_data_sorted['Average New Unique Images'],
                     label=f'{engine.capitalize()} - New Unique', marker=markers['Average New Unique Images'],
                     color=colors['New Unique'])

        # Plot Duplicates
        axes[i].plot(engine_data_sorted['Search Term Number'], engine_data_sorted['Average Duplicates'],
                     label=f'{engine.capitalize()} - Duplicates', marker=markers['Average Duplicates'],
                     color=colors['Duplicates'])

        # Plot Total (New Unique + Duplicates)
        total_urls = engine_data_sorted['Average New Unique Images'] + engine_data_sorted['Average Duplicates']
        axes[i].plot(engine_data_sorted['Search Term Number'], total_urls,
                     label=f'{engine.capitalize()} - Total', marker=markers['Total'],
                     color=colors['Total'], linestyle='--')

        # Set labels and title for each subplot
        axes[i].set_xlabel('Search Term Number')
        axes[i].set_title(f'{engine.capitalize()}')

    # Set common Y label
    axes[0].set_ylabel('Number of URLs')

    # Add overall title
    fig.suptitle(title)

    # Add legend
    axes[0].legend()
    axes[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    plt.savefig("new_vs_duplicate_urls_with_total.png")


def plot_total_unique_images_over_time(df, title):
    """
    Plot total unique images with a solid black line and total images considered with a dotted black line.
    Additionally, plot unique images per search engine as color-coded lines.

    :param df: DataFrame containing the benchmark2 statistics.
    :param title: Title of the plot.
    """
    # Calculate cumulative unique and considered images
    df_sorted = df.sort_values('Average Script Time (s)')
    df_sorted['Cumulative Unique'] = df_sorted['Average New Unique Images'].cumsum()
    df_sorted['Total Considered'] = df_sorted['Average New Unique Images'] + df_sorted['Average Duplicates']
    df_sorted['Cumulative Considered'] = df_sorted['Total Considered'].cumsum()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot total unique images as a solid black line
    ax.plot(df_sorted['Average Script Time (s)'], df_sorted['Cumulative Unique'],
            label='Total Unique Images', color='black', linestyle='-', linewidth=2)

    # Plot total images considered as a dotted black line
    ax.plot(df_sorted['Average Script Time (s)'], df_sorted['Cumulative Considered'],
            label='Total Images Considered', color='black', linestyle='--', linewidth=2)

    # Plot cumulative unique images per search engine as color-coded lines
    colors = {'bing': 'tab:blue', 'duckduckgo': 'tab:orange'}
    linestyles = {'bing': ':', 'duckduckgo': '-.'}

    for engine in df['Search Engine'].unique():
        engine_data = df_sorted[df_sorted['Search Engine'] == engine]
        ax.plot(engine_data['Average Script Time (s)'], engine_data['Cumulative Unique'],
                color=colors.get(engine, 'tab:gray'), label=f'{engine.capitalize()} Unique Images',
                linestyle=linestyles.get(engine, '-'), marker='o', markersize=5)

    # Add gridlines for better readability
    ax.grid(True, linestyle='--', alpha=0.6)

    # Set labels and title
    ax.set_ylabel('Cumulative Unique Images')
    ax.set_xlabel('Script Time (s)')
    ax.set_title(title)

    # Set the x and y axis limits to start at 0
    ax.set_xlim(left=0)  # Start x-axis at 0
    ax.set_ylim(bottom=0)  # Start y-axis at 0

    # Improve legend by grouping search engines under "Unique Images"
    ax.legend(title="Legend", loc='upper left')

    plt.tight_layout()
    plt.savefig("total_unique_images_over_time_corrected.png")
    # plt.show()


def plot_average_script_time(df, title):
    """
    Plot Average Script Time against Concurrent Downloads.

    :param df: DataFrame containing the data.
    :param title: Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['Concurrent Downloads'], df['Average Script Time (s)'], marker='o', linestyle='-')
    plt.xlabel('Concurrent Downloads')
    plt.ylabel('Average Script Time (s)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("average_script_time.png")


# ------------------- Main Function -------------------

def main():
    """
    Main function to load data and generate the charts.
    """
    # File paths to the Excel files
    stats_file_4_4 = 'stats_4-4.xlsx'
    stats_file_4_5 = 'stats_4-5.xlsx'
    stats_file_4_6 = 'stats_4-6.xlsx'

    # Load the data
    df_download = load_data(stats_file_4_4)
    df_benchmark2 = load_data(stats_file_4_5)
    df_script_time = load_data(stats_file_4_6)

    if df_download is not None:
        # Generate the successful downloads plot
        plot_successful_downloads(df_download, title="Successful Downloads vs Images Considered")

        # Generate the script times plot
        plot_script_times(df_download, title="Script Times for Image Collection and Download")

    if df_benchmark2 is not None:
        # Generate the new vs duplicate URLs plot
        plot_new_vs_duplicate_urls(df_benchmark2, title="New vs Duplicate URLs by Search Term Number")

        # Generate the total unique images over time plot
        plot_total_unique_images_over_time(df_benchmark2, title="Total Unique Images Over Time")

    if df_script_time is not None:
        # Generate the average script time plot
        plot_average_script_time(df_script_time, title="Average Script Time vs Concurrent Downloads")


# ------------------- Entry Point -------------------

if __name__ == "__main__":
    main()
