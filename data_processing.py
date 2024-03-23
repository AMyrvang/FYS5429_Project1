import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def process_temperature_data_for_country(file_path, area, output_folder="Data/Processed_Data"):
    """
    Processes and cleans temperature change data for a specified country. It filters the data,
    converts month names to numbers, creates a 'date' column, and saves the modified dataset to a CSV file.
    
    Parameters:
    - file_path (str): Path to the CSV file containing temperature data.
    - country_name (str): The name of the country or area for which to process data.
    - output_folder (str): The folder where the processed file will be saved. Defaults to 'Data/Processed_Data'.
    """
    # Load and filter dataset for the specified country\area
    df = pd.read_csv(file_path)
    filtered_df = df[df["Area"] == area]
    filtered_df = filtered_df[~filtered_df["Months"].str.contains("�|Meteorological year")]
    temp_change_df = filtered_df[filtered_df["Element"] == "Temperature change"]

    if not temp_change_df.empty:
        # Prepare the dataset
        cols_to_keep = ["Area", "Months"] + [col for col in temp_change_df.columns if col.startswith("Y")]
        clean_df = temp_change_df[cols_to_keep]

        melted_df = clean_df.melt(id_vars=["Area", "Months"], var_name="Year", value_name="Temperature Change")
        melted_df["Year"] = melted_df["Year"].apply(lambda x: x[1:])

        # Convert month names to numbers and create 'date' column
        month_to_num = {
            "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
            "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
        }
        melted_df["Months"] = melted_df["Months"].map(month_to_num)
        melted_df["date"] = pd.to_datetime(melted_df[["Year", "Months"]].assign(DAY=1))
        melted_df.sort_values(by=["date"], inplace=True)

        # Save processed data
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_file_name = os.path.join(output_folder, f"{area.replace(' ', '_')}_temperature_change.csv")
        melted_df.to_csv(output_file_name, index=False)
        print(f"Data for {area} saved to {output_file_name}")
    else:
        print(f"No temperature change data found for {area}.")


file_path = "Data/Temperature_Data/Environment_Temperature_change_E_All_Data_NOFLAG.csv"
country = "World"
process_temperature_data_for_country(file_path, country)

# Set theme
sns.set_theme(style="notebook", context="whitegrid", palette="colorblind")
cm = 1/2.54
params = {
    "legend.fontsize": "9",
    "font.size": "9",
    "figure.figsize": (8.647 * cm, 12.0 * cm), # figsize for two-column latex doc
    "axes.labelsize": "9",
    "axes.titlesize": "9",
    "xtick.labelsize": "9",
    "ytick.labelsize": "9",
    "legend.fontsize": "7",
    "lines.markersize": "3.0",
    "lines.linewidth": "1.0",
}
plt.rcParams.update(params)

# Load data
df = pd.read_csv("Data/Processed_Data/World_temperature_change.csv")
df["date"] = pd.to_datetime(df["date"])

# Plot data
plt.plot(df["date"], df["Temperature Change"], color="black")
plt.title("Temperature Change Over Time (1961-2019)")
plt.xlabel("Time [Years]")
plt.ylabel("Temperature change [\u00B0C]")

plt.savefig("Figs/Temperature_change.png")
