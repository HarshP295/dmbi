import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Set a professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
# Use the 'viridis' palette by default for color consistency
sns.set_palette('viridis')




def run_generalized_eda_pipeline(df):
    """
    Performs a comprehensive Exploratory Data Analysis (EDA) on any DataFrame
    by automatically detecting column types and generating relevant visualizations.
   
    The function saves all plots as PNG files in the current directory.


    Args:
        df (pd.DataFrame): The input DataFrame to analyze.
    """
    if df.empty:
        print("Error: The DataFrame is empty. Cannot run EDA.")
        return


    print("\n" + "="*50)
    print("--- Starting Generalized Exploratory Data Analysis (EDA) ---")
    print("="*50)
   
    # 1. DATA TYPE SEGREGATION
   
    # Identify numeric and object (string/categorical) columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
   
    print(f"Total Columns: {len(df.columns)}")
    print(f"Numeric Columns Found: {len(numeric_cols)}")
    print(f"Categorical/Object Columns Found: {len(object_cols)}")
    print("-" * 50)




    # 2. NUMERIC VISUALIZATIONS (Histograms, Boxplots, Correlation)
    if numeric_cols:
        print("Generating Numeric Visualizations...")
       
        # --- A. Histograms (Distribution) ---
        df[numeric_cols].hist(bins=20, figsize=(15, 10), edgecolor='black')
        plt.suptitle('Distribution of Numeric Columns (Histograms)', y=1.02, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig("eda_numeric_histograms.png")
        plt.close()
        print(" - Saved eda_numeric_histograms.png")


        # --- B. Boxplots (Outliers and Quartiles) ---
        num_plots = len(numeric_cols)
        fig, axes = plt.subplots(ncols=num_plots, figsize=(4 * num_plots, 6))
       
        # Handle case where there is only one numeric column
        if num_plots == 1:
            axes = [axes]


        for i, col in enumerate(numeric_cols):
            sns.boxplot(y=df[col], ax=axes[i], color=sns.color_palette()[i % 6])
            axes[i].set_title(col, fontsize=12)
            axes[i].set_ylabel(None)
       
        plt.suptitle('Outlier Detection (Boxplots)', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.savefig("eda_numeric_boxplots.png")
        plt.close()
        print(" - Saved eda_numeric_boxplots.png")


        # --- C. Correlation Heatmap (If more than one numeric feature exists) ---
        if len(numeric_cols) > 1:
            print(" - Generating Correlation Heatmap...")
            plt.figure(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black')
            plt.title('Correlation Heatmap of Numeric Features', fontsize=16)
            plt.tight_layout()
            plt.savefig("eda_correlation_heatmap.png")
            plt.close()
            print(" - Saved eda_correlation_heatmap.png")
       
        # --- D. Pairplot (Scatter matrix for quick bivariate look) ---
        if len(numeric_cols) < 6 and len(numeric_cols) > 1: # Limit for performance/readability
            print(" - Generating Pairplot...")
            sns.pairplot(df[numeric_cols])
            plt.suptitle('Pairplot of Numeric Columns', y=1.02, fontsize=16)
            plt.tight_layout()
            plt.savefig("eda_numeric_pairplot.png")
            plt.close()
            print(" - Saved eda_numeric_pairplot.png")




    # 3. CATEGORICAL VISUALIZATIONS (Count Plots)
    if object_cols:
        print("\nGenerating Categorical Visualizations...")
       
        # Plot only columns with a reasonable number of unique values (e.g., max 50 categories)
        plottable_object_cols = [col for col in object_cols if df[col].nunique() <= 50]
        non_plottable_count = len(object_cols) - len(plottable_object_cols)
       
        if non_plottable_count > 0:
            print(f" - Skipped {non_plottable_count} columns with > 50 unique categories (too dense for bar plots).")


        for i, col in enumerate(plottable_object_cols):
            # Calculate dynamic figure height based on category count
            num_categories = df[col].nunique()
            fig_height = min(12, max(6, num_categories * 0.4 + 1))


            plt.figure(figsize=(10, fig_height))
           
            # Use value_counts to get the top 50, then plot
            top_categories = df[col].value_counts().head(50)
           
            # Use countplot for general categorical data
            sns.barplot(y=top_categories.index, x=top_categories.values, palette='Pastel1', orient='h')
            plt.title(f'Frequency of {col} (Top {len(top_categories)})', fontsize=14)
            plt.xlabel('Count')
            plt.ylabel(col)
            plt.tight_layout()
            plt.savefig(f"eda_categorical_counts_{col.replace('/', '_').replace(' ', '_')}.png")
            plt.close()
            print(f" - Saved eda_categorical_counts_{col}.png")




    # 4. MIXED VISUALIZATIONS (Numeric vs Categorical)
    if numeric_cols and plottable_object_cols:
        print("\nGenerating Mixed (Numeric vs. Category) Visualizations...")
       
        # Violin Plot: Distribution of a numeric column grouped by a categorical column
        # Takes the first column from each list for a sample plot
        num_col = numeric_cols[0]
        cat_col = plottable_object_cols[0]
       
        print(f" - Generating Violin Plot for {num_col} vs {cat_col}...")


        plt.figure(figsize=(12, 6))
        # Use only top 10 categories for clarity in the violin plot
        top_cats = df[cat_col].value_counts().nlargest(10).index
        df_top = df[df[cat_col].isin(top_cats)]
       
        # Ensure the column used is not an ID-like column (e.g., Member_number)
        if df_top[num_col].nunique() > 10:
            sns.violinplot(x=cat_col, y=num_col, data=df_top, order=top_cats, inner='quartile')
            plt.title(f'{num_col} Distribution by Top 10 {cat_col}', fontsize=16)
            plt.xlabel(cat_col)
            plt.ylabel(num_col)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig("eda_mixed_violinplot.png")
            plt.close()
            print(" - Saved eda_mixed_violinplot.png")
        else:
             print(" - Skipping Violin Plot: The numeric column is likely an ID or has too few unique values.")




    # 5. DATA CLEANUP AND INTERPRETATION
    print("\nAll standard EDA plots have been generated and saved as PNG files.")
    print("Check your directory for the generated images.")
    print("="*50)




# --- Example Usage (To run the pipeline on a file) ---


def main():
    """
    Main execution function that loads the data and calls the EDA pipeline.
   
    NOTE: You must adjust the 'DATASET_FILE' name below to match your file.
    """
    # Define your file name here
    DATASET_FILE = 'Groceries_dataset.csv'
   
    print(f"Loading data from '{DATASET_FILE}' for EDA...")
    try:
        df = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print(f"Error: '{DATASET_FILE}' not found. Please ensure the file exists.")
        return
       
    # --- Groceries Dataset specific fixes for column headers (to make it runnable) ---
    if DATASET_FILE == 'Groceries_dataset.csv':
        if 'itemDescription' in df.columns:
            df = df.rename(columns={'itemDescription': 'ItemDescription'})
           
    # --- Run the generalized EDA ---
    run_generalized_eda_pipeline(df)




# If this script is run directly, execute the main function
if __name__ == '__main__':
    main()



