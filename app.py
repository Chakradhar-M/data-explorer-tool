import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set up the page configuration
st.set_page_config(page_title="Data Explorer Tool", layout="wide")

# Displaying the title of the tool
st.title("📊 Data Explorer Tool")

# Custom CSS to style the file uploader and sidebar widgets
st.markdown("""
    <style>
    .css-1aumxhk, .css-1lcbmhc {
        max-width: 300px;
        padding-right: 10px;
    }

    .stFileUploader > div {
        padding: 0.25rem;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar section for data input (file upload)
st.sidebar.header("📂 Data Input")
file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

sheet_name = None
df = None

# Check the file extension and read the appropriate file
if file:
    filename = file.name
    ext = os.path.splitext(filename)[-1].lower()  # Extract the file extension
    
    # Handle invalid file types
    if ext not in [".csv", ".xlsx"]:
        st.sidebar.error("❌ Unsupported file type. Please upload only .csv or .xlsx files.")
    else:
        # If the file is an Excel file, allow the user to select a sheet
        if ext == ".xlsx":
            xls = pd.ExcelFile(file)
            sheet_name = st.sidebar.radio("Select a sheet", options=xls.sheet_names, index=None)
            if sheet_name:
                st.sidebar.markdown(f"✅ Selected Sheet: **{sheet_name}**")
                df = pd.read_excel(xls, sheet_name=sheet_name)
        # If the file is a CSV, load it directly
        elif ext == ".csv":
            df = pd.read_csv(file)

        # Convert object columns to datetime where possible
        if df is not None:
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])  # Attempt conversion to datetime
                    except (ValueError, TypeError):
                        pass  # Ignore if conversion fails

# Separator line to indicate section change in the sidebar
if df is not None:
    st.sidebar.markdown("---")

# Sidebar - Action selection (checklist for different analyses)
if df is not None:
    st.sidebar.header("🛠️ Analysis Options")
    st.sidebar.markdown("Tick the boxes below to perform the corresponding analysis.")

    # Create checkboxes for different data analysis options
    show_sample_data = st.sidebar.checkbox("🔍 Display Sample Data")
    show_dimensions = st.sidebar.checkbox("📐 Dataset Dimensions")
    show_profile = st.sidebar.checkbox("📋 Column Profile")
    show_duplicates = st.sidebar.checkbox("🔁 Check Duplicates")
    show_missing = st.sidebar.checkbox("🕳️ Check Missing Values")
    show_date_range = st.sidebar.checkbox("📅 Date Column Range")
    show_dtype_dist = st.sidebar.checkbox("📊 Datatype Distribution Plot")
    show_statistics = st.sidebar.checkbox("📈 Descriptive Statistics")
    show_category_counts = st.sidebar.checkbox("📊 Category Counts")
    show_unique_vals = st.sidebar.checkbox("🔎 Unique Values by Column")

    # Display sample data
    if show_sample_data:
        st.subheader("🔍 Sample Data (First 5 Rows)")
        st.dataframe(df.head())  # Show the first 5 rows of the dataset
        st.markdown("---")

    # Display dataset dimensions (rows and columns count)
    if show_dimensions:
        st.subheader("📐 Dataset Dimensions")
        st.markdown("Shows the total number of rows and columns in the dataset.")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")  # Shape of the dataframe
        st.markdown("---")

    # Display column profile (data types, missing values, etc.)
    if show_profile:
        st.subheader("📋 Column Profile")
        st.markdown("Displays data type, total values, missing values, and unique values for each column.")
        
        # Create a table with column info
        profile_data = [
            [col, df[col].dtype, df[col].count(), df[col].isnull().sum(), df[col].nunique()]
            for col in df.columns
        ]
        profile_df = pd.DataFrame(profile_data, columns=["Column", "Data Type", "Total Values", "Missing Values", "Unique Values"])
        st.dataframe(profile_df)  # Show the column profile table
        st.markdown("---")

    # Check and display duplicate rows in the dataset
    if show_duplicates:
        st.subheader("🔁 Duplicate Rows")
        st.markdown("Checks for duplicate rows in the dataset and displays them if present.")
        duplicates = df[df.duplicated()]  # Find duplicate rows
        if not duplicates.empty:
            st.write(f"Found {duplicates.shape[0]} duplicate rows:")
            st.dataframe(duplicates)  # Show duplicate rows
        else:
            st.success("No duplicate rows found.")  # Display success message if no duplicates
        st.markdown("---")

    # Display rows with missing values
    if show_missing:
        st.subheader("🕳️ Missing Values")
        st.markdown("Identifies rows with missing values and displays them if found.")
        missing = df[df.isnull().any(axis=1)]  # Find rows with any missing value
        if not missing.empty:
            st.write(f"Found {missing.shape[0]} rows with missing values:")
            st.dataframe(missing)  # Show rows with missing values
        else:
            st.success("No missing values found.")  # Success message if no missing values
        st.markdown("---")

    # Display date range for date columns
    if show_date_range:
        st.subheader("📅 Date Column Range")
        st.markdown("Detects date columns and displays the oldest and latest dates for each.")
        date_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]'])  # Select date columns
        if not date_cols.empty:
            # Show the range for each date column
            date_summary = pd.DataFrame({
                "Column": date_cols.columns,
                "Oldest Date": [date_cols[col].min() for col in date_cols.columns],
                "Latest Date": [date_cols[col].max() for col in date_cols.columns]
            })
            st.dataframe(date_summary)  # Display the date summary
        else:
            st.warning("No date columns detected or parsed correctly.")  # Warning if no date columns are found
        st.markdown("---")

    # Display distribution of data types (bar plot)
    if show_dtype_dist:
        st.subheader("📊 Datatype Distribution")
        st.markdown("Visualizes the count of each data type in the dataset using a bar chart.")
        
        # Count occurrences of each data type
        dtype_counts = df.dtypes.value_counts()
        
        # Create a bar chart for data types
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = sns.barplot(x=dtype_counts.index.astype(str), y=dtype_counts.values, ax=ax)
        ax.bar_label(bars.containers[0], fmt='%d')  # Add labels on top of bars
        ax.set_xlabel("Data Types")
        ax.set_ylabel("Number of Columns")
        ax.set_title("Distribution of Column Data Types")
        st.pyplot(fig)  # Show the plot
        st.markdown("---")

    # Display descriptive statistics (summary stats for numerical columns)
    if show_statistics:
        st.subheader("📈 Descriptive Statistics (Numerical Columns)")
        st.markdown("Displays statistical summaries such as mean, std, min, and max for numerical columns.")
        st.dataframe(df.describe())  # Show descriptive statistics
        st.markdown("---")

    # Display category counts for categorical columns
    if show_category_counts:
        st.subheader("📊 Category Counts (Categorical Columns)")
        st.markdown("Shows counts of each value in categorical columns.")
        cat_cols = df.select_dtypes(include='object')  # Select categorical columns
        if cat_cols.shape[1] == 0:
            st.warning("No categorical columns found in the dataset.")  # Warning if no categorical columns
        else:
            for col in cat_cols.columns:
                st.markdown(f"**{col}**")
                st.dataframe(df[col].value_counts().reset_index().rename(columns={"index": col, col: "Count"}))  # Show category counts
        st.markdown("---")

    # Display unique values by column
    if show_unique_vals:
        st.subheader("🔎 Unique Values by Column")
        st.markdown("Displays **all** unique values for each column in a scrollable table. Useful for inspecting inconsistencies in data.")
        for col in df.columns:
            unique_vals = sorted(df[col].dropna().unique())  # Get unique values in each column
            if len(unique_vals) > 0:
                st.markdown(f"**{col}** → {len(unique_vals)} unique values")
                unique_df = pd.DataFrame(unique_vals, columns=["Unique Values"])
                row_height = 35
                min_height = 100
                max_height = 300
                height = min(max(min_height, row_height * len(unique_df)), max_height)  # Set table height dynamically
                st.dataframe(unique_df, height=height)  # Show unique values table
        st.markdown("---")
else:
    # Display a message when no file is uploaded yet
    st.info("👈 Please upload a dataset to get started.")
