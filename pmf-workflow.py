import os
import pandas as pd
import numpy as np
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import time
import math
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import LeaveOneOut
from scipy.stats import zscore
from itertools import cycle


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# INPUTS
input_file = 'ENIGMA_BigStudy_LIPIDS_POS_RAW_20230327_AgeGroups_20231110.csv' #'Coffee_Beans_HILIC_POS_RAW_20240105.csv'
input_batch_info_file = 'ENIGMA_BigStudy_LIPIDS_POS_RAW_20230327_AgeGroups_20231110_InputFiles.csv'#'Coffee_Beans_HILIC_POS_RAW_20240105_InputFiles.csv'

# OUTPUTS
output_file_prefix = 'ENIGMA_BigStudy_LIPIDS' #'Coffee_Beans_HILIC_POS'
report_file_name = 'ENIGMA_BigStudy_LIPIDS-report.pdf' #'Coffee_Beans_HILIC_POS-report'

# OUTPUT FOLDER
main_folder = 'ENIGMA_analysis' #'Coffee_analysis-POS'

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Creating the folders
import os

# Create the directory if it doesn't exist
if not os.path.exists(main_folder):
    os.makedirs(main_folder)

# Create the directory if it doesn't exist
if not os.path.exists(main_folder + '/figures'):
    os.makedirs(main_folder + '/figures')

# To work properly always save atleast png version of the figures
sufixes = ['.png', '.pdf'] # sufices for the image files to be saved in

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import pdf_reporter as pdf_rptr

# Reload the module to get the latest changes
# This will be deleted later, just for testing purposes
import importlib
importlib.reload(pdf_rptr)

#---------------------------------------------
# REPORTING
report_path = main_folder + '/' + report_file_name + '.pdf'
title_text = main_folder

report = pdf_rptr.Report(name = report_path, title = title_text)
report.initialize_report('processing')

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_data(data_input_file_name, separator = ';', encoding = 'ISO-8859-1'):
    """
    Load data matrix from a csv file.

    Parameters
    ----------

    data_input_file_name : str
        Name of the data file.
    separator : str
        Separator used in the data file. Default is ';'.
    encoding : str
        Encoding used in the data file. Default is 'ISO-8859-1'. (UTF-8 is also common.)
    """
    data_file = input_file  # Update with your file path
    data_df = pd.read_csv(data_input_file_name, sep = separator,  encoding=encoding )
    #---------------------------------------------
    # REPORTING
    text = 'Data were loaded from: ' + data_file + ' (data from Compound Discoverer).'
    report.add_together([('text', text), 
                        ('table', data_df),
                        'line'])
    return data_df

data_df = load_data(input_file, separator = ',', encoding = 'UTF-8')

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def add_cpdID(df, mz_col = 'm/z', rt_col = 'RT [min]'):
    """
    Add a column 'cpdID' to the data_df calculated from mz and RT. Matching ID - distinguished by adding "_1", "_2", etc.

    Parameters
    ----------
    df : DataFrame
        DataFrame with the data.
    mz_col : str
        Name of the column with m/z values. Default is 'm/z'.
    rt_col : str
        Name of the column with RT values. Default is 'RT [min]'.
    """
    #add cpdID column to the data_df calculated from mz and RT
    df['cpdID'] = 'M' + df[mz_col].round(0).astype(int).astype(str) + 'T' + (df[rt_col]*60).round(0).astype(int).astype(str)
    # Create a new column 'Duplicate_Count' to keep track of the duplicates
    df['Duplicate_Count'] = df.groupby('cpdID').cumcount()
    # Create a mask to identify duplicated compounds
    is_duplicate = df['Duplicate_Count'] > 0
    # Update the 'Compound_ID' column for duplicated compounds
    df.loc[is_duplicate, 'cpdID'] = df['cpdID'] + '_' + (df['Duplicate_Count']).astype(str)
    # Drop the 'Duplicate_Count' 
    df.drop('Duplicate_Count', axis=1, inplace=True)
    #check if there are rly no duplicates left
    remaining_duplicates = df[df['cpdID'].duplicated(keep=False)]
    print(remaining_duplicates.empty)

    #---------------------------------------------
    # REPORTING
    text = 'cpdID column was added to the data_df calculated from mz and RT. Matching ID - distinguished by adding "_1", "_2", etc.'
    report.add_together([('text', text),
                            'line'])
    return df
add_cpdID(data_df, 'm/z', 'RT [min]')


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Creating different matrices

def extract_variable_metadata(data_df, column_index_ranges = [(10, 15), (18, 23)]):
    """
    Extract variable metadata from the data_df. 

    Parameters
    ----------
    data_df : DataFrame
        DataFrame with the data.
    column_index_ranges : list
        List of tuples with the column index ranges to keep. Default is [(10, 15), (18, 23)].
    """
    variable_metadata = data_df[['cpdID', 'Name', 'Formula']]
    for column_index_range in column_index_ranges:
        variable_metadata = variable_metadata.join(data_df.iloc[:, column_index_range[0]:column_index_range[1]])

    #---------------------------------------------
    # REPORTING
    text = 'variable-metadata matrix was created.'
    report.add_together([('text', text),
                            'line'])
    return variable_metadata

variable_metadata = extract_variable_metadata(data_df)

def extract_data(data_df, prefix = 'Area:'):
    """
    Extract data matrix from the data_df.

    Parameters
    ----------
    data_df : DataFrame
        DataFrame with the data.
    prefix : str
        Prefix used in the column names to keep. Default is 'Area:'.
    """

    data = data_df[['cpdID']]
    # Select columns with names starting with "Area:"
    area_columns = data_df.filter(regex=r'^' + prefix, axis=1) #for those starting with Area:, for those including Area anywhere, you can use like='Area:'
    data = data.join(data_df[area_columns.columns])

    # Fill NaN values with zero (NaN values are problem when plotted, and we will treat 0s as missing values later)
    data.fillna(0, inplace=True)  # Replace NaN with 0

    #---------------------------------------------
    # REPORTING
    text = 'data matrix was created.'
    report.add_together([('text', text),
                            'line'])
    return data

data = extract_data(data_df)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_batch_info(input_batch_info_file):
    """
    Load batch_info matrix from a csv file.

    Parameters
    ----------
    input_batch_info_file : str
        Name of the batch info file.
    """
    #but better will be get it straight from the xml file (will do later)
    batch_info = pd.read_csv(input_batch_info_file, sep = ';')
    batch_info

    #---------------------------------------------
    # REPORTING
    text = 'batch_info matrix was loaded from: ' + input_batch_info_file
    report.add_together([('text', text),
                            'line'])
    return batch_info

batch_info = load_batch_info(input_batch_info_file)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# REORDER DATA BASED ON THE CREATION DATE, ADD A COLUMN 'Batch' TO THE BATCH_INFO BASED ON DISTINGUISHER
#---------------------------------------------
def batch_by_name(data, batch_info, distinguisher = 'Batch', format='%d.%m.%Y %H:%M'):
    """
    Reorder data based on the creation date and add a column 'Batch' to the batch_info based on the distinguisher.
    
    Parameters
    ----------
    data : DataFrame
        DataFrame with the data.
    batch_info : DataFrame
        DataFrame with the batch information.
    distinguisher : str
        Distinguisher used in the file names. Default is 'Batch'. Batches are distinguished in the File Name *_distinguisher_XXX.
    format : str
        Format of the date. Default is '%d.%m.%Y %H:%M'.
    """

    #Batches are distinguished in the File Name *_distinguisher_XXX
    batches_column = []

    names = data.columns[1:].to_list()
    batch_names = batch_info['File Name'].tolist()

    # Convert the 'Creation Date' column to datetime format
    batch_info['Creation Date'] = pd.to_datetime(batch_info['Creation Date'], format = format) #, dayfirst=True

    # Sort the DataFrame based on the 'Creation Date' column
    batch_info = batch_info.sort_values('Creation Date')

    for name in batch_names:
        split_name = re.split(r'[_|\\]', name) #split by _ or \
        for i, part in enumerate(split_name):
            if part == distinguisher:
                batches_column.append(split_name[i+1])
                break

    if len(batches_column) == 0:
        raise ValueError("No matches found in 'names' for the distinguisher: " + distinguisher + " in the file names.")
    else:
        batch_info['Batch'] = batches_column

    # Omit not found samples from batch_info
    not_found = []
    not_found_indexes = []
    new_data_order = []
    for i, id in enumerate(batch_info['Study File ID'].tolist()):
        found = False
        for name in names:
            # Check if the name contains the ID within brackets
            if re.search(f'\({id}\)', name):
                found = True
                new_data_order.append(name)
                names.pop(names.index(name))
                break
        if not found:
            not_found_indexes.append(i)
            not_found.append([id, batch_info['Sample Type'].tolist()[i]])

    print("Not found: " + str(len(not_found)) + " ;being: " + str(not_found))
    print("Names not identified: " + str(len(names)) + " ;being: " + str(names))

    # Reorder data
    data = data[['cpdID']+new_data_order]


    batch_info = batch_info.drop(not_found_indexes)
    batch_info = batch_info.reset_index(drop=True)

    #---------------------------------------------
    #REPORTING
    text0 = 'Batch information from file:'+ input_batch_info_file +' was used to (re)order samples.'
    text1 = 'Not found: ' + str(len(not_found)) + ' ;being: ' + str(not_found)
    text2 = 'Names not identified: ' + str(len(names)) + ' ;being: ' + str(names)
    report.add_together([('text', text0),
                        ('text', text1, 'italic'),
                        ('text', text2, 'italic'),
                        ('table', batch_info),
                        'line'])

    return data, batch_info

#data, batch_info = batch_by_date(data, batch_info, format = '%d.%m.%Y %H:%M')
data, batch_info = batch_by_name(data, batch_info, distinguisher = 'BigBatch', format = '%d.%m.%Y %H:%M')

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def extract_metadata(batch_info, group_columns_to_keep, prefix = 'Area:'):
    """
    Extract metadata from batch_info by choosing columns to keep. Keep your grouping columns such as AgeGroup, Sex, Diagnosis, etc. 

    Parameters
    ----------
    batch_info : DataFrame
        DataFrame with the batch information.
    group_columns_to_keep : list
        List of group columns to keep.
    prefix : str
        Prefix used in the column names to keep. Default is 'Area:'.
    """

    # Define columns to keep in the metadata matrix
    always_keep_colunms = ['Study File ID','File Name', 'Creation Date','Sample Type', 'Polarity', 'Batch']
    columns_to_keep = always_keep_colunms + group_columns_to_keep
    # Extract metadata from batch_info
    regex_pattern = re.compile(r'^'+ prefix)
    area_columns = data.filter(regex = regex_pattern, axis=1)
    metadata = batch_info[columns_to_keep].copy()
    metadata['Sample File'] = area_columns.columns
    metadata

    #---------------------------------------------
    # REPORTING
    text = 'metadata matrix was created from batch_info by choosing columns: ' + str(columns_to_keep) + '.'
    report.add_together([('text', text),
                            'line'])
    
    return metadata

metadata = extract_metadata(batch_info, group_columns_to_keep = ['Sex', 'AgeGroup'])

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Filtering out compounds with too many missing values

def filter_missing_values(data, QC_samples, qc_threshold = 0.8, sample_threshold = 0.5):
    """
    Filter out features with high number of missing values (nan or 0): A) within QC samples  B) all samples 

    Parameters
    ----------
    data : DataFrame
        DataFrame with the data.
    QC_samples : list
        List of QC samples. (QC_samples = metadata[metadata['Sample Type'] == 'Quality Control']['Sample File'].tolist())
    qc_threshold : float
        Threshold for the QC samples. Default is 0.8.
    sample_threshold : float
        Threshold for all samples. Default is 0.5.
    """ 
    is_qc_sample = [True if col in QC_samples else False for col in data.columns[1:]]
    #A) WITHIN QC SAMPLES ---------------------------------------------
    #if value is missing (nan or 0) in more than threshold% of QC samples, then it is removed
    QC_number_threshold = int(sum(is_qc_sample)*qc_threshold)
    print(QC_number_threshold)

    # Identify rows with missing (either missing or 0) values
    QC_missing_values = (data[data.columns[1:][is_qc_sample]].isnull() | (data[data.columns[1:][is_qc_sample]] == 0)).sum(axis=1)
    #print(missing_values[missing_values > 0])

    #Filter out rows with too much (over the threshold) missing values (either missing or 0)
    data = data.loc[QC_missing_values < QC_number_threshold, :]
    data

    #report how many features were removed
    print("Number of features removed for threshold (" + str(qc_threshold) + "%): within QC samples: "+ str(len(QC_missing_values[QC_missing_values > QC_number_threshold])) + " ;being: " + str(QC_missing_values[QC_missing_values > QC_number_threshold].index.tolist()))

    #B) ACROSS ALL SAMPLES ---------------------------------------------
    #if value is missing (nan or 0) in more than threshold% of samples, then it is removed
    number_threshold = int(len(data.columns[1:])*sample_threshold)

    # Identify rows with missing (either missing or 0) values
    missing_values = (data.isnull() | (data == 0)).sum(axis=1)
    #print(missing_values[missing_values > 0])

    print(missing_values)

    #Filter out rows with too much (over the threshold) missing values (either missing or 0)
    data = data.loc[missing_values < number_threshold, :]
    data = data.reset_index(drop=True)

    #report how many features were removed
    print("Number of features removed for threshold (" + str(sample_threshold) + "%): within all samples: "+ str(len(missing_values[missing_values > number_threshold])) + " ;being: " + str(missing_values[missing_values > number_threshold].index.tolist()))

    #---------------------------------------------
    #REPORTING
    text0 = 'Features with missing values over the threshold (' + str(qc_threshold*100) + '%) within QC samples were removed.'
    text1 = 'Number of features removed: ' + str(len(QC_missing_values[QC_missing_values > QC_number_threshold])) + ' ;being: '+ str(QC_missing_values[QC_missing_values > QC_number_threshold].index.tolist()[:10])[:-1] + ', ...'
    text2 = 'Features with missing values over the threshold (' + str(sample_threshold*100) + '%) within all samples were removed.'
    text3 = 'Number of features removed: ' + str(len(missing_values[missing_values > number_threshold])) + ' ;being: '+ str(missing_values[missing_values > number_threshold].index.tolist()[:10])[:-1] + ', ...'
    report.add_together([('text', text0),
                        ('text', text1, 'italic'),
                        'line',
                        ('text', text2),
                        ('text', text3, 'italic')])
    report.add_pagebreak()

    return data


#QC samples
QC_samples = metadata[metadata['Sample Type'] == 'Quality Control']['Sample File'].tolist()

data = filter_missing_values(data, QC_samples, qc_threshold = 0.8, sample_threshold = 0.5)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def visualize_boxplot(data, QC_samples, sufixes = ['.png', '.pdf']):
    """
    Create a boxplot of all samples.  Can depict if there is a problem with the data (retention time shift too big -> re-run alignment; batch effect, etc.)

    Parameters
    ----------
    data : DataFrame
        DataFrame with the data.
    QC_samples : list
        List of QC samples. (QC_samples = metadata[metadata['Sample Type'] == 'Quality Control']['Sample File'].tolist())
    sufixes : list
        List of sufices for the image files to be saved in. Default is ['.png', '.pdf'].

   
    """
    is_qc_sample = [True if col in QC_samples else False for col in data.columns[1:]]
    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(18, 12))

    # Create a box plot for the specific feature
    box = plt.boxplot(data.iloc[:,1:], showfliers=False, showmeans=True, meanline=True, medianprops={'color':'black'}, meanprops={'color':'blue'}, patch_artist=True, whiskerprops=dict(color='grey'), capprops=dict(color='yellow'))
    plt.title('Boxplot of all samples')

    #Color boxplots of QC samples in red and the rest in blue
    colors = ['red' if qc else 'lightblue' for qc in is_qc_sample]
    # Set the colors for the individual boxplots
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Customize the plot
    plt.xlabel('Sample Order')
    plt.ylabel('Peak Area')
    xx = np.arange(0, len(data.columns[1:])+1, 100)
    plt.xticks(xx, xx)

    # Save the plot
    plt_name = main_folder + '/figures/QC_samples_boxplot_first_view'
    for sufix in sufixes:
        plt.savefig(plt_name + sufix, bbox_inches='tight', dpi=300)

    # Show the plot
    plt.show()

    #---------------------------------------------
    #REPORTING
    text0 = 'The initial visualization of the data was created.'
    text1 = 'The boxplot of all samples was created and saved to: ' + plt_name
    report.add_together([('text', text0),
                        ('text', text1),
                        ('image', plt_name + '.png'),
                        'line'])
    
    return fig, ax


#filter our quality-control samples (QC)
QC_samples = metadata[metadata['Sample Type'] == 'Quality Control']['Sample File'].tolist()

visualize_boxplot(data, QC_samples, sufixes = sufixes)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def visualize_by_batch(df, QC_samples, main_folder, show = 'default', sufixes = ['.png', '.pdf'], batch = None, cmap = 'viridis'):
    """
    Visualize samples by batch. Highlight QC samples.

    Parameters
    ----------
    df : DataFrame
        DataFrame with the data.
    QC_samples : list
        List of QC samples. (QC_samples = metadata[metadata['Sample Type'] == 'Quality Control']['Sample File'].tolist())
    main_folder : str
        Name of the main folder.
    show : list or str
        List of features to show. Default is 'default'. Other options are 'all', 'none' or index of the feature.
    sufixes : list
        List of sufices for the image files to be saved in. Default is ['.png', '.pdf'].
    batch : list
        List of batches. Default is None. If None = all samples are in one batch else in list (e.g. ['001', '001', ...,  '002', '002', ...]).
    cmap : str
        Name of the colormap. Default is 'viridis'. (Other options are 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted', 'turbo'; ADD '_r' to get reversed colormap)
    """

    # If show isnt list or numpy array
    if type(show) != list and type(show) != np.ndarray: 
        if show == 'default':
            show = np.linspace(0, len(df.columns)-1, 5, dtype=int)
            print(show)
        elif show == 'all':
            show = np.arange(len(df.columns))
        elif show == 'none':
            show = []
        else:
            show = [show]

    if batch is None:
        batch = ['all_one_batch' for i in range(len(df.columns) - 1)]  # Dummy batch information (for cases where its all the same batch)

    unique_batches = []
    for b in batch:
        if b not in unique_batches:
            unique_batches.append(b)
    
    cmap = mpl.cm.get_cmap(cmap)

    # Create a dictionary that maps each unique batch to a unique index
    batch_to_index = {batch: index for index, batch in enumerate(unique_batches)}

    # Normalize the indices between 0 and 1
    normalized_indices = {batch: index / len(unique_batches) for batch, index in batch_to_index.items()}

    # Create a dictionary that maps each batch to a color
    batch_colors = [mpl.colors.rgb2hex(cmap(normalized_indices[batch])) for batch in unique_batches]

    batch_to_color = {batch_id: batch_colors[i % len(batch_colors)] for i, batch_id in enumerate(unique_batches)}

    # QC_samples mask
    is_qc_sample = [True if col in QC_samples else False for col in df.columns[1:]]
    # Create a boolean mask for the x array
    x_mask = np.full(len(df.columns) - 1, False)
    x_mask[is_qc_sample] = True

    # Color based on the batch (and if it is QC sample then always black)
    for feature in show:
        print(feature)

        # Colors and alphas and markers
        colors = ['black' if qc else batch_to_color[batch[i]] for i, qc in enumerate(is_qc_sample)]
        row_data = df.iloc[feature, 1:]
        alphas = [0.4 if qc else 0.1 if zero else 0.8 for qc, zero in zip(is_qc_sample, row_data == 0)]

        # Create a zero mask (for qc samples)
        zeros_mask = df.iloc[feature, 1:][is_qc_sample] == 0

        # Initialize a dictionary to store the zero counts for each batch
        zero_counts = {batch_id: 0 for batch_id in set(batch)}
        qc_zero_counts = {batch_id: 0 for batch_id in set(batch)}

        # Update the zero counts for each batch
        for i, (b, zero) in enumerate(zip(batch, row_data == 0)):
            if zero:
                zero_counts[b] += 1
                if is_qc_sample[i]:
                    qc_zero_counts[b] += 1
        
        # Create a gridspec object
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # 3:1 height ratio

        # Create a figure with specified size
        fig = plt.figure(figsize=(20, 4)) 

        # Create a scatter plot to display the QC sample points
        plt.subplot(gs[0])
        plt.scatter(range(len(df.columns) -1), df.iloc[feature, 1:], color= colors, alpha=alphas, marker='o') 
        plt.plot(np.arange(len(df.columns) - 1)[x_mask][~zeros_mask], df.iloc[feature, 1:][is_qc_sample][~zeros_mask], color='black', linewidth=1)
        plt.xticks(rotation=90)
        plt.xlabel('Samples in order')
        plt.ylabel('Peak Area')
        plt.title("cpID = " + df.iloc[feature, 0])

        # Create a table to display the zero counts
        plt.subplot(gs[1])
        plt.axis('tight')  # Remove axis
        plt.axis('off')  # Hide axis
        fig.subplots_adjust(bottom=-0.5) # Adjust the bottom of the figure to make room for the table

        # Calculate the total number of samples and QC samples for each batch
        total_samples = {batch_id: batch.count(batch_id) for batch_id in unique_batches}
        total_qc_samples = {batch_id: sum(1 for b, qc in zip(batch, is_qc_sample) if (b == batch_id and qc)) for batch_id in unique_batches}

        # Calculate the total number of QC samples
        total_qc_samples_count = sum(qc_zero_counts.values())
        total_qc_percentage = total_qc_samples_count / sum(is_qc_sample) * 100

        # Calculate the percentage of zeros for each batch
        zero_percentages = {batch: zero_counts[batch] / total_samples[batch] * 100 for batch in unique_batches}

        # Calculate the percentage of QC zeros for each batch
        qc_zero_percentages = {batch: qc_zero_counts[batch] / total_qc_samples[batch] * 100 if total_qc_samples[batch] > 0 else 0 for batch in unique_batches}

        # Format the zero counts and percentages as strings
        formatted_zero_counts = [f"{zero_counts[batch]} ({zero_percentages[batch]:.2f}%)" for batch in unique_batches]

        # Format the QC zero counts and percentages as strings
        formatted_qc_zero_counts = [f"{qc_zero_counts[batch]} ({qc_zero_percentages[batch]:.2f}%)" for batch in unique_batches]

        # Calculate the total number of missing values
        total_missing = sum(zero_counts.values()) 
        total_percentage = total_missing / len(batch) * 100
        
        # Append 'white' to the batch colors for the total
        table_batch_colors = batch_colors + ['white']

        # Append the total to the sorted lists
        col_labels = unique_batches + ['Total']

        # Append the total to the formatted zero counts
        formatted_zero_counts.append(f"{total_missing} ({total_percentage:.2f}%)")
        formatted_qc_zero_counts.append(f"{total_qc_samples_count} ({total_qc_percentage:.2f}%)")

        # Convert all colors in the list to RGBA format
        table_batch_colors_rgba = [convert_to_rgba(color, 0.6) for color in table_batch_colors]

        # Create the table 
        plt.table(cellText=[formatted_zero_counts, formatted_qc_zero_counts],  # Sorted values of the table
            #cellColours=[table_batch_colors, table_batch_colors],
            cellColours=[table_batch_colors_rgba, table_batch_colors_rgba], # Sorted colors of the table
            rowLabels=['All Samples', 'QC Samples'],  # Row label
            colLabels=col_labels,  # Sorted column labels
            cellLoc='center',  # Alignment of the data in the table
            fontsize=10,  # Font size
            loc='center')  # Position of the table

        # Add a text annotation for "Zero Counts"
        plt.text(x=-0.005, y=0.65, s='Zero Counts', fontsize=15, transform=plt.gca().transAxes, ha='right', va='center')

        plt_name = main_folder + '/figures/single_compound_' + str(feature) + '_batches_first_view'
        for sufix in sufixes:
            plt.savefig(plt_name + sufix, dpi=300, bbox_inches='tight')
        plt.show()

    #---------------------------------------------
    #REPORTING
    text = 'View of samples for a single compound with highlighted QC samples was created.'
    report.add_text(text)
    #add plot to the report
    images = [main_folder + '/figures/single_compound_' + str(feature) + '_batches_first_view' for feature in show]
    for image in images:
        report.add_image(image)
    report.add_pagebreak()



def convert_to_rgba(color, alpha=0.5):
    """
    Convert a color to RGBA format.

    Parameters
    ----------
    color : str
        Color in hexadecimal format.
    alpha : float
        Alpha value. Default is 0.5.
    """
    rgb = mcolors.hex2color(color)  # Convert hexadecimal to RGB
    return rgb + (alpha,)  # Concatenate RGB with alpha


# # Get batch information
batch = batch_info['Batch'].tolist()
batch = [value for value in batch if value is not None]

# or 
#batch = None # All samples in one batch

QC_samples = metadata[metadata['Sample Type'] == 'Quality Control']['Sample File'].tolist()

# Choose a compounds to plot
#show = [0, 1, 2, 3, 4]
# # add index of a feature with cpdID == M159T50_2
# coi = data[data['cpdID'] == 'M159T50_2'].index[0]
# indexes.append(coi)

visualize_by_batch(df = data, QC_samples = QC_samples, main_folder = main_folder, show = 'default', sufixes = sufixes, batch = batch, cmap = 'viridis') 

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def filter_blank_intensity_ratio(data, blank_samples, QC_samples, ratio = 20 ,setting = 'first'):
    """
    Filter out features with intensity sample/blank < ratio (default = 20/1).

    Parameters
    ----------
    data : DataFrame
        DataFrame with the data.
    blank_samples : list
        List of blank samples. (blank_samples = metadata[metadata['Sample Type'] == 'Blank']['Sample File'].tolist())
    QC_samples : list
        List of QC samples. (QC_samples = metadata[metadata['Sample Type'] == 'Quality Control']['Sample File'].tolist())
    ratio : float
        Ratio for the intensity sample/blank. Default is 20. 
    setting : str
        Setting used to calculate choose intensity of the blank. Default is 'first'. Other options are 'median', 'min', 'mean', 'first' or 'last'.
    """

    blank_threshold = ratio
    # filter out features (compounds) with intensity sample/blank < ratio (default = 20/1)
    # filters features by comparing the intensity of blank samples to the median intensity of QC samples. Features where the relative intensity (fold change) is not large when compared to the blank are removed.

    # Blank_samples mask
    is_blank_sample = [True if col in blank_samples else False for col in data.columns[1:]]
    # QC_samples mask
    is_qc_sample = [True if col in QC_samples else False for col in data.columns[1:]]

    # Different approaches
    if setting == 'median':
        #Calculate intensity sample/blank by taking median value
        blank_intensities = data[data.columns[1:][is_blank_sample]].median(axis=1)
    elif setting == 'min':
        #Calculate intensity sample/blank by taking min value
        blank_intensities = data[data.columns[1:][is_blank_sample]].min(axis=1)
    elif setting == 'mean':
        #Calculate intensity sample/blank by taking mean value
        blank_intensities = data[data.columns[1:][is_blank_sample]].mean(axis=1)
    elif setting == 'first':
        #Calculate intensity sample/blank by taking the first blank as reference value
        blank_intensities = data[data.columns[1:][is_blank_sample]].iloc[:, 0]
    elif setting == 'last':
        #Calculate intensity sample/blank by taking the last blank as reference value
        blank_intensities = data[data.columns[1:][is_blank_sample]].iloc[:, -1]
    else:
        raise ValueError("Setting not recognized. Choose from: 'median', 'min', 'mean', 'first' or 'last'. (Chooses blank to use as reference value)")

    intensity_sample_blank = data[data.columns[1:][is_qc_sample]].median(axis=1)/blank_intensities
    print(intensity_sample_blank)
    
    #Filter out features (compounds) with intensity sample/blank < blank_threshold
    data = data[intensity_sample_blank >= blank_threshold]
    #reset index
    data = data.reset_index(drop=True)

    #report how many features were removed
    print("Number of features removed: " + str(len(intensity_sample_blank[intensity_sample_blank < blank_threshold])) + " ;being: " + str(intensity_sample_blank[intensity_sample_blank < blank_threshold].index.tolist()))

    #---------------------------------------------
    #REPORTING
    text0 = 'Features with intensity sample/blank < ' + str(blank_threshold) + ' were removed.'
    text1 = 'Number of features removed: ' + str(len(intensity_sample_blank[intensity_sample_blank < blank_threshold])) + ' ;being: '+ str(intensity_sample_blank[intensity_sample_blank < blank_threshold].index.tolist()[:10])[:-1] + ', ...'
    report.add_together([('text', text0),
                        ('text', text1),
                        'line'])

    return data 

blank_samples = metadata[metadata['Sample Type'] == 'Blank']['Sample File'].tolist()
QC_samples = metadata[metadata['Sample Type'] == 'Quality Control']['Sample File'].tolist()

data = filter_blank_intensity_ratio(data, blank_samples = blank_samples, QC_samples = QC_samples, ratio = 20/1, setting = 'first')
data

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# THE FOLLOWING FUNCTIONS ARE USED FOR THE CORRECTION OF BATCH EFFECTS

# ------------------------------------------------------------------------
#Inspirations:
# Kirwan JA, Broadhurst DI, Davidson RL, Viant MR. Characterising and correcting batch variation in an automated direct infusion mass spectrometry (DIMS) metabolomics workflow. Anal Bioanal Chem. 2013 Jun;405(15):5147-57. doi: 10.1007/s00216-013-6856-7. Epub 2013 Mar 1. Erratum in: Anal Bioanal Chem. 2014 Sep;406(24):6075-6. PMID: 23455646.
# DOI 10.1007/s00216-013-6856-7
# and many others


# Big thanks to Gavin Lloyd for the consultations 


def log_transformation(df, invert=False):
    """
    Apply the log transformation to the data.

    Parameters
    ----------
    df : DataFrame
        DataFrame with the data.
    invert : bool
        Invert the log transformation. Default is False.
    """
    # Apply the log transformation to the data
    if invert:
        return np.exp(df)
    return np.log(df + 0.0001) # Add a small number to avoid log(0) 

def cubic_spline_smoothing(x, y, p, use_norm = True):
    """
    Fit a cubic spline to the data.
    
    Parameters
    ----------
    x : array
        Array with the x values.
    y : array
        Array with the y values.
    p : float
        Smoothing parameter.
    use_norm : bool
        Use normalization. Default is True. if True, then y is already z-scores Cause normalization was used before.
    """
    # Fit a cubic spline to the data
    if use_norm: # Data is already normalized (it is z_scores)
        weights = 1 / (1 + (y - 10)** 2) # 10 is added in normalization
        s = UnivariateSpline(x, y, w = weights, s=p)
    else:
        zscores = zscore(y, axis=0) 
        weights = 1 / (1 + (zscores ** 2))
        s = UnivariateSpline(x, y, w = weights, s=p)   
    return s

def normalize_zscores(data):
    """
    Normalize the z-scores for the data.

    Parameters
    ----------
    data : DataFrame
        DataFrame with the data.
    """
    # Calculate the z-scores for the data
    zscores = zscore(data, axis=0) + 10 # Add 10 to avoid division by zero (or close to zero)
    
    # Return the z-scores
    return zscores, zscores.mean(axis=0), zscores.std(axis=0)

def invert_zscores(zscores, mean, std):
    """
    Invert the z-scores for the data.

    Parameters
    ----------
    zscores : DataFrame
        DataFrame with the z-scores.
    mean : DataFrame
        DataFrame with the mean values.
    std : DataFrame
        DataFrame with the standard deviation values.
    """
    # Invert the z-scores
    mean = mean.to_numpy()
    std = std.to_numpy()
    data = (zscores - 1) * std + mean # Subtract 1 (10 was added, but then it was divided in the correction step)
    # Return the inverted z-scores
    return data   

def leave_one_out_cross_validation(x, y, p_values, use_norm = True):
    """
    Perform leave-one-out cross validation to find the best smoothing parameter.
    
    Parameters
    ----------
    x : array
        Array with the x values.
    y : array
        Array with the y values.
    p_values : array
        Array with the smoothing parameters.
    use_norm : bool
        Use normalization. Default is True.
    """
    # Initialize leave-one-out cross validation
    loo = LeaveOneOut()
    
    # Initialize the best smoothing parameter and the minimum mean squared error
    best_p = None
    min_mse = np.inf
    
    # For each smoothing parameter
    for p in p_values:
        # Initialize the mean squared error
        mse = 0
        
        # For each training set and test set
        for train_index, test_index in loo.split(x):
            # Split the data into a training set and a test set
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Fit a cubic spline to the training set
            s = cubic_spline_smoothing(x_train, y_train, p, use_norm)
            
            # Calculate the mean squared error on the test set
            mse += ((s(x_test) - y_test) ** 2).sum()
        
        # Calculate the mean squared error
        mse /= len(x)
        
        # If this is the best smoothing parameter so far, update the best smoothing parameter and the minimum mean squared error
        if mse < min_mse:
            best_p = p
            min_mse = mse
    # Return the best smoothing parameter
    return best_p

def custom_k_fold_cross_validation(x, y, p_values, k = 3, use_norm = True, min = 5, minloo = 5): 
    """
    Perform k-fold cross validation to find the best smoothing parameter.

    Parameters
    ----------
    x : array
        Array with the x values.
    y : array
        Array with the y values.
    p_values : array
        Array with the smoothing parameters.
    k : int
        Number of folds. Default is 3.
    use_norm : bool
        Use normalization. Default is True.
    min : int
        Minimum number of data points. Default is 5. 
    minloo : int
        Minimum number of data points for leave-one-out cross validation. Default is 5;
        
        (cubic_spline requires atleast 4 data-points to fit a cubic spline (degree 3), and one is left out (hence we use 5), might be lowered in future versions of the code, but then we have to use lesser degree spline)
    """
    # Initialize the best smoothing parameter and the minimum mean squared error
    best_p = None
    min_mse = np.inf

    # If there would be less data points then the minimum parameter, decrease the k until it works 
    while len(x)/k < min:
        k -= 1
        if k == 0:
            break
    if k <= 1:
        #Leave one out cross validation
        if len(x) < minloo:
            # Not enough data for leave-one-out cross validation
            return None
        else:
            return leave_one_out_cross_validation(x, y, p_values, use_norm)

    # For each smoothing parameter
    for p in p_values:
        # Initialize the mean squared error
        mse = 0
        
        # For each fold
        for i in range(k):

            # Split the data into a training set and a test set
            test_indices = np.arange(i, len(x), k) # Takes every k-th element starting from i

            x_test = x[test_indices]
            y_test = y[test_indices]

            x_train = np.delete(x, test_indices)
            y_train = np.delete(y, test_indices)

            # Fit a cubic spline to the training set
            s = cubic_spline_smoothing(x_train, y_train, p, use_norm)

            # Calculate the mean squared error on the test set
            mse += ((s(x_test) - y_test) ** 2).sum()

        # Calculate the mean squared error
        mse /= len(x)

        # If this is the best smoothing parameter so far, update the best smoothing parameter and the minimum mean squared error
        if mse < min_mse:
            best_p = p
            min_mse = mse
    # Return the best smoothing parameter
    return best_p


def qc_correction(df, variable_metadata, QC_samples, batch = None, p_values = 'default', show = 'default', use_log = True, use_norm = True, use_zeros = False, cmap  = 'viridis'):
    """

    Correct batch effects and systematic errors using the QC samples for interpolation. (Includes visualization of the results.)
   
    ----------

    Parameters
    ----------
    df : DataFrame
        DataFrame with the data.
    variable_metadata : DataFrame
        DataFrame with the variable metadata.
    QC_samples : list
        List of QC samples. (QC_samples = metadata[metadata['Sample Type'] == 'Quality Control']['Sample File'].tolist())
    batch : list
        List of batches. Default is None. If None = all samples are in one batch else in list (e.g. ['001', '001', ...,  '002', '002', ...]).
    p_values : array or str
        Array with the smoothing parameters. Default is 'default'. If 'default', then the default values are used.
    show : list or str
        List of features to show. Default is 'default'. Other options are 'all', 'none' or index of the feature.
    use_log : bool
        Apply log transformation. Default is True.
    use_norm : bool
        Use normalization. Default is True.
    use_zeros : bool
        Use zeros. Default is False.
    cmap : str
        Name of the colormap. Default is 'viridis'. (Other options are 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted', 'turbo'; ADD '_r' to get reversed colormap)
    ----------
    
    It is important to note that the function is computationally expensive and may take a long time to run.

    Additionally it is highly advised to use the default values for the p_values parameter and both logaritmic and normalization transformations (unless they were already applied to the data).


    """

    # If show isnt list or numpy array
    if type(show) != list and type(show) != np.ndarray: 
        if show == 'default':
            show = np.linspace(0, len(df.columns)-1, 5, dtype=int)
        elif show == 'all':
            show = np.arange(len(df.columns))
        elif show == 'none':
            show = []
        else:
            show = [show]

    # Find zeros in the data (if needed)
    if not use_zeros:
        # Mask for zeros
        is_zero = df == 0
        is_zero = is_zero.T # Transpose the mask to match the data

    feature_names = df.iloc[:, 0]
    # Transpose the dataframe
    df = df.iloc[:, 1:].copy()
    df = df.T

    #QC_samples
    is_qc_sample = [True if col in QC_samples else False for col in df.index]

    cmap = mpl.cm.get_cmap(cmap)   

    if batch is None:
        batch = ['all_one_batch' for i in range(len(df.index))] # Dummy batch information (for cases where its all the same batch) (not columns because of the transposition)
    
    if p_values != 'default':
        p_values_to_use = p_values

    # Batch information
    unique_batches = []
    for b in batch:
        if b not in unique_batches:
            unique_batches.append(b)
    
    # Create a dictionary that maps each unique batch to a unique index
    batch_to_index = {batch: index for index, batch in enumerate(unique_batches)}

    # Normalize the indices between 0 and 1
    normalized_indices = {batch: index / len(unique_batches) for batch, index in batch_to_index.items()}

    # Create a dictionary that maps each batch to a color
    batch_colors = [mpl.colors.rgb2hex(cmap(normalized_indices[batch])) for batch in unique_batches]

    batch_to_color = {batch_id: batch_colors[i % len(batch_colors)] for i, batch_id in enumerate(unique_batches)}

    # Colors and alphas and markers
    colors = ['black' if qc else batch_to_color[batch[i]] for i, qc in enumerate(is_qc_sample)]

    if use_log:
        # Apply log transformation to the data
        df = log_transformation(df)
    
    if use_norm:
        df, mean, std = normalize_zscores(df)

    start_time = time.time()

    # Create lists to store the plot names to plot into REPORT
    plot_names_orig = []
    plot_names_corr = []
    
    # For each feature
    chosen_p_values = []
    numbers_of_correctable_batches = []
    for feature in df.columns:
        # Iterate over batches
        splines = []
        is_correctable_batch = []
        number_of_correctable_batches = 0
        for batch_index, batch_name in enumerate(unique_batches):

            # This batch mask
            is_batch = [True if b == batch_name else False for b in batch]
            
            # Get the QC data mask
            qc_indexes = df.index.isin(QC_samples)

            if not use_zeros:
                # Use is_zero mask for this feature
                new_zero_value = df[is_zero][feature].max()

                # Mask for zeros
                isnt_zero = df[feature] > new_zero_value

                # Combine the masks
                qc_indexes = qc_indexes & isnt_zero

            #Combine the masks
            qc_indexes_batched = qc_indexes & is_batch

            qc_data = df[qc_indexes_batched] 

            x = np.arange(len(df))[qc_indexes_batched]
            y = qc_data[feature].values

            #Citation: "Recommended values of s depend on the weights, w. If the weights represent the inverse of the standard-deviation of y, then a good s value should be found in the range (m-sqrt(2*m),m+sqrt(2*m)) where m is the number of datapoints in x, y, and w." (standard-deviation is wrong tho, that doesn't work for each datapoint...) 
            if p_values == 'default':
                m = len(qc_data)
                p_values_to_use = np.linspace(m-math.sqrt(2*m), m+math.sqrt(2*m), 10)

            # Fit a cubic spline to the QC data
            p = custom_k_fold_cross_validation(x, y, p_values_to_use, 5, use_norm)
            chosen_p_values.append(p)
            if p is None:
                # This batch doesn't have enough data (might be due to filtering out zeros)
                s = None
                is_correctable_batch.append(False)
            else:
                s = cubic_spline_smoothing(x, y, p, use_norm)
                is_correctable_batch.append(True)
                number_of_correctable_batches += 1
            splines.append(s)

            #Print out progress
            percentage_done = round((feature + 1) / len(df.columns) * 100, 3)
            print(f'Progress: {percentage_done}%; last chosen p: {p}. Time estimate: {round((time.time() - start_time) / (feature + 1) * (len(df.columns) - feature - 1)/60, 2)}m       ', end='\r')

        numbers_of_correctable_batches.append(number_of_correctable_batches)        

        x = np.arange(len(df))
        y = df[feature].values

        reds = cycle(['red', 'tomato', 'firebrick'])

        # Plot the original QC data and the fitted spline
        if feature in show:
            
            # Colors and alphas and markers
            colors = ['black' if qc else batch_to_color[batch[i]] for i, qc in enumerate(is_qc_sample)]
            row_data = data.iloc[feature, 1:]
            alphas = [0.4 if qc else 0.1 if zero else 0.8 for qc, zero in zip(is_qc_sample, row_data == 0)]

            
            # Initialize a dictionary to store the zero counts for each batch
            zero_counts = {batch_id: 0 for batch_id in set(batch)}
            qc_zero_counts = {batch_id: 0 for batch_id in set(batch)}

            # Update the zero counts for each batch
            for i, (b, zero) in enumerate(zip(batch, row_data == 0)):
                if zero:
                    zero_counts[b] += 1
                    if is_qc_sample[i]:
                        qc_zero_counts[b] += 1

            # Create a gridspec object
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # 3:1 height ratio
            fig = plt.figure(figsize=(20, 4))

            plt.subplot(gs[0])
            plt.scatter(x, y, color = colors, alpha = alphas)
            for batch_index, batch_name in enumerate(unique_batches):
                is_batch = [True if b == batch_name else False for b in batch]
                x = np.arange(len(df))[is_batch]
                spline = splines[batch_index]
                if spline != None: # None means that there wasn't enough data for this batch
                    plt.plot(x, spline(x), color=next(reds), linewidth = 2) # Plot the fitted spline (only its relevant part)

            plt.xlabel('Sample Order')
            plt.ylabel('Peak Area Z-score')
            if len(feature_names) > 0:
                plt.title(feature_names[feature])
            else:
                plt.title(f'Feature: {feature}')

            # Create a table to display the zero counts
            plt.subplot(gs[1])
            plt.axis('tight')  # Remove axis
            plt.axis('off')  # Hide axis
            fig.subplots_adjust(bottom=-0.5) # Adjust the bottom of the figure to make room for the table

            # Calculate the total number of samples and QC samples for each batch
            total_samples = {batch_id: batch.count(batch_id) for batch_id in unique_batches}
            total_qc_samples = {batch_id: sum(1 for b, qc in zip(batch, is_qc_sample) if (b == batch_id and qc)) for batch_id in unique_batches}

            # Calculate the total number of QC samples
            total_qc_samples_count = sum(qc_zero_counts.values())
            total_qc_percentage = total_qc_samples_count / sum(is_qc_sample) * 100

            # Calculate the percentage of zeros for each batch
            zero_percentages = {batch: zero_counts[batch] / total_samples[batch] * 100 for batch in unique_batches}

            # Calculate the percentage of QC zeros for each batch
            qc_zero_percentages = {batch: qc_zero_counts[batch] / total_qc_samples[batch] * 100 if total_qc_samples[batch] > 0 else 0 for batch in unique_batches}

            # Format the zero counts and percentages as strings
            formatted_zero_counts = [f"{zero_counts[batch]} ({zero_percentages[batch]:.2f}%)" for batch in unique_batches]

            # Format the QC zero counts and percentages as strings
            formatted_qc_zero_counts = [f"{qc_zero_counts[batch]} ({qc_zero_percentages[batch]:.2f}%)" for batch in unique_batches]

            # Calculate the total number of missing values
            total_missing = sum(zero_counts.values()) 
            total_percentage = total_missing / len(batch) * 100
            
            # Append 'white' to the batch colors for the total
            table_batch_colors = batch_colors + ['white']

            # Append the total to the sorted lists
            col_labels = unique_batches + ['Total']

            # Append the total to the formatted zero counts
            formatted_zero_counts.append(f"{total_missing} ({total_percentage:.2f}%)")
            formatted_qc_zero_counts.append(f"{total_qc_samples_count} ({total_qc_percentage:.2f}%)")

            # Convert all colors in the list to RGBA format
            table_batch_colors_rgba = [convert_to_rgba(color, 0.6) for color in table_batch_colors]

            # Create the table 
            table = plt.table(cellText=[formatted_zero_counts, formatted_qc_zero_counts],  # Sorted values of the table
                #cellColours=[table_batch_colors, table_batch_colors],
                cellColours=[table_batch_colors_rgba, table_batch_colors_rgba], # Sorted colors of the table
                rowLabels=['All Samples', 'QC Samples'],  # Row label
                colLabels=col_labels,  # Sorted column labels
                cellLoc='center',  # Alignment of the data in the table
                fontsize=10,  # Font size
                loc='center')  # Position of the table
            
            # Change color of text in cells of uncorrectable batches
            cells_to_change_color = [(2, i) for i, correctable in enumerate(is_correctable_batch) if not correctable]
            for cell in cells_to_change_color:
                table.get_celld()[cell].set_text_props(fontweight='bold', color='red') # Set the text to bold and red if the batch wasn't correctable

            # Add a text annotation for "Zero Counts"
            plt.text(x=-0.005, y=0.65, s='Zero Counts', fontsize=15, transform=plt.gca().transAxes, ha='right', va='center')

            #Save the plot
            for sufix in sufixes:
                plt_name = main_folder + '/figures/QC_correction_' + str(feature) + '_original'
                plt.savefig(plt_name + sufix, dpi=300, bbox_inches='tight')
            plot_names_orig.append(plt_name +'.png')
            plt.show()

        #CORRECTION of the data for the feature BATCH by BATCH
        for batch_index, batch_name in enumerate(unique_batches):
            #This batch mask
            is_batch = [True if b == batch_name else False for b in batch]
            # Use fitted spline to correct all the data for the feature batch by batch
            x = np.arange(len(df[feature]))
            if splines[batch_index] is not None: # None means that there wasn't enough data for this batch
                df[feature][is_batch] = df[feature][is_batch] / abs(splines[batch_index](x[is_batch])) # !! the order of things matter a lot here; to only use relevant parts of the spline !!
            else:
                df[feature][is_batch] = [0 for i in df[feature][is_batch]] # If there wasn't enough data, then just set the values to 0

        # Plot the corrected data
        if feature in show: 
            # Create a gridspec object
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # 3:1 height ratio
            fig = plt.figure(figsize=(20, 4))

            plt.subplot(gs[0])
            plt.scatter(x, y, color = colors, alpha = alphas)
            plt.plot(x, [1 for i in range(len(x))], color='green', label='mean', alpha = 0.4)

            # Calculate min and max values for the y axis (excluding zeros)

            y_min = df[feature][df[feature] != 0].min()
            y_max = df[feature].max()
            offset = 0.01 * (y_max - y_min) # 1% of the range
            plt.ylim(y_min - offset, y_max + offset) # Set the y axis limits (no interpolation = zeros - are not shown)

            plt.xlabel('Sample Order')
            plt.ylabel('Peak Area Normalized')
            if len(feature_names) > 0:
                plt.title(feature_names[feature])
            else:
                plt.title(f'Feature: {feature}')

            # Create a table to display the zero counts
            plt.subplot(gs[1])
            plt.axis('tight')  # Remove axis
            plt.axis('off')  # Hide axis
            fig.subplots_adjust(bottom=-0.5) # Adjust the bottom of the figure to make room for the table
            
            # Create the table 
            table = plt.table(cellText=[formatted_zero_counts, formatted_qc_zero_counts],  # Sorted values of the table
                #cellColours=[table_batch_colors, table_batch_colors],
                cellColours=[table_batch_colors_rgba, table_batch_colors_rgba], # Sorted colors of the table
                rowLabels=['All Samples', 'QC Samples'],  # Row label
                colLabels=col_labels,  # Sorted column labels
                cellLoc='center',  # Alignment of the data in the table
                fontsize=10,  # Font size
                loc='center') # Position of the table
                
            # Change color of text in cells of uncorrectable batches
            cells_to_change_color = [(2, i) for i, correctable in enumerate(is_correctable_batch) if not correctable]
            for cell in cells_to_change_color:
                table.get_celld()[cell].set_text_props(fontweight='bold', color='red') # Set the text to bold and red if the batch wasn't correctable

            # Add a text annotation for "Zero Counts"
            plt.text(x=-0.005, y=0.65, s='Zero Counts', fontsize=15, transform=plt.gca().transAxes, ha='right', va='center')

            
            #Save the plot
            for sufix in sufixes:
                plt_name = main_folder + '/figures/QC_correction_' + str(feature) + '_corrected'
                plt.savefig(plt_name + sufix, dpi=300, bbox_inches='tight')
            plot_names_orig.append(plt_name +'.png')
            plt.show()

    # Create a dictionary with the chosen p values
    chosen_p_values = pd.Series(chosen_p_values).value_counts().to_dict()
    # Sort the dictionary by the number of occurrences
    chosen_p_values = {k: v for k, v in sorted(chosen_p_values.items(), key=lambda item: item[1], reverse=True)}


    if use_norm:
        # Calculate new std
        #std_new = df.std(axis=0)

        # Invert the normalization (based on the old mean and the new std (is it good or bad?)) 
        df = invert_zscores(df, mean, std)

    if use_log:
        # Invert the log transformation
        df= log_transformation(df, invert=True)


    # Transpose the dataframe back    
    df = df.T
    # Add cpdID as a first column (back)
    df.insert(0, 'cpdID', feature_names)
    
    # Create a dictionary to store the correction batch information
    correction_dict = {cpdID: corrected_batches for cpdID, corrected_batches in zip(feature_names, numbers_of_correctable_batches)}
    # Add the correction batch information to the variable metadata
    variable_metadata['corrected_batches'] = variable_metadata['cpdID'].map(correction_dict)

    #---------------------------------------------
    #REPORTING 1
    text0 = 'QC correction was performed.'
    texts = []
    text1 = 'In the correction process, the data ' + ('WAS' if use_log else "WASN'T") + ' log-transformed and ' + ('WAS' if use_norm else "WASN'T") +' normalized.'
    texts.append(('text', text1, 'italic'))
    if use_log or use_norm:
        text2 = 'This step was then inverted after the correction.'
        texts.append(('text', text2, 'italic'))
    text3 = "The smoothing parameter was optimized using variation of k-fold cross-validation from a range defined as <m-math.sqrt(2*m), m+math.sqrt(2*m)>. which was: <"+ str(int(m-math.sqrt(2*m))) + "; " + str(int(m+math.sqrt(2*m))) + ">; where m is the amount of data-points interpolated (in this case the number of QCs)"
    texts.append(('text', text3))
    together = [('text', text0, 'bold')]
    together.extend(texts)
    report.add_together(together)
    report.add_line()
    for plot_names in zip(plot_names_orig, plot_names_corr):
        report.add_together([('text', 'Original data:', 'bold'),
                            ('image', plot_names[0]),
                            ('text', 'Corrected data:', 'bold'),
                            ('image', plot_names[1]),
                            'line'])
    return df, variable_metadata


#---------------------------------------------
QC_samples = metadata[metadata['Sample Type'] == 'Quality Control']['Sample File'].tolist()

# BATCH INFORMATION
# # Get batch information
#
batch = batch_info['Batch'].tolist()
batch = [value for value in batch if value is not None]
# or
# # for testing purposes lets separate the data into 2 batches half and half
# batch = ['batch1' for i in range(len(df.columns))]
# batch[int(len(df.columns)/2):] = ['batch2' for i in range(len(df.columns) - int(len(df.columns)/2))]
# or 
#batch = None # All samples in one batch

data, variable_metadata = qc_correction(data, variable_metadata, QC_samples, batch = batch, p_values = 'default', show = 'default', use_log = True, use_norm = True, use_zeros = False)

print()
print('Done')

#---------------------------------------------
#REPORTING 2
report.add_pagebreak()

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def filter_match_variable_metadata(data, variable_metadata):
    """
    Help function to update variable_metadata based on the data.

    Filters out features from variable_metadata that were already filtered out in the data.

    Parameters
    ----------
    data : DataFrame
        DataFrame with the data.
    variable_metadata : DataFrame
        DataFrame with the variable metadata.
    """
    # Filter out features from variable_metadata that were already filtered out in the data
    variable_metadata = variable_metadata[variable_metadata['cpdID'].isin(data['cpdID'])]
    # Reset the index
    variable_metadata.reset_index(drop=True, inplace=True)
    return variable_metadata

variable_metadata = filter_match_variable_metadata(data, variable_metadata)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def filter_number_of_corrected_batches(data, variable_metadata, threshold = 0.8):
    """
    Filter out features that were corrected in less than the threshold number of batches.

    Parameters
    ----------
    data : DataFrame
        DataFrame with the data.
    variable_metadata : DataFrame
        DataFrame with the variable metadata.
    threshold : float
        Threshold for the number of batches. Default is 0.8. If less than 1, then it is interpreted as a percentage otherwise as a minimum number of batches corrected.
    """

    if threshold < 1:
        # If the threshold is less than 1, interpret it as a percentage
        percentage = True
        threshold = int(threshold * len(data.columns))
    else:
        percentage = False
    lngth = len(data.columns)
    # Filter out features (from both data and variable_metadata) that were corrected in less than the threshold number of batches
    corrected_batches_dict = {cpdID: corrected_batches for cpdID, corrected_batches in zip(variable_metadata['cpdID'], variable_metadata['corrected_batches'])}

    removed = data['cpdID'][data['cpdID'].map(corrected_batches_dict).fillna(0) < threshold].tolist()
    data = data[data['cpdID'].map(corrected_batches_dict).fillna(0) >= threshold]
    data.reset_index(drop=True, inplace=True)
    variable_metadata = variable_metadata[variable_metadata['cpdID'].isin(data['cpdID'])]
    variable_metadata.reset_index(drop=True, inplace=True)
    print('Number of features removed: ' + str(lngth - len(data.columns)) + ' ;being: ' + str(removed[:10])[:-1] + ', ...')
    #---------------------------------------------
    #REPORTING
    if percentage:
        text1 = 'Features that were corrected in less than ' + str(threshold * 100) + '% of batches were removed.'
    else:
        text1 = 'Features that were corrected in less than ' + str(threshold) + ' batches were removed.'
    text2 = 'Number of features removed: ' + str(lngth - len(data.columns)) + ' ;being: ' + str(removed[:10])[:-1] + ', ...'
    report.add_together([('text', text1),
                        ('text', text2),
                        'line'])
    return data, variable_metadata

data, variable_metadata = filter_number_of_corrected_batches(data, variable_metadata, threshold = 9)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def filter_rsd(data, variable_metadata, QC_samples, rsd_threshold = 20):
    """
    Filter out features with RSD% over the threshold.

    Parameters
    ----------
    data : DataFrame
        DataFrame with the data.
    variable_metadata : DataFrame
        DataFrame with the variable metadata.
    QC_samples : list
        List of QC samples. (QC_samples = metadata[metadata['Sample Type'] == 'Quality Control']['Sample File'].tolist())
    rsd_threshold : float
        Threshold for the RSD%. Default is 20.
    """

    #QC_samples mask
    is_qc_sample = [True if col in QC_samples else False for col in data.columns[1:]]
    print(len(is_qc_sample))

    #Calculate RSD for only QC samples
    qc_rsd = data[data.columns[1:][is_qc_sample]].std(axis=1)/data[data.columns[1:][is_qc_sample]].mean(axis=1)*100

    variable_metadata = filter_match_variable_metadata(data, variable_metadata)
    #Add RSD into the data
    variable_metadata['QC_RSD'] = qc_rsd

    #Filter out features (compounds) with RSD > rsd_threshold
    over_threshold = qc_rsd > rsd_threshold
    data = data[~over_threshold]

    #Plot some of the compounds with high RSD
    number_plotted = 4
    indexes = qc_rsd[qc_rsd > rsd_threshold].index.tolist()[:number_plotted]
    if len(indexes) < number_plotted:
        number_plotted = len(indexes)
    if len(indexes)  == 0:
        print("No compounds with RSD > " + str(rsd_threshold) + " were found.")
    else:   
        alphas = [0.7 if qc else 0.5 for qc in is_qc_sample]
        colors = ['grey' if qc else 'blue' for qc in is_qc_sample]
        for i in range(number_plotted):
            plt.scatter(range(len(data.columns[1:])), data.iloc[indexes[i],1:], color= colors, alpha=alphas, marker='o')
            plt.xlabel('Samples in order')
            plt.ylabel('Peak Area')
            plt.title("High RSD compound: cpID = " + data.iloc[indexes[i], 0])
            for sufix in sufixes:
                plt.savefig(main_folder + '/figures/QC_samples_scatter_' + str(indexes[i]) + '_high_RSD-deleted_by_correction' + sufix, dpi=400, bbox_inches='tight')
                plt.show()

    data = data.reset_index(drop=True)
    variable_metadata = filter_match_variable_metadata(data, variable_metadata)
    variable_metadata = variable_metadata.reset_index(drop=True)
    #report how many features were removed
    print("Number of features removed: " + str(len(qc_rsd[qc_rsd > rsd_threshold])) + " ;being: " + str(qc_rsd[qc_rsd > rsd_threshold].index.tolist()))

    #---------------------------------------------
    #REPORTING
    text0 = 'Features with RSD% over the threshold (' + str(rsd_threshold) + ') were removed.'
    if len(indexes) == 0:
        text1 = 'No compounds with RSD > ' + str(rsd_threshold) + ' were found.'
        report.add_together([('text', text0),
                        ('text', text1)])
    else:
        text1 = 'Number of features removed: ' + str(len(qc_rsd[qc_rsd > rsd_threshold])) + ' ;being: '+ str(qc_rsd[qc_rsd > rsd_threshold].index.tolist()[:10])[:-1] + ', ...'
        text2 = 'Examples of compounds with high RSD%:'
        report.add_together([('text', text0),
                        ('text', text1),
                        ('text', text2)])
        images = [main_folder + '/figures/QC_samples_scatter_' + str(indexes[i]) + '_high_RSD-deleted_by_correction' for i in range(number_plotted)]
        for image in images:
            report.add_image(image)
    report.add_line()
    return data, variable_metadata

QC_samples = metadata[metadata['Sample Type'] == 'Quality Control']['Sample File'].tolist()

data, variable_metadata = filter_rsd(data, variable_metadata, QC_samples, rsd_threshold = 20)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

variable_metadata = filter_match_variable_metadata(data, variable_metadata)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Saving the data before log-transformation (the question is if this even makes sense now, but it is here for now...)
data.to_csv(main_folder + '/' + output_file_prefix + '-data_before_log-transf.csv', index = False, sep = ';')  

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# FINISHING TOUCHES
#---------------------------------------------
# SAVE THE PROCESSED DATA
def save_data(data, variable_metadata, metadata, main_folder, output_file_prefix):
    """
    Save the processed data into files.

    Parameters
    ----------
    data : DataFrame
        DataFrame with the data.
    variable_metadata : DataFrame
        DataFrame with the variable metadata.
    metadata : DataFrame
        DataFrame with the metadata.
    main_folder : str
        Path to the main folder.
    output_file_prefix : str
        Prefix for the output files.
    """
    data = data.reset_index(drop=True)
    data.to_csv(main_folder + '/' + output_file_prefix +'-data.csv', sep = ';')

    variable_metadata = variable_metadata.reset_index(drop=True)
    variable_metadata.to_csv(main_folder + '/' + output_file_prefix+'-variable_metadata.csv', sep = ';')

    metadata = metadata.reset_index(drop=True)
    metadata.to_csv(main_folder + '/' + output_file_prefix+'-metadata.csv', sep = ';')
    #---------------------------------------------
    #REPORTING
    text0 = 'Processed data was saved into a files named as follows:'
    data_list = ['data_before_log-transf.csv', 'data.csv', 'variable_metadata.csv', 'metadata.csv']
    text1 = ''
    for file in data_list:
        text1 += '<br/>' + output_file_prefix + '-' + file
    report.add_together([('text', text0, 'bold'),
                        ('text', text1, 'italic', 'center'),
                        'line'])
    return data

data = save_data(data, variable_metadata, metadata, main_folder, output_file_prefix)

#----------------------------------------------
#Build the pdf file
report.finalize_report()

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------the end-----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------