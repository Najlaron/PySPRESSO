# General modules
import pandas as pd
import numpy as np
import os
import re
import time
import math
from itertools import cycle

# Plotting modules
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

# Statistics and ML modules
from scipy.stats import zscore
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA

# My modules
import pdf_reporter as pdf_rptr

# #---------------------------------------------
# # REPORTING
# report_path = main_folder + '/' + report_file_name + '.pdf'
# title_text = main_folder

# report = pdf_rptr.Report(name = report_path, title = title_text)
# report.initialize_report('processing')

class PMF: # Peak Matrix Filtering (and Correcting, Transforming, Normalizing, Statistics, etc.)
    """
    Class that contains all the methods to filter, correct and transform the peak matrix data.
    """

    def __init__(self):
        # Data variables (names, paths, etc.)
        self.report = None
        main_folder = None
        report_file_name = None
        self.sufixes = ['.png', '.pdf']

        # Statistics variables
        self.pca_data = None
        self.pca_per_var = None


    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ALL DATA HANDLING AND INITILAZITAION METHODS (like: loading, saving, initializing-report, etc.) (keywords like: loader_..., extracter_..., saver_..., but also without keywords)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _initalize_report(self, main_folder, report_file_name, report_type = 'processing'):
        """
        Initialize the report object.

        Parameters
        ----------
        main_folder : str
            Path to the main folder.
        report_file_name : str
            Name of the report file.
        report_type : str
            Settings of the report (title). Default is 'processing'. All options are 'processing', 'statistics'. (To add more options, add them to the pdf_reporter.py file)
        """
        #---------------------------------------------
        # REPORTING
        report_path = main_folder + '/' + report_file_name + '.pdf'
        title_text = main_folder
        report = pdf_rptr.Report(name = report_path, title = title_text)
        report.initialize_report(report_type)
        return report
    
    def loader_data(self, data_input_file_name, report, separator = ';', encoding = 'ISO-8859-1'):
        """
        Load data matrix from a csv file.

        Parameters
        ----------
        data_input_file_name : str
            Name of the data file.
        report : pdf_reporter.Report object
            Report object to add information to.
        separator : str
            Separator used in the data file. Default is ';'.
        encoding : str
            Encoding used in the data file. Default is 'ISO-8859-1'. (UTF-8 is also common.)
        """
        # Load the data from the csv file
        data = pd.read_csv(data_input_file_name, sep = separator,  encoding=encoding )
        #---------------------------------------------
        # REPORTING
        text = 'Data were loaded from: ' + data_input_file_name + ' (data from Compound Discoverer).'
        report.add_together([('text', text), 
                            ('table', data),
                            'line'])
        return data
    
    def add_cpdID(self, data, report, mz_col = 'm/z', rt_col = 'RT [min]'):
        """
        Add a column 'cpdID' to the data calculated from mz and RT. Matching ID - distinguished by adding "_1", "_2", etc.

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        report : pdf_reporter.Report object
            Report object to add information to.
        mz_col : str
            Name of the column with m/z values. Default is 'm/z'.
        rt_col : str
            Name of the column with RT values. Default is 'RT [min]'.
        """
        #add cpdID column to the data_df calculated from mz and RT
        data['cpdID'] = 'M' + data[mz_col].round(0).astype(int).astype(str) + 'T' + (data[rt_col]*60).round(0).astype(int).astype(str)
        # Create a new column 'Duplicate_Count' to keep track of the duplicates
        data['Duplicate_Count'] = data.groupby('cpdID').cumcount()
        # Create a mask to identify duplicated compounds
        is_duplicate = data['Duplicate_Count'] > 0
        # Update the 'Compound_ID' column for duplicated compounds
        data.loc[is_duplicate, 'cpdID'] = data['cpdID'] + '_' + (data['Duplicate_Count']).astype(str)
        # Drop the 'Duplicate_Count' 
        data.drop('Duplicate_Count', axis=1, inplace=True)
        #check if there are rly no duplicates left
        remaining_duplicates = data[data['cpdID'].duplicated(keep=False)]
        print(remaining_duplicates.empty)

        #---------------------------------------------
        # REPORTING
        text = 'cpdID column was added to the data calculated from mz and RT. Matching ID - distinguished by adding "_1", "_2", etc.'
        report.add_together([('text', text),
                                'line'])
        return data

    def extracter_variable_metadata(self, data, report, column_index_ranges = [(10, 15), (18, 23)]):
        """
        Extract variable metadata from the data. Based on the column index ranges. 

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        report : pdf_reporter.Report object
            Report object to add information to.
        column_index_ranges : list
            List of tuples with the column index ranges to keep. Default is [(10, 15), (18, 23)].
        """
        variable_metadata = data[['cpdID', 'Name', 'Formula']]
        for column_index_range in column_index_ranges:
            variable_metadata = variable_metadata.join(data.iloc[:, column_index_range[0]:column_index_range[1]])

        #---------------------------------------------
        # REPORTING
        text = 'variable-metadata matrix was created.'
        report.add_together([('text', text),
                                'line'])
        return data, variable_metadata
    
    def extracter_data(self, data, report, variable_metadata, prefix = 'Area:'):
        """
        Extract data matrix from the data_df.

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        report : pdf_reporter.Report object
            Report object to add information to.
        variable_metadata : DataFrame
            DataFrame with the variable metadata.
        prefix : str
            Prefix used in the column names to keep. Default is 'Area:'.
        """
        temp_data = data.copy()
        data = temp_data[['cpdID']]
        # Select columns with names starting with "Area:"
        area_columns = temp_data.filter(regex=r'^' + prefix, axis=1) #for those starting with Area:, for those including Area anywhere, you can use like='Area:'
        data = data.join(temp_data[area_columns.columns])
        del temp_data
        # Fill NaN values with zero (NaN values are problem when plotted, and we will treat 0s as missing values later)
        data.fillna(0, inplace=True)  # Replace NaN with 0
        #---------------------------------------------
        # REPORTING
        text = 'data matrix was created.'
        report.add_together([('text', text),
                                'line'])
        return data, variable_metadata
    
    def loader_batch_info(self, report, input_batch_info_file):
        """
        Load batch_info matrix from a csv file.

        Parameters
        ----------
        report : pdf_reporter.Report object
            Report object to add information to.
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

    def batch_by_name_reorder(self, data, report, batch_info, distinguisher = 'Batch', format='%d.%m.%Y %H:%M'):
        """
        Reorder data based on the creation date and add a column 'Batch' to the batch_info based on the distinguisher.
        
        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        report : pdf_reporter.Report object
            Report object to add information to.
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
        text0 = 'Batch information was used to (re)order samples.'
        text1 = 'Not found: ' + str(len(not_found)) + ' ;being: ' + str(not_found)
        text2 = 'Names not identified: ' + str(len(names)) + ' ;being: ' + str(names)
        report.add_together([('text', text0),
                            ('text', text1, 'italic'),
                            ('text', text2, 'italic'),
                            ('table', batch_info),
                            'line'])

        return data, batch_info

    def extracter_metadata(self, data, report, batch_info, group_columns_to_keep, prefix = 'Area:'):
        """
        Extract metadata from batch_info by choosing columns to keep. Keep your grouping columns such as AgeGroup, Sex, Diagnosis, etc. 

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        report : pdf_reporter.Report object
            Report object to add information to.
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
        return data, metadata

    def saver_all_datasets(data, report, variable_metadata, metadata, main_folder, output_file_prefix):
        """
        Save all the processed data into files.

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        report : pdf_reporter.Report object
            Report object to add information to.
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
        return data, variable_metadata, metadata
    
    def finalize_report(self, report):
        """
        Finalize the report. (Has to be called at the end of the script otherwise the pdf file will be "corrupted".)

        Parameters
        ----------
        report : pdf_reporter.Report object
            Report object to add information to.
        """
        report.finalize_report()
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ALL FILTERING METHODS (keyword: filter_...)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def _filter_match_variable_metadata(self, data, variable_metadata):
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
        return data, variable_metadata

    def filter_missing_values(self, data, report, variable_metadata, QC_samples, qc_threshold = 0.8, sample_threshold = 0.5):
        """
        Filter out features with high number of missing values (nan or 0): A) within QC samples  B) all samples 

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        report : pdf_reporter.Report object
            Report object to add information to.
        variable_metadata : DataFrame
            DataFrame with the variable metadata.
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

        return data, variable_metadata

    def filter_blank_intensity_ratio(self, data, report, variable_metadata ,blank_samples, QC_samples, ratio = 20 ,setting = 'first'):
        """
        Filter out features with intensity sample/blank < ratio (default = 20/1).

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        report : pdf_reporter.Report object
            Report object to add information to.
        variable_metadata : DataFrame
            DataFrame with the variable metadata.
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
        return data, variable_metadata 
    
    def filter_relative_standard_deviation(self, data, report, variable_metadata, QC_samples, rsd_threshold = 20, to_plot = False):
        """
        Filter out features with RSD% over the threshold.

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        report : pdf_reporter.Report object
            Report object to add information to.
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

        variable_metadata = self._match_variable_metadata(data, variable_metadata)
        #Add RSD into the data
        variable_metadata['QC_RSD'] = qc_rsd

        #Filter out features (compounds) with RSD > rsd_threshold
        over_threshold = qc_rsd > rsd_threshold
        data = data[~over_threshold]

        #Plot some of the compounds with high RSD
        if to_plot:
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
                    for sufix in self.sufixes:
                        plt.savefig(self.main_folder + '/figures/QC_samples_scatter_' + str(indexes[i]) + '_high_RSD-deleted_by_correction' + sufix, dpi=400, bbox_inches='tight')
                        plt.show()

        data = data.reset_index(drop=True)
        variable_metadata = self._match_variable_metadata(data, variable_metadata)
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
            images = [self.main_folder + '/figures/QC_samples_scatter_' + str(indexes[i]) + '_high_RSD-deleted_by_correction' for i in range(number_plotted)]
            for image in images:
                report.add_image(image)
        report.add_line()
        return data, variable_metadata
    
    def filter_number_of_corrected_batches(self, data, report, variable_metadata, threshold = 0.8):
        """
        Filter out features that were corrected in less than the threshold number of batches. (Ideally apply after the correction step.)

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        report : pdf_reporter.Report object
            Report object to add information to.
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
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ALL TRANSFORMING METHODS (keyword: transformer_...)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def _transformer_log(self, data, invert=False):
        """
        Help function to apply the log transformation to the data. 

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        invert : bool
            Invert the log transformation. Default is False.
        """
        # Apply the log transformation to the data
        if invert:
            return np.exp(data)
        return np.log(data + 0.0001) # Add a small number to avoid log(0) 

    def transformer_log(self, data, report, variable_metadata, invert=False):
        """
        Apply the log transformation to the data.

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        invert : bool
            Invert the log transformation. Default is False.
        """
        data.iloc[:, 1:] = self._transformer_log(data.iloc[:, 1:], invert)
        #---------------------------------------------
        #REPORTING
        if invert:
            text = 'Log transformation was inverted.'
        else:
            text = 'Log transformation was applied.'
        report.add_together([('text', text),
                            'line'])
        return data, variable_metadata
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ALL NORMALIZING METHODS (keyword: normalizer_...)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def _normalizer_zscores(self, data):
        """
        Help function to calculate the z-scores for the data.

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        """
        # Calculate the z-scores for the data
        zscores = zscore(data, axis=0) + 10 # Add 10 to avoid division by zero (or close to zero)
        
        # Return the z-scores
        return zscores, zscores.mean(axis=0), zscores.std(axis=0)
    
    def _invert_zscores(zscores, mean, std):
        """
        Help function to invert the z-scores for the data.

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
    
    def normalizer_zscores(self, data, report, variable_metadata):
        """
        Normalize the data using z-scores.

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        report : pdf_reporter.Report object
            Report object to add information to.
        variable_metadata : DataFrame
            DataFrame with the variable metadata.
        """
        # Calculate the z-scores for the data
        zscores, mean, std = self._normalizer_zscores(data.iloc[:, 1:])
        # Add the z-scores to the data
        data.iloc[:, 1:] = zscores
        #---------------------------------------------
        #REPORTING
        text = 'Data was normalized using z-scores. Mean and standard deviation were calculated for each feature.'
        report.add_together([('text', text),
                            'line'])
        return data, variable_metadata
    

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ALL CORRECTING METHODS (keyword: correcter_...)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _cubic_spline_smoothing(self, x, y, p, use_norm = True):
        """
        Help functin to fit a cubic spline to the data.
        
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

    def _leave_one_out_cross_validation(self, x, y, p_values, use_norm = True):
        """
        Help fucntion to perform leave-one-out cross validation to find the best smoothing parameter.
        
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
                s = self._cubic_spline_smoothing(x_train, y_train, p, use_norm)
                
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
    
    def _custom_k_fold_cross_validation(self, x, y, p_values, k = 3, use_norm = True, min = 5, minloo = 5): 
        """
        Help function to perform k-fold cross validation to find the best smoothing parameter.

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
                return self._leave_one_out_cross_validation(x, y, p_values, use_norm)

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
                s = self._cubic_spline_smoothing(x_train, y_train, p, use_norm)

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

    def correcter_qc_interpolation(self, data, report, variable_metadata, QC_samples, main_folder, sufixes = ['.png', '.pdf'], show = 'default', batch = None, p_values = 'default', use_log = True, use_norm = True, use_zeros = False, cmap  = 'viridis'):
        """

        Correct batch effects and systematic errors using the QC samples for interpolation. (Includes visualization of the results.)
    
        ----------

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        report : pdf_reporter.Report object
            Report object to add information to.
        variable_metadata : DataFrame
            DataFrame with the variable metadata.
        QC_samples : list
            List of QC samples. (QC_samples = metadata[metadata['Sample Type'] == 'Quality Control']['Sample File'].tolist())
        main_folder : str
            Main folder to save the figures.
        sufixes : list
            List of sufixes for the figures. Default is ['.png', '.pdf'].
        show : list or str
            List of features to show. Default is 'default'. Other options are 'all', 'none' or index of the feature.
        batch : list
            List of batches. Default is None. If None = all samples are in one batch else in list (e.g. ['001', '001', ...,  '002', '002', ...]).
        p_values : array or str
            Array with the smoothing parameters. Default is 'default'. If 'default', then the default values are used.
        use_log : bool
            Apply log transformation. Default is True.
        use_norm : bool
            Use normalization. Default is True.
        use_zeros : bool
            Use zeros (Not using zeros is better for interpolating sparse QC data). Default is False.
        cmap : str
            Name of the colormap. Default is 'viridis'. (Other options are 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted', 'turbo'; ADD '_r' to get reversed colormap)
        ----------
        
        It is important to note that the function is computationally expensive and may take a long time to run.

        Additionally it is highly advised to use the default values for the p_values parameter and both logaritmic and normalization transformations (unless they were already applied to the data).


        """
        # If show isnt list or numpy array
        if type(show) != list and type(show) != np.ndarray: 
            if show == 'default':
                show = np.linspace(0, len(data.columns)-1, 5, dtype=int)
            elif show == 'all':
                show = np.arange(len(data.columns))
            elif show == 'none':
                show = []
            else:
                show = [show]

        # Find zeros in the data (if needed)
        if not use_zeros:
            # Mask for zeros
            is_zero = data == 0
            is_zero = is_zero.T # Transpose the mask to match the data

        feature_names = data.iloc[:, 0]
        # Transpose the dataframe
        data = data.iloc[:, 1:].copy()
        data = data.T

        #QC_samples
        is_qc_sample = [True if col in QC_samples else False for col in data.index]

        cmap = mpl.cm.get_cmap(cmap)   

        if batch is None:
            batch = ['all_one_batch' for i in range(len(data.index))] # Dummy batch information (for cases where its all the same batch) (not columns because of the transposition)

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
            data = self._transformer_log(data)
        if use_norm:
            # Normalize the data using z-scores
            data, mean, std = self._normalizer_zscores(data)

        start_time = time.time()
        # Create lists to store the plot names to plot into REPORT
        plot_names_orig = []
        plot_names_corr = []
        
        # For each feature
        chosen_p_values = []
        numbers_of_correctable_batches = []
        for feature in data.columns:
            # Iterate over batches
            splines = []
            is_correctable_batch = []
            number_of_correctable_batches = 0
            for batch_index, batch_name in enumerate(unique_batches):
                # This batch mask
                is_batch = [True if b == batch_name else False for b in batch]
                # Get the QC data mask
                qc_indexes = data.index.isin(QC_samples)

                if not use_zeros:
                    # Use is_zero mask for this feature
                    new_zero_value = data[is_zero][feature].max()
                    # Mask for zeros
                    isnt_zero = data[feature] > new_zero_value
                    # Combine the masks
                    qc_indexes = qc_indexes & isnt_zero

                #Combine the masks
                qc_indexes_batched = qc_indexes & is_batch
                qc_data = data[qc_indexes_batched] 

                x = np.arange(len(data))[qc_indexes_batched]
                y = qc_data[feature].values

                #Citation: "Recommended values of s depend on the weights, w. If the weights represent the inverse of the standard-deviation of y, then a good s value should be found in the range (m-sqrt(2*m),m+sqrt(2*m)) where m is the number of datapoints in x, y, and w." (standard-deviation is wrong tho, that doesn't work for each datapoint...) 
                if p_values == 'default':
                    m = len(qc_data)
                    p_values_to_use = np.linspace(m-math.sqrt(2*m), m+math.sqrt(2*m), 10)

                # Fit a cubic spline to the QC data
                p = self._custom_k_fold_cross_validation(x, y, p_values_to_use, 5, use_norm)
                chosen_p_values.append(p)
                if p is None:
                    # This batch doesn't have enough data (might be due to filtering out zeros)
                    s = None
                    is_correctable_batch.append(False)
                else:
                    s = self._cubic_spline_smoothing(x, y, p, use_norm)
                    is_correctable_batch.append(True)
                    number_of_correctable_batches += 1
                splines.append(s)

                #Print out progress
                percentage_done = round((feature + 1) / len(data.columns) * 100, 3)
                print(f'Progress: {percentage_done}%; last chosen p: {p}. Time estimate: {round((time.time() - start_time) / (feature + 1) * (len(data.columns) - feature - 1)/60, 2)}m       ', end='\r')

            numbers_of_correctable_batches.append(number_of_correctable_batches)        

            x = np.arange(len(data))
            y = data[feature].values

            reds = cycle(['red', 'tomato', 'firebrick'])

            # Plot the original QC data and the fitted spline
            if feature in show:
                
                # Colors and alphas and markers
                colors = ['black' if qc else batch_to_color[batch[i]] for i, qc in enumerate(is_qc_sample)]
                #row_data = data.iloc[feature, 1:] # Previously this was looking back into the original data, now it works with the data already transposed
                row_data = data[feature].values
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
                    x = np.arange(len(data))[is_batch]
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
                table_batch_colors_rgba = [self._convert_to_rgba(color, 0.6) for color in table_batch_colors]

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
                x = np.arange(len(data[feature]))
                if splines[batch_index] is not None: # None means that there wasn't enough data for this batch
                    data[feature][is_batch] = data[feature][is_batch] / abs(splines[batch_index](x[is_batch])) # !! the order of things matter a lot here; to only use relevant parts of the spline !!
                else:
                    data[feature][is_batch] = [0 for i in data[feature][is_batch]] # If there wasn't enough data, then just set the values to 0

            # Plot the corrected data
            if feature in show: 
                # Create a gridspec object
                gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # 3:1 height ratio
                fig = plt.figure(figsize=(20, 4))

                plt.subplot(gs[0])
                plt.scatter(x, y, color = colors, alpha = alphas)
                plt.plot(x, [1 for i in range(len(x))], color='green', label='mean', alpha = 0.4)

                # Calculate min and max values for the y axis (excluding zeros)

                y_min = data[feature][data[feature] != 0].min()
                y_max = data[feature].max()
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
            data = self._invert_zscores(data, mean, std)

        if use_log:
            # Invert the log transformation
            data= self._transformer_log(data, invert=True)


        # Transpose the dataframe back    
        data = data.T
        # Add cpdID as a first column (back)
        data.insert(0, 'cpdID', feature_names)
        
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
        return data, variable_metadata
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ALL STATISTICS METHODS (keyword: statistics_...)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def statistics_correlation(self, data, report, metadata, column_name, output_file_prefix, sufixes = ['.png', '.pdf'], cmap = 'coolwarm'):
        """
        Calculate the correlation matrix of the data and create a heatmap.

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        report : pdf_reporter.Report object
            Report object to add information to.
        metadata : DataFrame
            DataFrame with the metadata.
        column_name : str
            Name of the column to group the data by. (From metadata, e.g. 'Sample Type' or 'Sex', 'Age', Diagnosis', etc.)
        output_file_prefix : str
            Prefix for the output file.
        sufixes : list
            List of sufices for the image files to be saved in. Default is ['.png', '.pdf'].
        cmap : str
            Name of the colormap. Default is 'coolwarm'. (Other options are 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted', 'turbo'; ADD '_r' to get reversed colormap)
        """
        # Group the data by 'column_name' and calculate the mean of each group
        grouped_means = data.iloc[:, 1:].groupby(metadata[column_name]).mean()

        # Calculate the correlation matrix of the group means
        correlation_matrix = grouped_means.T.corr()

        # Create a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, cmap=cmap, vmin=-1, vmax=1)
        name = output_file_prefix + '-group_correlation_matrix_heatmap-0'
        for sufix in sufixes:
            plt.savefig(name + sufix, bbox_inches='tight', dpi = 300)
        plt.show()

        #---------------------------------------------
        # REPORTING
        text = 'Group correlation matrix heatmap was created. Grouping is based on: ' + column_name
        report.add_together([
            ('text', text), 
            ('image', name), 
            'pagebreak'])
        return correlation_matrix
    
    def statistics_PCA(self, data, report, metadata, column_name):
        """
        Perform PCA on the data and visualize the results.

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.        
        """
        # Perform PCA
        pca = PCA()
        pca.fit(data.iloc[:,1:].T)
        pca_data = pca.transform(data.iloc[:, 1:].T)
        # Calculate the percentage of variance that each PC explains
        per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
        self.pca_per_var = per_var
        labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

        pca_df = pd.DataFrame(pca_data, columns=labels) 

        self.pca_data = pca_df
        #---------------------------------------------
        # REPORTING
        report.add_text('<b>PCA was performed.</b>')
        return pca_df
    



    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ALL VISUALIZING METHODS (keyword: visualizer_...)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _convert_to_rgba(self, color, alpha=0.5):
        """
        Help function to convert a color to RGBA format.

        Parameters
        ----------
        color : str
            Color in hexadecimal format.
        alpha : float
            Alpha value. Default is 0.5.
        """
        rgb = mcolors.hex2color(color)  # Convert hexadecimal to RGB
        return rgb + (alpha,)  # Concatenate RGB with alpha
    
    def visualizer_boxplot(self, data, report, QC_samples, main_folder, sufixes = ['.png', '.pdf']):
        """
        Create a boxplot of all samples.  Can depict if there is a problem with the data (retention time shift too big -> re-run alignment; batch effect, etc.)

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        report : pdf_reporter.Report object
            Report object to add information to.
        QC_samples : list
            List of QC samples. (QC_samples = metadata[metadata['Sample Type'] == 'Quality Control']['Sample File'].tolist())
        main_folder : str
            Path to the main folder.
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
    
    def visualizer_samples_by_batch(self, data, report, QC_samples, main_folder, sufixes = ['.png', '.pdf'], show = 'default', batch = None, cmap = 'viridis'):
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
        sufixes : list
            List of sufices for the image files to be saved in. Default is ['.png', '.pdf'].
        show : list or str
            List of features to show. Default is 'default'. Other options are 'all', 'none' or index of the feature.
        batch : list
            List of batches. Default is None. If None = all samples are in one batch else in list (e.g. ['001', '001', ...,  '002', '002', ...]).
        cmap : str
            Name of the colormap. Default is 'viridis'. (Other options are 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted', 'turbo'; ADD '_r' to get reversed colormap)
        """

        # If show isnt list or numpy array
        if type(show) != list and type(show) != np.ndarray: 
            if show == 'default':
                show = np.linspace(0, len(data.columns)-1, 5, dtype=int)
                print(show)
            elif show == 'all':
                show = np.arange(len(data.columns))
            elif show == 'none':
                show = []
            else:
                show = [show]

        if batch is None:
            batch = ['all_one_batch' for i in range(len(data.columns) - 1)]  # Dummy batch information (for cases where its all the same batch)

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
        is_qc_sample = [True if col in QC_samples else False for col in data.columns[1:]]
        # Create a boolean mask for the x array
        x_mask = np.full(len(data.columns) - 1, False)
        x_mask[is_qc_sample] = True

        # Color based on the batch (and if it is QC sample then always black)
        for feature in show:
            print(feature)

            # Colors and alphas and markers
            colors = ['black' if qc else batch_to_color[batch[i]] for i, qc in enumerate(is_qc_sample)]
            row_data = data.iloc[feature, 1:]
            alphas = [0.4 if qc else 0.1 if zero else 0.8 for qc, zero in zip(is_qc_sample, row_data == 0)]

            # Create a zero mask (for qc samples)
            zeros_mask = data.iloc[feature, 1:][is_qc_sample] == 0

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
            fig, ax = plt.subplots(figsize=(20, 4))

            # Create a scatter plot to display the QC sample points
            plt.subplot(gs[0])
            plt.scatter(range(len(data.columns) -1), data.iloc[feature, 1:], color= colors, alpha=alphas, marker='o') 
            plt.plot(np.arange(len(data.columns) - 1)[x_mask][~zeros_mask], data.iloc[feature, 1:][is_qc_sample][~zeros_mask], color='black', linewidth=1)
            plt.xticks(rotation=90)
            plt.xlabel('Samples in order')
            plt.ylabel('Peak Area')
            plt.title("cpID = " + data.iloc[feature, 0])

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
            table_batch_colors_rgba = [self._convert_to_rgba(color, 0.6) for color in table_batch_colors]

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
        return fig, ax

    def visualizer_PCA_scree_plot(self, data, report, output_file_prefix, sufixes = ['.png', '.pdf']):
        """
        Create a scree plot for PCA.

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        report : pdf_reporter.Report object
            Report object to add information to.
        output_file_prefix : str
            Prefix for the output file.
        sufixes : list
            List of sufices for the image files to be saved in. Default is ['.png', '.pdf'].
        """
        if self.pca_per_var is None:
            raise ValueError('PCA was not performed yet. Run PCA first.')
        else:
            per_var = self.pca_per_var
        # Create a scree plot
        fig, ax = plt.figure(figsize=(10, 10))
        plt.bar(x=range(1, 11), height=per_var[:10])

        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal Component')
        plt.title('Scree Plot')
        xticks = list(range(1, 11))
        plt.xticks(xticks)

        name = output_file_prefix + '_scree_plot-percentages'
        for sufix in sufixes:
            plt.savefig(name + sufix, bbox_inches='tight', dpi = 300)
        plt.show()
        plt.close()

        #---------------------------------------------
        # REPORTING
        text = 'Scree plot was created and added into: ' + name
        report.add_together([
            ('text', text),
            ('image', name)])
        return fig, ax
    
    def visualizer_PCA_run_order(self, data, report, output_file_prefix, sufixes = ['.png', '.pdf'], connected = True, colors_for_cmap = ['yellow', 'red']):
        """
        Create a PCA plot colored by run order. (To see if there is any batch effect)

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        report : pdf_reporter.Report object
            Report object to add information to.
        output_file_prefix : str
            Prefix for the output file.
        sufixes : list
            List of sufices for the image files to be saved in. Default is ['.png', '.pdf'].
        colors_for_cmap : list
            List of colors for the colormap. Default is ['yellow', 'red'].        
        """
        if self.pca_per_var is None:
            raise ValueError('PCA was not performed yet. Run PCA first.')
        else:
            per_var = self.pca_per_var
            pca_data = self.pca_data 

        # Create a yellow-to-blue colormap
        cmap = LinearSegmentedColormap.from_list("mycmap", colors_for_cmap)

        column_unique_values = data.columns[1:]
        num_unique_values = len(column_unique_values)
        colors = [cmap(i/num_unique_values) for i in range(num_unique_values)]
        sample_type_colors = dict(zip(column_unique_values, colors))

        # Create a new figure and axes
        fig, ax = plt.subplots()

        # Create custom legend handles for colors
        color_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=sample_type_colors[st], markersize=10) 
                        for st in sample_type_colors.keys()]

        # Plot the data
        for sample_type in sample_type_colors.keys():
            df_samples = pca_data[data.columns[1:] == sample_type]
            plt.scatter(df_samples['PC1'], df_samples['PC2'], color=sample_type_colors[sample_type], label=sample_type)

        plt.title('PCA Graph - run order colored')
        plt.xlabel('PC1 - {0}%'.format(per_var[0]))
        plt.ylabel('PC2 - {0}%'.format(per_var[1]))

        # Create a normalization object for the colorbar
        norm = Normalize(vmin=0, vmax=num_unique_values)

        # Create a ScalarMappable object for the colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Add the colorbar
        plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.2)

        if connected:
            for i in range(len(pca_data)-1):
                plt.plot([pca_data['PC1'].iloc[i], pca_data['PC1'].iloc[i+1]], [pca_data['PC2'].iloc[i], pca_data['PC2'].iloc[i+1]], color='black', linewidth=0.25)

        if connected:
            name = output_file_prefix + '_PCA_plot_colored_by_run_order-connected'
        else:
            name = output_file_prefix + '_PCA_plot_colored_by_run_order'
        for sufix in sufixes:
            plt.savefig(name + sufix, bbox_inches='tight', dpi = 300)
        plt.show()
        #---------------------------------------------
        # REPORTING
        if connected:
            text = 'PCA plot <b> colored by run order </b> was created and added into: ' + name + '. Line connecting the samples by run order was added.'
        else:
            text = 'PCA plot <b> colored by run order </b> was created and added into: ' + name
        report.add_together([
            ('text', text),
            ('image', name),])
        return fig, ax
    
    def visualizer_PCA_grouped(self, data, report, metadata, color_column, marker_column, output_file_prefix, sufixes = ['.png', '.pdf'], cmap = 'nipy_spectral', crossout_outliers = False):
        """
        Create a PCA plot with colors based on one column and markers based on another column from the metadata.

        Parameters
        ----------
        data : DataFrame
            DataFrame with the data.
        report : pdf_reporter.Report object
            Report object to add information to.
        metadata : DataFrame
            DataFrame with the metadata.
        color_column : str
            Name of the column to use for coloring. ! If you input 'None', then all will be grey. (From metadata, e.g. 'Age', etc.)
        marker_column : str
            Name of the column to use for markers. ! If you input 'None', then all markers will be same. (From metadata e.g. 'Diagnosis', etc.)
        output_file_prefix : str
            Prefix for the output file.
        sufixes : list
            List of sufices for the image files to be saved in. Default is ['.png', '.pdf'].
        cmap : str
            Name of the colormap. Default is 'nipy_spectral'. (Other options are 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted', 'turbo'; ADD '_r' to get reversed colormap)
        """
        if self.pca_data is None:
            raise ValueError('PCA was not performed yet. Run PCA first.')
        else:
            pca_data = self.pca_data
            per_var = self.pca_per_var
        ## PCA for both (with different markers)
        column_name = color_column
        second_column_name = marker_column

        cmap = mpl.colormaps[cmap]

        if crossout_outliers:
            pca_data['PC1_zscore'] = zscore(pca_data['PC1'])
            pca_data['PC2_zscore'] = zscore(pca_data['PC2'])

            # Identify outliers as any points where the absolute z-score is greater than 3
            outliers = pca_data[(np.abs(pca_data['PC1_zscore']) > 3) | (np.abs(pca_data['PC2_zscore']) > 3)]
        

        #get the color (for first column)
        if column_name is not None:
            column_unique_values = metadata[column_name].unique()
            num_unique_values = len(column_unique_values)
            colors = [cmap(i/num_unique_values) for i in range(num_unique_values)]
            class_type_colors = dict(zip(column_unique_values, colors))

        #get markers (for second column)
        if second_column_name is not None:
            second_column_unique_values = metadata[second_column_name].unique()
            markers = cycle(['o', 'v', 's', '^', '*', '<','p', '>', 'h', 'H', 'D', 'd', 'P', 'X'])
            class_type_markers = dict(zip(second_column_unique_values, markers))


        legend_elements = []
        if column_name is not None and second_column_name is not None:
            existing_combinations = set(zip(metadata[column_name], metadata[second_column_name]))
        elif column_name is not None:
            existing_combinations = set(metadata[column_name])
        elif second_column_name is not None:
            existing_combinations = set(metadata[second_column_name])
        else:
            raise ValueError('At least one of the columns must be specified. Either color_column or marker_column or both.')

        fig, ax = plt.subplots(figsize=(10, 8))

        # Iterate over unique combinations of the two columns
        for combination in existing_combinations:
            # Check if combination is a tuple (both columns provided) or single value (one column provided)
            if isinstance(combination, tuple):
                sample_type, class_type = combination
            else:
                # Assign the single value to the appropriate variable based on which column is provided
                if column_name is not None:
                    sample_type = combination
                    class_type = None  # Default or placeholder value
                else:
                    sample_type = None  # Default or placeholder value
                    class_type = combination

            # Adjust the selection logic to handle cases where one of the types is None
            if sample_type is not None and class_type is not None:
                df_samples = pca_data[(metadata[column_name] == sample_type) & (metadata[second_column_name] == class_type)]
            elif sample_type is not None:
                df_samples = pca_data[metadata[column_name] == sample_type]
            elif class_type is not None:
                df_samples = pca_data[metadata[second_column_name] == class_type]
            
            # Get the color and marker, handling cases where one of the types is None
            color = 'grey' if sample_type is None else class_type_colors.get(sample_type, 'grey')
            marker = 'o' if class_type is None else class_type_markers.get(class_type, 'o')
            
            # Construct label based on available types
            label = f"{sample_type or ''} - {class_type or ''}".strip(' -')
    
            # Plot the samples
            plt.scatter(df_samples['PC1'], df_samples['PC2'], color=color, marker=marker, label=sample_type + ' - ' + class_type, alpha=0.6)
            
            # Compute the covariance matrix and find the major and minor axis 
            covmat = np.cov(df_samples[['PC1', 'PC2']].values.T)
            lambda_, v = np.linalg.eig(covmat)
            lambda_ = np.sqrt(lambda_)
            
            # Draw an ellipse around the samples
            ell = Ellipse(xy=(np.mean(df_samples['PC1']), np.mean(df_samples['PC2'])),
                        width=lambda_[0]*2, height=lambda_[1]*2,
                        angle=np.rad2deg(np.arccos(v[0, 0])), edgecolor=color, lw=2, facecolor='none')
            plt.gca().add_artist(ell)

            # Add the legend element
            color = class_type_colors[sample_type]
            marker = class_type_markers[class_type]
            legend_elements.append(Line2D([0], [0], marker=marker, color='w', label=class_type + ' - ' + sample_type,
                                        markerfacecolor=color, markersize=10, alpha=0.6))

        if crossout_outliers:
            # Plot the outliers
            plt.scatter(outliers['PC1'], outliers['PC2'], color='black', marker='x', label='Outliers', alpha=0.6)

        # Create the legend
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        plt.title('PCA Graph')
        plt.xlabel('PC1 - {0}%'.format(per_var[0]))
        plt.ylabel('PC2 - {0}%'.format(per_var[1]))
        name = output_file_prefix + 'detailed_PCA'
        for sufix in sufixes:
            plt.savefig(name + sufix, bbox_inches='tight', dpi = 300)
        plt.show()
        #---------------------------------------------
        # REPORTING
        if column_name is not None and second_column_name is not None:
            text = 'Detailed PCA plot based on ' + column_name + '(colors) and ' + second_column_name + '(markers) was created and added into: ' + name
        elif column_name is not None:
            text = 'Detailed PCA plot based on ' + column_name + '(colors) was created and added into: ' + name
        elif second_column_name is not None:
            text = 'Detailed PCA plot based on ' + second_column_name + '(markers) was created and added into: ' + name
        report.add_together([
            ('text', text),
            ('image', name),])
        return fig, ax
