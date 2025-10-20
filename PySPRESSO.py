# General modules
import pandas as pd
import numpy as np
import os
import re
import time
import math
import json
import pickle
import warnings
# warnings.filterwarnings(
#     "ignore",
#     message=".*maximal number of iterations maxit.*s too small.*",
#     category=UserWarning
# )
from itertools import cycle, combinations

# Plotting modules
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.patches import Ellipse 
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from adjustText import adjust_text

# Statistics and ML modules
from scipy.stats import zscore, linregress, gaussian_kde, ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_predict, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score


# My modules
import pdf_reporter.pdf_reporter as pdf_rptr #since repo is a folder, now you can simply clone it

# |--------------------------------------------------------------------------------------------|
# |  PySPRESSO - (Python Statistical Processing and REporting for Scientific Studies in Omics) |
# |--------------------------------------------------------------------------------------------|

class Workflow: # WORKFLOW for Peak Matrix Filtering (and Correcting, Transforming, Normalizing, Statistics, etc.)
    """
    Class that contains all the methods to filter, correct and transform the peak matrix data.
    """

    def __init__(self):

        self.functions =  [] # List of functions to be applied in the order they are added
        self.name = 'Peak Matrix Filtering Workflow'
        # Data variables (names, paths, etc.)
        self.report = None
        self.report_file_name = self.name
        self.main_folder = self.name
        self.output_file_prefix = self.name
        self.suffixes = ['.png', '.pdf']

        # Data variables
        self.saves_count = 0 # Variable to keep track of the number of saves (to avoid overwriting)
        self.data = None
        self.variable_metadata = None
        self.metadata = None
        self.batch_info = None
        self.batch = None
        self.QC_samples = None
        self.blank_samples = None
        self.standard_samples = None
        self.dilution_series_samples = None
        self.dil_concentrations = None

        # Statistics variables
        self.pca_count = 0 # Variable to keep track of the number of PCA runs (to avoid overwriting) 
        self.pca = None
        self.pca_data = None
        self.pca_df = None
        self.pca_per_var = None
        self.pca_loadings = None
        self.pca_loadings_candidates = None
        
        self.fold_change = None
        self.fold_change_count = 0 # Variable to keep track of the number of fold_change runs (to avoid overwriting)

        self.plsda = None
        self.plsda_metadata = None
        self.plsda_response_column = None

        # Candidate features (contains significant features based on different methods - features that might be interesting for further analysis - e.g. biomarkers)
        self.candidates = pd.DataFrame(columns = ['feature', 'method', 'specifications', 'score', 'hits'])

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ALL THE FUNCTIONALITY TO INTERACT WITH WORKFLOW
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def show(self):
        """
        Print the list of functions to be applied.
        """
        print("---------------------------------")
        print("Workflow - " + self.name + "  - functions:")
        for i, (func, args, kwargs) in enumerate(self.functions):
            print(f"{i} - {func.__name__} with args: {args} and kwargs: {kwargs}")
        print("---------------------------------")

    def add_function(self, func_name, *args, **kwargs):
        """
        Add a method to the workflow.

        Parameters
        ----------
        func_name : str
            The name of the method to add to the workflow.
        args : tuple
            Positional arguments to pass to the method.
        kwargs : dict
            Keyword arguments to pass to the method.
        """
        if hasattr(self, func_name):
            func = getattr(self, func_name)
            self.functions.append((func, args, kwargs))
            print(f"Method {func_name} added to the workflow.")
        else:
            print(f"Method {func_name} does not exist in the workflow.")

    def run(self):
        """
        Run all methods in the workflow in the order they were added.
        """
        for func, args, kwargs in self.functions:
            print(f"Running {func.__name__}...")
            func(*args, **kwargs)

    def save_workflow(self, filename):
        """
        Save the workflow to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the workflow to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.functions, f)
        print(f"Workflow saved to {filename}.")

    def load_workflow(self, filename):
        """
        Load the workflow from a file.

        Parameters
        ----------
        filename : str
            The name of the file to load the workflow from.
        """
        with open(filename, 'rb') as f:
            self.functions = pickle.load(f)
        print(f"Workflow loaded from {filename}.")
        
    def rename(self, name):
        """
        Rename the workflow.

        Parameters
        ----------
        name : str
            New name for the workflow.
        """
        self.name = name
        print("Workflow renamed to: " + name)

    def set_name(self, name):
        """
        Set the name of the workflow.

        Parameters
        ----------
        name : str
            Name of the workflow.
        """
        self.name = name
        print("Workflow name set to: " + name)

    def set_report_file_name(self, report_file_name):
        """
        Set the report file name.

        Parameters
        ----------
        report_file_name : str
            Name of the report file.
        """
        self.report_file_name = report_file_name
        print("Report file name set.")
        
    def set_main_folder(self, main_folder):
        """
        Set the main folder.

        Parameters
        ----------
        main_folder : str
            Path to the main folder.
        """
        self.main_folder = main_folder
        print("Main folder set.")
    
    def set_output_file_prefix(self, output_file_prefix):
        """
        Set the output file prefix.

        Parameters
        ----------
        output_file_prefix : str
            Prefix for the output files.
        """
        self.output_file_prefix = output_file_prefix
        print("Output file prefix set.")
    
    def set_suffixes(self, suffixes):
        """
        Set the suffixes.

        Parameters
        ----------
        suffixes : list
            List of suffixes.
        """
        self.suffixes = suffixes
        print("suffixes set.")

    # SAMPLE LABELS ---------------------------------------------------------
    def set_batch(self, batch):
        """
        Set the batch.

        Parameters
        ----------
        batch : list
            List of batches: ['Batch1', 'Batch1', 'Batch1', 'Batch2', 'Batch2', ...]
        """
        if batch is None:
            # if cpID exists in the data 
            if 'cpdID' in self.data.columns:
                batch = ['all_one_batch' for i in range(len(self.data.columns)-1)] # Dummy batch information (for cases where its all the same batch) #-1 because of cpdID
            else: # if called before cpdID is added
                batch = ['all_one_batch' for i in range(len(self.data.columns))] # Dummy batch information (for cases where its all the same batch)

        self.batch = batch
        print("Batch set.")

    def set_QC_samples(self, QC_samples):
        """
        Set the QC samples.

        Parameters
        ----------
        QC_samples : list
            List of QC samples.
        """
        self.QC_samples = QC_samples
        print("QC samples set.")
    
    def set_dilution_series_samples(self, dilution_series_samples):
        """
        Set the dilution series samples.

        Parameters
        ----------
        dilution_series_samples : list
            List of dilution series_samples.
        """
        self.dilution_series_samples = dilution_series_samples
        print("Dilution series set.")

    def set_dilution_series_samples_by_name(self, dil_distinguisher = 'dQC', conc_distinguisher = None):
        """
        Set the dilution series samples by name.

        Parameters
        ----------
        dil_distinguisher : str
            Distinguisher used in the file names (columns in data) for dilution series. Default is 'dQC'. If distinguisher is in the name, then it is considered as dilution series sample.
        conc_distinguisher : str
            Distinguisher used in the file names (columns in data) before the concentration value is stated. (can be the same as dil_distinguisher if the value is right after it).
        """
        data = self.data
        dilution_series_samples = []
        concentrations = []
        # Look through values (names) in the specified column
        for name in data.columns[1:]:
            # find the distinguisher in the name
            if dil_distinguisher in name:
                dilution_series_samples.append(name)
            if conc_distinguisher is not None:
                # find the numeric value behind conc_distinguisher
                if conc_distinguisher in name:
                    pattern  = re.escape(conc_distinguisher) + r'(\d+)' #find the number after the conc_distinguisher
                    m = re.search(pattern, name)
                    if not m:
                        raise ValueError("No concentration value found in the name: " + name + " after the conc_distinguisher: " + conc_distinguisher)
                    else:
                        value = m.group(1)
                    concentrations.append(float(value))
        
        self.dilution_series_samples = dilution_series_samples
        self.dil_concentrations = concentrations
        print("Dilution series samples set.")

    def set_dil_concentrations(self, dil_concentrations):
        """
        Set the dilution concentrations.

        Parameters
        ----------
        dil_concentrations : list
            List of dilution concentrations.
        """
        self.dil_concentrations = dil_concentrations
        print("Dilution concentrations set.")

    def set_blank_samples(self, blank_samples):
        """
        Set the blank samples.

        Parameters
        ----------
        blank_samples : list
            List of blank samples.
        """
        self.blank_samples = blank_samples
        print("Blank samples set.")

    def set_standard_samples(self, standards):
        """
        Set the standards.

        Parameters
        ----------
        standards : list
            List of standards.
        """
        self.standard_samples = standards
        print("Standards set.")

    #-------------------------
    def initialize(self):
        """
        Perform all the initialization steps of the workflow.
        """
        self.report = self.initializer_report(self.main_folder, self.report_file_name)
        self.functions = [self.loader_data, self.add_cpdID, self.extracter_variable_metadata, self.extracter_data, self.loader_batch_info, self.batch_by_name_reorder, self.extracter_metadata]
        print("Workflow initialized.")
    
    def function_change_parameters(self, index, **kwargs):
        """
        Change the parameters of the function at the given index.

        Parameters
        ----------
        index : int
            Index of the function.
        kwargs : dict
            Dictionary with the parameters to be changed.
        """
        self.functions[index](**kwargs)
        print("Parameters of the function at index " + str(index) + " changed.")

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ALL DATA HANDLING AND INITILAZITAION METHODS (like: loading, saving, initializing-report, etc.) (keywords like: loader_..., extracter_..., saver_..., but also without keywords)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def _natural_sort_key(self, s):
        if isinstance(s, tuple):
            # Convert tuple to a string in the format "A-B"
            s = '-'.join(map(str, s))
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]
       
    def initializer_report(self, report_type = 'processing'):
        """
        Initialize the report object.

        Parameters
        ----------
        report_type : str
            Settings of the report (title). Default is 'processing'. All options are 'processing', 'statistics'. (To add more options, add them to the pdf_reporter.py file)
        """
        main_folder = self.main_folder
        report_file_name = self.report_file_name

        report_path = main_folder + '/' + report_file_name + '.pdf'
        title_text = main_folder

        report = pdf_rptr.Report(name = report_path, title = title_text)
        report.initialize_report(report_type)
        
        self.report = report

        print("Report initialized.")

        return self.report
    
    def initializer_folders(self):
        """
        Initialize the folders for the data and the report.
        """
        main_folder = self.main_folder

        # Initialize the folders for the data and the report
        if not os.path.exists(main_folder):
            os.makedirs(main_folder)
        if not os.path.exists(main_folder + '/figures'):
            os.makedirs(main_folder + '/figures')
        if not os.path.exists(main_folder + '/statistics'):
            os.makedirs(main_folder + '/statistics')

        print("Folders initialized.")

        return main_folder

    def loader_data(self, data_input_file_name, separator = ';', encoding = 'ISO-8859-1'):
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
        report = self.report

        # Load the data from the csv file
        self.data = pd.read_csv(data_input_file_name, sep = separator,  encoding=encoding )
        print("Data loaded.")
        #---------------------------------------------
        # REPORTING
        text = 'Data were loaded from: ' + data_input_file_name + ' (data from Compound Discoverer).'
        report.add_together([('text', text), 
                            ('table', self.data),
                            'line'])
        return self.data
    
    def add_cpdID(self, mz_col = 'm/z', rt_col = 'RT [min]', round_mz_col = 5, round_rt_col = 3):
        """
        Add a column 'cpdID' to the data calculated from mz and RT. Rounding up the numbers to specified given of numbers. Matching ID - distinguished by adding "_1", "_2", etc.

        Parameters
        ----------
        mz_col : str
            Name of the column with m/z values. Default is 'm/z'.
        rt_col : str
            Name of the column with RT values. Default is 'RT [min]'.
        round_mz_col : int
            Number of decimal places to round the m/z column. Default is 5. (Using 5 aligns with Compound Discoverer visualization.)
        round_rt_col : int
            Number of decimal places to round the RT column. Default is 3. (Using 3 aligns with Compound Discoverer visualization.)
        """
        data = self.data
        report = self.report

        #add cpdID column to the data_df calculated from mz and RT
        data['cpdID'] = 'M' + data[mz_col].round(round_mz_col).astype(float).astype(str) + '-T' + (data[rt_col]).round(round_rt_col).astype(float).astype(str)
        # Create a new column 'Duplicate_Count' to keep track of the duplicates
        data['Duplicate_Count'] = data.groupby('cpdID').cumcount()
        # Create a mask to identify duplicated compounds
        is_duplicate = data['Duplicate_Count'] > 0
        # Update the 'Compound_ID' column for duplicated compounds
        data.loc[is_duplicate, 'cpdID'] = data['cpdID'] + '_' + (data['Duplicate_Count']).astype(str)
        # Drop the 'Duplicate_Count' 
        data.drop('Duplicate_Count', axis=1, inplace=True)
        #check if there are rly no duplicates left
        #remaining_duplicates = data[data['cpdID'].duplicated(keep=False)]
        #print(remaining_duplicates.empty)
        print("Compound ID was added to the data")
        
        #---------------------------------------------
        # REPORTING
        text = 'cpdID column was added to the data calculated from mz and RT. Matching ID - distinguished by adding "_1", "_2", etc.'
        report.add_together([('text', text),
                                'line'])
        return self.data

    def extracter_variable_metadata(self, column_index_ranges = [(10, 15), (18, 23)]):
        """
        Extract variable metadata from the data. Based on the column index ranges. 

        Parameters
        ----------
        column_index_ranges : list
            List of tuples with the column index ranges to keep. Default is [(10, 15), (18, 23)].
        """
        data = self.data
        report = self.report

        self.variable_metadata = data[['cpdID', 'Name', 'Formula']]
        for column_index_range in column_index_ranges:
            self.variable_metadata = self.variable_metadata.join(data.iloc[:, column_index_range[0]:column_index_range[1]])

        print("Variable metadata was extracted from the data.")
        #---------------------------------------------
        # REPORTING
        text = 'variable-metadata matrix was created.'
        report.add_together([('text', text),
                                'line'])
        return self.variable_metadata
    
    def extracter_data(self, prefix = 'Area:'):
        """
        Extract data matrix from the data_df.

        Parameters
        ----------
        prefix : str
            Prefix used in the column names to keep. Default is 'Area:'.
        """
        data = self.data
        report = self.report

        temp_data = data.copy()
        data = temp_data[['cpdID']]
        # Select columns with names starting with "Area:"
        area_columns = temp_data.filter(regex=r'^' + prefix, axis=1) #for those starting with Area:, for those including Area anywhere, you can use like='Area:'
        data = data.join(temp_data[area_columns.columns])
        del temp_data
        # Fill NaN values with zero (NaN values are problem when plotted, and we will treat 0s as missing values later)
        data.fillna(0, inplace=True)  # Replace NaN with 0
        print("Important columns were kept in the data and rest filtered out.")
        
        #---------------------------------------------
        # REPORTING
        text = 'data matrix was created.'
        report.add_together([('text', text),
                                'line'])
        self.data = data
        return self.data
    
    def loader_batch_info(self, input_batch_info_file, separator = ';', encoding = 'ISO-8859-1'):
        """
        Load batch_info matrix from a csv file.

        Parameters
        ----------
        input_batch_info_file : str
            Name of the batch info file.
        separator : str
            Separator used in the data file. Default is ';'.
        encoding: str
            Encoding used in the data file. Default is 'ISO-8859-1'. (UTF-8 is also common.)
        """
        report = self.report

        #but better will be get it straight from the xml file (will do later)
        self.batch_info = pd.read_csv(input_batch_info_file, sep = separator , encoding=encoding)
        print("Batch info loaded.")

        #---------------------------------------------
        # REPORTING
        text = 'batch_info matrix was loaded from: ' + input_batch_info_file
        report.add_together([('text', text),
                                'line'])
        return self.batch_info

    def batch_by_name_reorder(self, distinguisher = 'Batch', format='%d.%m.%Y %H:%M'):
        """
        Reorder data based on the creation date and add a column 'Batch' to the batch_info based on the distinguisher.
        
        Parameters
        ----------
        distinguisher : str
            Distinguisher used in the file names. Default is 'Batch'. Batches are distinguished in the File Name *_distinguisher_XXX. If you have all in one batch and no such distinguisher, then use None.
        format : str
            Format of the date. Default is '%d.%m.%Y %H:%M'.
        """
        data = self.data
        report = self.report
        batch_info = self.batch_info

        #Batches are distinguished in the File Name *_distinguisher_XXX
        batches_column = []

        names = data.columns[1:].to_list()
        batch_names = batch_info['File Name'].tolist()

        # Convert the 'Creation Date' column to datetime format
        batch_info['Creation Date'] = pd.to_datetime(batch_info['Creation Date'], format = format) #, dayfirst=True
    
        # Sort the DataFrame based on the 'Creation Date' column
        batch_info = batch_info.sort_values('Creation Date')
        batch_info = batch_info.reset_index(drop=True)

        if distinguisher is None:
            # If all in one batch, use None
            batch_info['Batch'] = ['all_one_batch' for i in range(len(self.batch_info.index))] # Dummy batch information (for cases where its all the same batch) (not columns
            
            #---------------------------------------------
            #REPORTING
            text = 'All samples are in one batch.'
            report.add_together([('text', text),
                                'line'])
        else:
            for name in batch_names:
                split_name = re.split(r'[_|\\]', name) #split by _ or \
                for i, part in enumerate(split_name):
                    if part == distinguisher:
                        batches_column.append(split_name[i+1])
                        break

            if len(batches_column) == 0:
                raise ValueError("No matches found in 'names' for the distinguisher: " + distinguisher + " in the file names. If you have all in one batch, use None")
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

        print("New data order based on batch info:")
        print(new_data_order)

        print("Data reordered based on the creation date from batch info.")
        print("Not found: " + str(len(not_found)) + " ;being: " + str(not_found))
        print("Names not identified: " + str(len(names)) + " ;being: " + str(names))

        # Reorder data
        data = data[['cpdID'] + new_data_order]

        batch_info = batch_info.drop(not_found_indexes)
        batch_info = batch_info.reset_index(drop=True)

        self.batch_info = batch_info
        self.data = data

        #---------------------------------------------
        #REPORTING
        text0 = 'Batch information (creation date) was used to (re)order samples.'
        if distinguisher is None:
            text1 = 'All samples are in one batch. Thus "all_one_batch" was used as a batch information.'
        else:
            text1 = 'Batches are distinguished in the File Name *_' + distinguisher + '_XXX.'
        text2 = 'Not found: ' + str(len(not_found)) + ' ;being: ' + str(not_found)
        text3 = 'Names not identified: ' + str(len(names)) + ' ;being: ' + str(names)
        report.add_together([('text', text0),
                            ('text', text1, 'italic'),
                            ('text', text2, 'italic'),
                            ('text', text3, 'italic'),
                            ('table', batch_info),
                            'line'])

        return self.data, self.batch_info

    def extracter_metadata(self, group_columns_to_keep, prefix = 'Area:'):
        """
        Extract metadata from batch_info by choosing columns to keep. Keep your grouping columns such as AgeGroup, Sex, Diagnosis, etc. 

        Parameters
        ----------
        group_columns_to_keep : list
            List of group columns to keep.
        prefix : str
            Prefix used in the column names to keep. Default is 'Area:'.
        """
        data = self.data
        report = self.report
        batch_info = self.batch_info

        # Define columns to keep in the metadata matrix
        always_keep_colunms = ['Study File ID','File Name', 'Creation Date','Sample Type', 'Polarity', 'Batch']
        columns_to_keep = always_keep_colunms + group_columns_to_keep
        # Extract metadata from batch_info
        regex_pattern = re.compile(r'^'+ prefix)
        area_columns = data.filter(regex = regex_pattern, axis=1)
        metadata = batch_info[columns_to_keep].copy()
        metadata['Sample File'] = area_columns.columns
        self.metadata = metadata
        #---------------------------------------------
        # REPORTING
        text = 'metadata matrix was created from batch_info by choosing columns: ' + str(columns_to_keep) + '.'
        report.add_together([('text', text),
                                'line'])
        return self.metadata

    def saver_all_datasets(self, version_name = None, versionless = False):
        """
        Save all the processed data into files.

        Parameters
        ----------
        version_name : str
            Name of the version. Default is None. (If None, the version number is added to the file name to avoid overwriting the files if user wants to save multiple times, during workflow.)
        versionless : bool
            If True, save the files without version name/number. Default is False. 
        """
        data = self.data
        variable_metadata = self.variable_metadata
        metadata = self.metadata
        report = self.report
        main_folder = self.main_folder
        output_file_prefix = self.output_file_prefix
        candidates = self.candidates
        saves_count = self.saves_count + 1

        if versionless:
            version = ''
        elif version_name == None:
            version = '-v' + str(saves_count)
        else:
            if not isinstance(version_name, (str, int)):
                raise ValueError("Version name has to be a string or an int.")
            version_name = str(version_name)
            if version_name[0] not in "_-":
                version = "_" + version_name
            else:
                version = version_name
        
        #data
        data = data.reset_index(drop=True)
        data_name = main_folder + '/' + output_file_prefix + '-data' + version + '.csv'
        data.to_csv(data_name, sep = ';')

        #variable_metadata
        variable_metadata = variable_metadata.reset_index(drop=True)
        variable_metadata_name = main_folder + '/' + output_file_prefix + '-variable_metadata' + version + '.csv'
        variable_metadata.to_csv(variable_metadata_name, sep = ';')

        #metadata
        metadata = metadata.reset_index(drop=True)
        metadata_name = main_folder + '/' + output_file_prefix + '-metadata' + version + '.csv'
        metadata.to_csv(metadata_name, sep = ';')

        #fold_change
        if self.fold_change is not None:
            fold_change = self.fold_change.reset_index(drop=True)
            fold_change_name = main_folder + '/' + output_file_prefix + '-fold_change' + version + '.csv'
            fold_change.to_csv(fold_change_name, sep = ';')
        else:
            fold_change_name = 'No fold_change data to save.'
        
        #candidates
        if self.candidates is not None:
            candidates = candidates.reset_index(drop=True)
            candidates_name = main_folder + '/' + output_file_prefix + '-candidates' + version + '.csv'
            candidates.to_csv(candidates_name, sep = ';')
        else:
            candidates_name = 'No candidates data to save.'

        self.saves_count = saves_count

        print("All datasets were saved.")
        #---------------------------------------------
        #REPORTING
        text0 = 'Processed data was saved into a files named (path) as follows:'
        data_list = [data_name, variable_metadata_name, metadata_name, fold_change_name, candidates_name]
        text1 = ''
        for file in data_list:
            text1 += '<br/>' + file
        report.add_together([('text', text0, 'bold'),
                            ('text', text1, 'italic', 'center'),
                            'line'])
        return True
    
    def add_candidates(self, features, method, specification, scores):
        """
        Add candidate features to the candidates list.

        Parameters
        ----------
        features : list or np.ndarray or pd.Series
            List of features to add.
        method : str
            Method used to identify the features.
        specification : str
            Specification of the method - columns used in PLSDA, etc.
        scores : list or np.ndarray or pd.Series
            Scores of the features - PCA loading, PLSDA VIP, etc.
        """
        variable_metadata = self.variable_metadata
     
        # Ensure features and scores are iterable
        if not isinstance(features, (list, pd.Series, np.ndarray)):
            features = [features]
        if not isinstance(scores, (list, pd.Series, np.ndarray)):
            scores = [scores]


        # Repeat method and specification to match the length of features
        if isinstance(method, list) and len(method) != len(features):
            raise ValueError("Length of method list must match length of features.")
        if isinstance(specification, list) and len(specification) != len(features):
            raise ValueError("Length of specification list must match length of features.")
        
        if isinstance(method, str) or (isinstance(method, (list, pd.Series, np.ndarray)) and len(method) == 1):
            method = [method] * len(features)
        if isinstance(specification, str) or (isinstance(specification, (list, pd.Series, np.ndarray)) and len(specification) == 1):
            specification = [specification] * len(features)

        hits = [0] * len(features)  # Initialize hits to 0 for all features

        # Initialize self.candidates if it is None
        if self.candidates is None:
            self.candidates = pd.DataFrame(columns=['feature', 'method', 'specification', 'score', 'hits', 'Name', 'Formula', 'Annot. DeltaMass [ppm]', 'Annotation MW'])
        else:
            # Find features in variable_metadata - Name, Formula, ... (Matching is done by cpdID)
            names = variable_metadata[variable_metadata['cpdID'].isin(features)]['Name'].tolist()
            formulas = variable_metadata[variable_metadata['cpdID'].isin(features)]['Formula'].tolist()
            annotdeltamass = variable_metadata[variable_metadata['cpdID'].isin(features)]['Annot. DeltaMass [ppm]'].tolist()
            annotation_mw = variable_metadata[variable_metadata['cpdID'].isin(features)]['Annotation MW'].tolist()


        # Create a DataFrame for the new candidates
        new_candidates = pd.DataFrame({
            'feature': features,
            'method': method,
            'specification': specification,
            'score': scores,
            'hits': hits,
            'Name': names,
            'Formula': formulas,
            'Annot. DeltaMass [ppm]': annotdeltamass,
            'Annotation MW': annotation_mw
        })

        # Handle the case where self.candidates is empty
        if self.candidates.empty:
            self.candidates = new_candidates
        else:
            # Concatenate the new candidates with the existing ones
            self.candidates = pd.concat([self.candidates, new_candidates], ignore_index=True)

        # Calculate hits for the candidates
        self.candidates = self._candidates_calculate_hits()

        # Order the candidates
        self.candidates = self._candidates_order(how='method')

        return self.candidates
    
    def get_candidates_indexes(self):
        """
        Get the indexes of all the candidates. (For example to use them to create violin plots.)

        """
        candidates = self.candidates
        data = self.data

        # Get the indexes of the candidates
        if candidates is None:
            raise ValueError("No candidates found yet, they are identified during PCA, PLSDA, fold_change, etc.")
        else:
            # Find the indexes of the candidates in the data
            indexes = []
            for feature in candidates['feature']:
                # Find the index of the feature in the data
                index = data[data['cpdID'] == feature].index.tolist()
                indexes.append(index[0])
        return indexes

    def _candidates_calculate_hits(self):
        """
        Help function to calculate hits for the candidates.

        """
        # Calculate hits for the candidates
        hits = []
        for feature in self.candidates['feature']:
            # Count the number of times the feature appears in the candidates list
            hit = self.candidates[self.candidates['feature'] == feature].shape[0]
            hits.append(hit)
        self.candidates['hits'] = hits
        return self.candidates
    
    def _candidates_order(self, how = 'method'):
        """
        Help function to order the candidates.

        Parameters
        ----------
        how : str
            How to order the candidates. Default is 'method'. Other option is 'hits'.

        """
        # Order the candidates by method and score
        if how == 'method':
            self.candidates = self.candidates.sort_values(by=['method', 'specification', 'score', 'hits'], ascending=[False, False, False, False])
        # Order the candidates by hits and method
        elif how == 'hits':
            self.candidates = self.candidates.sort_values(by=['hits', 'method', 'specification', 'score'], ascending=[False, False, False, False])

        # Reset the index
        self.candidates.reset_index(drop=True, inplace=True)
        
        return self.candidates

    def finalizer_report(self):
        """
        Finalize the report. (Has to be called at the end of the script otherwise the pdf file will be "corrupted".)

        """
        self.report.finalize_report()
        print("Report was finalized and is ready for viewing.")
        return self.report
    
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
        return variable_metadata

    def filter_missing_values(self, qc_threshold = 0.8, sample_threshold = 0.5):
        """
        Filter out features with high number of missing values (nan or 0): A) within QC samples  B) all samples 

        Parameters
        ----------
        qc_threshold : float
            Threshold for the QC samples. Default is 0.8.
        sample_threshold : float
            Threshold for all samples. Default is 0.5.
        """ 
        data = self.data
        report = self.report
        QC_samples = self.QC_samples

        is_qc_sample = [True if col in QC_samples else False for col in data.columns[1:]]
        #A) WITHIN QC SAMPLES ---------------------------------------------
        #if value is missing (nan or 0) in more than threshold% of QC samples, then it is removed
        QC_number_threshold = int(sum(is_qc_sample)*qc_threshold)

        # Identify rows with missing (either missing or 0) values
        QC_missing_values = (data[data.columns[1:][is_qc_sample]].isnull() | (data[data.columns[1:][is_qc_sample]] == 0)).sum(axis=1)
        #print(missing_values[missing_values > 0])

        #Filter out rows with too much (over the threshold) missing values (either missing or 0)
        data = data.loc[QC_missing_values < QC_number_threshold, :]
        data

        #report how many features were removed
        print("Number of features removed for QC threshold (" + str(qc_threshold) + "%): within QC samples: "+ str(len(QC_missing_values[QC_missing_values > QC_number_threshold])) + " ;being: " + str(QC_missing_values[QC_missing_values > QC_number_threshold].index.tolist()))

        #B) ACROSS ALL SAMPLES ---------------------------------------------
        #if value is missing (nan or 0) in more than threshold% of samples, then it is removed
        number_threshold = int(len(data.columns[1:])*sample_threshold)

        # Identify rows with missing (either missing or 0) values
        missing_values = (data.isnull() | (data == 0)).sum(axis=1)
        #print(missing_values[missing_values > 0])

        #Filter out rows with too much (over the threshold) missing values (either missing or 0)
        data = data.loc[missing_values < number_threshold, :]
        
        #reset index
        data = data.reset_index(drop=True)
        #update data and variable_metadata
        self.data = data
        self.variable_metadata = self._filter_match_variable_metadata(data, self.variable_metadata)

        #report how many features were removed
        print("Number of features removed for sample threshold (" + str(sample_threshold) + "%): within all samples: "+ str(len(missing_values[missing_values > number_threshold])) + " ;being: " + str(missing_values[missing_values > number_threshold].index.tolist()))

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

        return self.data

    def filter_blank_intensity_ratio(self, ratio = 20, setting = 'first'):
        """
        Filter out features with intensity sample/blank < ratio (default = 20/1).

        Parameters
        ----------
        ratio : float
            Ratio for the intensity sample/blank. Default is 20. 
        setting : str
            Setting used to calculate choose intensity of the blank. Default is 'first'. Other options are 'median', 'min', 'mean', 'first' or 'last'.
        """
        data = self.data
        variable_metadata = self.variable_metadata
        report = self.report
        QC_samples = self.QC_samples
        blank_samples = self.blank_samples
        
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
        
        #Filter out features (compounds) with intensity sample/blank < blank_threshold
        data = data[intensity_sample_blank >= blank_threshold]
        
        #reset index
        data = data.reset_index(drop=True)
        #update data and variable_metadata
        self.data = data
        self.variable_metadata = self._filter_match_variable_metadata(data, variable_metadata)
        
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
    
    def filter_relative_standard_deviation(self, rsd_threshold = 20, to_plot = False):
        """
        Filter out features with QC samples with RSD% (relative standard deviation) over the threshold.

        Parameters
        ----------
        rsd_threshold : float
            Threshold for the RSD%. Default is 20.
        to_plot : bool or int
            If True, plot some of the compounds with high RSD. Default is False. (For True plots 4, for any other int, plots that amount (or all if there is less then inputed integer))
        """
        data = self.data
        variable_metadata = self._filter_match_variable_metadata(data, self.variable_metadata)
        report = self.report
        QC_samples = self.QC_samples

        #QC_samples mask
        is_qc_sample = [True if col in QC_samples else False for col in data.columns[1:]]
        
        #Calculate RSD for only QC samples
        qc_rsd = data[data.columns[1:][is_qc_sample]].std(axis=1)/data[data.columns[1:][is_qc_sample]].mean(axis=1)*100
        qc_rsd = qc_rsd.copy()

        #Add RSD into the data
        variable_metadata['QC_RSD'] = qc_rsd

        #Filter out features (compounds) with RSD > rsd_threshold
        over_threshold = qc_rsd > rsd_threshold
        data = data[~over_threshold]

        #Plot some of the compounds with high RSD
        if to_plot == False:
            number_plotted = 1 #this one will be only in report, but not shown or saved separately
        elif to_plot == True:
            number_plotted = 4
        elif type(to_plot) == int:
            number_plotted = int(to_plot)
        else:
            raise ValueError("to_plot has to be either True or an integer.")
        indexes = qc_rsd[qc_rsd > rsd_threshold].index.tolist()[:number_plotted]
        if len(indexes) < number_plotted:
            number_plotted = len(indexes)
        if len(indexes)  == 0:
            print("No compounds with RSD > " + str(rsd_threshold) + " were found.")
        else:   
            # plot only the QC samples
            for i in range(number_plotted):
                plt.figure(figsize=(10, 6))
                plt.scatter(range(len(data.iloc[indexes[i], 1:][is_qc_sample])),
                            data.iloc[indexes[i], 1:][is_qc_sample], 
                            label=data.iloc[indexes[i], 0], 
                            s=10, alpha=0.5)
                plt.xlabel('Samples in order')
                plt.ylabel('Peak Area')
                plt.title("High RSD compound: cpID = " + data.iloc[indexes[i], 0])
                for suffix in self.suffixes:
                    plt.savefig(self.main_folder + '/figures/QC_samples_scatter_' + str(indexes[i]) + '_high_RSD-deleted_by_correction' + suffix, dpi=400, bbox_inches='tight')
                plt.show()
        
        #reset index
        data = data.reset_index(drop=True)
        #update data and variable_metadata
        self.data = data
        self.variable_metadata = self._filter_match_variable_metadata(data, variable_metadata)

        #report how many features were removed
        print("Number of features removed: " + str(len(qc_rsd[qc_rsd > rsd_threshold])) + " ;being: " + str(qc_rsd[qc_rsd > rsd_threshold].index.tolist()))

        #---------------------------------------------
        #REPORTING
        text0 = 'Features with RSD% over the threshold (' + str(rsd_threshold) + ') were removed.'
        if qc_rsd[qc_rsd > rsd_threshold].empty:
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
        return self.data
    
    def filter_dilution_series_linearity(self, number_of_series, threshold=0.8, which_to_take="first", concentrations=False, to_plot=False):
        """
        Filter out features with low dilution series linearity. (R2 < threshold)
        
        Parameters
        ----------
        threshold : float
            Threshold for the R2. Default is 0.8.
        number_of_series : int
            Number of dilution series used. 
        which_to_take : str
            Which sample to take from the dilution series. Default is "best". Other options are "best", "worst", "first", "last", "mean" or "median".
        concentrations : bool or list
            If True, the concentrations need to be either already defined or added as a list (of the length of number of samples in each series or as all the samples in all series). Default is False.
        to_plot : bool or int
            If True, plot some of the compounds with low R2. Default is False. (For True plots 4, for any other int, plots that amount (or all if there is less than inputted integer))    
        """

        data = self.data
        variable_metadata = self.variable_metadata
        report = self.report
        dilution_series_samples = self.dilution_series_samples
        # if concentrations is list
        if type(concentrations) != list and concentrations != False:
            concentrations = self.dil_concentrations

        if dilution_series_samples is None:
            raise ValueError("Dilution series was not defined. Define it first.")
        if dilution_series_samples is False:
            raise ValueError("Dilution series were already removed. Cannot filter out features with low dilution series linearity.")
        
        # Dilution_series_samples mask
        is_dilution_series_sample = [True if col in dilution_series_samples else False for col in data.columns[1:]]

        # Calculate R2 for every feature (compound) in the dilution series samples
        r2 = []
        all_series_pred_y = []
        all_series_x = []
        all_series_y = []
        # Iterate over all features in dataframe
        for i in range(len(data)):
            # Get the peak areas for the feature
            y = np.array(data.iloc[i, 1:][is_dilution_series_sample].dropna().values)
            lngth = len(y)
            # Ensure y is not empty
            if len(y) == 0:
                r2.append(np.nan)
                continue

            # Each series has the same number of samples (len(y)/number_of_series)
            y = y.reshape((number_of_series, -1)) # Reshape the array to have the same number of samples in each series
            if concentrations:
                x = np.array(concentrations) 
                is_length_of_one = True
                # Check if the length of concentrations is the same as the number of samples in each series
                if len(x) != len(y[0]):
                    # Check if the length of concentrations is the same as the number of samples in all series together
                    if len(x) == lngth: # the length before reshaping
                        # Then reshape the x array as well
                        x = x.reshape((number_of_series, -1))
                        is_length_of_one = False
                    else:
                        raise ValueError("The length of concentrations is not the same as the number of samples in each/all series.")
            else:
                # Use the indices of the dilution series samples as the x-values
                x = np.arange(len(y[0]))
            
            # Perform linear regression
            this_series_r2 = []
            this_series_pred_y = []
            this_series_x = []
            for number, series in enumerate(y):
                # check if the length of x is the same as the length of y (otherwise the number and series work as indices of x as well)
                if is_length_of_one:
                    this_x = x
                else:
                    this_x = x[number]

                # Perform manual linear regression
                this_x = np.array(this_x, dtype=float)
                series = np.array(series, dtype=float)
                A = np.vstack([this_x, np.ones(len(this_x))]).T
                slope, intercept = np.linalg.lstsq(A, series, rcond=None)[0]
                y_pred = slope * this_x + intercept

                this_series_pred_y.append(y_pred)
                this_series_x.append(this_x)

                ss_res = np.sum((series - y_pred) ** 2)
                ss_tot = np.sum((series - np.mean(series)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
            
                # Append R-squared to the list
                this_series_r2.append(r_squared)
            all_series_pred_y.append(this_series_pred_y)
            all_series_x.append(this_series_x)
            all_series_y.append(y)
            if which_to_take == "best":
                r2.append(max(this_series_r2))
            elif which_to_take == "worst":
                r2.append(min(this_series_r2))
            elif which_to_take == "first":
                r2.append(this_series_r2[0])
            elif which_to_take == "last":
                r2.append(this_series_r2[-1])
            elif which_to_take == "mean":
                r2.append(np.mean(this_series_r2))
            elif which_to_take == "median":
                r2.append(np.median(this_series_r2))
            else:
                raise ValueError("which_to_take not recognized. Choose from: 'best', 'worst', 'first', 'last', 'mean' or 'median'. (Specify which series to take into account when comparing the R2 to the threshold)")
        
        r2 = pd.Series(r2, index=data.index)
        
        # Add R2 into the data
        variable_metadata['Dilution_Series_R2'] = r2

        # Plot some of the compounds with low R2
        images = []
        if to_plot:
            if to_plot is True:
                number_plotted = 4
            elif type(to_plot) == int:
                number_plotted = int(to_plot)
            else:
                raise ValueError("to_plot has to be either True/False or an integer.")
            indexes = r2[r2 < threshold].index.tolist()[:number_plotted]
            print(indexes)
            if len(indexes) < number_plotted:
                number_plotted = len(indexes)

            if len(indexes) == 0:
                print("No compounds with R2 < " + str(threshold) + " were found.")
            else:
                for i in indexes:
                    plt.figure()
                    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                    j = 0
                    for series, y_pred, this_x in zip(all_series_y[i], all_series_pred_y[i], all_series_x[i]):
                        plt.scatter(this_x, series, color= colors[j], alpha=0.5, marker='o')
                        plt.plot(this_x, y_pred, color=colors[j], alpha=0.5)
                        j += 1
                    plt.xlabel('Dilution factor')
                    plt.ylabel('Peak Area')
                    plt.title("Dilution series - LOW linearity compound: cpID = " + data.iloc[i, 0] + " ;R2 = " + str(r2[i])[:7] + '('+ which_to_take + ')')
                    for suffix in self.suffixes:
                        plt.savefig(self.main_folder + '/figures/dilution_series_linearity_' + str(i) + '_low_R2-deleted_by_correction' + suffix, dpi=400, bbox_inches='tight')
                    images.append(self.main_folder + '/figures/dilution_series_linearity_' + str(i) + '_low_R2-deleted_by_correction.png')
                    plt.show()
                    plt.close()

        # Plot some of the compounds with high R2
        if to_plot:
            if to_plot is True:
                number_plotted = 4
            elif type(to_plot) == int:
                number_plotted = int(to_plot)
            else:
                raise ValueError("to_plot has to be either True/False or an integer.")
            indexes = r2[r2 > threshold].index.tolist()[:number_plotted]
            if len(indexes) < number_plotted:
                number_plotted = len(indexes)

            if len(indexes) == 0:
                print("No compounds with R2 < " + str(threshold) + " were found.")
            else:
                for i in indexes:
                    plt.figure()
                    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                    j = 0
                    for series, y_pred, this_x in zip(all_series_y[i], all_series_pred_y[i], all_series_x[i]):
                        plt.scatter(this_x, series, color= colors[j], alpha=0.5, marker='o')
                        plt.plot(this_x, y_pred, color=colors[j], alpha=0.5)
                        j += 1
                    plt.xlabel('Dilution factor')
                    plt.ylabel('Peak Area')
                    # add the R2 value to the title
                    plt.title("Dilution series - HIGH linearity compound: cpID = " + data.iloc[i, 0] + " ;R2 = " + str(r2[i])[:7] + '(' + which_to_take + ')')
                    plt.show()      
    
        # Filter out features (compounds) with R2 < threshold
        under_threshold = r2 < threshold
        data = data[~under_threshold]
        # Reset index
        data = data.reset_index(drop=True)
        # Update data and variable_metadata
        self.data = data
        self.variable_metadata = self._filter_match_variable_metadata(data, variable_metadata)

        # Report how many features were removed
        print("Number of features removed: " + str(len(r2[r2 < threshold])) + " ;being: " + str(r2[r2 < threshold].index.tolist()))

        #---------------------------------------------
        # REPORTING
        text0 = 'Features with dilution series linearity (R2) under the threshold (' + str(threshold) + ') were removed.'
        if r2[r2 < threshold].empty:
            text1 = 'No compounds with R2 < ' + str(threshold) + ' were found.'
            report.add_together([('text', text0),
                            ('text', text1)])
        else:
            text1 = 'Number of features removed: ' + str(len(r2[r2 < threshold])) + ' ;being: '+ str(r2[r2 < threshold].index.tolist()[:10])[:-1] + ', ...'
            text2 = 'Examples of compounds with low R2:'
            report.add_together([('text', text0),
                            ('text', text1),
                            ('text', text2)])
            for image in images:
                report.add_image(image)
        report.add_line()
        return self.data
       
    def filter_number_of_corrected_batches(self, threshold = 0.8):
        """
        Filter out features that were corrected in less than the threshold number of batches. (Ideally apply after the correction step.)

        Parameters
        ----------
        threshold : float
            Threshold for the number of batches. Default is 0.8. If less than 1, then it is interpreted as a percentage otherwise as a minimum number of batches corrected.
        """
        data = self.data
        variable_metadata = self.variable_metadata
        report = self.report
        batch = self.batch_info

        nm_of_batches = len(batch['Batch'].dropna().unique())

        if threshold < 1:
            # If the threshold is less than 1, interpret it as a percentage
            percentage = True
            threshold = int(threshold * nm_of_batches)
        else:
            percentage = False
        lngth = len(data.columns)
        
        # Filter out features (from both data and variable_metadata) that were corrected in less than the threshold number of batches
        corrected_batches_dict = {cpdID: corrected_batches for cpdID, corrected_batches in zip(variable_metadata['cpdID'], variable_metadata['corrected_batches'])}
        removed = data['cpdID'][data['cpdID'].map(corrected_batches_dict).fillna(0) < threshold].tolist()
        data = data[data['cpdID'].map(corrected_batches_dict).fillna(0) >= threshold]
        
        data.reset_index(drop=True, inplace=True)
        self.data = data

        variable_metadata = variable_metadata[variable_metadata['cpdID'].isin(data['cpdID'])]
        variable_metadata.reset_index(drop=True, inplace=True)
        self.variable_metadata = self._filter_match_variable_metadata(data, variable_metadata)

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
        return self.data

    def drop_samples(self, column_indexes_to_drop, cpdID_as_zero = True):
        """
        Function for filtering out (deleting) specific samples. (Columns) Such as standards, ... and others you wanna omit from the analysis.

        Parameters
        ----------
        column_indexes_to_drop : list
            List of column indexes to drop.
        cpdID_as_zero : bool
            If True, cpdID is counted as a column indexed at 0; if false, 0 is index of first sample in data. Default is True.
        """
        data = self.data
        report = self.report
        metadata = self.metadata
        batch = self.batch
        batch_info = self.batch_info

        if cpdID_as_zero == True and 0 in column_indexes_to_drop:
            raise ValueError("cpdID cannot be dropped; your indexes cannot include 0 when cpdID_as_zero is True.")
        
        # Drop samples
        data = data.drop(data.columns[column_indexes_to_drop], axis=1)

        if cpdID_as_zero:
            column_indexes_to_drop = [i-1 for i in column_indexes_to_drop] # for metadata and batch_info we need to lower the indexes by 1 (there is no cpdID there)

        # Drop rows with dropped samples from metadata
        metadata = metadata.drop(metadata.index[column_indexes_to_drop])
        metadata.reset_index(drop=True, inplace=True)
        #if batch info is Not None, drop the samples from batch_info
        if batch_info is not None:
            # Update batch_info
            batch_info = batch_info.drop(column_indexes_to_drop)
            batch_info.reset_index(drop=True, inplace=True)
            # Update batch
            batch = [batch[i] for i in range(len(batch)) if i not in column_indexes_to_drop]
        # Update variable_metadata
        variable_metadata = self._filter_match_variable_metadata(data, self.variable_metadata)
        
        # Update the object variables
        self.data = data
        self.metadata = metadata
        self.batch_info = batch_info
        self.batch = batch
        self.variable_metadata = variable_metadata

        print('Specified samples: '+ str(column_indexes_to_drop) +' were removed from the data.')
        #---------------------------------------------
        #REPORTING
        text0 = 'Specified samples were removed from the data.'
        text1 = 'Number of samples removed: ' + str(len(column_indexes_to_drop)) + ' ;being: ' + str(column_indexes_to_drop[:10])[:-1] + ', ...'
        report.add_together([('text', text0),
                            ('text', text1),
                            'line'])          
        return self.data

    def delete_samples(self, column_indexes_to_drop, cpdID_as_zero = True):
        """
        Function for filtering out (deleting) specific samples. (Columns) Such as standards, ... and others you wanna omit from the analysis.
        
        Parameters
        ----------
        column_indexes_to_drop : list
            List of column indexes to drop.
        """
        return self.drop_samples(column_indexes_to_drop, cpdID_as_zero=cpdID_as_zero)

    def filter_out_samples(self, column_indexes_to_drop, cpdID_as_zero = True):
        """
        Function for filtering out (deleting) specific samples. (Columns) Such as standards, ... and others you wanna omit from the analysis.

        Parameters
        ----------
        column_indexes_to_drop : list
            List of column indexes to drop.
        
        """
        return self.drop_samples(column_indexes_to_drop)
    
    def filter_out_group_by_metadata_columns(self, metadata_column, value):
        """
        Function for filtering out (deleting) samples based on the metadata.
        
        Parameters
        ----------
        metadata_column : str
            Name of the metadata column.
        value : str
            Value of the metadata column.
        """
        metadata = self.metadata
        column_indexes_to_drop = metadata[metadata[metadata_column] == value].index.tolist()
        # Add 1 to the indexes to drop, because the first column is cpdID
        column_indexes_to_drop = [i+1 for i in column_indexes_to_drop]
        print(column_indexes_to_drop)
        return self.drop_samples(column_indexes_to_drop, cpdID_as_zero = True)

    def filter_out_blanks(self):
        """
        Function for filtering out (deleting) blank samples. 

        """
        data = self.data
        report = self.report
        if self.blank_samples == None:
            raise ValueError("No blank samples were defined.")
        elif self.blank_samples == False:
            raise ValueError("Blank samples were already removed.")
        blank_samples = self.blank_samples

        batch = self.batch
        batch_info = self.batch_info

        # Filter out blank samples
        # find indexes of blank samples
        blank_indexes = [data.columns.get_loc(col) for col in blank_samples] 

        # drop blank samples
        data = data.drop(data.columns[blank_indexes], axis=1)

        # WE NEED TO UPDATE OTHER DATA STRUCTURES AS WELL
        # lower the indexes of the blank samples by 1 (-1 because of cpdID column)

        if data.columns[0]  == 'cpdID':    
            blank_indexes = [i-1 for i in blank_indexes]
        
        # drop blank samples 
        self.batch = [batch[i] for i in range(len(batch)) if i not in blank_indexes]
        self.batch_info = batch_info.drop(blank_indexes)
        self.batch_info.reset_index(drop=True, inplace=True)
        self.blank_samples = False

        # drop rows with blank samples
        self.metadata = self.metadata.drop(self.metadata.index[blank_indexes])
        self.metadata.reset_index(drop=True, inplace=True)
        
        self.data = data
        self.variable_metadata = self._filter_match_variable_metadata(data, self.variable_metadata)
        self.variable_metadata.reset_index(drop=True, inplace=True)

        print("Blank samples were removed from the data.")
        #---------------------------------------------
        #REPORTING
        text = 'Blank samples were removed from the data.'
        report.add_together([('text', text),
                            'line'])
        return self.data
    
    def filter_out_dilution_series(self):
        """
        Function for filtering out (deleting) dilution series samples. 
        """
        data = self.data
        report = self.report
        if self.dilution_series_samples is None:
            raise ValueError("No dilution series were defined.")
        elif self.dilution_series_samples is False:
            raise ValueError("Dilution series were already removed.")
        dilution_series = self.dilution_series_samples

        batch = self.batch
        batch_info = self.batch_info

        # Filter out dilution series
        dilution_series_indexes = []
        for col in dilution_series:
            if col in data.columns:
                dilution_series_indexes.append(data.columns.get_loc(col))
            else:
                print(f"Column {col} not found, probably already deleted.")

        if dilution_series_indexes:
            # drop dilution series samples
            data = data.drop(data.columns[dilution_series_indexes], axis=1)

            # WE NEED TO UPDATE OTHER DATA STRUCTURES AS WELL
            # lower the indexes of the dilution series samples by 1 (-1 because of cpdID column)
            if data.columns[0] == 'cpdID':    
                dilution_series_indexes = [i-1 for i in dilution_series_indexes]

            # drop dilution series samples 
            self.batch = [batch[i] for i in range(len(batch)) if i not in dilution_series_indexes]
            self.batch_info = batch_info.drop(dilution_series_indexes)
            self.batch_info.reset_index(drop=True, inplace=True)
            self.dilution_series = False

            # drop rows with dilution series samples
            self.metadata = self.metadata.drop(self.metadata.index[dilution_series_indexes])
            self.metadata.reset_index(drop=True, inplace=True)

            self.data = data
            self.variable_metadata = self._filter_match_variable_metadata(data, self.variable_metadata)
            self.variable_metadata.reset_index(drop=True, inplace=True)

            print("Dilution series samples were removed from the data.")
            #---------------------------------------------
            #REPORTING
            text = 'Dilution series samples were removed from the data.'
            report.add_together([('text', text),
                                'line'])
        else:
            print("No dilution series samples were found to remove.")
        return self.data
    
    def filter_out_standards(self):
        """
        Function for filtering out (deleting) standards. 
        """
        data = self.data
        report = self.report
        if self.standard_samples is None:
            raise ValueError("No standards were defined.")
        elif self.standard_samples is False:
            raise ValueError("Standards were already removed.")
        standards = self.standard_samples

        batch = self.batch
        batch_info = self.batch_info

        # Filter out standards
        standard_indexes = []
        for col in standards:
            if col in data.columns:
                standard_indexes.append(data.columns.get_loc(col))
            else:
                print(f"Column {col} not found, probably already deleted.")

        if standard_indexes:
            # drop blank samples
            data = data.drop(data.columns[standard_indexes], axis=1)

            # WE NEED TO UPDATE OTHER DATA STRUCTURES AS WELL
            # lower the indexes of the standard samples by 1 (-1 because of cpdID column)
            if data.columns[0] == 'cpdID':    
                standard_indexes = [i-1 for i in standard_indexes]

            # drop blank samples 
            self.batch = [batch[i] for i in range(len(batch)) if i not in standard_indexes]
            self.batch_info = batch_info.drop(standard_indexes)
            self.batch_info.reset_index(drop=True, inplace=True)
            self.standard_samples = False

            # drop rows with blank samples
            self.metadata = self.metadata.drop(self.metadata.index[standard_indexes])
            self.metadata.reset_index(drop=True, inplace=True)

            self.data = data
            self.variable_metadata = self._filter_match_variable_metadata(data, self.variable_metadata)
            self.variable_metadata.reset_index(drop=True, inplace=True)

            print("Standards were removed from the data.")
            #---------------------------------------------
            #REPORTING
            text = 'Standards were removed from the data.'
            report.add_together([('text', text),
                                'line'])
        else:
            print("No standards were found to remove.")
        return self.data
    

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
            data = np.exp(data) 
            # return as pd.DataFrame
            return pd.DataFrame(data, index=data.index, columns=data.columns)
        data = np.log(data + 0.0001) # Add a small number to avoid log(0) 
        # return as pd.DataFrame
        return pd.DataFrame(data, index=data.index, columns=data.columns)

    def transformer_log(self, invert=False):
        """
        Apply the log transformation to the data.

        Parameters
        ----------
        invert : bool
            Invert the log transformation. Default is False.
        """
        data = self.data
        variable_metadata = self.variable_metadata
        report = self.report

        data.iloc[:, 1:] = self._transformer_log(data.iloc[:, 1:], invert)
        self.data = data
        #---------------------------------------------
        #REPORTING
        if invert:
            text = 'Log transformation was inverted.'
        else:
            text = 'Log transformation was applied.'
        report.add_together([('text', text),
                            'line'])
        return data
    
    def center_data(self, method='median'):
        """
        Center the data by subtracting either the mean or the median
        of each variable (column).

        Parameters
        ----------
        method : str, optional
            The method of centering to use:
            - 'mean'   : mean centering
            - 'median' : median centering (default)
        """
        data = self.data
        report = self.report

        # Select only numeric columns (skip sample identifiers, etc.)
        X = data.iloc[:, 1:]

        # Compute centering values
        if method == 'mean':
            center_values = np.mean(X, axis=0)
            center_type = 'mean'
        elif method == 'median':
            center_values = np.median(X, axis=0)
            center_type = 'median'
        else:
            raise ValueError("Invalid method. Use 'mean' or 'median'.")

        # Apply centering
        data.iloc[:, 1:] = X - center_values
        self.data = data

        #---------------------------------------------
        # REPORTING
        text = f'Data was centered using {center_type} centering.'
        report.add_together([('text', text), 'line'])

        return self.data
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ALL NORMALIZING METHODS (keyword: normalizer_...)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def normalizer_zscores(self):
        """
        Normalize the data using z-scores.

        """
        data = self.data
        report = self.report

        # Calculate the z-scores for the data
        zscores, mean, std = self._normalizer_zscores(data.iloc[:, 1:])
        # Add the z-scores to the data
        data.iloc[:, 1:] = zscores
        self.data = data
        #---------------------------------------------
        #REPORTING
        text = 'Data was normalized using z-scores. Mean and standard deviation were calculated for each feature.'
        report.add_together([('text', text),
                            'line'])
        return self.data
    
    def normalizer_pqn(self):
        """
        Normalize the data using Probabilistic Quotient Normalization (PQN).

        """
        data = self.data
        report = self.report

        # Step 1: Calculate the reference sample (median of all samples for each feature)
        reference_sample = data.iloc[:, 1:].median(axis=0)

        # Step 2: Calculate the quotients for each sample
        quotients = data.iloc[:, 1:].div(reference_sample, axis=1)

        # Step 3: Calculate the median quotient for each sample
        median_quotients = quotients.median(axis=1)

        # Step 4: Normalize each feature value in the sample by dividing by the median quotient
        data.iloc[:, 1:] = data.iloc[:, 1:].div(median_quotients, axis=0)

        self.data = data

        #---------------------------------------------
        # REPORTING
        text = 'Data was normalized using Probabilistic Quotient Normalization (PQN).'
        report.add_together([('text', text), 'line'])
        return self.data

    def scaler_pareto(self):
        """
        Normalize the data using Pareto scaling.

        """
        data = self.data
        report = self.report

        # Apply Pareto scaling to the data
        mean = np.mean(data.iloc[:, 1:], axis=0)
        std = np.std(data.iloc[:, 1:], axis=0, ddof=1)
        data.iloc[:, 1:] = (data.iloc[:, 1:] - mean) / np.sqrt(std)
        self.data = data

        #---------------------------------------------
        # REPORTING
        text = 'Data was normalized using Pareto scaling.'
        report.add_together([('text', text), 'line'])
        return self.data
        

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ALL CORRECTING METHODS (keyword: correcter_...)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _cubic_spline_smoothing(self, x, y, s_value, grow=10.0, max_tries=4, tol=1.0):
        """
        Fit a cubic smoothing spline with robust weights from z-scores.
        Returns a fitted UnivariateSpline or None if not enough signal.

        Parameters
        ----------
        x : array
            Array with the x values.
        y : array
            Array with the y values.
        s_value : float
            Smoothing parameter.
        grow : float
            Factor to increase s_value if fitting fails. Default is 10.0.
        max_tries : int
            Maximum number of attempts to fit the spline. Default is 4.
        tol : float
            Tolerance for increasing s_value. Default is 0.1.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        if y.size < 4 or not np.isfinite(y).all() or np.nanstd(y) == 0:
            return None

        # robust weights
        z = zscore(y, nan_policy='omit')
        z = np.where(np.isfinite(z), z, 0.0)
        w = 1.0 / (1.0 + z**2)

        # strictly increasing x
        if np.any(np.diff(x) <= 0):
            order = np.argsort(x)
            x, y, w = x[order], y[order], w[order]

        s_cur = float(s_value)

        for _ in range(max_tries):
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                try:
                    spline = UnivariateSpline(x, y, w=w, s=s_cur)
                except Exception:
                    spline = None

            bad = False
            # detect "too small s" warnings
            if wlist:
                text = " ".join(str(w.message) for w in wlist)
                if "maximal number of iterations maxit" in text and "s too small" in text:
                    bad = True

            # residual consistency check (relaxed for large s)
            if spline is not None:
                fp = spline.get_residual()  # weighted RSS
                if s_cur > 0 and np.isfinite(fp):
                    ratio = abs(fp - s_cur) / s_cur
                    # only enforce strict residual match for small s
                    if ratio > tol and s_cur < 1e5:
                        bad = True

             # success → return the spline
            if spline is not None and not bad:
                return spline
            
            # otherwise, relax smoothness and retry
            s_cur *= grow  

        # all attempts failed
        return None
                
    def _cv_best_smoothing_param(self, x, y, p_values, k=5, min_points=5, minloo=5):
        """
        Time-aware CV (contiguous folds). Falls back to LOO for tiny series.
        Returns the p (smoothing parameter) that minimizes CV MSE, or None.

        Parameters
        ----------
        x : array-like
            1D array of time points (e.g. injection order).
        y : array-like
            1D array of intensities (for a single feature).
        p_values : array-like
            Candidate smoothing parameters to evaluate.
        k : int
            Number of folds for K-Fold CV. Default is 5.
        min_points : int
            Minimum number of data points required to perform CV. Default is 5.
        minloo : int
            Minimum number of data points required to perform Leave-One-Out CV. Default is 5.
        """
        x = np.asarray(x); y = np.asarray(y)
        n = y.size

        # basic guards
        if n < min_points or not np.isfinite(y).all():
            return None
        if np.nanstd(y) == 0:
            return None

        # choose splitter: contiguous KFold or LOO
        splitter = LeaveOneOut() if (n < k or k <= 1 or n < minloo) else KFold(n_splits=min(k, n), shuffle=False)

        best_p, best_mse = None, np.inf
        for p in p_values:
            fold_err = 0.0
            n_splits = 0
            for tr_idx, te_idx in splitter.split(x):
                s = self._cubic_spline_smoothing(x[tr_idx], y[tr_idx], p)
                if s is None:
                    # if the model can’t fit on this fold, penalize heavily
                    fold_err += 1e12
                else:
                    yhat = s(x[te_idx])
                    fold_err += mean_squared_error(y[te_idx], yhat)
                n_splits += 1
            fold_mse = fold_err / max(1, n_splits)

            if fold_mse < best_mse:
                best_mse, best_p = fold_mse, p

        return best_p
    
    def _estimate_batchwise_s_ranges(self, data_log2, batch, is_qc_sample, n_features=100, k=5):
        """
        For each batch:
        - sample up to n_features features (evenly across columns),
        - run a wide log grid to find typical best s,
        - return a narrowed per-batch grid for the main loop.

        Parameters
        ----------
        data_log2 : DataFrame
            DataFrame with log2-transformed data (samples x features).
        batch : list or array
            List or array of batch ids aligned to rows of data_log2.
        is_qc_sample : array
            Boolean array aligned to rows of data_log2 indicating which samples are QC samples.
        n_features : int
            How many features to probe per batch. Default is 100.
        k : int
            Number of folds for K-Fold CV. Default is 5.

        """
        unique_batches = list(dict.fromkeys(batch))
        n_total_feats = data_log2.shape[1]

        # choose probe feature indices evenly spaced across all features
        probe_idx = np.linspace(0, n_total_feats - 1, min(n_features, n_total_feats), dtype=int)

        batch_p_ranges = {}
        batch_best_s = {} 

        for b in unique_batches:
            is_b = np.array([bb == b for bb in batch], dtype=bool)
            qc_mask_b = np.logical_and(is_qc_sample, is_b)

            best_s_vals = []
            for fi in probe_idx:
                y = data_log2.iloc[qc_mask_b, fi].values
                x = np.arange(y.size)

                # guard for tiny/noisy series
                if y.size < 5 or not np.isfinite(y).all() or np.nanstd(y) == 0:
                    continue

                # wide grid scaled by variance * n  (order-of-magnitude scan)
                var_y = max(np.nanvar(y), 1e-6)
                p_wide = np.logspace(-3, 3, 9) * var_y * y.size
                p_wide = np.clip(p_wide, 1.0, None) # never below 1.0 - avoids extreme overfitting (small s means very wiggly spline)

                p_best = self._cv_best_smoothing_param(x, y, p_wide, k=k)
                if p_best is not None and np.isfinite(p_best):
                    best_s_vals.append(p_best)

            batch_best_s[b] = best_s_vals 

            # derive narrowed grid for this batch
            if len(best_s_vals) >= 5:
                s10, s50, s90 = np.percentile(best_s_vals, [10, 50, 90])
                # keep a safety margin (+/- ~0.5–1 decade around middle)
                lower = max(s10 / 3.0, 1e-8)
                upper = s90 * 3.0
                upper = max(upper, lower * 10)  # ensure at least one decade span
                batch_p_ranges[b] = np.logspace(np.log10(lower), np.log10(upper), 7)
            elif len(best_s_vals) > 0:
                # too few points to be fancy—center on median with generous bounds
                s50 = float(np.median(best_s_vals))
                batch_p_ranges[b] = np.logspace(np.log10(s50 / 10.0), np.log10(s50 * 10.0), 7)
            else:
                # fallback: wide default
                batch_p_ranges[b] = np.logspace(-3, 3, 9)

        return batch_p_ranges, batch_best_s 
        
    def _leave_one_out_cross_validation(self, x, y, p_values):
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
                s = self._cubic_spline_smoothing(x_train, y_train, p)
                if s is None:
                    mse += 1e12
                else:
                    mse += ((s(x_test) - y_test) ** 2).sum()
            
            # Calculate the mean squared error
            mse /= len(x)
            
            # If this is the best smoothing parameter so far, update the best smoothing parameter and the minimum mean squared error
            if mse < min_mse:
                best_p = p
                min_mse = mse
        # Return the best smoothing parameter
        return best_p

    def correcter_qc_interpolation(self, show='default', p_values='default', delog=True, use_zeros=False, cmap='viridis'):
        """
        Correct batch effects and systematic errors using QC-sample interpolation.

        Workflow:
        1) Work in log2 space (always).
        2) Fit per-batch splines on QC-only points for each feature.
        3) Subtract the fitted drift and re-anchor to the GLOBAL QC median (log2).
        4) Optionally de-log to raw space if `delog=True`.
        5) Plot BEFORE/AFTER for selected features, overlaying the fitted spline(s).

        Parameters
        ----------
        show : 'default' | 'all' | 'none' | list[int|str] | int | str
            Which features to plot (indices or names). 'default' = 5 evenly spaced.
        p_values : 'default' | array-like
            Candidate smoothing parameters for spline fitting.
            If 'default', uses m ± sqrt(2m), where m = #QC points in that batch.
        delog : bool
            If True, return data back-transformed from log2 to raw intensities.
            If False, keep the output in log2 space (common in many workflows).
        use_zeros : bool
            If False (recommended), zero intensities are excluded from QC fitting.
        cmap : str
            Matplotlib colormap name for batches.
        """
        # ---------------------------
        # Inputs & basic prep
        # ---------------------------
        df_in = self.data                      # features x samples with 'cpdID' as col 0
        variable_metadata = self.variable_metadata
        report = self.report
        QC_samples = list(self.QC_samples)
        main_folder = self.main_folder
        suffixes = self.suffixes
        batch = self.batch

        feature_names = df_in.iloc[:, 0]
        # Work matrix: samples x features (no cpdID column)
        data = df_in.iloc[:, 1:].copy().T

        # Batch handling
        if batch is None:
            batch = ['all_one_batch'] * len(data.index)
        unique_batches = list(dict.fromkeys(batch))

        # QC flags
        is_qc_sample = np.array([idx in QC_samples for idx in data.index], dtype=bool)

        # Colors per batch / QC
        cmap_obj = mpl.cm.get_cmap(cmap)
        b2i = {b: i for i, b in enumerate(unique_batches)}
        norm_idx = {b: i / max(1, len(unique_batches)) for b, i in b2i.items()}
        batch_colors = [mpl.colors.rgb2hex(cmap_obj(norm_idx[b])) for b in unique_batches]
        batch_to_color = {bid: batch_colors[i % len(batch_colors)] for i, bid in enumerate(unique_batches)}
        point_colors = ['black' if qc else batch_to_color[batch[i]] for i, qc in enumerate(is_qc_sample)]

        # Zeros mask aligned to transposed matrix
        if not use_zeros:
            is_zero = (df_in.iloc[:, 1:].T == 0)
            is_zero = is_zero.reindex(index=data.index, columns=data.columns, fill_value=False)
        else:
            is_zero = None

        # ---------------------------
        # Normalize `show` (after transpose)
        # ---------------------------
        nfeat = data.shape[1]
        if isinstance(show, str):
            if show == 'default':
                show_idx = set(np.linspace(0, nfeat - 1, 5, dtype=int))
            elif show == 'all':
                show_idx = set(range(nfeat))
            elif show == 'none':
                show_idx = set()
            else:
                try:
                    show_idx = {int(show)}
                except ValueError:
                    name2idx = {c: i for i, c in enumerate(list(data.columns))}
                    idx = name2idx.get(show)
                    show_idx = set() if idx is None else {idx}
        else:
            show_idx = set(int(x) for x in (show.tolist() if isinstance(show, np.ndarray) else show))

        # ---------------------------
        # Forward transform: LOG2 only (always)
        # ---------------------------
        # Use eps to avoid -inf; keep consistent across features
        eps = 1e-9
        data = np.log2(np.maximum(data, eps))

        # ---------------------------
        # Estimate batchwise s ranges
        # ---------------------------
        # Learn per-batch p-grids from a quick exploration pass (order-of-magnitude scan)

        batch_p_ranges, batch_best_s = self._estimate_batchwise_s_ranges(
            data_log2=data,
            batch=batch,
            is_qc_sample=is_qc_sample,
            n_features=100,
            k=5
        )
        os.makedirs(f"{self.main_folder}/figures", exist_ok=True)

        for bname, s_vals in batch_best_s.items():
            if not s_vals:
                # nothing to plot for this batch
                continue

            s_vals = np.asarray(s_vals, dtype=float)
            s_vals = s_vals[np.isfinite(s_vals)]
            if s_vals.size == 0:
                continue

            # log10 for stable visualization
            log_s = np.log10(s_vals)

            # narrowed range we’ll use in the main loop
            p_grid = np.asarray(batch_p_ranges[bname], dtype=float)
            p_grid = p_grid[np.isfinite(p_grid)]
            if p_grid.size == 0:
                continue
            lo, hi = np.min(p_grid), np.max(p_grid)
            log_lo, log_hi = np.log10(lo), np.log10(hi)

            # edge diagnostics: how many best_s hit the wide search edges during exploration?
            # (we used 1e-3..1e3 * var(y) * n; detect if s is near min or max observed in that ‘wide’ scan)
            # crude proxy: 5% quantile near min OR 95% near max of observed best_s
            q5, q50, q95 = np.percentile(s_vals, [5, 50, 95])
            edge_low_hits  = (s_vals <= np.min(s_vals) * 1.0000001).sum()  # practically none; left for completeness
            edge_high_hits = (s_vals >= np.max(s_vals) / 1.0000001).sum()  # same note

            # build figure
            plt.figure(figsize=(8, 4.5))
            # histogram in log10 space
            bins = max(10, int(np.sqrt(log_s.size)))
            plt.hist(log_s, bins=bins, alpha=0.7, edgecolor='k')

            # overlay vertical lines for 10/50/90% and narrowed range
            s10, s50, s90 = np.percentile(s_vals, [10, 50, 90])
            for v, ls, lw in [(np.log10(s10), '--', 1.0),
                            (np.log10(s50), '-', 2.0),
                            (np.log10(s90), '--', 1.0)]:
                plt.axvline(v, ls=ls, lw=lw, color='C1')

            # shade the narrowed range to be used
            plt.axvspan(log_lo, log_hi, color='C2', alpha=0.15, label='Narrowed range')

            # labels and title
            plt.xlabel('log10(smoothing parameter s)')
            plt.ylabel('Count (features in exploration subset)')
            plt.title(f'Exploration of best s — batch: {bname}')

            # legend: include quantiles
            plt.legend(
                [f'10/50/90%: {s10:.2g} / {s50:.2g} / {s90:.2g}', 'Narrowed range'],
                frameon=False,
                loc='best'
            )

            # annotate simple edge signal if distribution is butting up against range
            # If median is within ~0.1 decade of either bound, warn in the title.
            warn = ''
            if (np.log10(s50) - log_lo) < 0.1:
                warn = ' (median near LOWER bound)'
            if (log_hi - np.log10(s50)) < 0.1:
                warn = ' (median near UPPER bound)'
            if warn:
                plt.title(f'Exploration of best s — batch: {bname}{warn}')

            out_path = f"{self.main_folder}/figures/s_exploration_{bname}.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"[s-explore] Saved: {out_path} (n={s_vals.size})")

        # ---------------------------
        # Correction
        # ---------------------------
        start_time = time.time()
        plot_names_orig = []
        plot_names_corr = []
        chosen_p_values = []
        numbers_of_correctable_batches = []

        for feat_idx, feature in enumerate(data.columns):
            splines = []
            is_correctable_batch = []
            num_corr_batches = 0

            # Global QC anchor in log space (across all batches)
            anchor_log = np.nanmedian(data.loc[is_qc_sample, feature]) if is_qc_sample.any() else 0.0

            # Fit per-batch QC splines
            for b_idx, bname in enumerate(unique_batches):
                is_batch = np.array([bb == bname for bb in batch], dtype=bool)
                qc_mask = is_qc_sample.copy()

                if not use_zeros:
                    nz_mask = ~is_zero[feature].values
                    qc_mask = np.logical_and(qc_mask, nz_mask)

                qc_batched = np.logical_and(qc_mask, is_batch)
                qc_y = data.loc[qc_batched, feature]

                x = np.arange(len(data))[qc_batched]
                y = qc_y.values

                # Choose candidate p-values
                if isinstance(p_values, str) and p_values == 'default':
                    # narrowed, batch-specific range learned in the exploration stage
                    p_vals = batch_p_ranges[bname]
                else:
                    # user-provided grid (array-like)
                    p_vals = p_values

                # Guard: if too few QC points in this batch, skip
                if len(y) < 5 or not np.isfinite(y).all() or np.nanstd(y) == 0:
                    p = None
                else:
                    p = self._cv_best_smoothing_param(x, y, p_vals, k=5)

                # Only record valid p's
                if p is not None and np.isfinite(p):
                    chosen_p_values.append(p)

                if p is None:
                    s = None
                    is_correctable_batch.append(False)
                else:
                    s = self._cubic_spline_smoothing(x, y, p)
                    is_correctable_batch.append(True)
                    num_corr_batches += 1

                splines.append(s)

            numbers_of_correctable_batches.append(num_corr_batches)

            # ---------------------------
            # BEFORE plot (selected only)
            # ---------------------------
            if feat_idx in show_idx:
                zeros_feat = (is_zero[feature].values if is_zero is not None
                            else np.zeros(len(data), dtype=bool))
                alphas = [0.4 if qc else (0.1 if z else 0.8) for qc, z in zip(is_qc_sample, zeros_feat)]

                gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
                fig = plt.figure(figsize=(20, 4))

                # top
                plt.subplot(gs[0])
                x_all = np.arange(len(data))
                y_before = data[feature].values
                plt.scatter(x_all, y_before, color=point_colors, alpha=alphas)

                reds = cycle(['red', 'tomato', 'firebrick'])
                for b_idx, bname in enumerate(unique_batches):
                    is_batch = np.array([bb == bname for bb in batch], dtype=bool)
                    xb = x_all[is_batch]
                    s = splines[b_idx]
                    if s is not None:
                        plt.plot(xb, s(xb), color=next(reds), linewidth=2)

                # baseline = global QC anchor in log space
                plt.axhline(anchor_log, color='0.5', linestyle='--', linewidth=1, alpha=0.8, label='QC anchor')

                plt.xlabel('Injection Order')
                plt.ylabel('log2 Peak Area (before)')
                plt.title(feature_names[feature] if len(feature_names) > 0 else f'Feature: {feature}')

                # ensure baseline visible
                ymin, ymax = np.nanmin(y_before), np.nanmax(y_before)
                ymin = min(ymin, anchor_log); ymax = max(ymax, anchor_log)
                pad = 0.01 * max(1e-6, (ymax - ymin))
                plt.ylim(ymin - pad, ymax + pad)
                plt.legend(loc='best', frameon=False)

                # bottom table
                plt.subplot(gs[1]); plt.axis('tight'); plt.axis('off'); fig.subplots_adjust(bottom=-0.5)
                zero_counts = {bid: 0 for bid in unique_batches}
                qc_zero_counts = {bid: 0 for bid in unique_batches}
                for i, (bcur, zflag) in enumerate(zip(batch, zeros_feat)):
                    if zflag:
                        zero_counts[bcur] += 1
                        if is_qc_sample[i]:
                            qc_zero_counts[bcur] += 1
                total_samples = {bid: batch.count(bid) for bid in unique_batches}
                total_qc_samples = {bid: sum(1 for bcur, qc in zip(batch, is_qc_sample) if (bcur == bid and qc))
                                    for bid in unique_batches}
                zero_percentages = {bid: (zero_counts[bid] / total_samples[bid] * 100) if total_samples[bid] else 0.0
                                    for bid in unique_batches}
                qc_zero_percentages = {bid: (qc_zero_counts[bid] / total_qc_samples[bid] * 100)
                                    if total_qc_samples[bid] > 0 else 0.0
                                    for bid in unique_batches}
                formatted_zero_counts = [f"{zero_counts[bid]} ({zero_percentages[bid]:.2f}%)" for bid in unique_batches]
                formatted_qc_zero_counts = [f"{qc_zero_counts[bid]} ({qc_zero_percentages[bid]:.2f}%)"
                                            for bid in unique_batches]
                total_missing = sum(zero_counts.values())
                total_percentage = total_missing / len(batch) * 100 if len(batch) else 0.0
                total_qc_zeros = sum(qc_zero_counts.values())
                total_qc_percentage = total_qc_zeros / sum(is_qc_sample) * 100 if sum(is_qc_sample) else 0.0
                col_labels = unique_batches + ['Total']
                table_batch_colors = batch_colors + ['white']
                table_batch_colors_rgba = [self._convert_to_rgba(color, 0.6) for color in table_batch_colors]
                formatted_zero_counts.append(f"{total_missing} ({total_percentage:.2f}%)")
                formatted_qc_zero_counts.append(f"{total_qc_zeros} ({total_qc_percentage:.2f}%)")
                table = plt.table(
                    cellText=[formatted_zero_counts, formatted_qc_zero_counts],
                    cellColours=[table_batch_colors_rgba, table_batch_colors_rgba],
                    rowLabels=['All Samples', 'QC Samples'],
                    colLabels=col_labels,
                    cellLoc='center', fontsize=10, loc='center'
                )
                cells_to_change_color = [(0, i) for i, ok in enumerate(is_correctable_batch) if not ok]
                for cell in cells_to_change_color:
                    if cell in table.get_celld():
                        table.get_celld()[cell].set_text_props(fontweight='bold', color='red')
                plt.text(x=-0.005, y=0.65, s='Zero Counts', fontsize=15,
                        transform=plt.gca().transAxes, ha='right', va='center')

                for suffix in suffixes:
                    plt_name = f"{main_folder}/figures/QC_correction_{feature}_original"
                    plt.savefig(plt_name + suffix, dpi=300, bbox_inches='tight')
                plot_names_orig.append(plt_name + '.png')
                plt.show()

            # ---------------------------
            # Apply correction per batch (log space)
            # ---------------------------
            x_all = np.arange(len(data))
            for b_idx, bname in enumerate(unique_batches):
                is_batch = np.array([bb == bname for bb in batch], dtype=bool)
                s = splines[b_idx]
                if s is None:
                    continue
                preds = s(x_all[is_batch])
                # subtract drift and re-anchor to global QC anchor in log space
                data.loc[is_batch, feature] = data.loc[is_batch, feature] - preds + anchor_log

            # ---------------------------
            # AFTER plot (selected only)
            # ---------------------------
            if feat_idx in show_idx:
                zeros_feat = (is_zero[feature].values if is_zero is not None
                            else np.zeros(len(data), dtype=bool))
                alphas = [0.4 if qc else (0.1 if z else 0.8) for qc, z in zip(is_qc_sample, zeros_feat)]

                gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
                fig = plt.figure(figsize=(20, 4))

                # top
                plt.subplot(gs[0])
                y_corr = data[feature].values
                plt.scatter(x_all, y_corr, color=point_colors, alpha=alphas)

                # overlay original splines (what was removed)
                preds_min, preds_max = np.inf, -np.inf
                for b_idx, bname in enumerate(unique_batches):
                    is_batch = np.array([bb == bname for bb in batch], dtype=bool)
                    xb = x_all[is_batch]
                    s = splines[b_idx]
                    if s is not None:
                        yb = s(xb)
                        preds_min = min(preds_min, np.nanmin(yb))
                        preds_max = max(preds_max, np.nanmax(yb))
                        plt.plot(xb, yb, color='green', linewidth=2, alpha=0.6,
                                label='fitted spline' if b_idx == 0 else None)

                # baseline = global QC anchor (log)
                plt.axhline(anchor_log, color='0.5', linestyle='--', linewidth=1, alpha=0.8, label='QC anchor')

                plt.xlabel('Injection Order')
                plt.ylabel('log2 Peak Area (corrected)')
                plt.title(feature_names[feature] if len(feature_names) > 0 else f'Feature: {feature}')

                # y-limits include corrected data, splines, and baseline
                corr_min = np.nanmin(y_corr[~zeros_feat]) if (~zeros_feat).any() else np.nanmin(y_corr)
                corr_max = np.nanmax(y_corr)
                y_min = min(corr_min, preds_min, anchor_log) if np.isfinite(preds_min) else min(corr_min, anchor_log)
                y_max = max(corr_max, preds_max, anchor_log) if np.isfinite(preds_max) else max(corr_max, anchor_log)
                pad = 0.01 * max(1e-6, (y_max - y_min))
                plt.ylim(y_min - pad, y_max + pad)

                plt.legend(loc='best', frameon=False)

                # bottom table
                plt.subplot(gs[1]); plt.axis('tight'); plt.axis('off'); fig.subplots_adjust(bottom=-0.5)
                zero_counts = {bid: 0 for bid in unique_batches}
                qc_zero_counts = {bid: 0 for bid in unique_batches}
                for i, (bcur, zflag) in enumerate(zip(batch, zeros_feat)):
                    if zflag:
                        zero_counts[bcur] += 1
                        if is_qc_sample[i]:
                            qc_zero_counts[bcur] += 1
                total_samples = {bid: batch.count(bid) for bid in unique_batches}
                total_qc_samples = {bid: sum(1 for bcur, qc in zip(batch, is_qc_sample) if (bcur == bid and qc))
                                    for bid in unique_batches}
                zero_percentages = {bid: (zero_counts[bid] / total_samples[bid] * 100) if total_samples[bid] else 0.0
                                    for bid in unique_batches}
                qc_zero_percentages = {bid: (qc_zero_counts[bid] / total_qc_samples[bid] * 100)
                                    if total_qc_samples[bid] > 0 else 0.0
                                    for bid in unique_batches}
                formatted_zero_counts = [f"{zero_counts[bid]} ({zero_percentages[bid]:.2f}%)" for bid in unique_batches]
                formatted_qc_zero_counts = [f"{qc_zero_counts[bid]} ({qc_zero_percentages[bid]:.2f}%)"
                                            for bid in unique_batches]
                total_missing = sum(zero_counts.values())
                total_percentage = total_missing / len(batch) * 100 if len(batch) else 0.0
                total_qc_zeros = sum(qc_zero_counts.values())
                total_qc_percentage = total_qc_zeros / sum(is_qc_sample) * 100 if sum(is_qc_sample) else 0.0
                col_labels = unique_batches + ['Total']
                table_batch_colors = batch_colors + ['white']
                table_batch_colors_rgba = [self._convert_to_rgba(color, 0.6) for color in table_batch_colors]
                formatted_zero_counts.append(f"{total_missing} ({total_percentage:.2f}%)")
                formatted_qc_zero_counts.append(f"{total_qc_zeros} ({total_qc_percentage:.2f}%)")
                table = plt.table(
                    cellText=[formatted_zero_counts, formatted_qc_zero_counts],
                    cellColours=[table_batch_colors_rgba, table_batch_colors_rgba],
                    rowLabels=['All Samples', 'QC Samples'],
                    colLabels=col_labels,
                    cellLoc='center', fontsize=10, loc='center'
                )
                cells_to_change_color = [(0, i) for i, ok in enumerate(is_correctable_batch) if not ok]
                for cell in cells_to_change_color:
                    if cell in table.get_celld():
                        table.get_celld()[cell].set_text_props(fontweight='bold', color='red')
                plt.text(x=-0.005, y=0.65, s='Zero Counts', fontsize=15,
                        transform=plt.gca().transAxes, ha='right', va='center')

                for suffix in suffixes:
                    plt_name = f"{main_folder}/figures/QC_correction_{feature}_corrected"
                    plt.savefig(plt_name + suffix, dpi=300, bbox_inches='tight')
                plot_names_corr.append(plt_name + '.png')
                plt.show()

            # progress
            pct = round((feat_idx + 1) / nfeat * 100, 3)
            eta_min = round((time.time() - start_time) / (feat_idx + 1) * (nfeat - feat_idx - 1) / 60, 2)
            print(f'Progress: {pct}%; last feature: {feature}. ETA ~ {eta_min}m       ', end='\r')

        # ---------------------------
        # Optional: back-transform (de-log)
        # ---------------------------
        if delog:
            data = np.power(2.0, data)

        # Back to features x samples with cpdID in col 0
        data = data.T
        data.insert(0, 'cpdID', feature_names)

        # chosen p-values (summary)
        _valid_p = [pv for pv in chosen_p_values if pv is not None and np.isfinite(pv)]
        chosen_p_values = pd.Series(_valid_p).value_counts().to_dict() if _valid_p else {}
        chosen_p_values = {k: v for k, v in sorted(chosen_p_values.items(), key=lambda kv: kv[1], reverse=True)}

        # number of corrected batches per feature
        correction_dict = {cpdID: ncb for cpdID, ncb in zip(feature_names, numbers_of_correctable_batches)}
        variable_metadata['corrected_batches'] = variable_metadata['cpdID'].map(correction_dict)

        # assign back
        self.data = data
        self.variable_metadata = self._filter_match_variable_metadata(data, variable_metadata)

        # REPORT
        text0 = 'QC-based drift correction (log2 space) was performed.'
        texts = []
        texts.append(('text', f'Output returned in {"raw" if delog else "log2"} space.', 'italic'))
        if isinstance(p_values, str) and p_values == 'default':
            texts.append(('text','Spline smoothness was selected by time-aware cross-validation over a batch-specific range - estimated from a small exploratory subset of features.' ))
        together = [('text', text0, 'bold')]
        together.extend(texts)
        report.add_together(together)
        report.add_line()
        for pn_orig, pn_corr in zip(plot_names_orig, plot_names_corr):
            report.add_together([('text', 'Original data:', 'bold'),
                                ('image', pn_orig),
                                ('text', 'Corrected data:', 'bold'),
                                ('image', pn_corr),
                                'line'])

        return self.data
        
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ALL STATISTICS METHODS (keyword: statistics_...)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def statistics_correlation_means(self, column_name, method = 'pearson', cmap = 'coolwarm', min_max = [-1, 1], plt_name_suffix = 'group_correlation_matrix_heatmap'):
        """
        Calculate the correlation matrix of the data and create a heatmap.

        Parameters
        ----------
        column_name : str
            Name of the column to group the data by. (From metadata, e.g. 'Sample Type' or 'Sex', 'Age', Diagnosis', etc.)
        method: str
            Method to calculate correlation. Default is 'pearson'. (Other options are 'kendall', 'spearman') For more information: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
        cmap : str
            Name of the colormap. Default is 'coolwarm'. (Other options are 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted', 'turbo'; ADD '_r' to get reversed colormap)
        min_max : list (of 2 elements)
            Minimum and maximum values for the colorbar. Default is [-1, 1].
        plt_name_suffix : str
            Suffix for the plot name. Default is 'group_correlation_matrix_heatmap'. (Useful when using multiple times - to avoid overwriting the previous plot)
        """
        data = self.data
        metadata = self.metadata
        report = self.report
        output_file_prefix = self.output_file_prefix
        main_folder = self.main_folder
        suffixes = self.suffixes
        
        if min_max[0] < -1:
            min_max[0] = -1
        if min_max[1] > 1:
            min_max[1] = 1
        if min_max[0] > min_max[1]:
            min_max = [-1, 1]


        #Group the data by 'column_name' and calculate the mean of each group
        grouped_means = data.iloc[:, 1:].groupby(metadata[column_name]).mean()

        #Calculate the correlation matrix of the group means
        correlation_matrix = grouped_means.T.corr()

        # Create a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, cmap=cmap, vmin=min_max[0], vmax=min_max[1])
        name = main_folder +'/statistics/'+output_file_prefix + '_' + plt_name_suffix
        for suffix in suffixes:
            plt.savefig(name + suffix, bbox_inches='tight', dpi = 300)
        plt.show()

        #---------------------------------------------
        # REPORTING
        text = 'Group correlation matrix heatmap was created. Grouping is based on: ' + column_name
        report.add_together([
            ('text', text), 
            ('image', name), 
            'pagebreak'])
        return correlation_matrix
    
    def statistics_PCA(self, n_components_for_candidates = 2):
        """
        Perform PCA on the data and visualize the results.

        (It stores the results in the object to be used in the future.)

        Parameters
        ----------
        n_components_for_candidates : int
            Number of components to use for the calculation of the candidates. Default is 2.
        """
        data = self.data
        report = self.report

        # Perform PCA
        pca = PCA()
        pca.fit(data.iloc[:,1:].T)
        pca_data = pca.transform(data.iloc[:, 1:].T)
        # Calculate the percentage of variance that each PC explains
        per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
        labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
        pca_df = pd.DataFrame(pca_data, columns=labels) 

        self.pca_per_var = per_var
        self.pca_data = pca_data
        self.pca_df = pca_df
        self.pca = pca

        self.pca_count += 1

        # Loadings 
        eigenvectors = pca.components_.T
        eigenvalues = pca.explained_variance_
        loadings = eigenvectors * np.sqrt(eigenvalues) # Loadings are the eigenvectors scaled by the square root of the eigenvalues        

        #index = feature_names = data.iloc[:, 0]
        loadings_df = pd.DataFrame(loadings, index=data.iloc[:, 0], columns=labels)
        self.pca_loadings = loadings_df

        # Add 99.5th percentile of the distances to the loadings dataframe BUT TAKE INTO ACCOUNT ONLY the first n PCs
        # check if n_components_for_candidates is less than the number of components
        if n_components_for_candidates > len(labels):
            n_components_for_candidates = len(labels)
        loadings_df['distance'] = np.sqrt(np.sum(loadings_df.iloc[:, :n_components_for_candidates]**2, axis=1)) # Calculate the distance of each loading from the origin and take into account only the first n PCs 
        loadings_df['distance'] = loadings_df['distance'] / np.max(loadings_df['distance']) # Normalize the distances
        loadings_df['distance'] = loadings_df['distance'] * 100
    
        #loadings_df['distance'] = loadings_df['distance'].round(1)

        # Get the candidates based on the distance
        candidate_loadings = loadings_df[loadings_df['distance'] > np.percentile(loadings_df['distance'], 99.5)].index
        # Get the candidate loadings scores
        candidate_loadings_scores = loadings_df[loadings_df['distance'] > np.percentile(loadings_df['distance'], 99.5)]['distance'].round(2)

        # Save candidate loadings
        self.pca_loadings_candidates = candidate_loadings

        # Add the candidate features to the list
        self.add_candidates(features = candidate_loadings.to_list(), method = 'PCA-loadings', specification = 'Analysis-' + str(self.pca_count) + '; PCA-components-'+str(n_components_for_candidates), scores = candidate_loadings_scores.to_list())

        #---------------------------------------------
        # REPORTING
        #report.add_text('<b>PCA was performed.</b>')
        # add text to report with PCA and count of PCA
        report.add_text('<b>PCA (' + str(self.pca_count) + ') was performed. </b>')
        return self.pca_data
    
    def _vip(self, model):
        """
        Help function to calculate the VIP scores for the PLS-DA model.

        Parameters
        ----------
        model : PLSRegression object
            (PLS-DA) model.
        """
        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_
        p, h = w.shape
        vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        for i in range(p):
            weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
            vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
        return vips
    
    
    def statistics_PLSDA(self, response_column_names, ignored_groups = None):
        """
        Perform PLSDA on the data and visualize the results.

        (It stores the results in the object to be used in the future.)

        Parameters
        ----------

        response_column_names : str or list
            Name of the column to use as a response variable. (From metadata, e.g. 'Sample Type' or list for multiple columns ['Sample Type', 'Diagnosis'], etc.)
        ignored_groups : None or list of lists 
            List of lists - groups to ignore in format [[Column Name, Group Name]]. Default is None. (e.g. [['Sample Type', 'QC'], ['Sample Type', 'Blank']]) - to ignore 'QC' and 'Blank' groups from 'Sample Type' column. If None, then no groups are ignored.

        """
        data = self.data.copy()
        metadata = self.metadata.copy()
        report = self.report

        # Define the number of components to be used
        n_comp = 2
        # Define the number of folds for cross-validation
        n_folds = 10
        # Random state for reproducibility
        rng = 42

         # Normalize ignored_groups argument into a list of [col, value]
        if ignored_groups is None:
            ignored_groups = []
        if isinstance(ignored_groups, list) and len(ignored_groups) > 0 and isinstance(ignored_groups[0], str):
            ignored_groups = [ignored_groups]

        # ----------- Filter metadata by ignored groups -----------
        for col_name, grp_name in ignored_groups:
            metadata = metadata[metadata[col_name] != grp_name].reset_index(drop=True)

        # ----------- Align data columns with (filtered) metadata -----------
        # Keep 'cpdID' plus all sample columns listed in metadata['Sample File']
        if 'Sample File' not in metadata.columns:
            raise ValueError("Expected 'Sample File' in metadata to align samples.")
        columns_to_keep = ['cpdID'] + metadata['Sample File'].tolist()
        data = data.loc[:, data.columns.isin(columns_to_keep)]

        # Reorder data columns to match metadata sample order (after 'cpdID')
        fixed_cols = ['cpdID']
        sample_cols = [c for c in metadata['Sample File'].tolist() if c in data.columns]
        data = pd.concat([data[fixed_cols], data[sample_cols]], axis=1)

        # ----------- Build response y (one-hot) -----------
        # If multiple response columns, concatenate values with underscores
        if isinstance(response_column_names, list):
            tmp_col = str(response_column_names)
            metadata[tmp_col] = metadata[response_column_names].apply(lambda x: '_'.join(x.map(str)), axis=1)
            response_col = tmp_col
        else:
            response_col = str(response_column_names)

        self.plsda_metadata = metadata.copy()
        y_labels = pd.Categorical(metadata[response_col])  # categorical labels
        y_strat = y_labels.codes                          # integer labels for stratification
        y_onehot = pd.get_dummies(y_labels)               # one-hot matrix (n_samples, n_classes)
        class_names = list(y_onehot.columns)

        # Prepare X (samples in rows) 
        # data: first column is 'cpdID', the rest are samples
        X_full = data.iloc[:, 1:].T.values     # shape: (n_samples, n_features)
        Y_full = y_onehot.values               # shape: (n_samples, n_classes)

        #  Safety: adjust n_folds if any class is too small 
        # Each class must have at least n_folds samples for StratifiedKFold
        class_counts = pd.Series(y_strat).value_counts().to_dict()
        min_class = min(class_counts.values()) if len(class_counts) else 0
        if min_class < n_folds:
            # reduce folds to the minimum class count (>=2)
            new_folds = max(2, min_class)
            if new_folds != n_folds:
                n_folds = new_folds

        # Cross-validated predictions (StratifiedKFold) 
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rng)
        y_cv = np.zeros_like(Y_full, dtype=float)  # CV predictions placeholder

        for train_idx, test_idx in skf.split(X_full, y_strat):
            model_cv = PLSRegression(n_components=n_comp)
            model_cv.fit(X_full[train_idx], Y_full[train_idx])
            y_cv[test_idx] = model_cv.predict(X_full[test_idx])

        # Final fit on all data (for VIPs & visualization) 
        model = PLSRegression(n_components=n_comp)
        model.fit(X_full, Y_full)

        # VIPs (uses your existing helper)
        vips = self._vip(model)  # expects shape (n_features,)
        candidate_mask = vips > np.percentile(vips, 99.5)
        candidate_vips = data.iloc[:, 0][candidate_mask]       # cpdID for selected features
        candidate_vips_scores = vips[candidate_mask]
        self.add_candidates(features=candidate_vips,
                            method='PLSDA-vips',
                            specification=str(response_column_names),
                            scores=candidate_vips_scores)

        # Metrics computed FROM CV predictions (your definition) 
        # Per-class metrics (each dummy column separately)
        r2_per_class = {}
        q2_per_class = {}
        for i, cname in enumerate(class_names):
            yt = Y_full[:, i]
            yp = y_cv[:, i]
            r2_i = r2_score(yt, yp)
            mse_i = mean_squared_error(yt, yp)
            var_i = np.var(yt)  # population variance (ddof=0), same as your code
            q2_i = 1.0 - (mse_i / var_i) if var_i > 0 else np.nan
            r2_per_class[cname] = float(r2_i)
            q2_per_class[cname] = float(q2_i)

        # Global metrics across all classes (flattened)
        yt_all = Y_full.ravel()
        yp_all = y_cv.ravel()
        r2_global = float(r2_score(yt_all, yp_all))
        mse_global = float(mean_squared_error(yt_all, yp_all))
        var_global = float(np.var(yt_all))
        q2_global = float(1.0 - (mse_global / var_global)) if var_global > 0 else np.nan

        # Cross-validated classification accuracy (argmax over class scores)
        y_true_int = Y_full.argmax(axis=1)
        y_pred_int = y_cv.argmax(axis=1)
        cv_accuracy = float((y_true_int == y_pred_int).mean())

        # Store & report 
        self.plsda_model = model
        self.plsda_response_column = response_column_names
        self.plsda_stats = {
            "n_components": n_comp,
            "n_folds": n_folds,
            "classes": class_names,
            "R2_global": r2_global,
            "Q2_global": q2_global,
            "CV_accuracy": cv_accuracy,
            "R2_per_class": r2_per_class,
            "Q2_per_class": q2_per_class,
        }

        # Console print 
        print("R2 (global):", r2_global)
        print("Q2 (global):", q2_global)
        print("Per-class R2:", r2_per_class)
        print("Per-class Q2:", q2_per_class)
        print("CV accuracy:", cv_accuracy)

        # ------------------------------
        # REPORTING
        text0 = f"<b>PLS-DA</b> was performed with the {str(response_column_names)} column(s) as the response."
        text1 = f"R2_global: {r2_global:.3f}, Q2_global: {q2_global:.3f}, CV accuracy: {cv_accuracy:.3f}."
        report.add_together([
            ('text', text0),
            ('text', text1),
            'line'
        ])

        return self.plsda_model

    def statistics_ttest(self, groups_column_name, group1, group2, p_value_correction_method = None, table_name_suffix = 'ttest_results'):
        """
        Perform t-test on two chosen groups of samples and return the results in a table.
        
        Parameters
        ----------

        groups_column_name : str
            Name of the column to group the data by. (From metadata, e.g. 'Sample Type')
        group1 : str
            Name of the first group to compare. (From metadata, e.g. 'Control')
        group2 : str
            Name of the second group to compare. (From metadata, e.g. 'Treatment')
        p_value_correction_method : str or None
            Method for p-value correction method. Default is None. (Options are: 'fdr_bh', 'bonferroni', 'holm-sidak', 'holm', 'simes-hochberg', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'; for more information: https://statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)
        table_name_suffix : str
            Suffix for the table name. Default is 'ttest_results'. (Useful when using multiple times - to avoid overwriting the previous table)
        """
        data = self.data.copy()
        metadata = self.metadata.copy()
        report = self.report
        output_file_prefix = self.output_file_prefix
        

        #Create masks for the two groups
        group1_mask = metadata[groups_column_name] == group1
        group2_mask = metadata[groups_column_name] == group2

        # Add False to the beginning of the masks to align them with the data (first column is cpdID)
        group1_mask = np.insert(group1_mask.to_numpy(), 0, False)
        group2_mask = np.insert(group2_mask.to_numpy(), 0, False)

        # Get the data for the two groups
        group1_data = data.iloc[:, group1_mask]
        group2_data = data.iloc[:, group2_mask]

        fold_change = group2_data.mean(axis=1) / group1_data.mean(axis=1)
        # Get p-values
        p_values = ttest_ind(group1_data, group2_data, axis=1)[1]

        if p_value_correction_method not in (None, "", "None", False):
            # Use correction for p-values 
            p_values = multipletests(p_values, method=p_value_correction_method)[1]
        
        # Create a DataFrame with the results
        p_values_table = pd.DataFrame({
            'cpdID': data['cpdID'],
            'Fold Change': fold_change,
            'p-value': p_values,
            'group': [f"{group1} vs {group2}"] * len(data)
        })
        # Sort the table by p-value
        p_values_table = p_values_table.sort_values(by='p-value', ascending=True).reset_index(drop=True)
        
        #-----------------------------------------------
        # REPORTING
        text0 = f'<b>t-test</b> was performed for the groups: {group1} vs {group2}.'
        text1 = 'The results are shown in the table below.'
        report.add_together([('text', text0),
                            ('text', text1),
                            ('table', p_values_table),
                            'line'])
        # Save the table to a CSV file
        csv_name = self.main_folder + '/statistics/' + output_file_prefix + '-' + group1 + '_vs_' + group2 + table_name_suffix +'.csv'
        p_values_table.to_csv(csv_name, index=False, sep=';')
        report.add_text(f'The t-test results were saved to: {csv_name}')
        
        return p_values_table


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
    
    def visualizer_boxplot(self, plt_name_suffix = 'boxplot_first_view'):
        """
        Create a boxplot of all samples.  Can depict if there is a problem with the data (retention time shift too big -> re-run alignment; batch effect, etc.)

        Parameters
        ----------
        plt_name_suffix : str
            Suffix for the plot name. Default is 'boxplot_first_view'. (Useful when using multiple times - to avoid overwriting the previous plot)
    
        """
        data = self.data
        report = self.report
        QC_samples = self.QC_samples
        blank_samples = self.blank_samples
        dilution_series_samples = self.dilution_series_samples
        standard_samples = self.standard_samples

        main_folder = self.main_folder
        suffixes = self.suffixes

        # QC_samples mask
        if QC_samples != False and QC_samples != None:
            is_qc_sample = [True if col in QC_samples else False for col in data.columns[1:]]
        else:
            is_qc_sample = [False for col in data.columns[1:]]
        # Blank samples mask
        print(blank_samples)
        if blank_samples != False and blank_samples != None:
            is_blank_sample = [True if col in blank_samples else False for col in data.columns[1:]]
        else:
            is_blank_sample = [False for col in data.columns[1:]]
        # Dilution series samples mask
        if dilution_series_samples != False and dilution_series_samples != None:
            is_dilution_series_sample = [True if col in dilution_series_samples else False for col in data.columns[1:]]
        else:
            is_dilution_series_sample = [False for col in data.columns[1:]]
        # Standard samples mask
        if standard_samples != False and standard_samples != None:
            is_standard_sample = [True if col in standard_samples else False for col in data.columns[1:]]
        else:
            is_standard_sample = [False for col in data.columns[1:]]

        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(18, 12))

        # Create a box plot for the specific feature
        box = plt.boxplot(data.iloc[:,1:], showfliers=False, showmeans=True, meanline=True, medianprops={'color':'black'}, meanprops={'color':'blue'}, patch_artist=True, whiskerprops=dict(color='grey'), capprops=dict(color='yellow'))
        plt.title('Boxplot of all samples')

        #Color boxplots of QC samples in red and the rest in blue
        colors = ['grey' if qc else 'darkred' if is_blank else 'blue' if is_dilution else 'darkgreen' if is_standard else 'lightblue' for qc, is_blank, is_dilution, is_standard in zip(is_qc_sample, is_blank_sample, is_dilution_series_sample, is_standard_sample)]
        # Set the colors for the individual boxplots
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        # Customize the plot
        plt.xlabel('Sample Order')
        plt.ylabel('Peak Area')
        xx = np.arange(0, len(data.columns[1:])+1, 100)
        plt.xticks(xx, xx)

        # Add a legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='QC samples', markerfacecolor='grey', markersize=10),
                        plt.Line2D([0], [0], marker='o', color='w', label='Blank samples', markerfacecolor='darkred', markersize=10),
                        plt.Line2D([0], [0], marker='o', color='w', label='Dilution series samples', markerfacecolor='blue', markersize=10),
                        plt.Line2D([0], [0], marker='o', color='w', label='Standard samples', markerfacecolor='darkgreen', markersize=10),
                        plt.Line2D([0], [0], marker='o', color='w', label='Samples', markerfacecolor='lightblue', markersize=10)]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=12, title='Sample types', title_fontsize='13', frameon=True)
        

        # Save the plot
        plt_name = main_folder + '/figures/QC_samples_' + plt_name_suffix
        for suffix in suffixes:
            plt.savefig(plt_name + suffix, bbox_inches='tight', dpi=300)

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
    
    def visualizer_samples_by_batch(self, show = 'default', cmap = 'viridis', plt_name_suffix = 'batches_first_view'):
        """
        Visualize samples by batch. Highlight QC samples.

        Parameters
        ----------
        show : list or str
            List of features to show. Default is 'default' -> 5 samples across the dataset. Other options are 'all', 'none' or index of the feature.
        cmap : str
            Name of the colormap. Default is 'viridis'. (Other options are 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted', 'turbo'; ADD '_r' to get reversed colormap)
        plt_name_suffix : str
            Suffix for the plot name. Default is 'batches_first_view'. (Useful when using multiple times - to avoid overwriting the previous plot)
        """
        data = self.data
        report = self.report
        QC_samples = self.QC_samples
        blank_samples = self.blank_samples
        dilution_series_samples = self.dilution_series_samples
        standard_samples = self.standard_samples
        main_folder = self.main_folder
        suffixes = self.suffixes
        batch = self.batch

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

        if batch is None:
            batch = ['all_one_batch' for i in range(len(data.columns) - 1)]  # Dummy batch information (for cases where its all the same batch)

        #unique_batches = []
        # for b in batch:
        #     if b not in unique_batches:
        #         unique_batches.append(b)
        unique_batches = list(dict.fromkeys(batch))  # Preserve the order of batches as they appear in the data
        
        cmap = mpl.cm.get_cmap(cmap)

        # Create a dictionary that maps each unique batch to a unique index
        batch_to_index = {batch: index for index, batch in enumerate(unique_batches)}

        # Normalize the indices between 0 and 1
        normalized_indices = {batch: index / len(unique_batches) for batch, index in batch_to_index.items()}

        # Create a dictionary that maps each batch to a color
        batch_colors = [mpl.colors.rgb2hex(cmap(normalized_indices[batch])) for batch in unique_batches]

        batch_to_color = {batch_id: batch_colors[i % len(batch_colors)] for i, batch_id in enumerate(unique_batches)}

        # QC_samples mask
        if QC_samples != False and QC_samples != None:
            is_qc_sample = [True if col in QC_samples else False for col in data.columns[1:]]
        else:
            is_qc_sample = [False for col in data.columns[1:]]
        # Blank samples mask
        print(blank_samples)
        if blank_samples != False and blank_samples != None:
            is_blank_sample = [True if col in blank_samples else False for col in data.columns[1:]]
        else:
            is_blank_sample = [False for col in data.columns[1:]]
        # Dilution series samples mask
        if dilution_series_samples != False and dilution_series_samples != None:
            is_dilution_series_sample = [True if col in dilution_series_samples else False for col in data.columns[1:]]
        else:
            is_dilution_series_sample = [False for col in data.columns[1:]]
        # Standard samples mask
        if standard_samples != False and standard_samples != None:
            is_standard_sample = [True if col in standard_samples else False for col in data.columns[1:]]
        else:
            is_standard_sample = [False for col in data.columns[1:]]

        # Create a boolean mask for the x array
        x_mask = np.full(len(data.columns) - 1, False)
        x_mask[is_qc_sample] = True

        # Color based on the batch (and if it is QC sample then always black)
        for feature in show:

            # Colors and alphas and markers
            colors = ['black' if qc else 'darkblue' if is_blank else 'darkred' if is_dilution else 'darkgreen' if is_standard else batch_to_color[b] for b, qc, is_blank, is_dilution, is_standard in zip(batch, is_qc_sample, is_blank_sample, is_dilution_series_sample, is_standard_sample)]
            row_data = data.iloc[feature, 1:]
            alphas = [0.5 if qc else 0.1 if zero else 0.8 for qc, zero in zip(is_qc_sample, row_data == 0)]

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
            fig = plt.figure(figsize=(20, 4))

            # Create a scatter plot to display the QC sample points
            plt.subplot(gs[0])
            plt.scatter(range(len(data.columns) -1), data.iloc[feature, 1:], color=colors, alpha=alphas, marker='o') 
            plt.plot(np.arange(len(data.columns) - 1)[x_mask][~zeros_mask], data.iloc[feature, 1:][is_qc_sample][~zeros_mask], color='black', linewidth=1)
            plt.xticks(rotation=90)
            plt.xlabel('Samples in order')
            plt.ylabel('Peak Area')
            plt.title("cpID = " + data.iloc[feature, 0])

            #add legend
            legend_elements = [Line2D([0], [0], marker='o', color='w', label='QC samples', markerfacecolor='black', markersize=10, alpha = 0.5),
                            Line2D([0], [0], marker='o', color='w', label='Blank samples', markerfacecolor='darkblue', markersize=10),
                            Line2D([0], [0], marker='o', color='w', label='Dilution series samples', markerfacecolor='darkred', markersize=10),
                            Line2D([0], [0], marker='o', color='w', label='Standard samples', markerfacecolor='darkgreen', markersize=10)]
            for batch_id, color in batch_to_color.items():
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label="Batch:" + str(batch_id), markerfacecolor=color, markersize=10))

            # Add the legend to the plot            
            plt.legend(handles=legend_elements, loc='upper left', fontsize=10, bbox_to_anchor=(1, 1), title='Sample Types', title_fontsize='13', frameon=False)
            
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
            
            plt_name = main_folder + '/figures/single_compound_' + str(feature) + '_' + plt_name_suffix
            for suffix in suffixes:
                plt.savefig(plt_name + suffix, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

        #---------------------------------------------
        #REPORTING
        text = 'View of samples for a single compound with highlighted QC samples was created.'
        report.add_text(text)
        #add plot to the report
        images = [main_folder + '/figures/single_compound_' + str(feature) + '_batches_first_view' for feature in show]
        for image in images:
            report.add_image(image)
        report.add_pagebreak()
        return fig

    def visualizer_PCA_scree_plot(self, plt_name_suffix='scree_plot-percentages'):
        """
        Create a scree plot for PCA.

        Parameters
        ----------
        plt_name_suffix : str
            Suffix for the plot name. Default is 'scree_plot-percentages'. (Useful when using multiple times - to avoid overwriting the previous plot)
        """
        report = self.report
        output_file_prefix = self.output_file_prefix
        main_folder = self.main_folder
        suffixes = self.suffixes
        if self.pca_data is None:
            raise ValueError('PCA was not performed yet. Run PCA first.')
        else:
            per_var = self.pca_per_var
            cumulative_variance = np.cumsum(per_var)

        # Create a scree plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.set_xlabel('Component')
        ax.set_ylabel('Variance (%)')
        ax.plot(range(1, 11), per_var[:10], color='tab:red', marker='o', label='Variance')
        ax.plot(range(1, 11), cumulative_variance[:10], color='tab:blue', marker='o', label='Cumulative Variance')

        for i, (ev, cv) in enumerate(zip(per_var[:10], cumulative_variance[:10]), start=1):
            ax.text(i, ev, f'{ev:.1f}', ha='center', va='bottom', fontsize=8, color='tab:red')
            ax.text(i, cv, f'{cv:.1f}', ha='center', va='bottom', fontsize=8, color='tab:blue')

        plt.xticks(range(1, 11)) 
        plt.title('Scree Plot')
        plt.legend(loc='upper left')
        fig.tight_layout()

        name = main_folder + '/statistics/' + output_file_prefix + '_' + plt_name_suffix
        for suffix in suffixes:
            plt.savefig(name + suffix, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

        #---------------------------------------------
        # REPORTING
        text = 'Scree plot was created and added into: ' + name
        report.add_together([
            ('text', text),
            ('image', name)])
        return fig
    
    def visualizer_PCA_run_order(self, connected = True, colors_for_cmap = ['yellow', 'red'], plt_name_suffix = 'PCA_plot_colored_by_run_order', annotate_samples = False, nm_to_annotate = 10):
        """
        Create a PCA plot colored by run order. (To see if there is any batch effect)

        Parameters
        ----------
        connected : bool
            If True, the samples will be connected by a line. Default is True.
        colors_for_cmap : list
            List of colors for the colormap. Default is ['yellow', 'red']. 
        plt_name_suffix : str
            Suffix for the plot name. Default is 'PCA_plot_colored_by_run_order'. (Useful when using multiple times - to avoid overwriting the previous plot)       
        annotate_samples : bool
            If True, the samples will be annotated with their names. Default is False.
        nm_to_annotate : int
            Number of samples to annotate. Default is 10. (Samples with the farthest distance from the origin will be annotated)
        """
        data = self.data
        report = self.report
        output_file_prefix = self.output_file_prefix
        main_folder = self.main_folder
        suffixes = self.suffixes

        if self.pca_data is None:
            raise ValueError('PCA was not performed yet. Run PCA first.')
        else:
            per_var = self.pca_per_var
            pca_df = self.pca_df

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
            df_samples = pca_df[data.columns[1:] == sample_type]
            plt.scatter(df_samples['PC1'], df_samples['PC2'], color=sample_type_colors[sample_type], label=sample_type)

        texts = []
        if annotate_samples:
            # Calculate distances from origin for all samples
            pca_df['distance'] = np.linalg.norm(pca_df[['PC1', 'PC2']].values, axis=1)
            # Select top N farthest samples
            farthest = pca_df.nlargest(nm_to_annotate, 'distance')
            for i, row in farthest.iterrows():
                sample_name = self.metadata.iloc[i]['Sample File']
                texts.append(ax.text(row['PC1'], row['PC2'], sample_name, fontsize=8, color='black', alpha=0.7, bbox=dict(facecolor="white", alpha=0.5, edgecolor="none")))
        # Adjust text to avoid overlap
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
        

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
            for i in range(len(pca_df)-1):
                plt.plot([pca_df['PC1'].iloc[i], pca_df['PC1'].iloc[i+1]], [pca_df['PC2'].iloc[i], pca_df['PC2'].iloc[i+1]], color='black', linewidth=0.25)

        if connected:
            name = main_folder +'/statistics/'+ output_file_prefix + '_' + plt_name_suffix + '_connected'
        else:
            name = main_folder +'/statistics/'+ output_file_prefix + '_' + plt_name_suffix
        for suffix in suffixes:
            plt.savefig(name + suffix, bbox_inches='tight', dpi = 300)
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
    
    def visualizer_PCA_loadings(self, color = 'blue', plt_name_suffix = 'PCA_loadings', components_to_show = [0, 1], annotate_candidates = True, axis_log = False):
        """
        Create a PCA loadings plot. (And suggest candidate features based on the 99.5th percentile of the distances from the origin)

        Parameters
        ----------
        color : str
            Color for the plot. Default is 'blue'.
        plt_name_suffix : str
            Suffix for the plot name. Default is 'PCA_loadings'. (Useful when using multiple times - to avoid overwriting the previous plot)
        components_to_show : list
            List of components to show. Default is [0, 1]. ([0, 1] means PC1 and PC2 - indexing from 0)
        annotate_candidates : bool
            If True, the candidates will be annotated in the plot. Default is True. !!! Candidates are those above 99.5th percentile of the distances from the origin based on the PC1 and PC2 loadings, so annotating them while plotting different PCs might be confusing!.
        axis_log : bool
            If True, the axes will be logarithmic. Default is True.        
        """
        data = self.data
        report = self.report
        output_file_prefix = self.output_file_prefix
        main_folder = self.main_folder
        suffixes = self.suffixes
        pca_loadings = self.pca_loadings
        pca_loadings_candidates = self.pca_loadings_candidates

        if self.pca_data is None:
            raise ValueError('PCA was not performed yet. Run PCA first.')    

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Check components_to_show
        if len(components_to_show) != 2:
            raise ValueError('components_to_show must be a list of two integers.')

        # Plot the loadings colored by chosen column
        plt.scatter(pca_loadings.iloc[:, components_to_show[0]], pca_loadings.iloc[:, components_to_show[1]], color = color, marker='o', alpha=0.5)
    
        # Loadings to annotate (those above 99.5th percentile)
        candidate_loadings_names = pca_loadings_candidates
        
        # Find the candidate loadings in the loadings dataframe
        candidate_loadings = [name for name in candidate_loadings_names if name in pca_loadings.index]
        # Get the candidate loadings scores
        candidate_loadings_scores = pca_loadings.loc[candidate_loadings].iloc[:, components_to_show]

        print(candidate_loadings_names)

        if annotate_candidates:
            texts = []  # List to store text annotations
            # Annotate
            if len(candidate_loadings) > 25:
                print('Too many candidates to annotate (more then 25). Skipping annotation.')
            else:
                for candidate in candidate_loadings:
                    # Get the index of the candidate in the loadings dataframe
                    candidate_index = pca_loadings.index.get_loc(candidate)
                    texts.append(ax.text(pca_loadings.iloc[candidate_index, components_to_show[0]], pca_loadings.iloc[candidate_index, components_to_show[1]], candidate, fontsize=10, color='darkred', alpha=1, bbox=dict(facecolor="white", alpha=0.5, edgecolor="none")))  # Semi-transparent white background))
                    # Annotate the significant points
            
            # Adjust text to avoid overlap
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='darkred', lw=0.8))


        if axis_log:
            # Logarithmic scale for the axes
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        # Add grid lines with bold lines for 0 values for both axes
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        ax.grid(color='grey', linewidth=0.5)
           
        # Add a title and labels
        ax.set_title('PCA Loadings')
        ax.set_xlabel('PC1 Loadings')
        ax.set_ylabel('PC2 Loadings')
        name = main_folder +'/statistics/'+ output_file_prefix + '_' + plt_name_suffix
        for suffix in suffixes:
            plt.savefig(name + suffix, bbox_inches='tight', dpi = 300)

        plt.show()

        #---------------------------------------------
        # REPORTING
        text0 = 'PCA loadings plot was created and added into: ' + name
        text1 = 'Candidate features (99.5th percentile) also highlighted in loadings graph are: ' + str(candidate_loadings)
        report.add_together([
            ('text', text0),
            ('image', name),
            ('text', text1, 'normal', 'center'),
            'pagebreak'])
        return fig
    
    def visualizer_PCA_grouped(self, color_column, marker_column, cmap = 'nipy_spectral', crossout_outliers = False, plt_name_suffix = 'PCA_grouped', graph_title = 'PCA graph', annotate_samples = False, nm_to_annotate = 10):
        """
        Create a PCA plot with colors based on one column and markers based on another column from the metadata.

        Parameters
        ----------
        color_column : str or None
            Name of the column to use for coloring. ! If you input 'None', then all will be grey. (From metadata, e.g. 'Age', etc.)
        marker_column : str or None
            Name of the column to use for markers. ! If you input 'None', then all markers will be same. (From metadata e.g. 'Diagnosis', etc.)
        cmap : str
            Name of the colormap. Default is 'nipy_spectral'. (Other options are 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted', 'turbo'; ADD '_r' to get reversed colormap)
        crossout_outliers : bool
            If True, the outliers will be marked with an 'X'. Default is False.
        plt_name_suffix : str
            Suffix for the plot name. Default is 'PCA_grouped'. (Useful when using multiple times - to avoid overwriting the previous plot)  
        graph_title : str
            Title of the graph. Default is 'PCA graph'.
        annotate_samples : bool
            If True, the samples will be annotated with their names. Default is False.
        nm_to_annotate : int
            Number of samples to annotate. Default is 10. (Samples with the farthest distance from the origin will be annotated)
        """
        metadata = self.metadata
        report = self.report
        output_file_prefix = self.output_file_prefix
        main_folder = self.main_folder
        suffixes = self.suffixes
        if self.pca_data is None:
            raise ValueError('PCA was not performed yet. Run PCA first.')
        else:
            pca_df = self.pca_df
            per_var = self.pca_per_var
        ## PCA for both (with different markers)
        column_name = color_column
        second_column_name = marker_column

        cmap = mpl.colormaps[cmap]

        if crossout_outliers:
            pca_df['PC1_zscore'] = zscore(pca_df['PC1'])
            pca_df['PC2_zscore'] = zscore(pca_df['PC2'])

            # Identify outliers as any points where the absolute z-score is greater than 3
            outliers = pca_df[(np.abs(pca_df['PC1_zscore']) > 3) | (np.abs(pca_df['PC2_zscore']) > 3)]
        
        #get the color (for first column)
        if column_name is not None:
            column_unique_values = metadata[column_name].unique()
            # Order them alphabetically, taking numbers into account
            column_unique_values = sorted(column_unique_values, key=self._natural_sort_key)

            num_unique_values = len(column_unique_values)
            color_indices = np.linspace(0.05, 0.95, num_unique_values)
            colors = [cmap(i) for i in color_indices]
            class_type_colors = dict(zip(column_unique_values, colors))

        #get markers (for second column)
        if second_column_name is not None:
            second_column_unique_values = metadata[second_column_name].unique()
            # Order them alphabetically, taking numbers into account
            second_column_unique_values = sorted(second_column_unique_values, key=self._natural_sort_key)

            markers = cycle(['o', 'v', 's', '^', '*', '<','p', '>', 'h', 'H', 'D', 'd', 'P', 'X'])
            class_type_markers = dict(zip(second_column_unique_values, markers))

        if column_name is not None and second_column_name is not None:
            existing_combinations = set(zip(metadata[column_name], metadata[second_column_name]))
            # Order them alphabetically, taking numbers into account
            existing_combinations = sorted(existing_combinations, key=self._natural_sort_key)

        elif column_name is not None:
            existing_combinations = set(metadata[column_name])
            # Order them alphabetically, taking numbers into account
            existing_combinations = sorted(existing_combinations, key=self._natural_sort_key)
        elif second_column_name is not None:
            existing_combinations = set(metadata[second_column_name])
            # Order them alphabetically, taking numbers into account
            existing_combinations = sorted(existing_combinations, key=self._natural_sort_key)
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
                df_samples = pca_df.loc[(metadata[column_name] == sample_type) & (metadata[second_column_name] == class_type)]
            elif sample_type is not None:
                df_samples = pca_df.loc[metadata[column_name] == sample_type]
            elif class_type is not None:
                df_samples = pca_df.loc[metadata[second_column_name] == class_type]

           
            # Get the color and marker, handling cases where one of the types is None
            color = 'grey' if sample_type is None else class_type_colors.get(sample_type, 'grey')
            marker = 'o' if class_type is None else class_type_markers.get(class_type, 'o')

            # Construct label based on available types
            label = f"{sample_type or ''} - {class_type or ''}".strip(' -')
    
            # Plot the samples
            plt.scatter(df_samples['PC1'], df_samples['PC2'], color=color, marker=marker, label=label, alpha=0.6, s=20)
            
            # Compute the covariance matrix and find the major and minor axis only if there are enough data points
            if len(df_samples) >= 3:
                covmat = np.cov(df_samples[['PC1', 'PC2']].values.T)
                if np.isinf(covmat).any() or np.isnan(covmat).any():
                    print(f"Skipping ellipse for combination {combination} due to invalid covariance matrix.")
                    continue
                lambda_, v = np.linalg.eig(covmat)
                lambda_ = np.sqrt(lambda_)

                # Draw an ellipse around the samples
                ell = Ellipse(xy=(np.mean(df_samples['PC1']), np.mean(df_samples['PC2'])),
                              width=lambda_[0] * 2, height=lambda_[1] * 2, 
                              angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])), edgecolor=color, lw=1, facecolor='none', alpha=0.6)
                plt.gca().add_artist(ell)

        texts = []
        if annotate_samples:
            # Calculate distances from origin for all samples
            pca_df['distance'] = np.linalg.norm(pca_df[['PC1', 'PC2']].values, axis=1)
            # Select top N farthest samples
            farthest = pca_df.nlargest(nm_to_annotate, 'distance')
            for i, row in farthest.iterrows():
                sample_name = self.metadata.iloc[i]['Sample File']
                texts.append(ax.text(row['PC1'], row['PC2'], sample_name, fontsize=8, color='black', alpha=0.7, bbox=dict(facecolor="white", alpha=0.5, edgecolor="none")))
        # Adjust text to avoid overlap
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

        # If outliers are to be crossed out
        if crossout_outliers:
            # Plot the outliers
            plt.scatter(outliers['PC1'], outliers['PC2'], color='black', marker='x', label='Outliers', alpha=0.6, s=20)

        # Create the legend
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        plt.title(graph_title)
        plt.xlabel('PC1 - {0}%'.format(per_var[0]))
        plt.ylabel('PC2 - {0}%'.format(per_var[1]))
        name = main_folder +'/statistics/'+ output_file_prefix + '_' + plt_name_suffix
        for suffix in suffixes:
            plt.savefig(name + suffix, bbox_inches='tight', dpi = 300)
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

    def visualizer_PLSDA(self, cmap = 'nipy_spectral', plt_name_suffix = 'PLS-DA', graph_title = 'PLS-DA graph'):
        """
        Visualize the results of PLS-DA.

        Parameters
        ----------
        cmap : str
            Name of the colormap. Default is 'nipy_spectral'. (Other options are 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted', 'turbo'; ADD '_r' to get reversed colormap)
        plt_name_suffix : str
            Suffix for the plot name. Default is 'PLS-DA'. (Useful when using multiple times - to avoid overwriting the previous plot)
        graph_title : str
            Title for the plot. Default is 'PLS-DA graph'.
        """
        metadata = self.plsda_metadata
        if metadata is None:
            raise ValueError('PLS-DA was not performed yet. Run PLS-DA first.') 
        report = self.report
        output_file_prefix = self.output_file_prefix
        main_folder = self.main_folder
        suffixes = self.suffixes

        if self.plsda_model is None:
            raise ValueError('PLS-DA was not performed yet. Run PLS-DA first.')
        else:
            model = self.plsda_model
            response_column_names = self.plsda_response_column

        if isinstance(response_column_names, list):
            missing = [col for col in response_column_names if col not in metadata.columns]
            if missing:
                # Combine columns to create a new one for visualization
                combined_col_name = str(response_column_names)
                metadata[combined_col_name] = metadata[response_column_names].apply(lambda x: '-'.join(x.map(str)), axis=1)
            else:
                combined_col_name = response_column_names[0] if len(response_column_names) == 1 else str(response_column_names)
        else:
            if response_column_names not in metadata.columns:
                combined_col_name = str(response_column_names)
                metadata[combined_col_name] = metadata[response_column_names].astype(str)
            else:
                combined_col_name = response_column_names

        cmap = mpl.cm.get_cmap(cmap)

        # Plot the data in the space of the first two components
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Get the unique values of the response column
        response_unique_values = metadata[str(response_column_names)].unique()
        # Order them alphabetically, taking numbers into account
        response_unique_values = sorted(response_unique_values, key=self._natural_sort_key)

        num_unique_values = len(response_unique_values)
        color_indices = np.linspace(0.05, 0.95, num_unique_values)
        colors = [cmap(i) for i in color_indices]
        response_colors = dict(zip(response_unique_values, colors))

        # Create a scatter plot of the data
        for response in response_unique_values:
            df_samples = model.x_scores_[metadata[str(response_column_names)] == response]
            ax.scatter(df_samples[:, 0], df_samples[:, 1], color=response_colors[response], label=response, alpha=0.6, s=20)
        
        # Draw ellipses around the samples
        for response in response_unique_values:
            df_samples = model.x_scores_[metadata[str(response_column_names)] == response]
            if len(df_samples) < 3: 
                print(f"Skipping ellipse for response {response} due to insufficient data points. 2 is possible, but looks weird, so 3 is the minimum.")
                continue
            covmat = np.cov(df_samples.T)
            if np.isinf(covmat).any() or np.isnan(covmat).any():
                print(f"Skipping ellipse for response {response} due to invalid covariance matrix.")
                continue
            lambda_, v = np.linalg.eig(covmat)
            lambda_ = np.sqrt(lambda_)
            ell = Ellipse(xy=(np.mean(df_samples[:, 0]), np.mean(df_samples[:, 1])),
                        width=lambda_[0] * 2, height=lambda_[1] * 2,
                        angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])), edgecolor=response_colors[response], lw=1, facecolor='none', alpha=0.6)
            ax.add_artist(ell)
            
        ax.set_xlabel('PLS-DA component 1')
        ax.set_ylabel('PLS-DA component 2')
        ax.set_title(graph_title)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        name = main_folder +'/statistics/'+output_file_prefix + '_' + plt_name_suffix
        for suffix in suffixes:
            plt.savefig(name + suffix, bbox_inches='tight', dpi = 300)
        plt.show()

        #---------------------------------------------
        # REPORTING
        text0 = '<b>PLS-DA</b> was visualized with the ' + str(response_column_names) + ' column as a response variable for the model.'
        text1 = 'PLS-DA plot was created and added into: ' + name
        report.add_together([
            ('text', text0),
            ('text', text1, 'italic', 'left', 8),
            ('image', name),
            'pagebreak'])
        return fig, ax
    
    def visualizer_violin_plots(self, column_names, indexes = 'all', save_into_pdf = True, cmap = 'nipy_spectral', plt_name_suffix = 'violin_plots', bw = 0.2, jitter = True, label_rotation = 0, show_all = False):
        """
        Visualize violin plots of all features grouped by a column from the metadata. (And save them into a single PDF file)

        Parameters
        ----------
        column_names : str or list
            Name of the column to use as a grouping factor. (From metadata, e.g. 'Sample Type' or list for multiple columns ['Sample Type', 'Diagnosis'], etc.)
        indexes : str, int or list
            Indexes of the features to create/save violin plots for. Default is 'all'. (Other options are either int or list of ints)
        save_into_pdf : bool
            If True, save each violin plot into a PDF file (multiple and later merged into a single PDF file). Default is True. (Usefull but slows down the process) If False, violin plots will be only shown.
        cmap : str
            Name of the colormap. Default is 'nipy_spectral'. (Other options are 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted', 'turbo'; ADD '_r' to get reversed colormap)
        plt_name_suffix : str
            Suffix for the plot name. Default is 'violin_plots'. (Useful when using multiple times - to avoid overwriting the previous plot)
        bw : float
            Bandwidth parameter for the kernel density estimation. Default is 0.2.
        jitter : bool
            If True, add jitter to the x-axis. Default is True
        label_rotation : int
            Rotation of the x-axis labels. Default is 0. (Useful for long labels, then use 90)
        show_all : bool
            If True, show all violin plots during the process. Default is False (to speed up the process).
        """
        data = self.data
        metadata = self.metadata
        report = self.report
        suffixes = self.suffixes

        if indexes == 'all':
            all_indexes = True
            indexes = range(len(data))
            if save_into_pdf == False:
                raise ValueError('Showing all violin plots without saving them into a PDF file is not recommended due to the large number of plots that will be created but not saved for later use.')
        elif isinstance(indexes, int):
            indexes = [indexes]
            all_indexes = False
        elif isinstance(indexes, list):
            # remove all duplicates from the list (there is no need to plot the same feature multiple times)
            indexes = list(set(indexes))
            all_indexes = False
        else:
            raise ValueError('Indexes must be either "all", int or list of ints.')

        cmap = mpl.cm.get_cmap(cmap)

        example_name = None

        # Transpose 'data' so that rows represent samples and columns represent compounds
        data_transposed = data.T

        if isinstance(column_names, list):
                # create temporary column for the response variable by concatenating the response columns
                metadata[str(column_names)] = metadata[column_names].apply(lambda x: '_'.join(x.map(str)), axis=1)
                column_names = str(column_names)

        # Add column to 'data_transposed'
        column = metadata[column_names].to_list()
        column.insert(0, None)
        data_transposed[column_names] = column

        column_unique_values = data_transposed[column_names].unique()
        
        # Order them alphabetically, taking numbers into account
        column_unique_values = sorted([val for val in column_unique_values if val is not None], key=self._natural_sort_key)

        num_unique_values = len(column_unique_values)
        color_indices = np.linspace(0.05, 0.95, num_unique_values)
        colors = [cmap(i) for i in color_indices]

        # Group 'data_transposed' by 'column_name'
        grouped = data_transposed.groupby(column_names)

        # Create x-axis labels with the number of points
        x_labels = [f"{val} ({len(group)})" for val, group in grouped if val is not None]

        # Delete this column from the data_transposed 
        data_transposed = data_transposed.drop([column_names], axis=1)

        pdf_files = []
        first_plot = True
        # Create a folder for the violin plots
        example = True
        for idx, index in enumerate(indexes):
            values = []
            for unique, group in grouped:
                if unique is not None:
                    sample_values = group.iloc[:, index].dropna().to_list()  # Drop missing values
                    if sample_values != [] and all(isinstance(x, (int, float)) for x in sample_values):  # Check if all values are numeric
                        values.append(sample_values)
                    else:
                        values.append([0])  # Add a list with a single zero if data is missing or non-numeric
                        print(unique)

            if values:  # Check if 'values' is not empty
                vp = plt.violinplot(values, showmeans=False, showmedians=False, showextrema=False, bw_method=bw)

                # Change the color and width of the violins
                for i in range(len(vp['bodies'])):
                    vp['bodies'][i].set_facecolor(colors[i])
                    vp['bodies'][i].set_edgecolor(colors[i])
                    vp['bodies'][i].set_linewidth(1)

                # Add custom mean and median lines
                for i, v in enumerate(values):
                    plt.plot([i + 1 - 0.2, i + 1 + 0.2], [np.mean(v)] * 2, color=colors[i], linewidth=0.5)  # mean line
                    plt.plot([i + 1 - 0.2, i + 1 + 0.2], [np.median(v)] * 2, color=colors[i], linewidth=0.5)  # median line

                # Scatter all the points with adaptive jitter
                if jitter:
                    for i in range(len(values)):
                        if len(values[i]) > 1:
                            kde = gaussian_kde(values[i])
                            densities = kde(values[i])
                            x_jitter = 0.05 * densities / densities.max()
                            x_jittered = np.random.normal(i + 1, x_jitter, size=len(values[i]))
                        else:
                            x_jittered = np.full(len(values[i]), i + 1)
                        plt.scatter(x_jittered, values[i], color=colors[i], s=5, alpha=1)
                else:
                    for i in range(len(values)):
                        plt.scatter(np.full(len(values[i]), i + 1), values[i], color=colors[i], s=5, alpha=1)
                
                plt.xticks(np.arange(1, len(column_unique_values) + 1), x_labels, rotation=label_rotation)
                cpd_title = data_transposed.loc['cpdID', index]
                plt.title(cpd_title)
            
                # Save the figure to a separate image file
                if save_into_pdf:
                    file_name = self.main_folder + '/statistics/' + self.report_file_name+ '-' + str(column_names) + '_' + plt_name_suffix +  f'_{index}.pdf'
                    plt.savefig(file_name, bbox_inches='tight')
                    pdf_files.append(file_name)

                if example:
                    example_name = self.main_folder + '/statistics/'+ str(column_names) + '-' + plt_name_suffix +'-example-' + str(index)
                    for suffix in suffixes:
                        plt.savefig(example_name + suffix, bbox_inches='tight', dpi = 300)
                    if first_plot == True or show_all == True:
                        plt.show() # Show the plot
                    returning_vp = vp
                    example = False
                
                #print progress
                ending = '\n' if idx == len(indexes) - 1 else '\r'
                print(f'Violin plots created: { (idx +1) / len(indexes) * 100:.2f}%', end=ending)
                plt.close()
              #---------------------------------------------
            first_plot = False

        # MERGING all the PDF files into a single PDF file 
        #(this way we should avoid loading all the plots into memory while creating all the plots)
        #(we will load them eventually when creating the final PDF file, but it will not slow down the process of creating the plots)   

        # Add all PDF files to a list
        if save_into_pdf:
            name = self.main_folder + '/statistics/' + self.report_file_name + '-'+ str(column_names)+ '_' + plt_name_suffix +'.pdf'
            # Merge all PDF files into a single PDF file
            report.merge_pdfs(pdf_files, name)

        #---------------------------------------------
        # REPORTING
        if indexes == 'all':
            text0 = 'Violin plots of <b>all features</b> grouped by '+ column_names +' column/s, were created'
            if save_into_pdf:
                text0 += ' and added into one file: <b>' + name + '</b>'
            else:
                text0 += '.'
        else:
            text0 = 'Violin plots of <b>selected features</b> grouped by '+ column_names +' column/s, were created'
            if save_into_pdf:
                text0 += ' and added into one file: <b>' + name + '</b>'
            else:
                text0 += '.'
        if example_name:
            text1 = 'Additionaly an example of such violin plot (shown below) was saved separately as: ' + example_name
        else:
            text1 = 'No example of violin plot was saved separately.'
        report.add_together([
            ('text', text0),
            ('text', text1),
            ('image', example_name),
            'line'])
        # Delete the temporary PDF files
        if save_into_pdf:
            for file in pdf_files:
                os.remove(file)

        return returning_vp
    
    
    def visualizer_fold_change(self, groups_column_name, group1, group2, color_left = 'green', color_right = 'blue', fold2_change_threshold=1, p_value_threshold=0.05, p_value_correction_method = '',plt_name_suffix='fold_change'):
        """
        Visualize fold change and p-value for two groups.

        Parameters
        ----------
        groups_column_name : str
            Name of the column in metadata that contains group labels.
        group1 : str
            Name of the first group.
        group2 : str
            Name of the second group.
        color_left : str
            Color for the left box. Default is 'green'.
        color_right : str
            Color for the right box. Default is 'blue'.
        fold2_change_threshold : float
            Threshold for the fold change. Default is 1.
        p_value_threshold : float
            Threshold for the p-value. Default is 0.05.
        p_value_correction_method : str
            Method for p-value correction. Default is '' (which means without a correction); others are: 'fdr_bh', 'bonferroni', 'holm', 'simes-hochberg', 'hommel', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'. (for more information see: https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html#statsmodels.stats.multitest.multipletests )
        plt_name_suffix : str
            Suffix for the plot name. Default is 'fold_change'.
        """
        data = self.data
        metadata = self.metadata
        report = self.report
        output_file_prefix = self.output_file_prefix
        main_folder = self.main_folder
        suffixes = self.suffixes
        fold_change_count = self.fold_change_count

        if p_value_threshold <= 0:
            raise ValueError('P-value threshold must be greater than 0.')
        
        if fold2_change_threshold < 0:
            raise ValueError('Fold change threshold must be greater than or equal to 0.')
        
        # Create masks for the two groups
        group1_mask = metadata[groups_column_name] == group1
        group2_mask = metadata[groups_column_name] == group2

        # Add False to the beginning of the masks to align them with the data (first column is cpdID)
        group1_mask = np.insert(group1_mask.to_numpy(), 0, False)
        group2_mask = np.insert(group2_mask.to_numpy(), 0, False)

        # Get the data for the two groups
        group1_data = data.iloc[:, group1_mask]
        group2_data = data.iloc[:, group2_mask]

        fold_change = group2_data.mean(axis=1) / group1_data.mean(axis=1)
        # Get p-values
        p_values = ttest_ind(group1_data, group2_data, axis=1)[1]

        if p_value_correction_method  != '':
            # Use correction for p-values 
            p_values = multipletests(p_values, method=p_value_correction_method)[1]

        # Create a scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot for all points
        plt.scatter(np.log2(fold_change), -np.log10(p_values), color='grey', alpha=0.5)

        # Highlight significant points
        significant_fc_right = (np.log2(fold_change) > (fold2_change_threshold))
        significant_fc_left = (np.log2(fold_change) < -(fold2_change_threshold))
        significant_pv = (p_values < p_value_threshold)
        significant_pv_right = (np.log2(fold_change) > 0) & (p_values < p_value_threshold)
        significant_pv_left = (np.log2(fold_change) < 0) & (p_values < p_value_threshold)
        significant_fc_right_and_pv = significant_fc_right & significant_pv
        significant_fc_left_and_pv = significant_fc_left & significant_pv

        # Add those significant for both fold changes and p-values into candidates
        # Get the feature names 
        feature_names = data.iloc[:, 0].to_numpy()
     
        # Calculate scores from -log10(p-values)
        scores = -np.log10(p_values)

        # Convert feature_names to a pandas Series
        feature_names_series = pd.Series(feature_names)

        fold_change_count += 1
        # Add candidates using the filtered feature names
        self.add_candidates(
            feature_names_series[significant_fc_right_and_pv].tolist(),  # Convert back to a list if needed
            method='FoldChange(right; ' + str(p_value_correction_method) +')',
            specification=str(group1 + ' - ' + group2) + ' Analysis-' + str(fold_change_count),
            scores=scores[significant_fc_right_and_pv]
        )

        self.add_candidates(
            feature_names_series[significant_fc_left_and_pv].tolist(),  # Convert back to a list if needed
            method='FoldChange(left; ' + str(p_value_correction_method) +')',
            specification=str(group1 + ' - ' + group2) + ' Analysis-' + str(fold_change_count),
            scores=scores[significant_fc_left_and_pv]
)
        # Scatter those significant for pv
        plt.scatter(np.log2(fold_change[significant_pv]), -np.log10(p_values[significant_pv]), color='grey', alpha=0.7, edgecolors="black")
        
        # Scatter those significant for fold changes right
        plt.scatter(np.log2(fold_change[significant_fc_right]), -np.log10(p_values[significant_fc_right]), color=color_right, alpha=0.6, edgecolors="black")

        # Scatter those significant for fold changes left
        plt.scatter(np.log2(fold_change[significant_fc_left]), -np.log10(p_values[significant_fc_left]), color=color_left, alpha=0.6, edgecolors="black")
        
        # Add lines for 0 for both axes
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)

        # Add lines for fold change threshold and p-value threshold
        plt.axvline(x=fold2_change_threshold, color=color_right, linestyle='--', alpha=0.5, linewidth=1)
        plt.axvline(x=-fold2_change_threshold, color=color_left, linestyle='--', alpha=0.5, linewidth=1)
        plt.axhline(y=-np.log10(p_value_threshold), color='black', linestyle='--', alpha=0.5, linewidth=1)


        # Identify indices of significant points
        significant_indices = [i for i in range(len(fold_change)) if significant_fc_right_and_pv[i] or significant_fc_left_and_pv[i]]

        # Check if the number of significant points exceeds the threshold
        if len(significant_indices) > 25:
            print('Too many candidates to annotate (more than 25). Skipping annotation.')
        else:
            # Annotate the significant points
            texts = [plt.annotate(
                feature_names[i], 
                (np.log2(fold_change[i]), -np.log10(p_values[i])), 
                fontsize=10, 
                alpha=1, 
                color="darkred",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none")  # Semi-transparent white background
            ) for i in significant_indices]
            
            # Adjust text to avoid overlap
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='darkred', lw=0.8))
       
        plt.xlabel('Log2 Fold Change')
        plt.ylabel('-Log10 P-Value')
        plt.title(f'Fold Change vs P-Value: {group1} vs {group2}')

        # Add values above the graph
        
        plt.text(0.99, 1.05, f'{sum(significant_fc_right_and_pv)} ↑ ({sum(significant_pv_right)} ≥  -Log10({str(p_value_threshold)}), {sum(significant_fc_right)} ≥  {str(fold2_change_threshold)}', color=color_right, transform=ax.transAxes, ha='right')
        plt.text(0.01, 1.05, f'{sum(significant_fc_left_and_pv)} ↓ ({sum(significant_pv_left)} ≥  -Log10({str(p_value_threshold)}), {sum(significant_fc_left)} ≥  {str(fold2_change_threshold)}', color=color_left, transform=ax.transAxes, ha='left')
        
        # Add fold_change and p-value to the DataFrame
        fold_change_df = pd.DataFrame({
            'cpdID': feature_names,
            'fold_change': fold_change,
            'p_value ('+ (p_value_correction_method)+ ')': p_values
        })
        # Order by lowest p-value
        fold_change_df = fold_change_df.sort_values(by='p_value ('+ (p_value_correction_method)+ ')', ascending=True)
        fold_change_df = fold_change_df.reset_index(drop=True)
        
        self.fold_change = fold_change_df
        self.fold_change_count = fold_change_count

        # Save the plot
        plt_name = main_folder + '/statistics/' + output_file_prefix + '_' + plt_name_suffix
        for suffix in suffixes:
            plt.savefig(plt_name + suffix, bbox_inches='tight', dpi=300)
        plt.show()

        #---------------------------------------------
        # REPORTING
        text = f'Fold change plot for {group1} vs {group2} was created and added into: {plt_name}'
        report.add_together([
            ('text', text),
            ('image', plt_name),
            'line'])

        return plt_name
    
    def visualizer_candidates(self, cmap = 'nipy_spectral', plt_name_suffix = 'candidates'):
        """
        Visualize all candidates found during the analysis.


        Parameters
        ----------
        cmap : str
            Name of the colormap. Default is 'nipy_spectral'. (Other options are 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted', 'turbo'; ADD '_r' to get reversed colormap)
        plt_name_suffix : str
            Suffix for the plot name. Default is 'candidates'. (Useful when using multiple times - to avoid overwriting the previous plot)
        
        """
        candidates = self.candidates # DataFrame with candidates
        report = self.report
        output_file_prefix = self.output_file_prefix
        main_folder = self.main_folder
        suffixes = self.suffixes
        if candidates is None:
            raise ValueError('No candidates were found. You need to run PCA, PLS-DA or other analysis that produces candidates first.')
        
        plot_names = []
        plots = []
        k = 0
        # For each group of candidates (method + specification) create a bar plot
        for method, group in candidates.groupby(['method', 'specification']):
            # Create a bar plot for each group of candidates
            fig, ax = plt.subplots(figsize=(10, 8))

            # Get the score for each candidate
            scores = group['score'].values
            # Get the name of each candidate
            names = group['feature'].values

            # Get the color for each candidate
            cmap = mpl.cm.get_cmap(cmap)
            num_candidates = len(scores)
            color_indices = np.linspace(0.05, 0.95, num_candidates)
            colors = [cmap(i) for i in color_indices]
            candidates_colors = dict(zip(names, colors))

            # Create a bar plot
            ax.barh(names, scores, color=[candidates_colors[name] for name in names], alpha=0.6)
            ax.set_xlabel(f'Score - {method}')
            ax.set_title(f'Candidates - {method}')
            ax.set_xlim(0, 1.05 * max(scores))
            ax.set_ylim(-1, num_candidates + 1)
            ax.set_yticks(np.arange(num_candidates))
            ax.set_yticklabels(names, rotation=0)
            ax.invert_yaxis()

            name = main_folder + '/statistics/' + output_file_prefix + '_' + plt_name_suffix + '_' + str(k)
            # Save the plot
            for suffix in suffixes:
                plt.savefig(name + suffix, bbox_inches='tight', dpi=300)
            plt.show()
            k += 1
        
        #---------------------------------------------
        # REPORTING
        text = 'Candidates were shown in the bar plots and added into: ' + str(plot_names)
        report.add_text(text)
        # Add the plots to the report
        for img in plots:
            report.add_image(img)
        report.add_line()    
        
        return plot_names, plots


