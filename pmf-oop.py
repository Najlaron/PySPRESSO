# My modules
import pdf_reporter as pdf_rptr
import pmf 

# Other modules

import os
import pandas as pd



class WorkFlow:
    """
    Workflow: This class contains all methods and variables to run the workflow. 
    ----------
    ----------
    
    It contains the following classes:
    
    Tasks:
    - Filter
    - Correcter
    - Visualizer

    Managers:
    - _Manager

    Handlers:
    - _DataHandler
    - _ReportHandler

    It stores the different tasks that need to be run in specific order during the workflow.
    """
    def __init__(self, name):
        self.name = name

        # Variables to store the different objects
        self._Manager = WorkFlow._Manager
        self._DataHandler = WorkFlow._DataHandler
        self._ReportHandler = WorkFlow._ReportHandler

        self.report = self._ReportHandler._initializeReport(self.reportFile, self.mainFolder)

        # Variables to store the input data
        self.inputDataFile = self._DataHandler(self._Manager.askUser('Please enter the name of the input data file: '))
        self.inputBatchInfo = self._DataHandler(self._Manager.askUser('Please enter the name of the input batch info file: '))
        self.mainFolder = self._Manager.askUser('Please enter desired name of the main folder: ')

        # Variables to store the different data matrices (+ initialization)
        self.data = self._DataHandler.addCpdID(self._DataHandler.loadData(self.inputDataFile)) #Load data and add cpdID
        self.batchInfo = self._DataHandler.loadBatchInfo(self.inputBatchInfo) #Load batch info
        self.data, self.batch = self._DataHandler.batchByName(self.data, self.batchInfo) #Sort data by batch and add batch column

        self.variableMetadata = self._DataExtracter.extractVariableMetadata(self.data) #Extract variable metadata
        self.metadata = self._DataExtracter.extractMetadata(self.batchInfo, self._Manager.askUser('Please enter the group columns to keep: ').split(',')) #Extract metadata from batch info by choosing columns to keep
        
        # Variables to store the different column names
        self.batchColumnName = None
        self.batch = None
        self.qcColumName = None
        self.qcSamples = None

        # Variables to store the different outputs
        self.outputPrefix = self._Manager.askUser('Please enter the desired output prefix - this will be used for all output files (e.g. "MyStudy" will generate outputs as "MyStudy-data.csv", etc.): ')
        self.reportFile = self._Manager.askUser('Please enter the desired name of the report file: ')
        self.pictureSuffixes = ['.png', '.pdf']        

        # Variable to store all the tasks (functions) that need to be run
        self.tasks = []

    #-------------------------------------------------
    def add_task(self, task):
        self.tasks.append(task)


    def run(self):
        for task in self.tasks:
            task.run()

    def save(self):
        pass

    #-----------------------------------------------------------------------------------------------------------------
    # TASKS = Classes - different types of tasks that can be run/autorun during the workflow
    class Filter:
        """
        Filter: This "task" class contains all methods to filter the data according to different criteria during the workflow.
        """
        def __init__(self, name):
            self.name = name

        def filterBlankIntensityRatio(self, data, variableMetadata, metadata, threshold = 20, setting = 'first'):
            """
            Function to filter the data based on the blank intensity ratio.

            Parameters:
            -----------
            data: Dataframe
                data matrix
            variableMetadata: Dataframe
                variable metadata matrix
            metadata: Dataframe
                metadata matrix
            threshold: float
                threshold for the blank intensity ratio; Default is 20
            setting: str
                setting for the blank intensity ratio; Default is 'first' (other options are 'median', 'mean', 'min', 'last')

            """
            data = pmf.filter_blank_intensity_ratio(data, variableMetadata, metadata, threshold, setting)
            variableMetadata = pmf.filter_match_variable_metadata(data, variableMetadata)
            return data, variableMetadata, metadata
        
        def filterMissingValues(self, data, variableMetadata, metadata, qc_threshold = 0.8, sample_threshold = 0.5):
            """
            Function to filter out features with high number of missing values (nan or 0): A) within QC samples B) all samples

            Parameters:
            -----------
            data: Dataframe 
                data matrix
            variableMetadata: Dataframe
                variable metadata matrix
            metadata: Dataframe
                metadata matrix
            qc_threshold: float
                threshold for the QC threshold; Default is 0.8
            sample_threshold: float
                threshold for the sample threshold; Default is 0.5
            """

            data = pmf.filter_missing_values(data, variableMetadata, metadata, qc_threshold, sample_threshold)
            variableMetadata = pmf.filter_match_variable_metadata(data, variableMetadata)
            return data, variableMetadata, metadata
        
        def filterRSD(self, data, variableMetadata, metadata, rsd_threshold = 20):
            """
            Function to filter out features with RSD% over the threshold.

            Parameters:
            -----------
            data : DataFrame
                DataFrame with the data.
            variable_metadata : DataFrame
                DataFrame with the variable metadata.
            QC_samples : list
                List of QC samples. (QC_samples = metadata[metadata['Sample Type'] == 'Quality Control']['Sample File'].tolist())
            rsd_threshold : float
                Threshold for the RSD%. Default is 20.
            """
            data = pmf.filter_rsd(data, variableMetadata, metadata, rsd_threshold)
            variableMetadata = pmf.filter_match_variable_metadata(data, variableMetadata)
            return data, variableMetadata, metadata
        
        def filterNumberOfCorrectedBatches(self, data, variableMetadata, metadata, threshold = 0.8):
            """
            Filter out features that were corrected in less than the threshold number of batches.

            Parameters:
            -----------
            data : DataFrame
                DataFrame with the data.
            variable_metadata : DataFrame
                DataFrame with the variable metadata.
            threshold : float
                Threshold for the number of batches. Default is 0.8. (If less than 1, then it is interpreted as a percentage otherwise as a minimum number of batches corrected.)
            """
            data = pmf.filter_number_of_corrected_batches(data, variableMetadata, metadata, threshold)
            variableMetadata = pmf.filter_match_variable_metadata(data, variableMetadata)
            return data, variableMetadata, metadata

        def run(self):
            print(f'Running filter {self.name}')


    class Correcter:
        """
        Correcter: This "task" class contains all methods to correct/change the data during the workflow.
        """
        def __init__(self, name):
            self.name = name

        def correctByQCs(self, data, variableMetadata, metadata, QC_samples, batch = None, p_values = 'default', show = 'default', use_log = True, use_norm = True, use_zeros = False, cmap = 'viridis'):
            """
            Correct batch effects and systematic errors using the QC samples for interpolation. (Includes visualization of the results.)

            Parameters:
            -----------
            df : DataFrame
                DataFrame with the data.
            variable_metadata : DataFrame
                DataFrame with the variable metadata.
            QC_samples : list
                List of QC samples. (QC_samples = metadata[metadata['Sample Type'] == 'Quality Control']['Sample File'].tolist())
            batch : list
                List of batches. Default is None. If None = all samples are in one batch else in list (e.g. ['001', '001', ..., '002', '002', ...]).
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

            It is important to note that the function is computationally expensive and may take a long time to run.

            Additionally it is highly advised to use the default values for the p_values parameter and both logaritmic and normalization transformations (unless they were already applied to the data)
            """
            data = pmf.qc_correction(data, variableMetadata, metadata, QC_samples, batch, p_values, show, use_log, use_norm, use_zeros, cmap)
            return data, variableMetadata, metadata
    
    class Visualizer:
        """
        Visualizer: This "task" class contains all methods to visualize the data during the workflow.
        """
        def __init__(self, name):
            self.name = name
    

        def visualizeBoxplot(self, data, QC_samples, sufixes = ['.png', '.pdf']):
            """
            Function to visualize the whole data using boxplots.

            Parameters:
            -----------
            data: Dataframe
                data matrix
            QC_samples: list
                list of QC samples
            sufixes: list
                list of sufixes for the pictures
            """
            fig, ax = pmf.visualize_boxplot(data, QC_samples, sufixes)
            return fig, ax
        
        def visualizeByBatch(self, data, QC_samples, main_folder, show = 'default', sufixes = ['.png', '.pdf'], batch = None, cmap = 'viridis'):
           """
            Function to visualize samples by batch. Highlight QC samples.

            Parameters:
            ----------
            data : DataFrame
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
           pmf.visualize_by_batch(data, QC_samples, main_folder, show = show, sufixes = sufixes, batch = batch, cmap = cmap)
           return None

    #-----------------------------------------------------------------------------------------------------------------
    # MANAGERS = Classes - different types of managers that can be used during the workflow
    class _Manager:
        """
        Manager: This "manager" class is used to communicate with the user if/when there is an error or variable that needs to be set by the user.
        """
        def __init__(self, name):
            self.name = name

        def askUser(self, question):
            print(question)
            return input()
        

    #-----------------------------------------------------------------------------------------------------------------
    # HANDLERS = Classes - different types of handlers that are used behind the scenes during the workflow
    class _DataHandler:
        """
        DataHandler: This "handler" class is used to handle the data (loading and saving) that is being used/changed druring the workflow.
        """
        def __init__(self, name):
            self.name = name

        def loadData(self, data_input_file_name, separator = ';', encoding = 'ISO-8859-1'):
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
            return pmf.load_data(data_input_file_name, separator, encoding)
        
        def loadBatchInfo(self, input_batch_info_file):
            """
            Load batch_info matrix from a csv file.

            Parameters
            ----------
            input_batch_info_file : str
                Name of the batch info file.
            """
            return pmf.load_batch_info(input_batch_info_file)

        def addCpdID(self, data, mz_col = 'm/z', rt_col = 'RT [min]'):
            """
            Add a column 'cpdID' to the data_df calculated from mz and RT. Matching ID - distinguished by adding "_1", "_2", etc.

            Parameters
            ----------
            data : DataFrame
                DataFrame with the data.
            mz_col : str
                Name of the column with m/z values. Default is 'm/z'.
            rt_col : str
                Name of the column with RT values. Default is 'RT [min]'.
            """
            return  pmf.add_cpdID(data, 'm/z', 'RT [min]')
        
        def batchByName(self, data, batch_info, batch_column = 'Batch', format = '%d.%m.%Y %H:%M'):
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
            data, batch = pmf.batch_by_name(data, batch_info, batch_column, format)
            return data, batch
        
    class _DataExtracter:
        """
        DataExtracter: This "handler" class is used to extract different data matrices from the data matrix that is being used during the workflow.
        """
        def __init__(self, name):
            self.name = name

        def extractVariableMetadata(self, data, column_index_ranges = [(10, 15), (18, 23)]):
            """
            Extract variable metadata from the data. 

            Parameters
            ----------
            data : DataFrame
                DataFrame with the data.
            column_index_ranges : list
                List of tuples with the column index ranges to keep. Default is [(10, 15), (18, 23)].
            """
            return pmf.extract_variable_metadata(data, column_index_ranges)
        
        def extractData(self, data, prefix= 'Area:'):
            """
            Extract data matrix from the data.

            Parameters
            ----------
            data : DataFrame
                DataFrame with the data.
            prefix : str
                Prefix used in the column names to keep. Default is 'Area:'.
            """
            return pmf.extract_data(data, prefix)

        def extractMetadata(self, batch_info, group_columns_to_keep, prefix = 'Area:'):
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
            return pmf.extract_metadata(batch_info, group_columns_to_keep, prefix)

    class _ReportHandler:
        """
        ReportHandler: This "handler" class is used to handle the report that is being generated during the workflow.
        """
        def __init__(self, name):
            self.name = name

        def _initializeReport(report_file_name, main_folder, type = 'processing'):
            """
            Function to initialize the report for processing/statistics.

            report_file_name: name of the report file
            main_folder: name of the main folder
            type: 'processing' or 'statistics'
            """
            report_path = main_folder + '/' + report_file_name + '.pdf'
            title_text = main_folder

            report = pdf_rptr.Report(name = report_path, title = title_text)
            report.initialize_report(type)
            return report
        
    