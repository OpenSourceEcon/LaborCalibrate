'''
------------------------------------------------------------------------
This module creates the Calibrate class and define functions to load data
on consumption and hours worked by age from STATA files, to manipulate
the data to be then used for chi_n_vec calculation in line with formula (7.8)
in chapter 7 of the OG textbook.

This Python script imports the following module(s):
    elliputil.py
    constants.py
    utilities.py

This Python script defines the following function(s):
    load_large_dta()
    get_ndata()
    get_cdata()
    get_chi_n_vec()

------------------------------------------------------------------------
'''

import numpy as np
import pandas as pd
import elliputil as elp
import utilities as utils
import constants

'''
------------------------------------------------------------------------
    Class(es)
------------------------------------------------------------------------
'''
# This will be moved to class file later on
class Calibrate:

    def __init__(self, S, hours_worked_file_complete_name, consumption_file_complete_name,
                 hours_worked_file_parameter, hours_worked_file_parameter2,
                 consumption_file_parameter, consumption_file_parameter2, consumption_file_year,
                 population_by_age_file_name, population_by_age_file_parameter, population_by_age_sheet_name,
                 files_path, years, time_endowment):
        self.S = S #In a future version, S will not be a calibration parameter but a household parameter
        self.hours_worked_file_complete_name = hours_worked_file_complete_name
        self.consumption_file_complete_name = consumption_file_complete_name
        self.hours_worked_file_parameter = hours_worked_file_parameter
        self.hours_worked_file_parameter2 = hours_worked_file_parameter2
        self.consumption_file_parameter = consumption_file_parameter
        self.consumption_file_parameter2 = consumption_file_parameter2
        self.consumption_file_year = consumption_file_year
        self.population_by_age_file_name = population_by_age_file_name
        self.population_by_age_file_parameter = population_by_age_file_parameter
        self.population_by_age_sheet_name = population_by_age_sheet_name
        self.files_path = files_path
        self.years = years
        self.time_endowment = time_endowment
        self.WDATA = 56.840  #14.21 Euro per hour times 4000 hours a year
        self.missing_ages = 20
        self.slope_cdata_years = 10
        self.slope_ndata_years = 7

    '''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
    '''

    # load data on hours worked from stata .dta files. This may be moved
    # to utilities' class
    def load_large_dta(self, fname_ndata):
        '''
        --------------------------------------------------------------------
        This function loads 4 datasets for hours worked for Italy from STATA
        data file
        --------------------------------------------------------------------
        INPUTS:
        fname_ndata  = path for the files containing hours worked data

        OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
            pd.read_stata()
            pd.DataFrame()

        OBJECTS CREATED WITHIN FUNCTION:
        hw_year = panda's dataframes containing average weekly hours worked from
                  the STATA files for a given year

        FILES CREATED BY THIS FUNCTION: None

        RETURNS: hw_year
        --------------------------------------------------------------------
        '''

        reader = pd.read_stata(fname_ndata, iterator=True)
        hw_year = pd.DataFrame()

        try:
            chunk = reader.get_chunk(100*1000)
            while len(chunk) > 0:
                hw_year =hw_year.append(chunk, ignore_index=True)
                chunk = reader.get_chunk(100*1000)
        except (StopIteration, KeyboardInterrupt):
            pass

        return hw_year

    def get_ndata(self):

        '''
        --------------------------------------------------------------------
        This function calculates average weekly hours worked by an individual
        as a fraction of total time endowment, does extrapolation for individuals
        aged 80+ and then averages over the 4 year values.
        --------------------------------------------------------------------
        INPUTS:
        S              = integer in [3, 80], number of periods an individual
                         lives
        hours_worked_file_complete_name
                       = string, complete name of the hours worked STATA file
                         that includes year and file extension
        hours_worked_file_parameter
                       = string, head of the dataset for hours worked in the
                         STATA file
        time_endowment = integer, numer of total weekly hours that can be used
                         for work
        files_path     = string, path where the source STATA files are saved
        years          = string, elements of the years_list
        missing_ages   = string, number of missing ages in the hours worked survey
                        for Italy
        slope_ndata_years
                       = number of years used for calculation of the slope for
                        the purpose of data extrapolation

        OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
            ITA_lhw_2008.dta
            ITA_lhw_2010.dta
            ITA_lhw_2012.dta
            ITA_lhw_2014.dta
            load_large_dta()

        OBJECTS CREATED WITHIN FUNCTION:
        frac_dict      = length 4 dict, hours worked in a year as a fraction
                         of total time endowment for each individual {hw_avg_frac_2008,
                         hw_avg_frac_2010, hw_avg_frac_2012, hw_avg_frac_2014}
        n_dict         = length 4 dict containing (80,) arrays of hours worked
                         as a fraction of total time endowment in a year
        fname_ndata    = path for the files containing hours worked data
        hwdata         = panda's dataframes containing average weekly hours worked from
                         the STATA files for a given year
        frac_variable_name
                       = string, name of labels of frac_dict elements
        n_year         = (S,) vector of hours worked as a fraction of total time
                         endowment in a year
        ndata_source   = (S-1,) vector of hours worked data from the STATA file,
                         averaged over 4 years (discarding observations for age 80
                         due to an outlier value in 2010)
        slope_ndata    = scalar, slope of the trend of average out of 4 years
                         hours worked by individual as a fraction of total time
                         endowment
        ndata          = (S,) vector, elementwise average (i.e. out of 4 years'values)
                         of n_year
        temp_n_value   = scalar, equals ndata if ndata>0 and last positive value
                         if ndata<0

        FILES CREATED BY THIS FUNCTION: None

        RETURNS: ndata
        ---------------------------------------------------------------------
        '''

        frac_dict = {}
        n_dict = {}

        it = 0
        print(self.years)
        for year in self.years:
#            print(year)

            fname_ndata = self.files_path+self.hours_worked_file_complete_name[it]
            hwdata = self.load_large_dta(fname_ndata)
#            print("hwdata:", hwdata)
            frac_variable_name = "hw_avg_frac" + "_" + year

            frac_dict[frac_variable_name] = hwdata[self.hours_worked_file_parameter[it]] \
            * constants.WEEKS_IN_A_YEAR/self.time_endowment

            # Populate n_year discarding observations for age 80 due to an outlier value in 2010
            n_year = np.zeros(self.S-1)
            n_year[:(self.S-self.missing_ages-1)] = frac_dict[frac_variable_name][self.missing_ages+1:self.S]

            n_dict[year] = n_year

            it = it + 1

        # calculate average out of n2008, n2010, n2012 and n2014 elementwise and store
        # the result in ndata
        ndata_source = np.zeros(self.S-1)
        it = 0
        for year in self.years:
            ndata_source = np.sum([ndata_source, n_dict[year]], axis=0)

            it = it + 1

        ndata_source = ndata_source / len(self.years)
#        print("ndata_source:", ndata_source)

        #Fit a line to ages 80-100
        slope_ndata = (ndata_source[self.S-(self.missing_ages+2)] - ndata_source[self.S-(self.missing_ages+self.slope_ndata_years)])/(self.slope_ndata_years)

        ndata = np.zeros(self.S)
        ndata[:79] = ndata_source
        ndata[(self.S-self.missing_ages-1):] = ndata[(self.S-self.missing_ages-2)] + slope_ndata * range(self.missing_ages+1)

        # Replace possibly negative values for elderly with the reasonable positive value
        temp_n_value = ndata[0]
        for s in range(self.S-1):
            if(ndata[s] > 0):
                temp_n_value = ndata[s]
            else:
                ndata[s] = temp_n_value
        print("ndata:", ndata)

        # The following lines serve for plotting cdata vector using the
        # utils.create_graph_calib function
        vector_name = 'ndata'
        utils.create_graph_calib(ndata, vector_name)

        return ndata

    def get_cdata(self):

        '''
        --------------------------------------------------------------------
        This function calculates average annual consumption expenditure by
        an individual, smoothes the raw data by fitting the polynomial trend
        and does extrapolation for individuals aged 80+.
        --------------------------------------------------------------------
        INPUTS:
        S              = integer in [3, 80], number of periods an individual
                         lives
        consumption_file_complete_name
                       = string, complete name of the STATA file containing data
                         on consumption and income by age that includes year
                         and file extension
        consumption_file_parameter
                       = string, head of the dataset for consumption in the
                         STATA file
        files_path     = string, path where the source STATA files are saved
        missing_ages   = string, number of missing ages in the hours worked survey
                        for Italy
        slope_cdata_years
                       = number of years used for calculation of the slope for
                        the purpose of data extrapolation

        OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
            ITA_inc_exp_v2.dta
            load_large_dta()
            constants.MONTHS_IN_A_YEAR
            constants.THOUSAND
            np.linspace()
            np.polyfit()
            utils.create_graph_calib()


        OBJECTS CREATED WITHIN FUNCTION:
        fname_cdata   = path for the files containing data on consumption and income
        cydata        = panda's dataframes containing average monthly consumption and income
                        from the STATA files for a base year
        cxord         = (S-missing_ages,) vector of x-coordinates of the  sample
                        points for a polynomial fit of the cdata
        cdata_source  = (S-missing_ages,) vector of source consumption data from
                        the STATA file, expressed annually
        cpoly         = (5,) vector containing polynomial coefficients, ordered
                        from the highest power to the lowest (zero) power
        slope_cdata   = scalar, slope of the trend of average monthly consumption
                        expenditure by individual in a given year
        cdata         = (S,) vector, average annual consumption expenditure by
                        individual in thousands EUR, smoothed using polynomial function
        temp_c_value  = scalar, equals cdata if cdata>0 and last positive value
                         if cdata<0
        vector_name   = name of the series to be plotted
        cdata_graph   = (S+missing_ages,) cdata vector to be plotted

        FILES CREATED BY THIS FUNCTION: cdata.png

        RETURNS: cdata
        ---------------------------------------------------------------------
        '''
        fname_cdata = self.files_path+self.consumption_file_complete_name
        cydata = self.load_large_dta(fname_cdata)
#        print("cydata:", cydata)

        cdata_source = np.zeros(self.S-self.missing_ages)

        #consumption is expressed in thousands euros
        cdata_source = constants.MONTHS_IN_A_YEAR * (cydata[self.consumption_file_parameter][self.missing_ages+1:self.S+1]/constants.THOUSAND)

        cxord = np.linspace(1, self.S-self.missing_ages, self.S-self.missing_ages)
        cpoly = np.polyfit(cxord, cdata_source, 4, rcond=None, full=False, w=None, cov=False)
#        print("cpoly:", cpoly)

        cdata = np.zeros(self.S)

        cdata[:(self.S-self.missing_ages)] = ((cpoly[0] * (cxord ** 4)) + (cpoly[1] * (cxord ** 3)) + (cpoly[2] * (cxord ** 2)) \
            + (cpoly[3] * cxord) + cpoly[4])
        print('cdata after cpoly:', cdata)

        #Fit a line to ages 80-100 for avg consumption expenditure
        slope_cdata = (cdata[self.S-(self.missing_ages+1)] - cdata[self.S-(self.missing_ages+self.slope_cdata_years)])/(self.slope_cdata_years)
        #print("slope_cdata:", slope_cdata)

        cdata[(self.S-self.missing_ages):] = cdata[(self.S-self.missing_ages-1)] + slope_cdata * range(self.missing_ages)
        print('cdata before replacing:', cdata)


        # Replace negative values for elderly with the reasonable positive value
        temp_c_value = cdata[0]
        for s in range(self.S):
            if(cdata[s] > 0):
                temp_c_value = cdata[s]
            else:
                cdata[s] = temp_c_value
        print('cdata after replacing:', cdata)

        #The following lines serve for plotting cdata vector using the
        # utils.create_graph_calib function
        vector_name = 'cdata'
        # cdata_graph = np.zeros(self.S + self.missing_ages)
        # cdata_graph[self.missing_ages:] = cdata
        utils.create_graph_calib(cdata, vector_name)

        return cdata

    def get_chi_n_vec(self, params):

        '''
        --------------------------------------------------------------------
        This function calibrates chi_n_vec(s) in line with eq. (7.8) based
        on actual data for hours worked, consumption by age and wage for Italy.
        --------------------------------------------------------------------

        INPUTS:
        WDATA  = scalar, data on annual gross wage in thousands of euros
        params = length 7 tuple, (sigma, l_tilde, ellip_graph, b_ellip_init,
                 upsilon_init, Frisch_elast, CFE_scale)

        OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
            get_cdata()
            get_ndata()
            elp.fit_ellip_CFE()

        OBJECTS CREATED WITHIN FUNCTION:
        cdata        = (S,) vector, average annual consumption expenditure by
                       individual in thousands EUR, smoothed using polynomial function
        ndata        = (S,) vector, elementwise average (i.e. out of 4 years'
                       values) of 80 elements of n_year
        ellip_init   = (2,) vector, initial guesses for b and upsilon
        cfe_params   = (2,) vector, parameters for CFE disutility of labor
                       parameters: Frisch elasticity and scale parameter
        b_ellip      = scalar > 0, scale parameter in elliptical disutility
                       of labor function
        upsilon      = scalar > 1, shape parameter in elliptical disutility
                       of labor function
        chi_n_vec    = (S,) vector, values for chi^n_s

        FILES CREATED BY THIS FUNCTION: None

        RETURNS: chi_n_vec
        ---------------------------------------------------------------------
        '''
        cdata = self.get_cdata()
        ndata = self.get_ndata()

        """
        S = 10
        if S == 10:
            cdata = np.array([3.98753917, 10.43735981, 14.30110456, 15.73431751,
                              15.14497752, 13.19349825, 10.79272809, 9.1079502,
                              8.41754415, 7.36851578])

            ndata = np.array([0.24494488, 0.40675908, 0.41317181, 0.4082234,
                              0.3838976, 0.24507573, 0.10400452, 0.04430578,
                              0.040039, 0.03372474])

        elif S == 80:
            cdata = self.get_cdata()
            ndata = self.get_ndata()
        """

        sigma, l_tilde, ellip_graph, b_ellip_init, upsilon_init, Frisch_elast, CFE_scale = params


        ellip_init = np.array([b_ellip_init, upsilon_init])


        cfe_params = np.array([Frisch_elast, CFE_scale])

        b_ellip, upsilon = elp.fit_ellip_CFE(ellip_init, cfe_params, l_tilde,
                                             ellip_graph)

        chi_n_vec = (self.WDATA * cdata**(-sigma)) / ((b_ellip / l_tilde) * (ndata / l_tilde)
        ** (upsilon - 1) * (1 - (ndata / l_tilde) ** upsilon) ** ((1 - upsilon) / upsilon))

        """
        if S == 80:
            vector_name ="chi_n_vec"

            chi_n_vec_graph = np.zeros(self.S+self.missing_ages)
            chi_n_vec_graph[self.missing_ages:] = chi_n_vec

            utils.create_graph_calib(chi_n_vec_graph, vector_name)
        """
        vector_name = 'chi_n_vec_hat'
        utils.create_graph_calib(chi_n_vec, vector_name)

        return chi_n_vec

    def get_avg_ydata(self):
        '''
        --------------------------------------------------------------------
        This function calculates average annual income by an individual
        and does extrapolation for individuals aged 80+.
        --------------------------------------------------------------------
        INPUTS:
        S              = integer in [3, 80], number of periods an individual
                         lives
        consumption_file_year
                       = year of the data for average income by age of an individual
        consumption_file_complete_name
                       = string, complete name of the STATA file containing data
                         on consumption and income by age that includes year
                         and file extension
        consumption_file_parameter2
                       = string, head of the dataset for income in the
                         STATA file
        files_path     = string, path where the source STATA files are saved
        years          = string, elements of the years_list
        missing_ages   = string, number of missing ages in the hours worked survey
                         for Italy
        slope_cdata_years
                       = number of years used for calculation of the slope for
                         the purpose of data extrapolation

        OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
            ITA_inc_exp_v2.dta
            ITA_Eurostat_pop_by_age.xslx
            load_large_dta()
            constants.MONTHS_IN_A_YEAR
            constants.THOUSAND

        OBJECTS CREATED WITHIN FUNCTION:
        fname_cdata   = path for the files containing data on consumption and income
        cydata        = panda's dataframes containing average consumption and income
                        from the STATA files for a given year
        slope_ydata   = scalar, slope of the trend of annual income by individual
                        in a given year
        ydata         = (S,) vector, annual income by individual in thousands EUR
        temp_y_value  = scalar, equals ydata if ydata>0 and last positive value
                        if ydata<0
        pop_source    = (80,) list containing Eurostat population data by age
                        in Italy, average number of persons in 2012
        pop           = (S,) vector, number of individuals aged 21-100
        temp_pop_value= scalar, equals pop if pop>0 and last positive value
                        if pop<0
        avg_ydata     = scalar, average annual income in thousands EUR

        FILES CREATED BY THIS FUNCTION: None

        RETURNS: avg_ydata
        ---------------------------------------------------------------------
        '''
        fname_cdata = self.files_path+self.consumption_file_complete_name
        cydata = self.load_large_dta(fname_cdata)

        #Fit a line to ages 80-100 for avg income
        slope_ydata = ((constants.MONTHS_IN_A_YEAR * cydata[self.consumption_file_parameter2][self.S]/constants.THOUSAND)
                - (constants.MONTHS_IN_A_YEAR * cydata[self.consumption_file_parameter2][self.S-self.slope_cdata_years]/constants.THOUSAND))/(self.slope_cdata_years)
        # print("slope_y:", slope_y)

        #calculate annual income
        ydata = np.zeros(self.S)
        ydata[:(self.S-self.missing_ages)] =  constants.MONTHS_IN_A_YEAR * (cydata[self.consumption_file_parameter2][self.missing_ages+1:self.S+1]/constants.THOUSAND)
        ydata[(self.S-self.missing_ages-1):] = ydata[(self.S-self.missing_ages-1)] + slope_ydata * range(self.missing_ages+1)
        # print("ydata:", ydata)

        # Replace negative values for elderly with the reasonable positive value
        temp_y_value = ydata[0]
        for s in range(self.S):
            if(ydata[s] > 0):
                temp_y_value = ydata[s]
            else:
                ydata[s] = temp_y_value
        # print('ydata after replacing:', ydata)

        population_file_name = self.files_path+self.population_by_age_file_name
        df = pd.ExcelFile(population_file_name).parse(self.population_by_age_sheet_name)
        pop_source=[]
        pop_source.append(df[self.population_by_age_file_parameter])
        print("pop_source:", pop_source)

        #Conversion of a list-type data frame (pop_source) into a numpy array (pop)
        # and express popultaion in thonusands of persons
        pop = (np.array(pop_source)/constants.THOUSAND).flatten()
        print("pop:", pop)

        print("y data before updating with pop: ", ydata)
        #calculate weighted annual income by age
        ydata = ydata * pop
        print("ydata:", ydata)

        #Calculate aggregate annual income
        avg_ydata = (ydata.sum(axis=0)) / (pop.sum(axis=0))
        print("avg_ydata:", avg_ydata)

        return avg_ydata
