'''
------------------------------------------------------------------------
This module defines parameters that are used for calibration of the
chi_n_vec based on data on hours worked from the STATA files for 4 years:
2008, 2010, 2012 and 2014.

This module defines the following function(s):
    process_calibration_parameters()
------------------------------------------------------------------------
'''

import json
import Calibrate


def process_calibration_parameters():
    '''
    --------------------------------------------------------------------
    This function defines parameters that are used for calibration of
    the chi_n_vec
    --------------------------------------------------------------------
    INPUTS:
    S           = integer in [3, 80], number of periods an individual
                  lives
    hours_worked_file_name
                = string, name of the hours worked STATA file
    hours_worked_file_years
                = string, years included in the names of the STATA files
                  with hours worked data
    file_extensions
                = string, STATA data file extension
    consumption_file_name
                = string, name of the STATA file that contains data on
                  average monthly consumption expenditure and income
    hours_worked_file_parameter_name
                = string, header of the dataset for hours worked in the
                  STATA file
    hours_worked_file_parameter_name2
                = string, header of the dataset for number of persons
                  in the STATA file
    time_endowment
                = integer, numer of total weekly hours that can be used
                  for work
    files_path  = string, path where the source STATA files are saved


    OBJECTS CREATED WITHIN FUNCTION:
    hours_worked_file_complete_name
                = string, complete name of the hours worked STATA file
                  that includes year and file extension
    consumption_file_complete_name
                = string, complete name of the consumption and income STATA
                  file that includes file extension
    hours_worked_file_parameter
                = string, header of the dataset for hours worked in the
                  STATA file
    hours_worked_file_parameter2
                = string, header of the dataset for number of individuals
                  in the STATA file
    consumption_file_parameter
                = string, header of the dataset for average monthly
                  consumption expenditure in the STATA file
    consumption_file_parameter2
                = string, head of the dataset for income in the STATA file
    consumption_file_year
                = year of the data for average income by age of an individual
    years       = string, elements of the years_list
    years_list  = list, years of hours worked data

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: calibration_params
    --------------------------------------------------------------------
    '''

    with open("parameters.json") as json_data_file:
        data = json.load(json_data_file)

    calibration_data = data['calibration']
    print(calibration_data)


    S = calibration_data['S']
    hours_worked_file_name = calibration_data['hours_worked_file_name']
    hours_worked_file_years = calibration_data['hours_worked_file_years']
    file_extensions = calibration_data['file_extensions']
    consumption_file_name = calibration_data['consumption_file_name']
    hours_worked_file_parameter_name = calibration_data['hours_worked_file_parameter']
    hours_worked_file_parameter_name2 = calibration_data['hours_worked_file_parameter2']
    population_by_age_file_name = calibration_data['population_by_age_file_name']
    population_by_age_file_parameter = calibration_data['population_by_age_file_parameter']
    population_by_age_sheet_name = calibration_data['population_by_age_sheet_name']


    time_endowment = calibration_data['time_endowment']

    files_path = calibration_data['files_path']

    years = hours_worked_file_years.split(" ")
    hours_worked_file_complete_name = []
    hours_worked_file_parameter = []
    hours_worked_file_parameter2 = []
    consumption_file_parameter = []
    consumption_file_parameter2 = []
    years_list = []

    for year in years:
        file_name = hours_worked_file_name + "_" + year + "." + file_extensions
        hours_worked_file_complete_name.append(file_name)
        parameter_name = hours_worked_file_parameter_name + "_" + year
        parameter_name2 = hours_worked_file_parameter_name2 + "_" + year
        hours_worked_file_parameter.append(parameter_name)
        hours_worked_file_parameter2.append(parameter_name2)
        years_list.append(year)

    consumption_file_parameter = calibration_data['consumption_file_parameter']
    consumption_file_parameter2 = calibration_data['consumption_file_parameter2']
    consumption_file_year = calibration_data['consumption_file_year']

    print(years_list)

    print(hours_worked_file_complete_name)
    print(hours_worked_file_parameter)
    print(hours_worked_file_parameter2)

    consumption_file_complete_name = consumption_file_name + "." + file_extensions

    print(consumption_file_complete_name)

    calibration_params = Calibrate.Calibrate(S, hours_worked_file_complete_name, consumption_file_complete_name,
        hours_worked_file_parameter, hours_worked_file_parameter2, consumption_file_parameter, consumption_file_parameter2,
        consumption_file_year, population_by_age_file_name, population_by_age_file_parameter, population_by_age_sheet_name,
        files_path, years_list, time_endowment)

    print(calibration_params)

    return calibration_params

process_calibration_parameters()
