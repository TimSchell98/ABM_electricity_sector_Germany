# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 21:30:34 2023

@author: Tim Schell
"""


from ABM_GES_6 import ESS
import pandas as pd
import numpy as np
import time

startTime = time.time()

if __name__ == '__main__':
    
    start_year=2009
    end_year=2021
    
    #parameters to be analized in excel
    grid_resolution = 20
    deviation = 0.3
    zooms = 0

    """
    1 run 2009 - 2021 with prognisis: 3 min (laptop)/ 2 min (Desktop)
    10 variables
    12 grid resolution
    -> 120 runs * 3 min (laptop) = 378 min  -> 6.3 stunden
    
    total time = variables * grid resolution * 3.15 (laptop) * zooms
    
    
    """

    print("Estimated runtime in minutes:")
    runtime = (grid_resolution*10)*2 + (grid_resolution*10)*3.15 * zooms
    print(runtime)

    min_index = 0    

    elec_prices_empirical = pd.read_excel("Strompreis.xlsx",sheet_name="Empirical")
    
    installed_power_empirical = pd.read_excel("Installed_power.xlsx")


    #create parameter_list_full
    #parameter_list = pd.read_excel('Parameters_to_be_analysed.xlsx', sheet_name= "Complete")
    parameter_list = pd.read_excel('Parameters_to_be_analysed.xlsx', sheet_name= "Part")
    
    params_list = []
    
    
    for x in range (0,zooms+1):
        
        elec_prices = pd.DataFrame()
       
        electricity_costs = []
        
        nrmsd_list = []
        
        parameter_list_steps = pd.DataFrame(index = np.arange(grid_resolution))
        #loop for filling parameter_list_full
        for i in range (0,parameter_list.shape[0]):
            
            #pull start and end value out of parameter_list
            #check if there is a value in the dataframe
            if np.isnan(parameter_list.iloc[i,2]):
                #if no than calculate start_value
                start_val = parameter_list.iloc[i,1] * (1-deviation)
            else:
                #if yes, set start_value
                start_val = parameter_list.iloc[i,2]
            
            
            if np.isnan(parameter_list.iloc[i,3]):
                end_val = parameter_list.iloc[i,1] * (1+deviation)
            
            else:
                #if yes set end value
                end_val = parameter_list.iloc[i,3]
            
            #calculate delta from start and end value
            delta = round((end_val-start_val)/(grid_resolution-1),6)
            #create and write steps into parameter_list_steps
            for j in range (0,grid_resolution):
                parameter_list_steps.loc[j,i] = round(start_val+delta*j,6)
        parameter_list_steps.rename(columns = parameter_list.name, inplace = True)
        
        
        #create dataframe with every possible combination
        #create base
        parameter_list_full = pd.DataFrame(columns = [parameter_list.name],index = np.arange(grid_resolution*parameter_list.shape[0]))
        #fill parameter_list_full with standard values
        for i in range(0,parameter_list.shape[0]*grid_resolution):
            for j in range(0,parameter_list.shape[0]):
                parameter_list_full.iloc[i,j] = parameter_list.iloc[j,1]
        
        #fill parameter_list_full with steps     
        for i in range(0,parameter_list.shape[0]):
            for j in range(0,grid_resolution):
                parameter_list_full.iloc[j+i*grid_resolution,i] = parameter_list_steps.iloc[j,i]
    
        
    
    
        for y in range(0,len(parameter_list_full)):
        
            #create parameter_dict for the dynamic definition of parameters
            parameter_dict = {}
            for column_index, column_name in enumerate(parameter_list_full.columns):
                parameter_dict[column_name[0]] = parameter_list_full[column_name[0]].iloc[y].item()    
        
            ess = ESS(start_year=start_year,end_year=end_year, **parameter_dict)
            
            
            
            #init agents
            ess.init_agents()
            
            # Initialize profiles
            pv_generation_profile, wind_generation_profile, load_profile = ess.init_profiles()
            
            #run simulation
            ess.run_sim(pv_generation_profile, wind_generation_profile, load_profile)
            
            
            electricity_prices = ess.agents["Government"]["data"]["Electricity_cost"]
            
            installed_power_coal = ess.agents["Coal"]["data"]["Installed_power"]
            installed_power_gas = ess.agents["Gas_CC"]["data"]["Installed_power"] + ess.agents["Gas_CT"]["data"]["Installed_power"]
            installed_power_solar = ess.agents["Solar"]["data"]["Installed_power"]
            installed_power_wind = ess.agents["Wind"]["data"]["Installed_power"]
            
            elec_prices = pd.concat([elec_prices, electricity_prices], ignore_index = True, axis=1)
            
            #trim the installed_power_empirical df
            installed_power_empirical = installed_power_empirical.iloc[:len(installed_power_coal)]
            
            df = pd.DataFrame({y: electricity_prices})
            electricity_costs.append(df)
    
            data = {'elec_cost': [0], 'installed_power_coal': [0], 'installed_power_gas': [0], 'installed_power_solar': [0], 'isntalled_power_wind': [0]}
            nrmsd = pd.DataFrame(data = data)
            
            
            
            #ToDo: calculate NRMSD in function
            #NRMSD berechnung, electricity cost
            no_of_calculations = len(electricity_prices)
            stepwidth = 1
    
            model_data = np.array(electricity_prices)
            empirical_data = np.array(elec_prices_empirical)
            nominator_single_values = np.zeros(no_of_calculations)
            denominator_single_values = np.zeros(no_of_calculations)
    
            for i in range(no_of_calculations):
                nominator_single_values[-i - 1] = np.square(model_data[-i * stepwidth - 1] - empirical_data[-i * stepwidth - 1])
                denominator_single_values[-i - 1] = empirical_data[-i * stepwidth - 1]
    
            nrmsd_elec_prices = (np.sqrt(nominator_single_values.sum() / len(nominator_single_values))) / (denominator_single_values.sum() / len(denominator_single_values))
            
            #save in df
            nrmsd.iloc[0,0] = nrmsd_elec_prices    
        
    
            #NRMSD berechnung, installed power coal
            no_of_calculations = len(installed_power_coal)
            stepwidth = 1
    
            model_data = np.array(installed_power_coal)
            empirical_data = np.array(installed_power_empirical["Coal"])
            nominator_single_values = np.zeros(no_of_calculations)
            denominator_single_values = np.zeros(no_of_calculations)
    
            for i in range(no_of_calculations):
                nominator_single_values[-i - 1] = np.square(model_data[-i * stepwidth - 1] - empirical_data[-i * stepwidth - 1])
                denominator_single_values[-i - 1] = empirical_data[-i * stepwidth - 1]
    
            nrmsd_installed_power_coal = (np.sqrt(nominator_single_values.sum() / len(nominator_single_values))) / (denominator_single_values.sum() / len(denominator_single_values))
            
            #save in df
            nrmsd.iloc[0,1] = nrmsd_installed_power_coal
            
        
            #NRMSD berechnung, installed power gas
            no_of_calculations = len(installed_power_gas)
            stepwidth = 1
    
            model_data = np.array(installed_power_gas)
            empirical_data = np.array(installed_power_empirical["Gas"])
            nominator_single_values = np.zeros(no_of_calculations)
            denominator_single_values = np.zeros(no_of_calculations)
    
            for i in range(no_of_calculations):
                nominator_single_values[-i - 1] = np.square(model_data[-i * stepwidth - 1] - empirical_data[-i * stepwidth - 1])
                denominator_single_values[-i - 1] = empirical_data[-i * stepwidth - 1]
    
            nrmsd_installed_power_gas = (np.sqrt(nominator_single_values.sum() / len(nominator_single_values))) / (denominator_single_values.sum() / len(denominator_single_values))
            
            #save in df
            nrmsd.iloc[0,2] = nrmsd_installed_power_gas
            
            
            #NRMSD berechnung, installed power solar
            no_of_calculations = len(installed_power_solar)
            stepwidth = 1
    
            model_data = np.array(installed_power_solar)
            empirical_data = np.array(installed_power_empirical["PV"])
            nominator_single_values = np.zeros(no_of_calculations)
            denominator_single_values = np.zeros(no_of_calculations)
    
            for i in range(no_of_calculations):
                nominator_single_values[-i - 1] = np.square(model_data[-i * stepwidth - 1] - empirical_data[-i * stepwidth - 1])
                denominator_single_values[-i - 1] = empirical_data[-i * stepwidth - 1]
    
            nrmsd_installed_power_solar = (np.sqrt(nominator_single_values.sum() / len(nominator_single_values))) / (denominator_single_values.sum() / len(denominator_single_values))
            
            #save in df
            nrmsd.iloc[0,3] = nrmsd_installed_power_solar
            
            
            #NRMSD berechnung, installed power wind
            no_of_calculations = len(installed_power_wind)
            stepwidth = 1
    
            model_data = np.array(installed_power_wind)
            empirical_data = np.array(installed_power_empirical["Wind"])
            nominator_single_values = np.zeros(no_of_calculations)
            denominator_single_values = np.zeros(no_of_calculations)
    
            for i in range(no_of_calculations):
                nominator_single_values[-i - 1] = np.square(model_data[-i * stepwidth - 1] - empirical_data[-i * stepwidth - 1])
                denominator_single_values[-i - 1] = empirical_data[-i * stepwidth - 1]
    
            nrmsd_installed_power_wind = (np.sqrt(nominator_single_values.sum() / len(nominator_single_values))) / (denominator_single_values.sum() / len(denominator_single_values))
            
            #save in df
            nrmsd.iloc[0,4] = nrmsd_installed_power_wind
            
            nrmsd_total = nrmsd.mean().mean()
            #safe nrmsd
            nrmsd_list.append(nrmsd_total)
        
        
        # Concatenate the list of DataFrames into a single DataFrame
        result_df = pd.concat(electricity_costs, axis=1)
        
        #append nrmsd list to dataframe
        parameter_list_full['nrmsd'] = nrmsd_list
        
        #Find the index of the row with the lowest nrmsd
        min_index = parameter_list_full.iloc[:, -1].idxmin()
    
        #Extract the row with the lowest nrmsd
        row_with_lowest_last_value = parameter_list_full.iloc[min_index]
        
        #save parameter values and nrsmd
        params = row_with_lowest_last_value
        params_df = pd.DataFrame({y: params})
        params_list.append(params_df)
        
        #delete nrmsd
        row_with_lowest_last_value = row_with_lowest_last_value.drop('nrmsd')
        
        #transpose list
        new_standards = row_with_lowest_last_value.T
        
        #set index
        new_standards.index = parameter_list.index
        
        #update defaults
        parameter_list["default"] = new_standards
        
        #set start and end values in parameter list
        #Find NRMSD index of line which has the minimal NRMSD.
        NRMSD_index = min_index
                 
        #calculate index of parameter which is to be improved       
        parameter_index = int(NRMSD_index/grid_resolution)

        #check if optimal value is last of first value
        if NRMSD_index < grid_resolution*parameter_list.shape[0]-1 and NRMSD_index > 0:
            #if best parameter value is not an edge value, use previous value and next value as new start and end values
            if parameter_list_full.iloc[NRMSD_index-1,parameter_index] < parameter_list_full.iloc[NRMSD_index, parameter_index] and parameter_list_full.iloc[NRMSD_index+1,parameter_index] > parameter_list_full.iloc[NRMSD_index,parameter_index]:
                parameter_list.iloc[parameter_index,2] = parameter_list_full.iloc[NRMSD_index-1,parameter_index]
                parameter_list.iloc[parameter_index,3] = parameter_list_full.iloc[NRMSD_index+1,parameter_index]
            #check if best parameter value is edge value, if yes then move start or end value by given amount
            #check if best parameter is first value
            if parameter_list_full.iloc[NRMSD_index-1,parameter_index] > parameter_list_full.iloc[NRMSD_index,parameter_index] or NRMSD_index <= 0:
                parameter_list.iloc[parameter_index,2] = round(parameter_list_full.iloc[NRMSD_index,parameter_index]*(1-(deviation/(grid_resolution-1))),6) 
                parameter_list.iloc[parameter_index,3] = parameter_list_full.iloc[NRMSD_index+1,parameter_index]
            #check if best parameter is last value
            if parameter_list_full.iloc[NRMSD_index+1,parameter_index] < parameter_list_full.iloc[NRMSD_index,parameter_index] or NRMSD_index >= grid_resolution*parameter_list.shape[0]-1:
                parameter_list.iloc[parameter_index,2] = parameter_list_full.iloc[NRMSD_index-1,parameter_index]
                parameter_list.iloc[parameter_index,3] = round(parameter_list_full.iloc[NRMSD_index,parameter_index]*(1+(deviation/(grid_resolution-1))),6) 
    
    #save final paramter lists with NRMSD
    parameter_list_nrmsd = pd.concat(params_list, axis=1)
    
    #todo: ich könnte die parameter_liste direkt in die excel speichern, dann muss ich das nicht manuell machen
    #ich muss aber dran denken, dass die start und end_val = leer/nan sein müssen
        
    #print final computing time
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(round(executionTime,2)))