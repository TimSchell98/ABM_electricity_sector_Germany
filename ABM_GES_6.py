# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 12:36:34 2023

@author: Tim Schell

Agent based modell of the electricity generation system of Germany.
"""

import time
startTime = time.time()
import random
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import pypsa

class ESS:
    def __init__(self, start_year=2009, end_year=2021, governmental_funding = 0, cogeneration_funding = 35, co2_emissions_coal = 0.385, energy_content_coal = 29.75,
                 co2_emissions_gas = 0.201, fuel_factor_uranium = 8.9, conversion_uranium = 19.5, 
                 coal_lifetime = 40, gas_lifetime = 30, nuclear_lifetime = 50, solar_lifetime = 25, wind_lifetime = 25
                 ,enrichment_uranium = 67.1,fuel_fabrication_uranium = 300, energy_content_uranium = 360
                
                ,coal_max	=   6090
                ,gas_ct_max=	2376
                ,gas_cc_max=	4653
                ,solar_max	=   16740
                ,wind_max	=   9782
                
                ,coal_slope	  =  -5.042422
                ,coal_shift	  =  1.592889
                
                ,gas_ct_slope=	-4
                ,gas_ct_shift=	1.2
                
                ,gas_cc_slope=	-1
                ,gas_cc_shift=	0.624
                
                ,solar_slope	=-2.41616
                ,solar_shift	=1.294222
                
                ,wind_slope	 =   -3.054543
                ,wind_shift	 =   1.55
                
                ,battery_lifetime = 10
                ,battery_slope = -3
                ,battery_shift = 2.5
                ,battery_max = 10000
                
                ,hydrogen_lifetime = 20
                ,hydrogen_slope = -3
                ,hydrogen_shift = 1
                ,hydrogen_max = 250000
                
                ,prognosis = True
                
                ,coal_block_size = 400
                ,gas_block_size = 200
                ,wind_block_size = 5
                ,solar_block_size = 10
                ,battery_block_size = 100
                ,hydrogen_block_size = 10000
                 ):
        """
        Initiate all the variables and lists
        """
        self.start_year = start_year
        self.end_year = end_year
        self.agents = {}

        #prognosis, if true every year two market sims are done
        self.prognosis = prognosis
        
        #if prognosis is false, max investment has to be doubled
        if self.prognosis == False:
            print("Prognosis = False")
            self.coal_max = coal_max*2
            self.coal_max_init = coal_max*2
            self.gas_ct_max = gas_ct_max*2
            self.gas_ct_max_init = gas_ct_max*2
            self.gas_cc_max = gas_cc_max*2
            self.gas_cc_max_init = gas_cc_max*2
            self.solar_max = solar_max*2
            self.solar_max_init = solar_max*2
            self.wind_max = wind_max*2
            self.wind_max_init = wind_max*2
            self.battery_max = battery_max*2
            self.battery_max_init = battery_max*2
            self.hydrogen_max = hydrogen_max*2
            self.hydrogen_max_init = hydrogen_max*2
            
        if self.prognosis == True:
            print("Prognosis = True")
            self.coal_max = coal_max
            self.coal_max_init = coal_max
            self.gas_ct_max = gas_ct_max
            self.gas_ct_max_init = gas_ct_max
            self.gas_cc_max = gas_cc_max
            self.gas_cc_max_init = gas_cc_max
            self.solar_max = solar_max
            self.solar_max_init = solar_max
            self.wind_max = wind_max
            self.wind_max_init = wind_max
            self.battery_max = battery_max
            self.battery_max_init = battery_max
            self.hydrogen_max = hydrogen_max
            self.hydrogen_max_init = hydrogen_max
        
        #for calibration        
        self.coal_slope = coal_slope
        self.coal_shift = coal_shift
        
        self.gas_ct_slope = gas_ct_slope
        self.gas_ct_shift = gas_ct_shift
        
        self.gas_cc_slope = gas_cc_slope
        self.gas_cc_shift = gas_cc_shift
        
        self.solar_slope = solar_slope
        self.solar_shift = solar_shift
        
        self.wind_slope = wind_slope
        self.wind_shift = wind_shift
        
        self.battery_slope = battery_slope
        self.battery_shift = battery_shift
        self.battery_shift_init = battery_shift
        
        self.hydrogen_slope = hydrogen_slope
        self.hydrogen_shift = hydrogen_shift
        self.hydrogen_shift_init = hydrogen_shift
        
        #if a run has to be run again, the seed list can be imported, if not the seed list is safed, to ensure the run can be run again
        self.shuffle_seeds = []
        
        #cogeneration funding for gas power plants
        self.cogeneration_funding = cogeneration_funding
                
        #coal specs
        #Co2 emissions from one MWh thermal of coal, Umweltbundesamt
        self.co2_emissions_coal = co2_emissions_coal
        #energy content of one tonne of coal, Steinkohle einheiten
        self.energy_content_coal = energy_content_coal
        
        #gas specs
        #Co2 emissions from one MWh thermal of gas, Umweltbundesamt
        self.co2_emissions_gas = co2_emissions_gas  
        
        #nuclear specs
        self.fuel_factor_uranium = fuel_factor_uranium #[kg uranium/kg fuel]
        self.conversion_uranium = conversion_uranium #[Euro/kg Uranium]
        self.enrichment_uranium = enrichment_uranium #[Euro/kg Uranium]
        self.fuel_fabrication_uranium = fuel_fabrication_uranium #[Euro/kg Fuel] 
        self.energy_content_uranium = energy_content_uranium #[MWh/kg Fuel]
        
        #lifetime of power plants
        #quelle: https://energy.ec.europa.eu/system/files/2020-10/final_report_levelised_costs_0.pdf
        self.coal_lifetime = coal_lifetime
        self.gas_lifetime = gas_lifetime
        self.nuclear_lifetime = nuclear_lifetime
        self.solar_lifetime = solar_lifetime
        self.wind_lifetime = wind_lifetime

        #governmental funding for renewable energy
        self.Governmental_funding = governmental_funding
        
        #import LIBOR index list
        self.LIBOR_index = pd.read_excel("LIBOR.xlsx", sheet_name= "2009-2050")
        #self.LIBOR_index = pd.read_excel("LIBOR.xlsx", sheet_name= "2009-2050 Crisis")
        
        #import risk markup list
        self.Risk_markup = pd.read_excel("Risk_markup.xlsx")
        self.Risk_markup_init = self.Risk_markup
        
        #import investment costs as CAPEX
        self.Capex = pd.read_excel("Capex_nrel.xlsx")
        
        #define storage specs
        self.battery_lifetime = battery_lifetime
        self.hydrogen_lifetime = hydrogen_lifetime
        
        #import co2 certificate costs and ressource cost as lists
        self.Co2_certificate_costs = pd.read_excel("co2_certificate_costs.xlsx")
        self.coal_costs = pd.read_excel("ressource_costs.xlsx", sheet_name = "coal")
        self.gas_costs = pd.read_excel("ressource_costs.xlsx", sheet_name = "gas")
        self.uranium_costs = pd.read_excel("ressource_costs.xlsx", sheet_name = "uranium")

        #import operational costs
        self.Operational_costs = pd.read_excel("Operational_costs.xlsx", sheet_name = "Variable O and M costs")
        self.Operational_costs = self.Operational_costs.iloc[3]

        #import fix costs
        self.Fix_costs = pd.read_excel("Operational_costs.xlsx", sheet_name = "Fixed O and M costs")
        self.Fix_costs = self.Fix_costs.iloc[4]
        
        #import tax and network charge 2009 - 2021
        self.tax = pd.read_excel("Strompreis.xlsx")
        
        #import electricity demand
        self.Electricity_demand = pd.read_excel("Power demand.xlsx")
        
        #initialize the loans dataframe to store the loans given
        loans = {'Agent_name': [0], 'Active': [True], 'Runtime': [0], 'Amount': [0], 'Interest_rate': [0], 'Payback': [0]}
        self.loans = pd.DataFrame(data = loans)

        #fallback producer list
        self.Fallback_generator_supply = []
        
        #oil marginal cost
        self.oil_marginal_cost = pd.read_excel("oil_cost.xlsx")
        
        #block size of new power plants
        self.coal_block_size = coal_block_size
        self.gas_block_size = gas_block_size
        self.solar_block_size = solar_block_size
        self.wind_block_size = wind_block_size
        
        self.battery_block_size = battery_block_size
        self.hydrogen_block_size = hydrogen_block_size


    def add_agent(self, agent_name, agent_function, columns):
        """
        Add an agent to the model with a descriptive name, name of function and data columns
        """
        self.agents[agent_name] = {
            'function': agent_function,
            'data': pd.DataFrame(columns=columns, index=np.arange(self.end_year-self.start_year+1))
        }

    def init_agents(self):
        """
        This function defines all the agents with the corresponding varibles
        Also the values of the varibles of the first timestep are initilized
        """
        
        #ToDo: Muss Braunkohle auch als agent rein? die Daten zu steinkohle sind schon sehr anders

        #import agents names, functions and initial values
        agents_list = pd.read_excel("initial_values.xlsx")
        self.agents_list = agents_list

        #iterate through the data and add agents dynamically
        for index, row in agents_list.iterrows():
            #get agent name
            agent_name = row['Agent name']
            #get function name
            function_name = row['Function name']
            #get variable names
            variable_names = [row[i] for i in range(2, len(row), 2)]
            #remove nan from variable names list
            variable_names = [item for item in variable_names if not pd.isna(item)]
            #add agents with the agent name, function name and variable names
            self.add_agent(agent_name, getattr(self, function_name), variable_names)

        #fill dataframes of agents with 0 instead of nan
        for agent_name, agent_dict in self.agents.items():
            agent_dict['data'].fillna(0, inplace=True)

        #init variables first value
        #extract agent names from dataframe
        agent_names = agents_list.iloc[:, 0]
        #extract values from dataframe
        values = agents_list.iloc[:, 3::2]
        #transpone dataframe
        values = values.T
        #rename columns to agent names
        values = values.rename(columns=agent_names)
        #convert dataframe to dict
        values = values.to_dict(orient='list')
        #remove nan from dict
        values = {key: [value for value in values if not pd.isna(value)] for key, values in values.items()}
        #write values to agents
        for agent_name, agent_dict in self.agents.items():
            if agent_name in values:
                agent_dict['data'].loc[0] = values[agent_name]
        
        #Add dataframes for investments
        for agent_name, agent_dict in self.agents.items():
            agent_dict['Investments'] = pd.DataFrame(columns=['Block_size', 'Lifetime', 'Status'], index = np.arange(self.end_year-self.start_year+1))
            agent_dict['Investments'].fillna(0, inplace=True)
            
        #create dataframe for investments, ToDo: Split storage agent into battery and hydrogen so that these have there own dataframe in the agent_dict
        self.Storage_investments = pd.DataFrame(columns=['Block_size_battery', 'Lifetime_battery', "Status_battery", "Block_size_hydrogen", 'Lifetime_hydrogen', 'Status_hydrogen'], index = np.arange(self.end_year-self.start_year+1))
        self.Storage_investments.fillna(0, inplace=True)
        
        #init first values for storages
        self.Storage_investments.loc[0,"Block_size_battery"] = self.agents["Storage"]["data"].loc[0, "Battery_cap"]
        self.Storage_investments.loc[0,"Lifetime_battery"] = self.battery_lifetime-1
        self.Storage_investments.loc[0,"Status_battery"] = True
        
        self.Storage_investments.loc[0,"Block_size_hydrogen"] = self.agents["Storage"]["data"].loc[0, "Hydrogen_cap"]
        self.Storage_investments.loc[0,"Lifetime_hydrogen"] = self.hydrogen_lifetime-1
        self.Storage_investments.loc[0,"Status_hydrogen"] = True        
        
        #create function name list for the create_agent_list randomizing function
        #ToDo: Import these from excel list
        self.essencial_list = [self.government, self.bank]
        self.producer_list = [self.coal, self.gas_ct, self.gas_cc, self.nuclear, self.solar, self.wind, self.hydro, self.biomass, self.oil]
        self.storage_list = [self.storage]
        self.consumer_list = [self.consumer_1, self.consumer_2, self.consumer_3]
        
        #create function name list
        self.agent_list =  self.producer_list + self.storage_list + self.consumer_list + self.essencial_list

        #create function name list for the create_agent_list randomizing function, prognosis
        self.agent_list_prognosis = [self.coal_prognosis, self.gas_ct_prognosis, self.gas_cc_prognosis, self.solar_prognosis, self.wind_prognosis, self.storage_prognosis]

        #TODo: producer 1 zu coal etc

        #calculate deprication of every agent
        self.coal_deprication = self.agents["Coal"]["data"].loc[0, 'Installed_power_initial']/self.coal_lifetime
        self.gas_ct_deprication = self.agents["Gas_CT"]["data"].loc[0, 'Installed_power_initial']/self.gas_lifetime
        self.gas_cc_deprication = self.agents["Gas_CC"]["data"].loc[0, 'Installed_power_initial']/self.gas_lifetime
        self.solar_deprication = self.agents["Solar"]["data"].loc[0, 'Installed_power_initial']/self.solar_lifetime
        self.wind_deprication = self.agents["Wind"]["data"].loc[0, 'Installed_power_initial']/self.wind_lifetime

    def get_agent_list(self):
        """
        Returns the list of agents
        """
        return list(self.agents.keys())
    
    def get_agent_name(self, agent_function):
        """
        Returns the name of the agent given its function
        """
        for name, agent_dict in self.agents.items():
            if agent_dict['function'] == agent_function:
                return name
        return "Unknown Agent"

    def create_agent_list(self):
        """
        This function creates a order in which the agents are called
        At first the Producers are called in a random order, then the consumers are called in a random order
        """      

        
        #ToDo: Implement functionality to save the order or the randomizing key to run the same order again.
        """
        # Set the random seed if provided
        if seed is not None:
            random.seed(seed)
        else:
            #Safe these seeds and implement a functionality with which the list of seeds can be used here.
            random_seed = random.randint(0, 1000)
            random.seed(random_seed)
       
        random.shuffle(producer_list, random=random.Random(producer_seed))
        random.shuffle(consumer_list, random=random.Random(consumer_seed))
        self.shuffle_seeds.append((producer_seed, consumer_seed))
        
        """
        
        #shuffle the lists
        random.shuffle(self.producer_list)
        random.shuffle(self.consumer_list)
        
        #append the lists to each other in the order in which the agent types should be called
        agent_list =  self.producer_list + self.storage_list + self.consumer_list + self.essencial_list
    
        return agent_list
    
    
    def create_agent_list_prognosis(self):
        """
        This function creates a order in which the agents are called
        At first the Producers are called in a random order, then the consumers are called in a random order
        """      

        #shuffle the lists
        random.shuffle(self.producer_list_prognosis)
        
        #append the lists to each other in the order in which the agent types should be called
        agent_list_prognosis =  self.essencial_list + self.producer_list_prognosis
    
        return agent_list_prognosis

    def sigmoid(self, x, slope, shift):
        """
        This function calculates the y value of a sigmoid function given the x value.
        """
        # Compute the y value using the sigmoid function
        
        y = 1 / (1 + np.exp(slope * (x + shift)))

        
        return y
    
    def loan_list(self):
        """
        This function saves and procedes the loans that the bank lends to the agents.
        The loan are saved in a list comprising of the lifttime, amount, interest rate and which agent is the client of the loan.
        Every timestep the lifetime is lowered by 1. When the lifttime hits zero the credit is no longer active. 
        While the credit is active the amount that has to be paid by the agent is calculated and substracted from the income.
        
        'Agent_name': [0], 'Active': [True], 'Runtime': [0], 'Amount': [0], 'Interest_rate': [0], 'Payback': [0]}
        """

        #loop through the loans dataframe
        for i in range(1, len(self.loans)):
            
            #check if loan is still active
            if self.loans.iloc[i,1] == True:
                
                #substract payback from income of agent
                self.agents[str(self.loans.iloc[i,0])]["data"].loc[self.timestep, "Payback"] = self.agents[str(self.loans.iloc[i,0])]["data"].loc[self.timestep, "Payback"] - int(self.loans.iloc[i,5])
        
                #substract 1 from lifetime
                if self.loans.iloc[i,2] > 0:
                    self.loans.iloc[i,2] = self.loans.iloc[i,2] - 1
                    
                    #if lifetime reaches 0, switch to false
                    if self.loans.iloc[i,2] == 0:
                        self.loans.iloc[i,1] = False

    def power_plants_list(self, producer_name):
        """
        This function loops throu the dataframes of the investments of the agents and decreases the lifetime by one every year.
        Also if the lifetime of 0 is reached, the status of that block is switched to False
        """
    
        # Loop through the investments dataframe
        for i in range(len(self.agents[producer_name]['Investments'])):
    
            # Check if power plant is still active
            if self.agents[producer_name]['Investments'].loc[i, 'Status']:
    
                # Subtract one year from lifetime
                self.agents[producer_name]['Investments'].loc[i, 'Lifetime'] -= 1
    
                # If lifetime reaches 0, switch to false
                if self.agents[producer_name]['Investments'].loc[i, 'Lifetime'] == 0:
                    self.agents[producer_name]['Investments'].loc[i, 'Status'] = False


    def init_profiles(self):
        """
        This function imports the profiles that are used in the PyPSA function
        """
        #Load PV profile
        pv_profile = pd.read_excel('Generations_profile.xlsx', sheet_name='2017 Generationsprofil Solar')
        
        self.pv_generation_profile = pv_profile.iloc[:, 0].tolist()
        #pv_profile[2:300].plot()
        plt.show()

        #Load wind profile
        wind_profile = pd.read_excel('Generations_profile.xlsx', sheet_name='2017 Generationsprofil wind')
        self.wind_generation_profile = wind_profile.iloc[:, 0].tolist()
        #wind_profile[2:300].plot()
        plt.show()

        #Load load profile
        load_profile_df = pd.read_excel('load_data_smard.xlsx')
        self.load_factor_profile = load_profile_df.iloc[:, 0].tolist()
        #load_profile_df[2:300].plot()
        plt.show()
        
        #create p_max profile for hydro and biomass
        self.hydro_profile = [0.8] * 8759
        self.biomass_profile= [0.6] * 8759
        
        #create p_min profiles for gas
        self.gas_cc_profile = pd.read_excel('gas profile.xlsx', sheet_name= "cc")
        #self.gas_cc_profile = self.gas_cc_profile['KWK profile'].tolist()
        self.gas_cc_profile = self.gas_cc_profile['KWK profile']

        #ToDo: Alle übertragungen von self. variablen sind überflüssig
        return self.pv_generation_profile, self.wind_generation_profile, self.load_factor_profile
    
    
    def run_network(self, i, total_demand, pv_generation_profile, wind_generation_profile, load_factor_profile):
        """
        This function runs a PyPSA network simulation of the producers.
        The average electricity costs are calculated and given to the consumers in the next time step
        The total electricity supplied is also calculated
        """
        
        #Calculate load profile with total demand and factor profile
        self.load_profile = [round(value * total_demand, 4) for value in self.load_factor_profile]
        
        #create total generation profile of gas profile
        gas_profile_generation = self.gas_cc_profile * self.agents["Gas_CC"]["data"].loc[self.timestep-1, "Installed_power"]
        
        #create a temporary gas profile
        gas_cc_profile_temp = self.gas_cc_profile
        #loop throug every element of the profiles
        for i in range(0,len(self.load_profile)):
            #check wheather of not the load is smaller than the generation of the gas pp 
            if self.load_profile[i] <= gas_profile_generation[i]:
                #if yes, calculate the proportion of demand/installed power and set this as the new element in the gas profile
                gas_cc_profile_temp[i] = self.load_profile[i]/self.agents["Gas_CC"]["data"].loc[self.timestep-1, "Installed_power"]

        #Create an empty network
        network = pypsa.Network()
        network.set_snapshots(range(len(self.load_factor_profile)))
    
        #Add bus
        network.add('Bus', name='Main_bus')
        
        #Add producers as generators
        network.add('Generator', name='Coal', bus='Main_bus', p_nom=self.agents["Coal"]["data"].loc[self.timestep-1, "Installed_power"], marginal_cost=self.agents["Coal"]["data"].loc[self.timestep-1, "Marginal_cost"])
        network.add('Generator', name='Gas_CT', bus='Main_bus', p_nom=self.agents["Gas_CT"]["data"].loc[self.timestep-1, "Installed_power"],  marginal_cost=self.agents["Gas_CT"]["data"].loc[self.timestep-1, "Marginal_cost"])
        network.add('Generator', name='Gas_CC', bus='Main_bus', p_nom=self.agents["Gas_CC"]["data"].loc[self.timestep-1, "Installed_power"],  marginal_cost=self.agents["Gas_CC"]["data"].loc[self.timestep-1, "Marginal_cost"]) #, p_min_pu = gas_cc_profile_temp
        network.add('Generator', name='Nuclear', bus='Main_bus', p_nom=self.agents["Nuclear"]["data"].loc[self.timestep-1, "Installed_power"], marginal_cost=self.agents["Nuclear"]["data"].loc[self.timestep-1, "Marginal_cost"])
        network.add('Generator', name='Solar', bus='Main_bus', p_nom=self.agents["Solar"]["data"].loc[self.timestep-1, "Installed_power"], p_max_pu=self.pv_generation_profile,marginal_cost=self.agents["Solar"]["data"].loc[self.timestep-1, "Marginal_cost"])
        network.add('Generator', name='Wind', bus='Main_bus', p_nom=self.agents["Wind"]["data"].loc[self.timestep-1, "Installed_power"], p_max_pu=self.wind_generation_profile, marginal_cost=self.agents["Wind"]["data"].loc[self.timestep-1, "Marginal_cost"])
        network.add('Generator', name='Hydro', bus='Main_bus', p_nom=self.agents["Hydro"]["data"].loc[self.timestep-1, "Installed_power"], p_max_pu = self.hydro_profile, marginal_cost=self.agents["Hydro"]["data"].loc[self.timestep-1, "Marginal_cost"])
        network.add('Generator', name='Biomass', bus='Main_bus', p_nom=self.agents["Biomass"]["data"].loc[self.timestep-1, "Installed_power"], p_max_pu = self.biomass_profile, marginal_cost=self.agents["Biomass"]["data"].loc[self.timestep-1, "Marginal_cost"])
        network.add('Generator', name='Oil', bus='Main_bus', p_nom=self.agents["Oil"]["data"].loc[self.timestep-1, "Installed_power"], marginal_cost=self.agents["Oil"]["data"].loc[self.timestep-1, "Marginal_cost"])
         
        #fallback generator
        network.add('Generator', name='Fallback_generator', bus='Main_bus', p_nom=0, p_nom_extendable = True, marginal_cost=2000)
        
        #Add storage battery
        network.add('Bus', name = 'Storagebus_battery')
        network.add('Store', name = 'Battery_storage', bus = 'Storagebus_battery', e_nom = self.agents["Storage"]["data"].loc[self.timestep-1, "Battery_cap"])
        network.add('Link', name = 'Load_battery', bus0 = 'Main_bus',bus1 = 'Storagebus_battery', p_nom = self.agents["Storage"]["data"].loc[self.timestep-1, "Battery_in"],efficiency = self.agents["Storage"]["data"].loc[self.timestep-1, "Battery_eff"]**0.5)
        network.add('Link', name = 'Unload_battery', bus1 = 'Main_bus',bus0 = 'Storagebus_battery', p_nom = self.agents["Storage"]["data"].loc[self.timestep-1, "Battery_out"],efficiency = self.agents["Storage"]["data"].loc[self.timestep-1, "Battery_eff"]**0.5)

        #Add storage hydrogen
        network.add('Bus', name = 'Storagebus_hydrogen')
        network.add('Store', name = 'Hydrogen_storage', bus = 'Storagebus_hydrogen', e_nom = self.agents["Storage"]["data"].loc[self.timestep-1, "Hydrogen_cap"])
        network.add('Link', name = 'Load_hydrogen', bus0 = 'Main_bus',bus1 = 'Storagebus_hydrogen', p_nom = self.agents["Storage"]["data"].loc[self.timestep-1, "Hydrogen_in"],efficiency = self.agents["Storage"]["data"].loc[self.timestep-1, "Hydrogen_eff"]**0.5)
        network.add('Link', name = 'Unload_hydrogen', bus1 = 'Main_bus',bus0 = 'Storagebus_hydrogen', p_nom = self.agents["Storage"]["data"].loc[self.timestep-1, "Hydrogen_out"],efficiency = self.agents["Storage"]["data"].loc[self.timestep-1, "Hydrogen_eff"]**0.5)

        #Add storage water pump
        network.add('Bus', name = 'Storagebus_pump')
        network.add('Store', name = 'Pump_storage', bus = 'Storagebus_pump', e_nom = self.agents["Storage"]["data"].loc[self.timestep-1, "Pump_cap"])
        network.add('Link', name = 'Load_pump', bus0 = 'Main_bus',bus1 = 'Storagebus_pump', p_nom = self.agents["Storage"]["data"].loc[self.timestep-1, "Pump_in"],efficiency = self.agents["Storage"]["data"].loc[self.timestep-1, "Pump_eff"]**0.5)
        network.add('Link', name = 'Unload_pump', bus1 = 'Main_bus',bus0 = 'Storagebus_pump', p_nom = self.agents["Storage"]["data"].loc[self.timestep-1, "Pump_out"],efficiency = self.agents["Storage"]["data"].loc[self.timestep-1, "Pump_eff"]**0.5)
        
        #Add consumers as load
        network.add('Load', name='Sum_consumers', bus='Main_bus', p_set=self.load_profile)
        
        #Optimize network
        network.optimize(solver_name='gurobi')

        #save links and stores
        self.links = network.links
        self.stores = network.stores
        
        #safe generator lists
        self.generators_producer_p = network.generators_t.p
        
        self.generators_producer = network.generators_t

        #safe to excel
        #network.generators.to_excel("generators.xlsx")

        #calculation of electricity cost volume-related
        #calculate the sum of rows
        row_sums = network.generators_t.p.sum(axis=1)
        
        #electricity costs from marginal cost and generated electricity
        electricity_cost = network.buses_t.marginal_price["Main_bus"] * row_sums
        
        #calculate sum of electricity_cost
        total_cost = sum(electricity_cost)

        #calculte total_generated amount
        total_generated_amount = sum(row_sums)

        #calculate electricy_cost
        self.Electricity_cost_pure = total_cost/total_generated_amount       

        #print and save power that the fallback generator generated
        print("Fallback generator provided: ")
        print(sum(network.generators_t.p["Fallback_generator"]))
        self.Fallback_generator_supply.append(sum(network.generators_t.p["Fallback_generator"]))

        #create dataframes and fill with values, total generated power
        data_generated_power = {'Coal'      : [round(sum(network.generators_t.p["Coal"]),2)],
                                'Gas_CT'    : [round(sum(network.generators_t.p["Gas_CT"]),2)],
                                'Gas_CC'    : [round(sum(network.generators_t.p["Gas_CC"]),2)],
                                'Nuclear'   : [round(sum(network.generators_t.p["Nuclear"]),2)],
                                'Solar'     : [round(sum(network.generators_t.p["Solar"]),2)],
                                'Wind'      : [round(sum(network.generators_t.p["Wind"]),2)],
                                'Hydro'     : [round(sum(network.generators_t.p["Hydro"]),2)],
                                'Biomass'   : [round(sum(network.generators_t.p["Biomass"]),2)],
                                'Oil'       : [round(sum(network.generators_t.p["Oil"]),2)]
                                }

        self.generated_power = pd.Series(data = data_generated_power)
        
        #calculate the agents income
        self.coal_income_total =    ((network.buses_t.marginal_price["Main_bus"] - self.agents["Coal"]["data"].loc[self.timestep-1, "Marginal_cost"]) * network.generators_t.p["Coal"]).sum()
        self.nuclear_income_total = ((network.buses_t.marginal_price["Main_bus"] - self.agents["Nuclear"]["data"].loc[self.timestep-1, "Marginal_cost"]) * network.generators_t.p["Nuclear"]).sum()
        self.solar_income_total =   ((network.buses_t.marginal_price["Main_bus"] - self.agents["Solar"]["data"].loc[self.timestep-1, "Marginal_cost"]) * network.generators_t.p["Solar"]).sum()
        self.wind_income_total =    ((network.buses_t.marginal_price["Main_bus"] - self.agents["Wind"]["data"].loc[self.timestep-1, "Marginal_cost"]) * network.generators_t.p["Wind"]).sum()   
        self.hydro_income_total =   ((network.buses_t.marginal_price["Main_bus"] - self.agents["Hydro"]["data"].loc[self.timestep-1, "Marginal_cost"]) * network.generators_t.p["Hydro"]).sum()
        self.biomass_income_total = ((network.buses_t.marginal_price["Main_bus"] - self.agents["Biomass"]["data"].loc[self.timestep-1, "Marginal_cost"]) * network.generators_t.p["Biomass"]).sum()

        #gas income: addition of cogeneration funding
        self.gas_ct_income_total = ((network.buses_t.marginal_price["Main_bus"] - self.agents["Gas_CT"]["data"].loc[self.timestep-1, "Marginal_cost"] + self.cogeneration_funding) * network.generators_t.p["Gas_CT"]).sum()
        self.gas_cc_income_total = ((network.buses_t.marginal_price["Main_bus"] - self.agents["Gas_CC"]["data"].loc[self.timestep-1, "Marginal_cost"]) * network.generators_t.p["Gas_CC"]).sum()

        #calculate generation profile of coal, gas ct and gas cc 
        #if installed power is 0 profile will be nan -> change to 0
        if self.agents["Coal"]["data"].loc[self.timestep-1, "Installed_power"] > 0:
            self.coal_profile = network.generators_t.p["Coal"] / self.agents["Coal"]["data"].loc[self.timestep-1, "Installed_power"]
        else:    
            self.coal_profile = 0
            
        if self.agents["Gas_CT"]["data"].loc[self.timestep-1, "Installed_power"] > 0:
            self.gas_ct_profile = network.generators_t.p["Gas_CT"] / self.agents["Gas_CT"]["data"].loc[self.timestep-1, "Installed_power"]
        else:
            self.gas_ct_profile = 0
        
        if self.agents["Gas_CC"]["data"].loc[self.timestep-1, "Installed_power"] > 0:
            self.gas_cc_profile = network.generators_t.p["Gas_CC"] / self.agents["Gas_CC"]["data"].loc[self.timestep-1, "Installed_power"]
        else:
            self.gas_cc_profile = 0
        
        #save marginal costs list of main bus       
        self.marginal_costs = network.buses_t.marginal_price["Main_bus"]
        
        #safe marginal cost of every bus
        self.marginal_cost = network.buses_t.marginal_price

        #calculate profit of storages
        #pump storage
        self.pump_profit = sum(network.links_t.p1["Unload_pump"] * network.buses_t.marginal_price["Main_bus"] - network.links_t.p0["Load_pump"] * network.buses_t.marginal_price["Main_bus"])

        #battery storage
        self.battery_profit = sum(network.links_t.p1["Unload_battery"] * network.buses_t.marginal_price["Main_bus"] - network.links_t.p0["Load_battery"] * network.buses_t.marginal_price["Main_bus"])

        #hydrogen storage
        self.hydrogen_profit = sum(network.links_t.p1["Unload_hydrogen"] * network.buses_t.marginal_price["Main_bus"] - network.links_t.p0["Load_hydrogen"] * network.buses_t.marginal_price["Main_bus"])
        
        #calulate profiles of storages
        #battery
        self.battery_load = network.links_t.p0["Load_battery"]
        self.battery_unload = network.links_t.p1["Unload_battery"]
        self.battery_profile = network.links_t.p0["Load_battery"]/max(network.links_t.p0["Load_battery"]) - network.links_t.p1["Unload_battery"]/ min(network.links_t.p1["Unload_battery"])
        
        #hydrogen
        self.hydrogen_load = network.links_t.p0["Load_hydrogen"]
        self.hydrogen_unload = network.links_t.p1["Unload_hydrogen"]
        self.hydrogen_profile = network.links_t.p0["Load_hydrogen"]/max(network.links_t.p0["Load_hydrogen"]) - network.links_t.p1["Unload_hydrogen"]/ min(network.links_t.p1["Unload_hydrogen"])

        #water pump
        self.water_profile = network.links_t.p0["Load_pump"]/max(network.links_t.p0["Load_pump"]) - network.links_t.p1["Unload_pump"]/ min(network.links_t.p1["Unload_pump"])
        
        #save storages in excel
        #storages = pd.DataFrame({'Battery_profile': self.battery_profile, 'Battery_load': self.battery_load, 'Battery_unload': self.battery_unload, 'Hydrogen_profile': self.hydrogen_profile, 'Hydrogen_load': self.hydrogen_load, 'Hydrogen_unload': self.hydrogen_unload, 'Marginal_cost': self.marginal_costs})
        #storages.to_excel('storages.xlsx', index=False)    

        #save marginal costs in excel
        #self.marginal_cost.to_excel(f"data/marginal_costs_{self.timestep}.xlsx", index=False, sheet_name=f'Sheet_name_{self.timestep}')
        
        #save producer in excel
        #self.generators_producer_p.to_excel(f"data/producers_{self.timestep}.xlsx", index=False, sheet_name=f'Sheet_name_{self.timestep}')

    def run_sim(self,pv_generation_profile, wind_generation_profile, load_factor_profile):
        """
        Function that runs the simulation from the start year to the end year
        """
        #init dataframes to gather PyPSA Data
        data_generated_power = {'Coal': [0], 'Gas_CT': [0], 'Gas_CC': [0], 'Nuclear': [0], 'Solar': [0], 'Wind': [0], 'Hydro': [0], 'Biomass': [0], 'Oil': [0] }
        self.generated_power = pd.DataFrame(data = data_generated_power)
        
        #calculate estimated run time
        if self.prognosis  == True:
            estimated_run_time = (self.end_year-self.start_year)*2*17.5/60
        if self.prognosis == False:
            estimated_run_time = (self.end_year-self.start_year)*17.5/60
        
        print("\nEstimated runtime in minutes:")
        print(estimated_run_time)
        print("\n")
        
        #loop from start year to end year, dont simulate first year (only start values)
        for self.time in range(self.start_year+1, self.end_year+1):
            #calculate timestep
            self.timestep = self.time - self.start_year
            
            #increase maximum investment of renewables in the future
            self.solar_max = self.solar_max_init + (self.solar_max_init * (1 / (2 + np.exp(-0.3 * (self.time - 2035)))))
            self.wind_max = self.wind_max_init + (self.wind_max_init * (1 / (2 + np.exp(-0.3 * (self.time - 2035)))))
    
    
            #add strategies here, ToDo: exclude strategies into functions
            if self.time >= 2027 and self.time <= 2037:
                print("\nCrisis started")
            
                #strategy 1: governmental securities
                #reduce risk markup -> reduces shift
                #safe risk markups and after crisis set riskmarkup back to normal
                #self.Risk_markup["Production Coal"] = 0.04
                #self.Risk_markup["Production Gas"] = 0.03
                #self.Risk_markup["Production Solar"] = 0
                #self.Risk_markup["Production Wind"] = 0
            
            
                #strategy 2: increase maximum investment
                #self.coal_max = 6090
                #self.gas_ct_max = 2376 
                #self.gas_cc_max = 4653
                #self.solar_max = 16740*2
                #self.wind_max = 9782*2 
            
            
                #strategy 3: increase funding
                #self.governmental_funding = 25
                #self.cogeneration_funding = 60
                
                
                #strategy 4: increase storage
                #self.battery_max = 10000*2
                #self.hydrogen_max = 250000*2
                #self.battery_shift = 2.5
                #self.hydrogen_shift = 1.5


                
                #strategy 5: decrease demand
                #self.Electricity_demand.iloc[self.timestep,0] = self.Electricity_demand.iloc[self.timestep,0] * 0.8
                #self.Electricity_demand.loc[self.timestep,"Industry demand"] = self.Electricity_demand.loc[self.timestep,"Industry demand"] * 0.8
                #self.Electricity_demand.loc[self.timestep,"Commerce demand"] = self.Electricity_demand.loc[self.timestep,"Commerce demand"] * 0.8
                #self.Electricity_demand.loc[self.timestep,"Private demand"] = self.Electricity_demand.loc[self.timestep,"Private demand"] * 0.8               
                
                
                #strategy 6:increase lifetime

            """
            if self.time == 2033:
                agent_names = ["Coal", "Gas_CT", "Gas_CC", "Solar", "Wind"]
                
                #loop throu agent_names
                for producer_name in agent_names:
                
                    # Loop through the investments dataframe
                    for i in range(len(self.agents[producer_name]['Investments'])):
                
                        # Check if power plant is still active
                        if self.agents[producer_name]['Investments'].loc[i, 'Status'] == True and self.agents[producer_name]['Investments'].loc[i, 'Lifetime'] < 5:
                
                            # Subtract one year from lifetime
                            self.agents[producer_name]['Investments'].loc[i, 'Lifetime'] += 3
            """
        
            #reset strategies
            if self.time == 2038:
                print("\nCrisis ended")
                
                #strategy 1: governmental securities
                #self.Risk_markup["Production Coal"] = 0.06
                #self.Risk_markup["Production Gas"] = 0.05
                #self.Risk_markup["Production Solar"] = 0.02
                #self.Risk_markup["Production Wind"] = 0.03
                
                #strategy 2: increase maximum invetsment
                #self.coal_max = 6090
                #self.gas_ct_max = 2376
                #self.gas_cc_max = 4653
                #self.solar_max = 16740
                #self.wind_max = 9782
                
                
                #strategy 3: increase funding
                #self.governmental_funding = 0
                #self.cogeneration_funding = 35
                
                
                #strategy 4: increase storage
                #self.battery_max = 10000
                #self.hydrogen_max = 250000
                #self.battery_shift = 2.5
                #self.hydrogen_shift = 1
    
    
            #   ---Run year as prognosis---
            
            if self.prognosis == True:
                
                print("\n\nPrognosis")
                print("Time:", self.time)
                print("Timestep:", self.timestep)
                
                #caculate total demand to use in PyPSA as load, Timestep-1 so that the inital values are used
                total_demand = self.agents["Consumer 1"]["data"].loc[self.timestep-1, "Power_demand"] + self.agents["Consumer 2"]["data"].loc[self.timestep-1, "Power_demand"] + self.agents["Consumer 3"]["data"].loc[self.timestep-1, "Power_demand"]
    
                #Run the network simulation and gather data
                #ToDo: übergabe von self. variablen brauche ich nicht
                self.run_network(self.timestep-1, total_demand, pv_generation_profile, wind_generation_profile, load_factor_profile)
                
                #add taxes and network costs to electricity price
                self.Electricity_cost = self.Electricity_cost_pure + self.agents["Government"]["data"].loc[self.timestep-1, "Tax"] 

                #call agents in the order of the list
                for function_name in self.agent_list_prognosis:
                    function_name()
                
            
            #   ---Run again as real year---  
            
            print("\n\nTime:", self.time)
            print("Timestep:", self.timestep)
            
            #caculate total demand to use in PyPSA as load, Timestep-1 so that the inital values are used
            total_demand = self.agents["Consumer 1"]["data"].loc[self.timestep-1, "Power_demand"] + self.agents["Consumer 2"]["data"].loc[self.timestep-1, "Power_demand"] + self.agents["Consumer 3"]["data"].loc[self.timestep-1, "Power_demand"]
            
            #run PyPSA a second time
            self.run_network(self.timestep-1, total_demand, pv_generation_profile, wind_generation_profile, load_factor_profile)
            
            #add taxes and network costs to electricity price
            self.Electricity_cost = self.Electricity_cost_pure + self.agents["Government"]["data"].loc[self.timestep-1, "Tax"] 
            
            #call investment/loan payback function
            self.loan_list()
            
            #call power plant list
            self.power_plants_list("Coal")
            self.power_plants_list("Gas_CT")
            self.power_plants_list("Gas_CC")
            self.power_plants_list("Solar")
            self.power_plants_list("Wind")
            
            #call agents in the order of the list
            for function_name in self.agent_list:
                function_name()
                

    def government(self):
        """
        This agent represents the government of authority agent
        In this function variables like tax can be changed
        """
        print("\nGoverment called")
        
        #update tax
        self.agents["Government"]["data"].loc[self.timestep, "Tax"] = self.tax.iloc[self.timestep,0]

        #fill Electricity_cost with with calculated PyPSA data
        self.agents["Government"]["data"].loc[self.timestep, "Electricity_cost"] = self.Electricity_cost
    
        #fill Electricity_cost_pure with with calculated PyPSA data
        self.agents["Government"]["data"].loc[self.timestep, "Electricity_cost_pure"] = self.Electricity_cost_pure
    
        #update total emissions
        self.agents["Government"]["data"].loc[self.timestep, "Total_emissions"] = self.agents["Coal"]["data"].loc[self.timestep, "Total_emissions"]+self.agents["Gas_CT"]["data"].loc[self.timestep, "Total_emissions"]+self.agents["Gas_CC"]["data"].loc[self.timestep, "Total_emissions"]+self.agents["Nuclear"]["data"].loc[self.timestep, "Total_emissions"] + self.agents["Solar"]["data"].loc[self.timestep, "Total_emissions"] + self.agents["Wind"]["data"].loc[self.timestep, "Total_emissions"] + self.agents["Hydro"]["data"].loc[self.timestep, "Total_emissions"] + self.agents["Biomass"]["data"].loc[self.timestep, "Total_emissions"] + self.agents["Oil"]["data"].loc[self.timestep, "Total_emissions"]

        #update government total generated power
        self.agents["Government"]["data"].loc[self.timestep, "Total_generated_power"] = self.agents["Coal"]["data"].loc[self.timestep, "Generated_power_total"]+ self.agents["Gas_CT"]["data"].loc[self.timestep, "Generated_power_total"]+ self.agents["Gas_CC"]["data"].loc[self.timestep, "Generated_power_total"] + self.agents["Nuclear"]["data"].loc[self.timestep, "Generated_power_total"] + self.agents["Solar"]["data"].loc[self.timestep, "Generated_power_total"] + self.agents["Wind"]["data"].loc[self.timestep, "Generated_power_total"] + self.agents["Hydro"]["data"].loc[self.timestep, "Generated_power_total"] + self.agents["Biomass"]["data"].loc[self.timestep, "Generated_power_total"]
        
        #update government total installed power
        self.agents["Government"]["data"].loc[self.timestep, "Total_installed_power"] = self.agents["Coal"]["data"].loc[self.timestep, "Installed_power"] + self.agents["Gas_CT"]["data"].loc[self.timestep, "Installed_power"] + self.agents["Gas_CC"]["data"].loc[self.timestep, "Installed_power"] + self.agents["Nuclear"]["data"].loc[self.timestep, "Installed_power"] + self.agents["Solar"]["data"].loc[self.timestep, "Installed_power"] + self.agents["Wind"]["data"].loc[self.timestep, "Installed_power"] + self.agents["Hydro"]["data"].loc[self.timestep, "Installed_power"] + self.agents["Biomass"]["data"].loc[self.timestep, "Installed_power"]
        
        #update government total demanded power
        self.agents["Government"]["data"].loc[self.timestep, "Total_demanded_power"] = self.agents["Consumer 1"]["data"].loc[self.timestep, "Power_demand"] + self.agents["Consumer 2"]["data"].loc[self.timestep, "Power_demand"] + self.agents["Consumer 3"]["data"].loc[self.timestep, "Power_demand"]
        

    def bank(self):
        """
        This agent represents a central bank, that provides loans to every other agent
        """
        print("\nBank called")
        
        #update money
        self.agents["Bank"]["data"].loc[self.timestep,"Money"] = self.agents["Bank"]["data"].loc[self.timestep-1,"Money"]

        #update LIBOR
        self.agents["Bank"]["data"].loc[self.timestep,"LIBOR_index"] = self.LIBOR_index.iloc[self.timestep,0]
        
        #update margin
        self.agents["Bank"]["data"].loc[self.timestep,"Margin"] = self.agents["Bank"]["data"].loc[self.timestep-1,"Margin"]

        #update interest rate
        self.agents["Bank"]["data"].loc[self.timestep,"Interest_rate"] = self.agents["Bank"]["data"].loc[self.timestep,"LIBOR_index"] + self.agents["Bank"]["data"].loc[self.timestep,"Margin"]

    def coal_prognosis(self):
        """
        Function of Coal.
        This producer resembels the combined coal power plants
        """
        print("\nCoal prognosis")
        
        #update installed_power
        self.agents["Coal"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Coal"]["data"].loc[self.timestep-1, "Installed_power"]

        #update efficiency
        self.agents["Coal"]["data"].loc[self.timestep, "Efficiency"] = self.agents["Coal"]["data"].loc[self.timestep-1, "Efficiency"]
        
        #update marginal cost
        #calculate co2 costs
        #calculate t co2 emittet by producing 1 MWh
        co2_mwh = self.co2_emissions_coal / self.agents["Coal"]["data"].loc[self.timestep, "Efficiency"]
        
        #calculate co2 costs
        co2_costs = self.Co2_certificate_costs.iloc[self.timestep,0] * co2_mwh
    
        #calculate mwh per tonne of coal
        mwh_t_coal = (self.energy_content_coal * self.agents["Coal"]["data"].loc[self.timestep, "Efficiency"])/3.6
    
        #calculate ressource costs
        ressource_cost = self.coal_costs.iloc[self.timestep,0]/mwh_t_coal
    
        #calculate marginal costs  
        marginal_cost = ressource_cost + co2_costs + self.Operational_costs["Coal"]
        
        #write marginal cost in dataframe
        self.agents["Coal"]["data"].loc[self.timestep, "Marginal_cost"] = marginal_cost
    
        #calculate outlook of profit
        income_outlook = sum((self.coal_profile * 100) * (self.marginal_costs-self.agents["Coal"]["data"].loc[self.timestep-1, "Marginal_cost"]))

        #calculate interest rate 
        interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Coal"]).iloc[0])
        
        #amount
        amount = 100 * self.Capex.loc[self.timestep-1, "Coal"]
    
        #payback
        payback = float(amount * interest_rate)   

        #caluate profit with income outlook and fix costs
        profit = income_outlook-(self.Fix_costs.loc["Coal"]*100) - payback
        
        #form the cash_flow list
        cash_flows = [profit] * self.coal_lifetime
        
        #insert investment cost as first value
        cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Coal"])

        #calculate the rate of return
        irr = npf.irr(cash_flows)
        
        #if irr is nan (profit is negative), mirror irr
        if irr != irr:
            cash_flows = [-profit] * self.coal_lifetime
            cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Coal"])
            irr = npf.irr(cash_flows)
            irr = -irr - 2
        
        #save irr
        self.agents["Coal"]["data"].loc[self.timestep, "Irr prognose"] = irr

        #calcualte shift, increases when interste rate rises
        if self.LIBOR_index.iloc[self.timestep,0] > 0.025:
            crisis_shift = 100*((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"]+self.Risk_markup["Production Coal"].iloc[0])-0.025)**2
        else:
            crisis_shift = 0

        shift = self.coal_shift-crisis_shift

        #for scenario coal phase-out, no more investments
        #if self.time <= 2030:
        if self.time <= 2050:   

            #calculate power_increase with sigmoid function
            power_increase = self.coal_max * self.sigmoid(irr,self.coal_slope,shift)
            
            #round power increase to a block size
            power_increase = (power_increase // self.coal_block_size) * self.coal_block_size       

        else: 
            power_increase = 0

        #check if power increase is more than 0
        if power_increase > 0:
            #safe power increase in power increase dataframe
            self.agents["Coal"]['Investments'].loc[self.timestep, 'Block_size'] +=  power_increase
        
            #set lifetime
            self.agents["Coal"]['Investments'].loc[self.timestep, 'Lifetime'] = self.coal_lifetime
            
            #set status
            self.agents["Coal"]['Investments'].loc[self.timestep, 'Status'] = True
       
            #calculate interest rate 
            interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Coal"]).iloc[0])
            
            #amount
            amount = power_increase * self.Capex.loc[self.timestep, "Coal"]
        
            #payback
            payback = float(amount * interest_rate)
        
            #take on loan and write that into the loan list
            loans = {'Agent_name': ["Coal"], 'Active': [True], 'Runtime': [40], 'Amount': [amount], 'Interest_rate': [interest_rate], 'Payback': [payback]}
            self.loans_temp = pd.DataFrame(data = loans)
            
            #add to loans list
            self.loans = pd.concat([self.loans, self.loans_temp], ignore_index=True)
       
        #calculate the total installed power
        self.agents["Coal"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Coal"]["data"].loc[self.timestep-1, "Installed_power_initial"] + self.agents["Coal"]["Investments"].loc[self.agents["Coal"]["Investments"]['Status'] == True, 'Block_size'].sum()

        #for coal phase out, variant 2, after 2030 no more coal power plants
        #if self.time >=2030:
            
            #self.agents["Coal"]["data"].loc[self.timestep, "Installed_power_initial"] = 0
            
            # Loop through the investments dataframe
            #for i in range(len(self.agents["Coal"]['Investments'])):
        
                #self.agents["Coal"]['Investments'].loc[i, 'Status'] == False
                

    def coal(self):
        """
        Function of Coal.
        This producer resembels the combined coal power plants
        """
        print("\nCoal called")
    
        #check if initial installed power is higher than 0
        if self.agents["Coal"]['data'].loc[self.timestep-1, 'Installed_power_initial'] > 0:
            
            #initial installed power is reduced by a fixed amount every year
            self.agents["Coal"]['data'].loc[self.timestep, 'Installed_power_initial'] = self.agents["Coal"]["data"].loc[self.timestep-1, 'Installed_power_initial'] - self.coal_deprication
        
        
            #strategy 8 increase lifetime:
            #if self.time >= 2033 and self.time <= 2036:
        
                #self.agents["Coal"]['data'].loc[self.timestep, 'Installed_power_initial'] = self.agents["Coal"]["data"].loc[self.timestep-1, 'Installed_power_initial']
            
            #else:
                
                #self.agents["Coal"]['data'].loc[self.timestep, 'Installed_power_initial'] = self.agents["Coal"]["data"].loc[self.timestep-1, 'Installed_power_initial'] - self.coal_deprication            
                
        else: 
            self.agents["Coal"]["data"].loc[self.timestep, 'Installed_power_initial'] = 0 
        
        #calculate the total installed power
        self.agents["Coal"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Coal"]["data"].loc[self.timestep, "Installed_power_initial"] + self.agents["Coal"]["Investments"].loc[self.agents["Coal"]["Investments"]['Status'] == True, 'Block_size'].sum()

        #update efficiency, increase by 0.1% every year
        #todo: function einbauen die dafür sorgt, dass der wirkungsgrad bis 41% steigt und danach so bleibt
        self.agents["Coal"]["data"].loc[self.timestep, "Efficiency"] = self.agents["Coal"]["data"].loc[self.timestep-1, "Efficiency"] + 0.001    
        
        #update marginal cost
        #calculate co2 costs
        #calculate t co2 emittet by producing 1 MWh
        co2_mwh = self.co2_emissions_coal / self.agents["Coal"]["data"].loc[self.timestep, "Efficiency"]
        
        #calculate co2 costs
        co2_costs = self.Co2_certificate_costs.iloc[self.timestep,0] * co2_mwh
    
        #calculate mwh per tonne of coal
        mwh_t_coal = (self.energy_content_coal * self.agents["Coal"]["data"].loc[self.timestep, "Efficiency"])/3.6
    
        #calculate ressource costs
        ressource_cost = self.coal_costs.iloc[self.timestep,0]/mwh_t_coal
    
        #calculate marginal costs  
        marginal_cost = ressource_cost + co2_costs + self.Operational_costs["Coal"]
        
        #write marginal cost in dataframe
        self.agents["Coal"]["data"].loc[self.timestep, "Marginal_cost"] = marginal_cost

        #fill generated_power_total with calculated PyPSA data
        self.agents["Coal"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.loc["Coal"]
        
        #set income with the income of the last timestep from PyPSA
        self.agents["Coal"]["data"].loc[self.timestep, "Income"] = self.coal_income_total

        #set expenses from the fix costs
        self.agents["Coal"]["data"].loc[self.timestep, "Expenses"] = self.agents["Coal"]["data"].loc[self.timestep, "Expenses"]-self.Fix_costs.loc["Coal"]*self.agents["Coal"]["data"].loc[self.timestep, "Installed_power"]

        #calculate profit
        self.agents["Coal"]["data"].loc[self.timestep, "Profit"] = self.agents["Coal"]["data"].loc[self.timestep, "Income"] + self.agents["Coal"]["data"].loc[self.timestep, "Payback"] + self.agents["Coal"]["data"].loc[self.timestep, "Expenses"]

        #update money
        self.agents["Coal"]["data"].loc[self.timestep, "Money"] = self.agents["Coal"]["data"].loc[self.timestep-1, "Money"] + self.agents["Coal"]["data"].loc[self.timestep, "Profit"] 

        #investment function
        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy_total = self.agents["Coal"]["data"].loc[self.timestep, "Installed_power"] * 365 * 24
        
        #calculate average load in the last time step
        self.agents["Coal"]["data"].loc[self.timestep, "Load"] = round(self.agents["Coal"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy_total,4)
        
        #check if load is bigger than 1
        if self.agents["Coal"]["data"].loc[self.timestep, "Load"] > 1:
            #if yes set load to 1
            self.agents["Coal"]["data"].loc[self.timestep, "Load"] = 1
        
        #calculate outlook of profit
        income_outlook = sum((self.coal_profile * 100) * (self.marginal_costs-self.agents["Coal"]["data"].loc[self.timestep-1, "Marginal_cost"]))
        
        #calculate interest rate 
        interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Coal"]).iloc[0])
        
        #amount
        amount = 100 * self.Capex.loc[self.timestep-1, "Coal"]
    
        #calculate interest
        interest = float(amount * interest_rate)
        
        #caluate profit with income outlook and fix costs
        profit = income_outlook-(self.Fix_costs.loc["Coal"]*100) - interest
        
        #form the cash_flow list
        cash_flows = [profit] * self.coal_lifetime
        
        #insert investment cost as first value
        cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Coal"])

        #calculate the rate of return
        irr = npf.irr(cash_flows)

        #if irr is nan (profit is negative), mirror irr
        if irr != irr:
            cash_flows = [-profit] * self.coal_lifetime
            cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Coal"])
            irr = npf.irr(cash_flows)
            irr = -irr - 2

        #save irr
        self.agents["Coal"]["data"].loc[self.timestep, "Irr"] = irr

        #check if crisis is happening
        if self.LIBOR_index.iloc[self.timestep,0] > 0.025:
            crisis_shift = 100*((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"]+self.Risk_markup["Production Coal"].iloc[0])-0.025)**2
        else:
            crisis_shift = 0

        #calcualte shift, increases when interste rate rises
        shift = self.coal_shift-crisis_shift
        
        #if self.time <= 2030:
        if self.time <= 2050:   
        
            #calculate power_increase with sigmoid function
            power_increase = self.coal_max * self.sigmoid(irr,self.coal_slope,shift)
    
            #round power increase to a block size
            power_increase = (power_increase // self.coal_block_size) * self.coal_block_size

        else: 
            power_increase = 0

        #check if power increase is more than 0
        if power_increase > 0:
            #safe power increase in power increase dataframe
            self.agents["Coal"]['Investments'].loc[self.timestep, 'Block_size'] = self.agents["Coal"]['Investments'].loc[self.timestep, 'Block_size'] + power_increase
           
            #set lifetime
            self.agents["Coal"]['Investments'].loc[self.timestep, 'Lifetime'] = self.coal_lifetime
            
            #set status
            self.agents["Coal"]['Investments'].loc[self.timestep, 'Status'] = True
       
            #calculate interest rate 
            interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Coal"]).iloc[0])
            
            #amount
            amount = power_increase * self.Capex.loc[self.timestep, "Coal"]
        
            #calculate interest
            interest = float(amount * interest_rate)    
        
            #take on loan and write that into the loan list
            loans = {'Agent_name': ["Coal"], 'Active': [True], 'Runtime': [self.coal_lifetime], 'Amount': [amount], 'Interest_rate': [interest_rate], 'Payback': [interest]}
            self.loans_temp = pd.DataFrame(data = loans)
            
            #add to loans list
            self.loans = pd.concat([self.loans, self.loans_temp], ignore_index=True)
       
        #calculate the total installed power
        self.agents["Coal"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Coal"]["data"].loc[self.timestep, "Installed_power_initial"] + self.agents["Coal"]["Investments"].loc[self.agents["Coal"]["Investments"]['Status'] == True, 'Block_size'].sum()
        
        #calculate t co2 emittet by producing 1 MWh
        co2_mwh = self.co2_emissions_coal / self.agents["Coal"]["data"].loc[self.timestep, "Efficiency"]
        
        #calculate total emissions with calculated PyPSA data
        self.agents["Coal"]["data"].loc[self.timestep, "Total_emissions"] = self.agents["Coal"]["data"].loc[self.timestep, "Generated_power_total"] * co2_mwh 
        
        #for coal phase out, variant 2, after 2030 no more coal power plants
        #if self.time >=2030:
            
            #self.agents["Coal"]["data"].loc[self.timestep, "Installed_power_initial"] = 0
            
            # Loop through the investments dataframe
            #for i in range(len(self.agents["Coal"]['Investments'])):
        
                #self.agents["Coal"]['Investments'].loc[i, 'Status'] == False
        
        
    def gas_ct_prognosis(self):
        """
        prognosis of Gas_CT.
        This producer resembels the combined gas power plants Combustion Turbine
        """
        print("\nGas CT prognosis")
        
        #update installed_power
        self.agents["Gas_CT"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Gas_CT"]["data"].loc[self.timestep-1, "Installed_power"]

        #update efficiency
        self.agents["Gas_CT"]["data"].loc[self.timestep, "Efficiency"] = self.agents["Gas_CT"]["data"].loc[self.timestep-1, "Efficiency"]
        
        #update marginal cost
        #calculate co2 costs
        #calculate t co2 emittet by producing 1 MWh
        co2_mwh = self.co2_emissions_gas / self.agents["Gas_CT"]["data"].loc[self.timestep, "Efficiency"]
        
        #calculate co2 costs
        co2_costs = self.Co2_certificate_costs.iloc[self.timestep,0] * co2_mwh
        
        #calculate ressource costs
        ressource_cost = self.gas_costs.iloc[self.timestep,0] / self.agents["Gas_CT"]["data"].loc[self.timestep, "Efficiency"]

        #calculate marginal costs  
        marginal_cost = ressource_cost + co2_costs + self.Operational_costs["Gas_CT"]
        
        #write marginal cost in dataframe
        self.agents["Gas_CT"]["data"].loc[self.timestep, "Marginal_cost"] = marginal_cost

        #investment function
    
        #calculate interest rate 
        interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Gas"]).iloc[0])
        
        #amount
        amount = 100 * self.Capex.loc[self.timestep-1, "Gas CT"]
    
        #calculate interest
        interest = float(amount * interest_rate)    
    
        #calculate outlook of profit
        income_outlook = sum((self.gas_ct_profile * 100) * (self.marginal_costs-self.agents["Gas_CT"]["data"].loc[self.timestep-1, "Marginal_cost"]))
    
        #caluate profit with income outlook and fix costs
        profit = income_outlook-(self.Fix_costs.loc["Gas_CT"]*100) - interest

        #form the cash_flow list
        cash_flows = [profit] * self.gas_lifetime

        #insert investment cost as first value
        cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Gas CT"])
        
        #calculate the rate of return
        irr = npf.irr(cash_flows)

        #if irr is nan (profit is negative), mirror irr
        if irr != irr:
            cash_flows = [-profit] * self.gas_lifetime
            cash_flows.insert(0, 0-50*self.Capex.loc[self.timestep-1, "Gas CT"])
            irr = npf.irr(cash_flows)
            irr = -irr - 2

        #save irr
        self.agents["Gas_CT"]["data"].loc[self.timestep, "Irr prognose"] = irr

        #check if crisis is happening
        if self.LIBOR_index.iloc[self.timestep,0] > 0.025:
            crisis_shift = 100*((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"]+self.Risk_markup["Production Gas"].iloc[0])-0.025)**2
        
        else:
            crisis_shift = 0

        #calcualte shift, increases when interste rate rises
        shift = self.gas_ct_shift-crisis_shift

        #calculate power_increase with sigmoid function
        power_increase = self.gas_ct_max * self.sigmoid(irr,self.gas_ct_slope,shift)

        #round power increase to a block size
        power_increase = (power_increase // self.gas_block_size) * self.gas_block_size   

        #check if power increase is higher than 0
        if power_increase > 0:
    
            #safe power increase in power increase dataframe
            self.agents["Gas_CT"]['Investments'].loc[self.timestep, 'Block_size'] +=  power_increase
        
            #set lifetime
            self.agents["Gas_CT"]['Investments'].loc[self.timestep, 'Lifetime'] = self.gas_lifetime
            
            #set status
            self.agents["Gas_CT"]['Investments'].loc[self.timestep, 'Status'] = True        
    
            #calculate interest rate 
            interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Gas"]).iloc[0])
        
            #amount
            amount = power_increase * self.Capex.loc[self.timestep, "Gas CT"]
        
            #payback
            payback = float(amount * interest_rate)
        
            #take on loan and write that into the loan list
            loans = {'Agent_name': ["Gas_CT"], 'Active': [True], 'Runtime': [self.gas_lifetime], 'Amount': [amount], 'Interest_rate': [interest_rate], 'Payback': [payback]}
            self.loans_temp = pd.DataFrame(data = loans)
            
            #add to loans list
            self.loans = pd.concat([self.loans, self.loans_temp], ignore_index=True)
           
        #calculate the total installed power
        self.agents["Gas_CT"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Gas_CT"]["data"].loc[self.timestep-1, "Installed_power_initial"] + self.agents["Gas_CT"]["Investments"].loc[self.agents["Gas_CT"]["Investments"]['Status'] == True, 'Block_size'].sum()

    def gas_ct(self):
        """
        Function of Gas_CT.
        This producer resembels the combined gas power plants Combustion Turbine
        """
        print("\nGas CT called")
              
        #check if initial installed power is higher than 0
        if self.agents["Gas_CT"]['data'].loc[self.timestep-1, 'Installed_power_initial'] > 0:
        
            
            #initial installed power is reduced by a fixed amount every year
            self.agents["Gas_CT"]['data'].loc[self.timestep, 'Installed_power_initial'] = self.agents["Gas_CT"]["data"].loc[self.timestep-1, 'Installed_power_initial'] - self.gas_ct_deprication
        
            #strategy 8 increase lifetime:
            #if self.time >= 2033 and self.time <= 2036:
        
                #self.agents["Gas_CT"]['data'].loc[self.timestep, 'Installed_power_initial'] = self.agents["Gas_CT"]["data"].loc[self.timestep-1, 'Installed_power_initial']
            
            #else:
                
                #self.agents["Gas_CT"]['data'].loc[self.timestep, 'Installed_power_initial'] = self.agents["Gas_CT"]["data"].loc[self.timestep-1, 'Installed_power_initial'] - self.gas_ct_deprication           
                
        else: 
            self.agents["Gas_CT"]["data"].loc[self.timestep, 'Installed_power_initial'] = 0 
        
        #calculate the total installed power
        self.agents["Gas_CT"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Gas_CT"]["data"].loc[self.timestep, "Installed_power_initial"] + self.agents["Gas_CT"]["Investments"].loc[self.agents["Gas_CT"]["Investments"]['Status'] == True, 'Block_size'].sum()
        
        #update efficiency, increase by 0.1% every year
        #todo: function einbauen die dafür sorgt, dass der wirkungsgrad bis 41% steigt und danach so bleibt
        self.agents["Gas_CT"]["data"].loc[self.timestep, "Efficiency"] = self.agents["Gas_CT"]["data"].loc[self.timestep-1, "Efficiency"] + 0.001    
        
        #update marginal cost
        #calculate co2 costs
        #calculate t co2 emittet by producing 1 MWh
        co2_mwh = self.co2_emissions_gas / self.agents["Gas_CT"]["data"].loc[self.timestep, "Efficiency"]
        
        #calculate co2 costs
        co2_costs = self.Co2_certificate_costs.iloc[self.timestep,0] * co2_mwh
        
        #calculate ressource costs
        ressource_cost = self.gas_costs.iloc[self.timestep,0] / self.agents["Gas_CT"]["data"].loc[self.timestep, "Efficiency"]

        #calculate marginal costs  
        marginal_cost = ressource_cost + co2_costs + self.Operational_costs["Gas_CT"]
        
        #write marginal cost in dataframe
        self.agents["Gas_CT"]["data"].loc[self.timestep, "Marginal_cost"] = marginal_cost

        #fill generated_power_total with calculated PyPSA data
        self.agents["Gas_CT"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.loc["Gas_CT"]
        
        #set income with the income of the last timestep from PyPSA
        self.agents["Gas_CT"]["data"].loc[self.timestep, "Income"] = self.gas_ct_income_total

        #set expenses from the fix costs
        self.agents["Gas_CT"]["data"].loc[self.timestep, "Expenses"] = self.agents["Gas_CT"]["data"].loc[self.timestep, "Expenses"]-self.Fix_costs.loc["Gas_CT"]*self.agents["Gas_CT"]["data"].loc[self.timestep, "Installed_power"]

        #calculate profit
        self.agents["Gas_CT"]["data"].loc[self.timestep, "Profit"] = self.agents["Gas_CT"]["data"].loc[self.timestep, "Income"] + self.agents["Gas_CT"]["data"].loc[self.timestep, "Payback"] + self.agents["Gas_CT"]["data"].loc[self.timestep, "Expenses"]

        #update money
        self.agents["Gas_CT"]["data"].loc[self.timestep, "Money"] = self.agents["Gas_CT"]["data"].loc[self.timestep-1, "Money"] + self.agents["Gas_CT"]["data"].loc[self.timestep, "Profit"] 

        #investment function
        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy_total = self.agents["Gas_CT"]["data"].loc[self.timestep, "Installed_power"] * 365 * 24
        
        #calculate average load in the last time step
        self.agents["Gas_CT"]["data"].loc[self.timestep, "Load"] = round(self.agents["Gas_CT"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy_total,4)
    
        #check if load is bigger than 1
        if self.agents["Gas_CT"]["data"].loc[self.timestep, "Load"] > 1:
            #if yes set load to 1
            self.agents["Gas_CT"]["data"].loc[self.timestep, "Load"] = 1
    
        #calculate interest rate 
        interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Gas"]).iloc[0])
        
        #amount
        amount = 100 * self.Capex.loc[self.timestep-1, "Gas CT"]
    
        #calculate interest
        interest = float(amount * interest_rate)      
    
        #calculate outlook of profit
        income_outlook = sum((self.gas_ct_profile * 100) * (self.marginal_costs-self.agents["Gas_CT"]["data"].loc[self.timestep-1, "Marginal_cost"]))
    
        #caluate profit with income outlook and fix costs
        profit = income_outlook-(self.Fix_costs.loc["Gas_CT"]*100) - interest
        
        #form the cash_flow list
        cash_flows = [profit] * self.gas_lifetime

        #insert investment cost as first value
        cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Gas CT"])

        #calculate the rate of return
        irr = npf.irr(cash_flows)

        #if irr is nan (profit is negative), mirror irr
        if irr != irr:
            cash_flows = [-profit] * self.gas_lifetime
            cash_flows.insert(0, 0-50*self.Capex.loc[self.timestep-1, "Gas CT"])
            irr = npf.irr(cash_flows)
            irr = -irr - 2
        
        #save irr
        self.agents["Gas_CT"]["data"].loc[self.timestep, "Irr"] = irr    

        #check if crisis is happening
        if self.LIBOR_index.iloc[self.timestep,0] > 0.025:
            crisis_shift = 100*((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"]+self.Risk_markup["Production Gas"].iloc[0])-0.025)**2
        else:
            crisis_shift = 0

        #calcualte shift, increases when interste rate rises
        shift = self.gas_ct_shift-crisis_shift

        #calculate power_increase with sigmoid function
        power_increase = self.gas_ct_max * self.sigmoid(irr,self.gas_ct_slope,shift)

        #round power increase to a block size
        power_increase = (power_increase // self.gas_block_size) * self.gas_block_size

        #check if power increase is more than 0
        if power_increase > 0:
            
            #safe power increase in power increase dataframe
            self.agents["Gas_CT"]['Investments'].loc[self.timestep, 'Block_size'] += power_increase
           
            #set lifetime
            self.agents["Gas_CT"]['Investments'].loc[self.timestep, 'Lifetime'] = self.gas_lifetime
            
            #set status
            self.agents["Gas_CT"]['Investments'].loc[self.timestep, 'Status'] = True
            
            #calculate interest rate 
            interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Gas"]).iloc[0])
    
            #amount
            amount = power_increase * self.Capex.loc[self.timestep, "Gas CT"]
    
            #payback
            payback = float(amount * interest_rate)
    
            #take on loan and write that into the loan list
            loans = {'Agent_name': ["Gas_CT"], 'Active': [True], 'Runtime': [self.gas_lifetime], 'Amount': [amount], 'Interest_rate': [interest_rate], 'Payback': [payback]}
            self.loans_temp = pd.DataFrame(data = loans)
            
            #add to loans list
            self.loans = pd.concat([self.loans, self.loans_temp], ignore_index=True)
           
        #calculate the total installed power
        self.agents["Gas_CT"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Gas_CT"]["data"].loc[self.timestep, "Installed_power_initial"] + self.agents["Gas_CT"]["Investments"].loc[self.agents["Gas_CT"]["Investments"]['Status'] == True, 'Block_size'].sum()      

        #calculate t co2 emittet by producing 1 MWh
        co2_mwh = self.co2_emissions_gas / self.agents["Gas_CT"]["data"].loc[self.timestep, "Efficiency"]
        
        #calculate total emissions with calculated PyPSA data
        self.agents["Gas_CT"]["data"].loc[self.timestep, "Total_emissions"] = self.agents["Gas_CT"]["data"].loc[self.timestep, "Generated_power_total"] * co2_mwh 
        
        
    def gas_cc_prognosis(self):
        """
        prognosis of Gas_CC.
        This producer resembels the combined gas power plants Combined Cycle
        """
        print("\nGas CC prognosis")
        
        #update installed_power
        self.agents["Gas_CC"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Gas_CC"]["data"].loc[self.timestep-1, "Installed_power"]

        #update efficiency
        self.agents["Gas_CC"]["data"].loc[self.timestep, "Efficiency"] = self.agents["Gas_CC"]["data"].loc[self.timestep-1, "Efficiency"]
        
        #update marginal cost
        
        #calculate co2 costs
        #calculate t co2 emittet by producing 1 MWh
        co2_mwh = self.co2_emissions_gas / self.agents["Gas_CC"]["data"].loc[self.timestep, "Efficiency"]
        
        #calculate co2 costs
        co2_costs = self.Co2_certificate_costs.iloc[self.timestep,0] * co2_mwh
        
        #calculate ressource costs
        ressource_cost = self.gas_costs.iloc[self.timestep,0] / self.agents["Gas_CC"]["data"].loc[self.timestep, "Efficiency"]

        #calculate marginal costs  
        marginal_cost = ressource_cost + co2_costs + self.Operational_costs["Gas_CC"]
        
        #write marginal cost in dataframe
        self.agents["Gas_CC"]["data"].loc[self.timestep, "Marginal_cost"] = marginal_cost
    
        #investment function
        #calculate outlook of profit
        income_outlook = sum((self.gas_cc_profile * 100) * (self.marginal_costs-self.agents["Gas_CC"]["data"].loc[self.timestep-1, "Marginal_cost"]))
    
        #calculate interest rate 
        interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Gas"]).iloc[0])
        
        #amount
        amount = 100 * self.Capex.loc[self.timestep-1, "Gas CC"]
    
        #calculate interest
        interest = float(amount * interest_rate)
    
        #caluate profit with income outlook and fix costs
        profit = income_outlook-(self.Fix_costs.loc["Gas_CC"]*100) - interest
        
        #form the cash_flow list
        cash_flows = [profit] * self.gas_lifetime

        #insert investment cost as first value
        cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Gas CC"])

        #calculate the rate of return
        irr = npf.irr(cash_flows)

        #if irr is nan (profit is negative), mirror irr
        if irr != irr:
            cash_flows = [-profit] * self.gas_lifetime
            cash_flows.insert(0, 0-50*self.Capex.loc[self.timestep-1, "Gas CC"])
            irr = npf.irr(cash_flows)
            irr = -irr - 2
        
        #save irr
        self.agents["Gas_CC"]["data"].loc[self.timestep, "Irr prognose"] = irr    

        #check if crisis is happening
        if self.LIBOR_index.iloc[self.timestep,0] > 0.025:
            crisis_shift = 100*((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"]+self.Risk_markup["Production Gas"].iloc[0])-0.025)**2
        else:
            crisis_shift = 0

        #calcualte shift, increases when interste rate rises
        shift = self.gas_cc_shift-crisis_shift

        #calculate power_increase with sigmoid function
        power_increase = self.gas_cc_max * self.sigmoid(irr,self.gas_cc_slope,shift)

        #round power increase to a block size
        power_increase = (power_increase // self.gas_block_size) * self.gas_block_size 

        #check if power increase is more than 0
        if power_increase > 0:

            #safe power increase in power increase dataframe
            self.agents["Gas_CC"]['Investments'].loc[self.timestep, 'Block_size'] +=  power_increase
        
            #set lifetime
            self.agents["Gas_CC"]['Investments'].loc[self.timestep, 'Lifetime'] = self.gas_lifetime
            
            #set status
            self.agents["Gas_CC"]['Investments'].loc[self.timestep, 'Status'] = True            

            #calculate interest rate 
            interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Gas"]).iloc[0])
        
            #amount
            amount = power_increase * self.Capex.loc[self.timestep, "Gas CC"]
        
            #payback
            payback = float(amount * interest_rate)
        
            #take on loan and write that into the loan list
            loans = {'Agent_name': ["Gas_CC"], 'Active': [True], 'Runtime': [self.gas_lifetime], 'Amount': [amount], 'Interest_rate': [interest_rate], 'Payback': [payback]}
            self.loans_temp = pd.DataFrame(data = loans)
            
            #add to loans list
            self.loans = pd.concat([self.loans, self.loans_temp], ignore_index=True)
           
        #calculate the total installed power
        self.agents["Gas_CC"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Gas_CC"]["data"].loc[self.timestep-1, "Installed_power_initial"] + self.agents["Gas_CC"]["Investments"].loc[self.agents["Gas_CC"]["Investments"]['Status'] == True, 'Block_size'].sum()


    def gas_cc(self):
        """
        Function of Gas_CC.
        This producer resembels the combined gas power plants Combined Cycle
        """
        print("\nGas CC called")
        
        #check if initial installed power is higher than 0
        if self.agents["Gas_CC"]['data'].loc[self.timestep-1, 'Installed_power_initial'] > 0:
            #initial installed power is reduced by a fixed amount every year
            self.agents["Gas_CC"]['data'].loc[self.timestep, 'Installed_power_initial'] = self.agents["Gas_CC"]["data"].loc[self.timestep-1, 'Installed_power_initial'] - self.gas_cc_deprication
        
            #strategy 8 increase lifetime:
            #if self.time >= 2033 and self.time <= 2036:
        
                #self.agents["Gas_CC"]['data'].loc[self.timestep, 'Installed_power_initial'] = self.agents["Gas_CC"]["data"].loc[self.timestep-1, 'Installed_power_initial']
            
            #else:
                
                #self.agents["Gas_CC"]['data'].loc[self.timestep, 'Installed_power_initial'] = self.agents["Gas_CC"]["data"].loc[self.timestep-1, 'Installed_power_initial'] - self.gas_cc_deprication           
                
        else: 
            self.agents["Gas_CC"]["data"].loc[self.timestep, 'Installed_power_initial'] = 0 
        
        #calculate the total installed power
        self.agents["Gas_CC"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Gas_CC"]["data"].loc[self.timestep, "Installed_power_initial"] + self.agents["Gas_CC"]["Investments"].loc[self.agents["Gas_CC"]["Investments"]['Status'] == True, 'Block_size'].sum()

        #update efficiency, increase by 0.1% every year
        #todo: function einbauen die dafür sorgt, dass der wirkungsgrad bis 41% steigt und danach so bleibt
        self.agents["Gas_CC"]["data"].loc[self.timestep, "Efficiency"] = self.agents["Gas_CC"]["data"].loc[self.timestep-1, "Efficiency"] + 0.001    
             
        #update marginal cost
        
        #calculate co2 costs
        #calculate t co2 emittet by producing 1 MWh
        co2_mwh = self.co2_emissions_gas / self.agents["Gas_CC"]["data"].loc[self.timestep, "Efficiency"]
        
        #calculate co2 costs
        co2_costs = self.Co2_certificate_costs.iloc[self.timestep,0] * co2_mwh
        
        #calculate ressource costs
        ressource_cost = self.gas_costs.iloc[self.timestep,0] / self.agents["Gas_CC"]["data"].loc[self.timestep, "Efficiency"]

        #calculate marginal costs  
        marginal_cost = ressource_cost + co2_costs + self.Operational_costs["Gas_CC"]
        
        #write marginal cost in dataframe
        self.agents["Gas_CC"]["data"].loc[self.timestep, "Marginal_cost"] = marginal_cost

        #fill generated_power_total with calculated PyPSA data
        self.agents["Gas_CC"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.loc["Gas_CC"]
        
        #set income with the income of the last timestep from PyPSA
        self.agents["Gas_CC"]["data"].loc[self.timestep, "Income"] = self.gas_cc_income_total

        #set expenses from the fix costs
        self.agents["Gas_CC"]["data"].loc[self.timestep, "Expenses"] = self.agents["Gas_CC"]["data"].loc[self.timestep, "Expenses"]-self.Fix_costs.loc["Gas_CC"]*self.agents["Gas_CC"]["data"].loc[self.timestep, "Installed_power"]

        #calculate profit
        self.agents["Gas_CC"]["data"].loc[self.timestep, "Profit"] = self.agents["Gas_CC"]["data"].loc[self.timestep, "Income"] + self.agents["Gas_CC"]["data"].loc[self.timestep, "Payback"] + self.agents["Gas_CC"]["data"].loc[self.timestep, "Expenses"]

        #update money
        self.agents["Gas_CC"]["data"].loc[self.timestep, "Money"] = self.agents["Gas_CC"]["data"].loc[self.timestep-1, "Money"] + self.agents["Gas_CC"]["data"].loc[self.timestep, "Profit"] 

        #investment function
        
        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy_total = self.agents["Gas_CC"]["data"].loc[self.timestep, "Installed_power"] * 365 * 24
        
        #calculate average load in the last time step
        self.agents["Gas_CC"]["data"].loc[self.timestep, "Load"] = round(self.agents["Gas_CC"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy_total,4)
        
        #check if load is bigger than 1
        if self.agents["Gas_CC"]["data"].loc[self.timestep, "Load"] > 1:
            #if yes set load to 1
            self.agents["Gas_CC"]["data"].loc[self.timestep, "Load"] = 1
        
        #investment function
        #calculate outlook of profit
        income_outlook = sum((self.gas_cc_profile * 100) * (self.marginal_costs-self.agents["Gas_CC"]["data"].loc[self.timestep-1, "Marginal_cost"])) + sum(self.gas_cc_profile * 100) * 35

        #calculate interest rate 
        interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Gas"]).iloc[0])
        
        #amount
        amount = 100 * self.Capex.loc[self.timestep-1, "Gas CC"]
    
        #calculate interest
        interest = float(amount * interest_rate)  
    
        #caluate profit with income outlook and fix costs
        profit = income_outlook-(self.Fix_costs.loc["Gas_CC"]*100) - interest
        
        #form the cash_flow list
        cash_flows = [profit] * self.gas_lifetime

        #insert investment cost as first value
        cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Gas CC"])

        #calculate the rate of return
        irr = npf.irr(cash_flows)

        #if irr is nan (profit is negative), mirror irr
        if irr != irr:
            cash_flows = [-profit] * self.gas_lifetime
            cash_flows.insert(0, 0-50*self.Capex.loc[self.timestep-1, "Gas CC"])
            irr = npf.irr(cash_flows)
            irr = -irr - 2
        
        #save irr
        self.agents["Gas_CC"]["data"].loc[self.timestep, "Irr"] = irr    

        #check if crisis is happening
        if self.LIBOR_index.iloc[self.timestep,0] > 0.025:
            crisis_shift = 100*((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"]+self.Risk_markup["Production Gas"].iloc[0])-0.025)**2
        else:
            crisis_shift = 0

        #calcualte shift, increases when interste rate rises
        shift = self.gas_cc_shift-crisis_shift

        #calculate power_increase with sigmoid function
        power_increase = self.gas_cc_max * self.sigmoid(irr,self.gas_cc_slope,shift)

        #round power increase to a block size
        power_increase = (power_increase // self.gas_block_size) * self.gas_block_size

        #check if power increase is more than 0
        if power_increase > 0:

            #safe power increase in power increase dataframe
            self.agents["Gas_CC"]['Investments'].loc[self.timestep, 'Block_size'] += power_increase
           
            #set lifetime
            self.agents["Gas_CC"]['Investments'].loc[self.timestep, 'Lifetime'] = self.gas_lifetime
            
            #set status
            self.agents["Gas_CC"]['Investments'].loc[self.timestep, 'Status'] = True            

            #calculate interest rate 
            interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Gas"]).iloc[0])
    
            #amount
            amount = power_increase * self.Capex.loc[self.timestep, "Gas CC"]
    
            #payback
            payback = float(amount * interest_rate)
    
            #take on loan and write that into the loan list
            loans = {'Agent_name': ["Gas_CC"], 'Active': [True], 'Runtime': [self.gas_lifetime], 'Amount': [amount], 'Interest_rate': [interest_rate], 'Payback': [payback]}
            self.loans_temp = pd.DataFrame(data = loans)
            
            #add to loans list
            self.loans = pd.concat([self.loans, self.loans_temp], ignore_index=True)
      
        #calculate the total installed power
        self.agents["Gas_CC"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Gas_CC"]["data"].loc[self.timestep, "Installed_power_initial"] + self.agents["Gas_CC"]["Investments"].loc[self.agents["Gas_CC"]["Investments"]['Status'] == True, 'Block_size'].sum()
        
        #calculate t co2 emittet by producing 1 MWh
        co2_mwh = self.co2_emissions_gas / self.agents["Gas_CC"]["data"].loc[self.timestep, "Efficiency"]
        
        #calculate total emissions with calculated PyPSA data
        self.agents["Gas_CC"]["data"].loc[self.timestep, "Total_emissions"] = self.agents["Gas_CC"]["data"].loc[self.timestep, "Generated_power_total"] * co2_mwh 
        
        
    def nuclear(self):
        """
        Function of Nuclear.
        This producer resembels the combined nuclear power plants
        """
        print("\nNuclear called")
        
        #update marginal cost
        #calculate ressource costs
        ressource_cost = (self.uranium_costs.iloc[self.timestep,0] * self.fuel_factor_uranium + self.fuel_factor_uranium * self.conversion_uranium + self.fuel_factor_uranium * self.enrichment_uranium + self.fuel_fabrication_uranium) / self.energy_content_uranium
        #calculate marginal costs
        marginal_cost = ressource_cost + self.Operational_costs["Nuclear"]
        #write marginal cost in dataframe
        self.agents["Nuclear"]["data"].loc[self.timestep, "Marginal_cost"] = marginal_cost
        
        #update installed_power, generation decreased, because of nuclear phase out 9% a year
        if self.time < 2023:
            self.agents["Nuclear"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Nuclear"]["data"].loc[self.timestep-1, "Installed_power"]*(1-0.09)
        else:
            self.agents["Nuclear"]["data"].loc[self.timestep, "Installed_power"] = 0

        #fill generated_power_total with calculated PyPSA data
        self.agents["Nuclear"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.loc["Nuclear"]

        #set income with the income of the last timestep from PyPSA
        self.agents["Nuclear"]["data"].loc[self.timestep, "Income"] = self.nuclear_income_total

        #set expenses from the fix costs
        self.agents["Nuclear"]["data"].loc[self.timestep, "Expenses"] = self.agents["Nuclear"]["data"].loc[self.timestep, "Expenses"] - self.Fix_costs.loc["Nuclear"] * self.agents["Nuclear"]["data"].loc[self.timestep, "Installed_power"]

        #calculate profit
        self.agents["Nuclear"]["data"].loc[self.timestep, "Profit"] = self.agents["Nuclear"]["data"].loc[self.timestep, "Income"] + self.agents["Nuclear"]["data"].loc[self.timestep, "Payback"] + self.agents["Nuclear"]["data"].loc[self.timestep, "Expenses"]

        #update money
        self.agents["Nuclear"]["data"].loc[self.timestep, "Money"] = self.agents["Nuclear"]["data"].loc[self.timestep-1, "Money"] + self.agents["Nuclear"]["data"].loc[self.timestep, "Profit"] 

        #investment function
        
        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy_total = self.agents["Nuclear"]["data"].loc[self.timestep-1, "Installed_power"] * 365 * 24
        
        #calculate average load
        average_load = round(self.agents["Nuclear"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy_total,4)
        
        #fill data
        self.agents["Nuclear"]["data"].loc[self.timestep, "Load"] = average_load
        
        #no investment function because of nuclear phase out
        
        #calculate total emissions with calculated PyPSA data, nuclear energy has no emissions in the operation
        self.agents["Nuclear"]["data"].loc[self.timestep, "Total_emissions"] = 0
        
        #update efficiency, because of the nuclear phase-out stays the same
        self.agents["Nuclear"]["data"].loc[self.timestep, "Efficiency"] = self.agents["Nuclear"]["data"].loc[self.timestep-1, "Efficiency"]

        
    def solar_prognosis(self):
        """
        Prognosis of Solar.
        This producer resembels the combined pv power plants
        """
        print("\nSolar prognosis")
        
        #update installed_power
        self.agents["Solar"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Solar"]["data"].loc[self.timestep-1, "Installed_power"]

        #update efficiency
        self.agents["Solar"]["data"].loc[self.timestep, "Efficiency"] = self.agents["Solar"]["data"].loc[self.timestep-1, "Efficiency"]
        
        #update marginal cost
        #write marginal cost in dataframe
        self.agents["Solar"]["data"].loc[self.timestep, "Marginal_cost"] = 0

        #investment function
        #calculate total energy
        total_energy = [value * 100 for value in self.pv_generation_profile]
        
        #calculate outlook of profit
        income_outlook = sum(total_energy * (self.marginal_costs-self.agents["Solar"]["data"].loc[self.timestep-1, "Marginal_cost"]))
    
        #calculate interest rate 
        interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Solar"]).iloc[0])
        
        #amount
        amount = 100 * self.Capex.loc[self.timestep-1, "Solar Utility"]
    
        #calculate interest
        interest = float(amount * interest_rate)  
    
        #caluate profit with income outlook and fix costs
        profit = income_outlook-(self.Fix_costs.loc["Solar Utility"]*100) - interest
        
        #form the cash_flow list
        cash_flows = [profit] * self.solar_lifetime

        #insert investment cost as first value
        cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Solar Utility"])

        #calculate the rate of return
        irr = npf.irr(cash_flows)

        #if irr is nan (profit is negative), mirror irr
        if irr != irr:
            cash_flows = [-profit] * self.solar_lifetime
            cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Solar Utility"])
            irr = npf.irr(cash_flows)
            irr = -irr - 2
        
        #save irr
        self.agents["Solar"]["data"].loc[self.timestep, "Irr prognose"] = irr    

        #check if crisis is happening
        if self.LIBOR_index.iloc[self.timestep,0] > 0.025:
            crisis_shift = 100*((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"]+self.Risk_markup["Production Solar"].iloc[0])-0.025)**2
        else:
            crisis_shift = 0

        #calcualte shift, increases when interste rate rises
        shift = self.solar_shift-crisis_shift

        #calculate power_increase with sigmoid function
        power_increase = self.solar_max * self.sigmoid(irr,self.solar_slope,shift)

        #round power increase to a block size
        power_increase = (power_increase // self.solar_block_size) * self.solar_block_size       
        
        #check if power increase is more than 0
        if power_increase > 0:
            #safe power increase in power increase dataframe
            self.agents["Solar"]['Investments'].loc[self.timestep, 'Block_size'] +=  power_increase
            
            #set lifetime
            self.agents["Solar"]['Investments'].loc[self.timestep, 'Lifetime'] = self.solar_lifetime
            
            #set status
            self.agents["Solar"]['Investments'].loc[self.timestep, 'Status'] = True
    
            #calculate interest rate 
            interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Solar"]).iloc[0])
    
            #amount
            amount = power_increase * self.Capex.loc[self.timestep, "Solar Utility"]
    
            #payback
            payback = float(amount * interest_rate)
    
            #take on loan and write that into the loan list
            loans = {'Agent_name': ["Solar"], 'Active': [True], 'Runtime': [self.solar_lifetime], 'Amount': [amount], 'Interest_rate': [interest_rate], 'Payback': [payback]}
            self.loans_temp = pd.DataFrame(data = loans)
            
            #add to loans list
            self.loans = pd.concat([self.loans, self.loans_temp], ignore_index=True)

            #calculate the total installed power
            self.agents["Solar"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Solar"]["data"].loc[self.timestep-1, "Installed_power_initial"] + self.agents["Solar"]["Investments"].loc[self.agents["Solar"]["Investments"]['Status'] == True, 'Block_size'].sum()

    
    def solar(self):
        """
        Function of Solar.
        This producer resembels the combined pv power plants
        """
        print("\nSolar called")
        
        #check if initial installed power is higher than 0
        if self.agents["Solar"]["data"].loc[self.timestep-1, 'Installed_power_initial'] - self.solar_deprication  > 0:
            #initial installed power is reduced by a fixed amount every year
            self.agents["Solar"]['data'].loc[self.timestep, 'Installed_power_initial'] = self.agents["Solar"]["data"].loc[self.timestep-1, 'Installed_power_initial'] - self.solar_deprication
        
            #strategy 8 increase lifetime:
            #if self.time >= 2033 and self.time <= 2036:
        
                #self.agents["Solar"]['data'].loc[self.timestep, 'Installed_power_initial'] = self.agents["Solar"]["data"].loc[self.timestep-1, 'Installed_power_initial']
            
            #else:
                
                #self.agents["Solar"]['data'].loc[self.timestep, 'Installed_power_initial'] = self.agents["Solar"]["data"].loc[self.timestep-1, 'Installed_power_initial'] - self.solar_deprication           
        else: 
            self.agents["Solar"]["data"].loc[self.timestep, 'Installed_power_initial'] = 0
        
        #calculate the total installed power
        self.agents["Solar"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Solar"]["data"].loc[self.timestep, "Installed_power_initial"] + self.agents["Solar"]["Investments"].loc[self.agents["Solar"]["Investments"]['Status'] == True, 'Block_size'].sum()

        #update efficiency, increase by 0.1% every year
        #todo: function einbauen die dafür sorgt, dass der wirkungsgrad bis 41% steigt und danach so bleibt
        self.agents["Solar"]["data"].loc[self.timestep, "Efficiency"] = self.agents["Solar"]["data"].loc[self.timestep-1, "Efficiency"] + 0.001    

        #fill generated_power_total with calculated PyPSA data
        self.agents["Solar"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.loc["Solar"]
        
        #set income with the income of the last timestep from PyPSA
        self.agents["Solar"]["data"].loc[self.timestep, "Income"] = self.solar_income_total

        #set expenses from the fix costs
        self.agents["Solar"]["data"].loc[self.timestep, "Expenses"] = self.agents["Solar"]["data"].loc[self.timestep, "Expenses"]-self.Fix_costs.loc["Solar Utility"]*self.agents["Solar"]["data"].loc[self.timestep, "Installed_power"]

        #calculate profit
        self.agents["Solar"]["data"].loc[self.timestep, "Profit"] = self.agents["Solar"]["data"].loc[self.timestep, "Income"] + self.agents["Solar"]["data"].loc[self.timestep, "Payback"] + self.agents["Solar"]["data"].loc[self.timestep, "Expenses"]

        #update money
        self.agents["Solar"]["data"].loc[self.timestep, "Money"] = self.agents["Solar"]["data"].loc[self.timestep-1, "Money"] + self.agents["Solar"]["data"].loc[self.timestep, "Profit"] 

        #investment function
        
        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy = [round(value * self.agents["Solar"]["data"].loc[self.timestep, "Installed_power"], 4) for value in self.pv_generation_profile]
        theoratical_energy_total = sum(theoratical_energy)

        #calculate load and fill data
        self.agents["Solar"]["data"].loc[self.timestep, "Load"] = round(self.agents["Solar"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy_total,4)
        
        #check if load is bigger than 1
        if self.agents["Solar"]["data"].loc[self.timestep, "Load"] > 1:
            #if yes set load to 1
            self.agents["Solar"]["data"].loc[self.timestep, "Load"] = 1
        
        #investment function
        #calculate outlook of profit
        income_outlook = sum(([value * 100 for value in self.pv_generation_profile]) * (self.marginal_costs-self.agents["Solar"]["data"].loc[self.timestep-1, "Marginal_cost"]))

        #calculate interest rate 
        interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Solar"]).iloc[0])
        
        #amount
        amount = 100 * self.Capex.loc[self.timestep-1, "Solar Utility"]
    
        #calculate interest
        interest = float(amount * interest_rate)      
    
        #caluate profit with income outlook and fix costs
        profit = income_outlook-(self.Fix_costs.loc["Solar Utility"]*100) - interest

        #form the cash_flow list
        cash_flows = [profit] * self.solar_lifetime

        #insert investment cost as first value
        cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Solar Utility"])

        #calculate the rate of return
        irr = npf.irr(cash_flows)

        #if irr is nan (profit is negative), mirror irr
        if irr != irr:
            cash_flows = [-profit] * self.solar_lifetime
            cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Solar Utility"])
            irr = npf.irr(cash_flows)
            irr = -irr - 2
       
        #save irr
        self.agents["Solar"]["data"].loc[self.timestep, "Irr"] = irr     

        #check if crisis is happening
        if self.LIBOR_index.iloc[self.timestep,0] > 0.025:
            crisis_shift = 100*((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"]+self.Risk_markup["Production Solar"].iloc[0])-0.025)**2
        else:
            crisis_shift = 0

        #calcualte shift, increases when interste rate rises
        shift = self.solar_shift-crisis_shift

        #calculate power_increase with sigmoid function
        power_increase = self.solar_max * self.sigmoid(irr,self.solar_slope,shift)

        #round power increase to a block size
        power_increase = (power_increase // self.solar_block_size) * self.solar_block_size
        
        #check if power increase is more than 0
        if power_increase > 0:
            
            #safe power increase in power increase dataframe
            self.agents["Solar"]['Investments'].loc[self.timestep, 'Block_size'] += power_increase
               
            #set lifetime
            self.agents["Solar"]['Investments'].loc[self.timestep, 'Lifetime'] = self.solar_lifetime
            
            #set status
            self.agents["Solar"]['Investments'].loc[self.timestep, 'Status'] = True
    
            #calculate interest rate 
            interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Solar"]).iloc[0])
    
            #amount
            amount = power_increase * self.Capex.loc[self.timestep, "Solar Utility"]
    
            #payback
            payback = float(amount * interest_rate)
    
            #take on loan and write that into the loan list
            loans = {'Agent_name': ["Solar"], 'Active': [True], 'Runtime': [self.solar_lifetime], 'Amount': [amount], 'Interest_rate': [interest_rate], 'Payback': [payback]}
            self.loans_temp = pd.DataFrame(data = loans)
            
            #add to loans list
            self.loans = pd.concat([self.loans, self.loans_temp], ignore_index=True)
       
        #calculate the total installed power
        self.agents["Solar"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Solar"]["data"].loc[self.timestep, "Installed_power_initial"] + self.agents["Solar"]["Investments"].loc[self.agents["Solar"]["Investments"]['Status'] == True, 'Block_size'].sum()

        #calculate total emissions with calculated PyPSA data
        self.agents["Solar"]["data"].loc[self.timestep, "Total_emissions"] = self.agents["Solar"]["data"].loc[self.timestep, "Generated_power_total"] * 0
        
            
    def wind_prognosis(self):
        """
        Prognosis of Wind.
        This producer resembels the combined wind power plants
        """
        print("\nWind prognosis")
        
        #update installed_power
        self.agents["Wind"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Wind"]["data"].loc[self.timestep-1, "Installed_power"]

        #update efficiency
        self.agents["Wind"]["data"].loc[self.timestep, "Efficiency"] = self.agents["Wind"]["data"].loc[self.timestep-1, "Efficiency"]
        
        #update marginal cost
        #write marginal cost in dataframe
        self.agents["Wind"]["data"].loc[self.timestep, "Marginal_cost"] = 0

        #investment function
        #calulate total energy
        total_energy = [value * 100 for value in self.wind_generation_profile]
        
        #calculate outlook of profit
        income_outlook = sum(total_energy * (self.marginal_costs-self.agents["Wind"]["data"].loc[self.timestep-1, "Marginal_cost"]))
    
        #calculate interest rate 
        interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Wind"]).iloc[0])
        
        #amount
        amount = 100 * self.Capex.loc[self.timestep-1, "Wind Onshore"]
    
        #calculate interest
        interest = float(amount * interest_rate)  
    
        #caluate profit with income outlook and fix costs
        profit = income_outlook-(self.Fix_costs.loc["Wind onshore"]*100) - interest
        
        #form the cash_flow list
        cash_flows = [profit] * self.wind_lifetime

        #insert investment cost as first value
        cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Wind Onshore"])

        #calculate the rate of return
        irr = npf.irr(cash_flows)
        
        #if irr is nan (profit is negative), mirror irr
        if irr != irr:
            cash_flows = [-profit] * self.wind_lifetime
            cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Wind Onshore"])
            irr = npf.irr(cash_flows)
            irr = -irr - 2
        
        #save irr
        self.agents["Wind"]["data"].loc[self.timestep, "Irr prognose"] = irr    

        #check if crisis is happening
        if self.LIBOR_index.iloc[self.timestep,0] > 0.025:
            crisis_shift = 100*((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"]+self.Risk_markup["Production Wind"].iloc[0])-0.025)**2
        else:
            crisis_shift = 0

        #calcualte shift, increases when interste rate rises
        shift = self.wind_shift-crisis_shift

        #calculate power_increase with sigmoid function
        power_increase = self.wind_max * self.sigmoid(irr,self.wind_slope,shift)

        #round power increase to a block size
        power_increase = (power_increase // self.wind_block_size) * self.wind_block_size       
        
        #check if power increase is more than 0
        if power_increase > 0:
            
            #safe power increase in power increase dataframe
            self.agents["Wind"]['Investments'].loc[self.timestep, 'Block_size'] +=  power_increase
            
            #set lifetime
            self.agents["Wind"]['Investments'].loc[self.timestep, 'Lifetime'] = self.wind_lifetime
            
            #set status
            self.agents["Wind"]['Investments'].loc[self.timestep, 'Status'] = True
    
            #calculate interest rate 
            interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Wind"]).iloc[0])
    
            #amount
            amount = power_increase * self.Capex.loc[self.timestep, "Wind Onshore"]
    
            #payback
            payback = float(amount * interest_rate)
    
            #take on loan and write that into the loan list
            loans = {'Agent_name': ["Wind"], 'Active': [True], 'Runtime': [self.wind_lifetime], 'Amount': [amount], 'Interest_rate': [interest_rate], 'Payback': [payback]}
            self.loans_temp = pd.DataFrame(data = loans)
            
            #add to loans list
            self.loans = pd.concat([self.loans, self.loans_temp], ignore_index=True)
       
        #calculate the total installed power
        self.agents["Wind"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Wind"]["data"].loc[self.timestep-1, "Installed_power_initial"] + self.agents["Wind"]["Investments"].loc[self.agents["Wind"]["Investments"]['Status'] == True, 'Block_size'].sum()

    
    def wind(self):
        """
        Function of Wind.
        This producer resembels the combined wind power plants
        """
        print("\nWind called")
        
        #check if initial installed power is higher than 0
        if self.agents["Wind"]["data"].loc[self.timestep-1, 'Installed_power_initial'] - self.coal_deprication > 0:
            #initial installed power is reduced by a fixed amount every year
            self.agents["Wind"]['data'].loc[self.timestep, 'Installed_power_initial'] = self.agents["Wind"]["data"].loc[self.timestep-1, 'Installed_power_initial'] - self.coal_deprication
        
            #strategy 8 increase lifetime:
            #if self.time >= 2033 and self.time <= 2036:
        
                #elf.agents["Wind"]['data'].loc[self.timestep, 'Installed_power_initial'] = self.agents["Wind"]["data"].loc[self.timestep-1, 'Installed_power_initial']
            
            #else:
                
                #self.agents["Wind"]['data'].loc[self.timestep, 'Installed_power_initial'] = self.agents["Wind"]["data"].loc[self.timestep-1, 'Installed_power_initial'] - self.wind_deprication           
       
        else: 
            self.agents["Wind"]["data"].loc[self.timestep, 'Installed_power_initial'] = 0        
        
        #calculate the total installed power
        self.agents["Wind"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Wind"]["data"].loc[self.timestep, "Installed_power_initial"] + self.agents["Wind"]["Investments"].loc[self.agents["Wind"]["Investments"]['Status'] == True, 'Block_size'].sum()

        #update efficiency, increase by 0.1% every year
        #todo: function einbauen die dafür sorgt, dass der wirkungsgrad bis 41% steigt und danach so bleibt
        self.agents["Wind"]["data"].loc[self.timestep, "Efficiency"] = self.agents["Wind"]["data"].loc[self.timestep-1, "Efficiency"] + 0.001    
             
        #print("marginal costs")
        #print(self.agents["Wind"]["data"].loc[self.timestep, "Marginal_cost"])
    
        #fill generated_power_total with calculated PyPSA data
        self.agents["Wind"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.loc["Wind"]
        
        #set income with the income of the last timestep from PyPSA
        self.agents["Wind"]["data"].loc[self.timestep, "Income"] = self.wind_income_total
    
        #print("Income")
        #print(self.agents["Wind"]["data"].loc[self.timestep, "Income"])
    
        #set expenses from the fix costs
        self.agents["Wind"]["data"].loc[self.timestep, "Expenses"] = self.agents["Wind"]["data"].loc[self.timestep, "Expenses"]-self.Fix_costs.loc["Wind onshore"]*self.agents["Wind"]["data"].loc[self.timestep, "Installed_power"]

        #calculate profit
        self.agents["Wind"]["data"].loc[self.timestep, "Profit"] = self.agents["Wind"]["data"].loc[self.timestep, "Income"] + self.agents["Wind"]["data"].loc[self.timestep, "Payback"] + self.agents["Wind"]["data"].loc[self.timestep, "Expenses"]

        #update money
        self.agents["Wind"]["data"].loc[self.timestep, "Money"] = self.agents["Wind"]["data"].loc[self.timestep-1, "Money"] + self.agents["Wind"]["data"].loc[self.timestep, "Profit"] 

        #investment function
        
        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy = [round(value * self.agents["Wind"]["data"].loc[self.timestep, "Installed_power"], 4) for value in self.wind_generation_profile]
        theoratical_energy_total = sum(theoratical_energy)

        #calculate load and fill data
        self.agents["Wind"]["data"].loc[self.timestep, "Load"] = round(self.agents["Wind"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy_total,4)

        #check if load is bigger than 1
        if self.agents["Wind"]["data"].loc[self.timestep, "Load"] > 1:
            #if yes set load to 1
            self.agents["Wind"]["data"].loc[self.timestep, "Load"] = 1
            
        #investment function
        #calulate total energy
        total_energy = [value * 100 for value in self.wind_generation_profile]
        
        #calculate outlook of profit
        income_outlook = sum(total_energy * (self.marginal_costs-self.agents["Wind"]["data"].loc[self.timestep-1, "Marginal_cost"]))
    
        #calculate interest rate 
        interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Wind"]).iloc[0])
        
        #amount
        amount = 100 * self.Capex.loc[self.timestep-1, "Wind Onshore"]
    
        #calculate interest
        interest = float(amount * interest_rate)  
    
        #caluate profit with income outlook and fix costs
        profit = income_outlook-(self.Fix_costs.loc["Wind onshore"]*100) - interest
        
        #form the cash_flow list
        cash_flows = [profit] * self.wind_lifetime

        #insert investment cost as first value
        cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Wind Onshore"])

        #calculate the rate of return
        irr = npf.irr(cash_flows)
            
        #if irr is nan (profit is negative), mirror irr
        if irr != irr:
            cash_flows = [-profit] * self.wind_lifetime
            cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Wind Onshore"])
            irr = npf.irr(cash_flows)
            irr = -irr - 2

        #save irr
        self.agents["Wind"]["data"].loc[self.timestep, "Irr"] = irr    

        #check if crisis is happening
        if self.LIBOR_index.iloc[self.timestep,0] > 0.025:
            crisis_shift = 100*((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"]+self.Risk_markup["Production Wind"].iloc[0])-0.025)**2
        else:
            crisis_shift = 0

        #calcualte shift, increases when interste rate rises
        shift = self.wind_shift-crisis_shift

        #calculate power_increase with sigmoid function
        power_increase = self.wind_max * self.sigmoid(irr,self.wind_slope,shift)

        #round power increase to a block size
        power_increase = (power_increase // self.wind_block_size) * self.wind_block_size
        
        #check if power increase is more than 0
        if power_increase > 0:
            #safe power increase in power increase dataframe
            self.agents["Wind"]['Investments'].loc[self.timestep, 'Block_size'] += power_increase
           
            #set lifetime
            self.agents["Wind"]['Investments'].loc[self.timestep, 'Lifetime'] = self.wind_lifetime
            
            #set status
            self.agents["Wind"]['Investments'].loc[self.timestep, 'Status'] = True

            #calculate interest rate 
            interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Wind"]).iloc[0])
    
            #amount
            amount = power_increase * self.Capex.loc[self.timestep, "Wind Onshore"]
    
            #payback
            payback = float(amount * interest_rate)
    
            #take on loan and write that into the loan list
            loans = {'Agent_name': ["Wind"], 'Active': [True], 'Runtime': [self.wind_lifetime], 'Amount': [amount], 'Interest_rate': [interest_rate], 'Payback': [payback]}
            self.loans_temp = pd.DataFrame(data = loans)
            
            #add to loans list
            self.loans = pd.concat([self.loans, self.loans_temp], ignore_index=True)
       
       
        #calculate the total installed power
        self.agents["Wind"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Wind"]["data"].loc[self.timestep, "Installed_power_initial"] + self.agents["Wind"]["Investments"].loc[self.agents["Wind"]["Investments"]['Status'] == True, 'Block_size'].sum()

        #calculate total emissions with calculated PyPSA data
        self.agents["Wind"]["data"].loc[self.timestep, "Total_emissions"] = self.agents["Wind"]["data"].loc[self.timestep, "Generated_power_total"] * 0
    
    
    def hydro(self):
        """
        Function of Hydro.
        This producer resembels the combined water power plants
        """
        print("\nHydro called")
        
        #update marginal cost
        self.agents["Hydro"]["data"].loc[self.timestep, "Marginal_cost"] = self.agents["Hydro"]["data"].loc[self.timestep-1, "Marginal_cost"]
        
        #update installed_power
        self.agents["Hydro"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Hydro"]["data"].loc[self.timestep-1, "Installed_power"]

        #fill generated_power_total with calculated PyPSA data
        self.agents["Hydro"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.loc["Hydro"]

        #set income with the income of the last timestep from PyPSA
        self.agents["Hydro"]["data"].loc[self.timestep, "Income"] = self.hydro_income_total

        #set expenses from the fix costs
        self.agents["Hydro"]["data"].loc[self.timestep, "Expenses"] = self.agents["Hydro"]["data"].loc[self.timestep, "Expenses"]-self.Fix_costs.loc["Hydropower"]*self.agents["Hydro"]["data"].loc[self.timestep, "Installed_power"]

        #calculate profit
        self.agents["Hydro"]["data"].loc[self.timestep, "Profit"] = self.agents["Hydro"]["data"].loc[self.timestep, "Income"] + self.agents["Hydro"]["data"].loc[self.timestep, "Payback"] + self.agents["Hydro"]["data"].loc[self.timestep, "Expenses"]

        #update money
        self.agents["Hydro"]["data"].loc[self.timestep, "Money"] = self.agents["Hydro"]["data"].loc[self.timestep-1, "Money"] + self.agents["Hydro"]["data"].loc[self.timestep, "Profit"] 

        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy = self.agents["Hydro"]["data"].loc[self.timestep, "Installed_power"] * 365 * 24
        
        #calculate load and fill data
        self.agents["Hydro"]["data"].loc[self.timestep, "Load"] = round(self.agents["Hydro"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy,4)

        #no inverstment function because capacity in germany has been reached

        #update efficiency
        self.agents["Hydro"]["data"].loc[self.timestep, "Efficiency"] = self.agents["Hydro"]["data"].loc[self.timestep-1, "Efficiency"]

        #calculate total emissions with calculated PyPSA data, hydro has no emissions in operation
        self.agents["Hydro"]["data"].loc[self.timestep, "Total_emissions"] = 0
    
        
    def biomass(self):
        """
        Function of Biomass.
        This producer resembels the combined biomass power plants
        """
        print("\nBiomass called")
        
        #update marginal cost
        self.agents["Biomass"]["data"].loc[self.timestep, "Marginal_cost"] = self.Operational_costs["Biomass"] + 5 #difficult to calculate ressource and co2 costs -> 5 euro estimate for both -> 230 kg CO2 per MWh
        
        #update installed_power
        self.agents["Biomass"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Biomass"]["data"].loc[self.timestep-1, "Installed_power"]

        #fill generated_power_total with calculated PyPSA data
        self.agents["Biomass"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.loc["Biomass"]

        #set income with the income of the last timestep from PyPSA
        self.agents["Biomass"]["data"].loc[self.timestep, "Income"] = self.biomass_income_total

        #set expenses from the fix costs
        self.agents["Biomass"]["data"].loc[self.timestep, "Expenses"] = self.agents["Biomass"]["data"].loc[self.timestep, "Expenses"]-self.Fix_costs.loc["Biomass"]*self.agents["Biomass"]["data"].loc[self.timestep, "Installed_power"]

        #calculate profit
        self.agents["Biomass"]["data"].loc[self.timestep, "Profit"] = self.agents["Biomass"]["data"].loc[self.timestep, "Income"] + self.agents["Biomass"]["data"].loc[self.timestep, "Payback"] + self.agents["Biomass"]["data"].loc[self.timestep, "Expenses"]

        #update money
        self.agents["Biomass"]["data"].loc[self.timestep, "Money"] = self.agents["Biomass"]["data"].loc[self.timestep-1, "Money"] + self.agents["Biomass"]["data"].loc[self.timestep, "Profit"] 

        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy = self.agents["Biomass"]["data"].loc[self.timestep, "Installed_power"] * 365 * 24
        
        #calculate load and fill data
        self.agents["Biomass"]["data"].loc[self.timestep, "Load"] = round(self.agents["Biomass"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy,4)

        #update efficiency
        self.agents["Biomass"]["data"].loc[self.timestep, "Efficiency"] = self.agents["Biomass"]["data"].loc[self.timestep-1, "Efficiency"]

        #calculate total emissions with calculated PyPSA data, esstimated that emissions is always 230 kg CO2 per MWh
        self.agents["Biomass"]["data"].loc[self.timestep, "Total_emissions"] = self.agents["Biomass"]["data"].loc[self.timestep, "Generated_power_total"] * 230
            
        
    def oil(self):
        """
        Function of Oil.
        This producer resembels the combined oil power plants
        """
        print("\nOil called")
        
        #update marginal cost
        self.agents["Oil"]["data"].loc[self.timestep, "Marginal_cost"] = self.oil_marginal_cost.iloc[self.timestep,0]
        
        #update installed_power
        self.agents["Oil"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Oil"]["data"].loc[self.timestep-1, "Installed_power"]

        #fill generated_power_total with calculated PyPSA data
        self.agents["Oil"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.loc["Oil"]

        #calculate Income with calculated PyPSA data
        self.agents["Oil"]["data"].loc[self.timestep, "Income"] = self.agents["Oil"]["data"].loc[self.timestep, "Generated_power_total"]*self.Electricity_cost_pure

        #update money
        self.agents["Oil"]["data"].loc[self.timestep, "Money"] = self.agents["Oil"]["data"].loc[self.timestep-1, "Money"] + self.agents["Oil"]["data"].loc[self.timestep, "Income"] 

        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy = self.agents["Oil"]["data"].loc[self.timestep, "Installed_power"] * 365 * 24
        
        #calculate load and fill data
        self.agents["Oil"]["data"].loc[self.timestep, "Load"] = round(self.agents["Oil"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy,4)

        #update efficiency
        self.agents["Oil"]["data"].loc[self.timestep, "Efficiency"] = self.agents["Oil"]["data"].loc[self.timestep-1, "Efficiency"]

        #calculate total emissions with calculated PyPSA data, esstimated that emissions is always 890 kg CO2 per MWh
        self.agents["Oil"]["data"].loc[self.timestep, "Total_emissions"] = self.agents["Oil"]["data"].loc[self.timestep, "Generated_power_total"] * 890
    
        
    def storage_prognosis(self):
        """
        This function represents the prognosis of the storage agent.
        Water pump, battery and hydrogen
        """
        print("\nStorage prognosis called")
         
        #ToDo: Fix kosten und payback einfügen
    
        #Battery storage
        print("\nBattery storage")

        #update efficiency
        self.agents["Storage"]["data"].loc[self.timestep, "Battery_eff"] = self.agents["Storage"]["data"].loc[self.timestep-1, "Battery_eff"]
        
        #update income
        
        #update expenses
        #opex = 0.75% of capex
        
        #update payback
        
        #update profit
        
        #update money

        #calculate profit outlook
        profit = sum((-self.battery_profile*100)*self.marginal_costs)
        
        #check if profit is nan, if yes set to 0
        if profit != profit:
            profit = 0

        #form the cash_flow list
        cash_flows = [profit] * self.battery_lifetime
        
        #insert investment cost as first value
        cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Battery"])
        
        #calculate the rate of return
        irr = npf.irr(cash_flows)
        
        #if irr is nan (profit is negative), mirror irr
        if irr != irr:
            cash_flows = [-profit] * self.battery_lifetime
            cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Battery"])
            irr = npf.irr(cash_flows)
            irr = -irr - 2
        
        #calculate capacity_increase with sigmoid function
        capacity_increase = self.battery_max*self.sigmoid(irr, self.battery_slope, self.battery_shift)

        #round power increase to a block size
        capacity_increase = (capacity_increase // self.battery_block_size) * self.battery_block_size  

        #check if power increase is more than 0
        if capacity_increase > 0:
            #safe power increase in power increase dataframe
            self.Storage_investments.loc[self.timestep, 'Block_size_battery'] +=  capacity_increase
    
            #set lifetime
            self.Storage_investments.loc[self.timestep, 'Lifetime_battery'] = self.battery_lifetime
    
            #set status
            self.Storage_investments.loc[self.timestep, 'Status_battery'] = True
   
        #update capacity
        self.agents["Storage"]["data"].loc[self.timestep, "Battery_cap"] = self.Storage_investments.loc[self.Storage_investments['Status_battery'] == True, 'Block_size_battery'].sum()
        
        #update power in   
        self.agents["Storage"]["data"].loc[self.timestep, "Battery_in"] = self.agents["Storage"]["data"].loc[self.timestep, "Battery_cap"]/2

        #update power out     
        self.agents["Storage"]["data"].loc[self.timestep, "Battery_out"] = self.agents["Storage"]["data"].loc[self.timestep, "Battery_cap"]/2


        #Hydrogen storage
        print("\nHydrogen storage")

        #update efficiency
        self.agents["Storage"]["data"].loc[self.timestep, "Hydrogen_eff"] = self.agents["Storage"]["data"].loc[self.timestep-1, "Hydrogen_eff"]
    
        #update income
        
        #update expenses
        #opex = 2.75% of capex
        
        #update payback
        
        #update profit
        
        #update money

        #calculate profit outlook
        profit = sum((-self.hydrogen_profile*100)*self.marginal_costs)
        
        #check if profit is nan, if yes set to 0
        if profit != profit:
            profit = 0

        #form the cash_flow list
        cash_flows = [profit] * self.hydrogen_lifetime
        
        #insert investment cost as first value
        cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Hydrogen"])

        #calculate the rate of return
        irr = npf.irr(cash_flows)

        #if irr is nan (profit is negative), mirror irr
        if irr != irr:
            cash_flows = [-profit] * self.hydrogen_lifetime
            cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Hydrogen"])
            irr = npf.irr(cash_flows)
            irr = -irr - 2

        #calculate capacity_increase with sigmoid function
        capacity_increase = self.hydrogen_max*self.sigmoid(irr, self.hydrogen_slope, self.hydrogen_shift)

        #round power increase to a block size
        capacity_increase = (capacity_increase // self.hydrogen_block_size) * self.hydrogen_block_size  
        
        #check if power increase is more than 0
        if capacity_increase > 0:
            #safe power increase in power increase dataframe
            self.Storage_investments.loc[self.timestep, 'Block_size_hydrogen'] +=  capacity_increase
    
            #set lifetime
            self.Storage_investments.loc[self.timestep, 'Lifetime_hydrogen'] = self.hydrogen_lifetime
    
            #set status
            self.Storage_investments.loc[self.timestep, 'Status_hydrogen'] = True

        #update capacity
        self.agents["Storage"]["data"].loc[self.timestep, "Hydrogen_cap"] = self.Storage_investments.loc[self.Storage_investments['Status_hydrogen'] == True, 'Block_size_hydrogen'].sum()
        
        #update power in   
        self.agents["Storage"]["data"].loc[self.timestep, "Hydrogen_in"] = self.agents["Storage"]["data"].loc[self.timestep, "Hydrogen_cap"]/100

        #update power out     
        self.agents["Storage"]["data"].loc[self.timestep, "Hydrogen_out"] = self.agents["Storage"]["data"].loc[self.timestep, "Hydrogen_cap"]/100

        #water pump storage
        #update capacity  
        self.agents["Storage"]["data"].loc[self.timestep, "Pump_cap"] = self.agents["Storage"]["data"].loc[self.timestep-1, "Pump_cap"]

        #update power in
        self.agents["Storage"]["data"].loc[self.timestep, "Pump_in"] = self.agents["Storage"]["data"].loc[self.timestep-1, "Pump_in"]

        #update power out        
        self.agents["Storage"]["data"].loc[self.timestep, "Pump_out"] = self.agents["Storage"]["data"].loc[self.timestep-1, "Pump_out"]

        #update efficiency
        self.agents["Storage"]["data"].loc[self.timestep, "Pump_eff"] = self.agents["Storage"]["data"].loc[self.timestep-1, "Pump_eff"]

        
    def storage(self):
        """
        This agent represents the storage of the electricity grid.
        Water pump, battery and hydrogen
        """
        print("\nStorage called")
         
        #ToDo: Fix kosten und payback einfügen
    
        #Battery storage
        print("\nBattery storage")
        #update capacity   
        self.agents["Storage"]["data"].loc[self.timestep, "Battery_cap"] = self.agents["Storage"]["data"].loc[self.timestep-1, "Battery_cap"]

        #update power in   
        self.agents["Storage"]["data"].loc[self.timestep, "Battery_in"] = self.agents["Storage"]["data"].loc[self.timestep-1, "Battery_in"]

        #update power out     
        self.agents["Storage"]["data"].loc[self.timestep, "Battery_out"] = self.agents["Storage"]["data"].loc[self.timestep-1, "Battery_out"]

        #update efficiency
        self.agents["Storage"]["data"].loc[self.timestep, "Battery_eff"] = self.agents["Storage"]["data"].loc[self.timestep-1, "Battery_eff"]
        
        #update income
        
        #update expenses
        #opex = 0.75% of capex
        
        #update payback
        
        #update profit
        
        #update money

        #calculate profit outlook
        profit = sum((-self.battery_profile*100)*self.marginal_costs)
        
        #check if profit is nan, if yes set to 0
        if profit != profit:
            profit = 0

        #form the cash_flow list
        cash_flows = [profit] * self.battery_lifetime
        
        #insert investment cost as first value
        cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Battery"])
        
        #calculate the rate of return
        irr = npf.irr(cash_flows)
        
        #if irr is nan (profit is negative), mirror irr
        if irr != irr:
            cash_flows = [-profit] * self.battery_lifetime
            cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Battery"])
            irr = npf.irr(cash_flows)
            irr = -irr - 2
        
        #calculate capacity_increase with sigmoid function
        capacity_increase = self.battery_max*self.sigmoid(irr, self.battery_slope, self.battery_shift)

        #round power increase to a block size
        capacity_increase = (capacity_increase // self.battery_block_size) * self.battery_block_size  

        #check if power increase is more than 0
        if capacity_increase > 0:
            #safe power increase in power increase dataframe
            self.Storage_investments.loc[self.timestep, 'Block_size_battery'] +=  capacity_increase
    
            #set lifetime
            self.Storage_investments.loc[self.timestep, 'Lifetime_battery'] = self.battery_lifetime
    
            #set status
            self.Storage_investments.loc[self.timestep, 'Status_battery'] = True
   
        #update capacity
        self.agents["Storage"]["data"].loc[self.timestep, "Battery_cap"] = self.Storage_investments.loc[self.Storage_investments['Status_battery'] == True, 'Block_size_battery'].sum()
        
        #update power in   
        self.agents["Storage"]["data"].loc[self.timestep, "Battery_in"] = self.agents["Storage"]["data"].loc[self.timestep, "Battery_cap"]/2

        #update power out     
        self.agents["Storage"]["data"].loc[self.timestep, "Battery_out"] = self.agents["Storage"]["data"].loc[self.timestep, "Battery_cap"]/2

        # Loop through the investments dataframe
        for i in range(len(self.Storage_investments)):
    
            # Check if power plant is still active
            if self.Storage_investments.loc[i, 'Status_battery'] == True:
    
                # Subtract one year from lifetime
                self.Storage_investments.loc[i, 'Lifetime_battery'] -= 1
    
                # If lifetime reaches 0, switch to false
                if self.Storage_investments.loc[i, 'Lifetime_battery'] == 0:
                    self.Storage_investments.loc[i, 'Status_battery'] = False


        #Hydrogen storage
        print("\nHydrogen storage")
        #update capacity   
        self.agents["Storage"]["data"].loc[self.timestep, "Hydrogen_cap"] = self.agents["Storage"]["data"].loc[self.timestep-1, "Hydrogen_cap"]

        #update power in   
        self.agents["Storage"]["data"].loc[self.timestep, "Hydrogen_in"] = self.agents["Storage"]["data"].loc[self.timestep-1, "Hydrogen_in"]

        #update power out     
        self.agents["Storage"]["data"].loc[self.timestep, "Hydrogen_out"] = self.agents["Storage"]["data"].loc[self.timestep-1, "Hydrogen_out"]

        #update efficiency
        self.agents["Storage"]["data"].loc[self.timestep, "Hydrogen_eff"] = self.agents["Storage"]["data"].loc[self.timestep-1, "Hydrogen_eff"]
    
        #update income
        
        #update expenses
        #opex = 2.75% of capex
        
        #update payback
        
        #update profit
        
        #update money

        #calculate profit outlook
        profit = sum((-self.hydrogen_profile*100)*self.marginal_costs)
        
        #check if profit is nan, if yes set to 0
        if profit != profit:
            profit = 0

        #form the cash_flow list
        cash_flows = [profit] * self.hydrogen_lifetime
        
        #insert investment cost as first value
        cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Hydrogen"])

        #calculate the rate of return
        irr = npf.irr(cash_flows)

        #if irr is nan (profit is negative), mirror irr
        if irr != irr:
            cash_flows = [-profit] * self.hydrogen_lifetime
            cash_flows.insert(0, 0-100*self.Capex.loc[self.timestep-1, "Hydrogen"])
            irr = npf.irr(cash_flows)
            irr = -irr - 2

        #calculate capacity_increase with sigmoid function
        capacity_increase = self.hydrogen_max*self.sigmoid(irr, self.hydrogen_slope, self.hydrogen_shift)

        #round power increase to a block size
        capacity_increase = (capacity_increase // self.hydrogen_block_size) * self.hydrogen_block_size  
        
        #check if power increase is more than 0
        if capacity_increase > 0:
            #safe power increase in power increase dataframe
            self.Storage_investments.loc[self.timestep, 'Block_size_hydrogen'] +=  capacity_increase
    
            #set lifetime
            self.Storage_investments.loc[self.timestep, 'Lifetime_hydrogen'] = self.hydrogen_lifetime
    
            #set status
            self.Storage_investments.loc[self.timestep, 'Status_hydrogen'] = True

        #update capacity
        self.agents["Storage"]["data"].loc[self.timestep, "Hydrogen_cap"] = self.Storage_investments.loc[self.Storage_investments['Status_hydrogen'] == True, 'Block_size_hydrogen'].sum()
        
        #update power in   
        self.agents["Storage"]["data"].loc[self.timestep, "Hydrogen_in"] = self.agents["Storage"]["data"].loc[self.timestep, "Hydrogen_cap"]/100

        #update power out     
        self.agents["Storage"]["data"].loc[self.timestep, "Hydrogen_out"] = self.agents["Storage"]["data"].loc[self.timestep, "Hydrogen_cap"]/100


        # Loop through the investments dataframe
        for i in range(len(self.Storage_investments)):
    
            # Check if power plant is still active
            if self.Storage_investments.loc[i, 'Status_hydrogen'] == True:
    
                # Subtract one year from lifetime
                self.Storage_investments.loc[i, 'Lifetime_hydrogen'] -= 1
    
                # If lifetime reaches 0, switch to false
                if self.Storage_investments.loc[i, 'Lifetime_hydrogen'] == 0:
                    self.Storage_investments.loc[i, 'Status_hydrogen'] = False



        #water pump storage
        #update capacity  
        self.agents["Storage"]["data"].loc[self.timestep, "Pump_cap"] = self.agents["Storage"]["data"].loc[self.timestep-1, "Pump_cap"]

        #update power in
        self.agents["Storage"]["data"].loc[self.timestep, "Pump_in"] = self.agents["Storage"]["data"].loc[self.timestep-1, "Pump_in"]

        #update power out        
        self.agents["Storage"]["data"].loc[self.timestep, "Pump_out"] = self.agents["Storage"]["data"].loc[self.timestep-1, "Pump_out"]

        #update efficiency
        self.agents["Storage"]["data"].loc[self.timestep, "Pump_eff"] = self.agents["Storage"]["data"].loc[self.timestep-1, "Pump_eff"]


    def consumer_1(self):
        """
        Function of consumer 1.
        This consumer resembels the combined industrial consumers
        """
        print("\nConsumer 1 called")
        
        #ToDo: make demand increase depended of average_price, but "real" demand isnt just dependend of cost, how could i implement this?
        #maybe the demand has the same base level (because the demand doesnt realy change much in the last 20 years) only the cost de or increase the demand.
        #implement a electricity price, at which the consumers change there behaviour and invest in energy efficiency and own generatition
        
        #update power demand
        self.agents["Consumer 1"]["data"].loc[self.timestep, "Power_demand"] = self.Electricity_demand.loc[self.timestep,"Industry demand"]
        
        #update income, if not touched, next income same as this years
        self.agents["Consumer 1"]["data"].loc[self.timestep, "Income"] = self.agents["Consumer 1"]["data"].loc[self.timestep-1, "Income"]

        #substract electricity costs from income, ToDo: wird negativ
        self.agents["Consumer 1"]["data"].loc[self.timestep, "Income"] = self.agents["Consumer 1"]["data"].loc[self.timestep, "Income"] - self.agents["Consumer 1"]["data"].loc[self.timestep, "Power_demand"]*self.agents["Government"]["data"].loc[self.timestep-1, "Electricity_cost"]

        #add income to money, other expensise decrease income by 70%, Assumption
        self.agents["Consumer 1"]["data"].loc[self.timestep, "Money"] = self.agents["Consumer 1"]["data"].loc[self.timestep-1, "Money"]+self.agents["Consumer 1"]["data"].loc[self.timestep, "Income"]*0.3


    def consumer_2(self):
        """
        Function of consumer 2.
        This consumer resembels the combined commerce consumers
        """
        print("\nConsumer 2 called")
        
        #update power demand
        self.agents["Consumer 2"]["data"].loc[self.timestep, "Power_demand"] = self.Electricity_demand.loc[self.timestep,"Commerce demand"]
        
        #update income
        self.agents["Consumer 2"]["data"].loc[self.timestep, "Income"] = self.agents["Consumer 2"]["data"].loc[self.timestep-1, "Income"]

        #substract electricity costs from income
        self.agents["Consumer 2"]["data"].loc[self.timestep, "Income"] = self.agents["Consumer 2"]["data"].loc[self.timestep, "Income"] - self.agents["Consumer 2"]["data"].loc[self.timestep, "Power_demand"]*self.agents["Government"]["data"].loc[self.timestep-1, "Electricity_cost"]

        #get income, other expensise decrease income by 70%, Assumption
        self.agents["Consumer 2"]["data"].loc[self.timestep, "Money"] = self.agents["Consumer 2"]["data"].loc[self.timestep-1, "Money"]+self.agents["Consumer 2"]["data"].loc[self.timestep, "Income"]*0.3

    
    def consumer_3(self):
        """
        Function of consumer 3.
        This consumer resembels the combined private consumers
        """
        print("\nConsumer 3 called")

        #ToDo: der demand geht seit 2009 runter, aber durch e autos etc, wird der strombedarf in der zukunft steigen...
        #ToDo: Investment funktion einfügen, wenn der strompreis so hoch ist, dass sich ein investment in verringerter demand oder eigene erzeugung lohnt

        #update power demand
        self.agents["Consumer 3"]["data"].loc[self.timestep, "Power_demand"] = self.Electricity_demand.loc[self.timestep,"Private demand"]
        
        #update income
        #Assumption average income of households increases by 3% every year
        self.agents["Consumer 3"]["data"].loc[self.timestep, "Income"] = self.agents["Consumer 3"]["data"].loc[self.timestep-1, "Income"]*1.03

        #substract electricity costs from income
        self.agents["Consumer 3"]["data"].loc[self.timestep, "Income"] = self.agents["Consumer 3"]["data"].loc[self.timestep, "Income"] - self.agents["Consumer 3"]["data"].loc[self.timestep, "Power_demand"]*self.agents["Government"]["data"].loc[self.timestep-1, "Electricity_cost"]

        #get income, other expensise decrease income by 80%, Assumption
        self.agents["Consumer 3"]["data"].loc[self.timestep, "Money"] = self.agents["Consumer 3"]["data"].loc[self.timestep-1, "Money"]+self.agents["Consumer 3"]["data"].loc[self.timestep, "Income"]*0.2
        
        #ToDo: implement own generation with generation profil
        
        
if __name__ == '__main__':

    #init class
    start_year=2009
    end_year=2021
    
    plot_data = True
    
    ess = ESS(start_year=start_year,end_year=end_year)
    
    #init agents
    ess.init_agents()
    
    # Initialize profiles
    pv_generation_profile, wind_generation_profile, load_factor_profile = ess.init_profiles()
    
    #print all the agents in the model
    #print("Agent List:", ess.get_agent_list())
    
    #run simulation
    ess.run_sim(pv_generation_profile, wind_generation_profile, load_factor_profile)
    
    #loan list
    loan_list = ess.loans
    
    Fallback_generator_supply = ess.Fallback_generator_supply

    coal_profile=ess.coal_profile
    gas_ct_profile=ess.gas_ct_profile
    gas_cc_profile=ess.gas_cc_profile
    
    #investments dataframes
    Investment_coal = ess.agents["Coal"]['Investments']
    Investment_gas_ct = ess.agents["Gas_CT"]['Investments']
    Investment_gas_cc = ess.agents["Gas_CC"]['Investments']
    Investment_solar = ess.agents["Solar"]['Investments']
    Investment_wind = ess.agents["Wind"]['Investments']
    
    Storage_investments = ess.Storage_investments

    hydrogen_profile=ess.hydrogen_profile
    battery_profile=ess.battery_profile
    
    battery_load = ess.battery_load
    battery_unload = ess.battery_unload
    
    hydrogen_load = ess.hydrogen_load
    hydrogen_unload = ess.hydrogen_unload
    
    water_profile=ess.water_profile

    marginal_costs = ess.marginal_costs
    marginal_cost = ess.marginal_cost
    
    agents_list = ess.agents_list
    
    generators_producer_1=ess.generators_producer_p
    
    generators_producer = ess.generators_producer
    
    links = ess.links
    stores = ess.stores
    
    #plot some variables, if bool is True
    if plot_data == True:
    
        
        #Generated Power
        ess.agents["Coal"]["data"]["Generated_power_total"][0:end_year-start_year].plot(title="Generated Power")
        ess.agents["Gas_CT"]["data"]["Generated_power_total"][0:end_year-start_year].plot()
        ess.agents["Gas_CC"]["data"]["Generated_power_total"][0:end_year-start_year].plot()
        ess.agents["Nuclear"]["data"]["Generated_power_total"][0:end_year-start_year].plot()
        ess.agents["Solar"]["data"]["Generated_power_total"][0:end_year-start_year].plot()
        ess.agents["Wind"]["data"]["Generated_power_total"][0:end_year-start_year].plot()
        ess.agents["Hydro"]["data"]["Generated_power_total"][0:end_year-start_year].plot()
        ess.agents["Biomass"]["data"]["Generated_power_total"][0:end_year-start_year].plot()
        ess.agents["Oil"]["data"]["Generated_power_total"][0:end_year-start_year].plot()
        plt.show()
        
        #plot empirical installed data, for calibration
        installed_power_empirical = pd.read_excel("Installed_power.xlsx")
        installed_power_empirical.plot()
        plt.show()
        
        #for every agent:
        ess.agents["Coal"]["data"]["Installed_power"][0:end_year-start_year].plot(title="Installed_power_coal")
        installed_power_empirical["Coal"].plot()
        plt.show()
        
        gas_total = ess.agents["Gas_CT"]["data"]["Installed_power"] + ess.agents["Gas_CC"]["data"]["Installed_power"]
        gas_total.plot(title="Installed_power_gas")
        installed_power_empirical["Gas"].plot()
        plt.show()
        
        
        ess.agents["Solar"]["data"]["Installed_power"][0:end_year-start_year].plot(title="Installed_power_Solar")
        installed_power_empirical["PV"].plot()
        plt.show()
        
        ess.agents["Wind"]["data"]["Installed_power"][0:end_year-start_year].plot(title="Installed_power_wind")
        installed_power_empirical["Wind"].plot()
        plt.show()
        
        #installed power
        ess.agents["Coal"]["data"]["Installed_power"][0:end_year-start_year].plot(title="Installed Power")
        ess.agents["Gas_CT"]["data"]["Installed_power"][0:end_year-start_year].plot()
        ess.agents["Gas_CC"]["data"]["Installed_power"][0:end_year-start_year].plot()
        ess.agents["Nuclear"]["data"]["Installed_power"][0:end_year-start_year].plot()
        ess.agents["Solar"]["data"]["Installed_power"][0:end_year-start_year].plot()
        ess.agents["Wind"]["data"]["Installed_power"][0:end_year-start_year].plot()
        ess.agents["Hydro"]["data"]["Installed_power"][0:end_year-start_year].plot()
        ess.agents["Biomass"]["data"]["Installed_power"][0:end_year-start_year].plot()
        ess.agents["Oil"]["data"]["Installed_power"][0:end_year-start_year].plot()
        plt.show()   
        
        #investments
        ess.agents["Coal"]['Investments']["Block_size"][0:end_year-start_year].plot(title="Investments coal")
        ess.agents["Gas_CT"]['Investments']["Block_size"][0:end_year-start_year].plot()
        ess.agents["Gas_CC"]['Investments']["Block_size"][0:end_year-start_year].plot()
        ess.agents["Solar"]['Investments']["Block_size"][0:end_year-start_year].plot()
        ess.agents["Wind"]['Investments']["Block_size"][0:end_year-start_year].plot()
        plt.show()   
        
        #Load
        ess.agents["Coal"]["data"]["Load"][1:end_year-start_year].plot(title="Load")
        ess.agents["Gas_CT"]["data"]["Load"][1:end_year-start_year].plot()
        ess.agents["Gas_CC"]["data"]["Load"][1:end_year-start_year].plot()
        ess.agents["Nuclear"]["data"]["Load"][1:end_year-start_year].plot()
        ess.agents["Solar"]["data"]["Load"][1:end_year-start_year].plot()
        ess.agents["Wind"]["data"]["Load"][1:end_year-start_year].plot()
        ess.agents["Hydro"]["data"]["Load"][1:end_year-start_year].plot()
        ess.agents["Biomass"]["data"]["Load"][1:end_year-start_year].plot()
        ess.agents["Oil"]["data"]["Load"][1:end_year-start_year].plot()
        plt.show()
        
        #co2 Costs
        #ess.Co2_certificate_costs.plot(title="CO2 Costs")
        #plt.show()
        
        #ressource costs
        #ess.gas_costs.plot(legend = True)
        #ess.coal_costs.plot()
        #ess.uranium_costs.plot()
        #plt.show()
    
        
        #storage
        ess.agents["Storage"]["data"]["Battery_cap"][0:end_year-start_year].plot(title="Storage capacity battery")
        plt.show()
        ess.agents["Storage"]["data"]["Hydrogen_cap"][0:end_year-start_year].plot(title="Storage capacity hydrogen")
        plt.show()
        
    
        
        #Marginal costs
        ess.agents["Coal"]["data"]["Marginal_cost"][0:end_year-start_year].plot(title="Marginal costs")
        ess.agents["Gas_CT"]["data"]["Marginal_cost"][0:end_year-start_year].plot()
        ess.agents["Gas_CC"]["data"]["Marginal_cost"][0:end_year-start_year].plot()
        ess.agents["Nuclear"]["data"]["Marginal_cost"][0:end_year-start_year].plot()
        ess.agents["Solar"]["data"]["Marginal_cost"][0:end_year-start_year].plot()
        ess.agents["Wind"]["data"]["Marginal_cost"][0:end_year-start_year].plot()
        ess.agents["Hydro"]["data"]["Marginal_cost"][0:end_year-start_year].plot()
        ess.agents["Biomass"]["data"]["Marginal_cost"][0:end_year-start_year].plot()
        ess.agents["Oil"]["data"]["Marginal_cost"][0:end_year-start_year].plot()
        plt.show()
        
        #Total_emissions
        ess.agents["Coal"]["data"]["Total_emissions"][0:end_year-start_year].plot(title="Emissions")
        ess.agents["Gas_CT"]["data"]["Total_emissions"][0:end_year-start_year].plot()
        ess.agents["Gas_CC"]["data"]["Total_emissions"][0:end_year-start_year].plot()
        ess.agents["Nuclear"]["data"]["Total_emissions"][0:end_year-start_year].plot()
        ess.agents["Solar"]["data"]["Total_emissions"][0:end_year-start_year].plot()
        ess.agents["Wind"]["data"]["Total_emissions"][0:end_year-start_year].plot()
        ess.agents["Hydro"]["data"]["Total_emissions"][0:end_year-start_year].plot()
        ess.agents["Biomass"]["data"]["Total_emissions"][0:end_year-start_year].plot()
        ess.agents["Oil"]["data"]["Total_emissions"][0:end_year-start_year].plot()
        ess.agents["Government"]["data"]["Total_emissions"][0:end_year-start_year].plot()
        plt.show()
        
        #Income
        ess.agents["Coal"]["data"]["Income"][1:end_year-start_year].plot(title="Income")
        ess.agents["Gas_CT"]["data"]["Income"][1:end_year-start_year].plot()
        ess.agents["Gas_CC"]["data"]["Income"][1:end_year-start_year].plot()
        ess.agents["Nuclear"]["data"]["Income"][1:end_year-start_year].plot()
        ess.agents["Solar"]["data"]["Income"][1:end_year-start_year].plot()
        ess.agents["Wind"]["data"]["Income"][1:end_year-start_year].plot()
        ess.agents["Hydro"]["data"]["Income"][1:end_year-start_year].plot()
        ess.agents["Biomass"]["data"]["Income"][1:end_year-start_year].plot()
        ess.agents["Oil"]["data"]["Income"][1:end_year-start_year].plot()
        plt.show()
        
        #Payback
        ess.agents["Coal"]["data"]["Payback"][1:end_year-start_year].plot(title="Payback")
        ess.agents["Gas_CT"]["data"]["Payback"][1:end_year-start_year].plot()
        ess.agents["Gas_CC"]["data"]["Payback"][1:end_year-start_year].plot()
        ess.agents["Nuclear"]["data"]["Payback"][1:end_year-start_year].plot()
        ess.agents["Solar"]["data"]["Payback"][1:end_year-start_year].plot()
        ess.agents["Wind"]["data"]["Payback"][1:end_year-start_year].plot()
        ess.agents["Hydro"]["data"]["Payback"][1:end_year-start_year].plot()
        ess.agents["Biomass"]["data"]["Payback"][1:end_year-start_year].plot()
        ess.agents["Oil"]["data"]["Payback"][1:end_year-start_year].plot()
        plt.show()
        
        #Expenses
        ess.agents["Coal"]["data"]["Expenses"][1:end_year-start_year].plot(title="Expenses")
        ess.agents["Gas_CT"]["data"]["Expenses"][1:end_year-start_year].plot()
        ess.agents["Gas_CC"]["data"]["Expenses"][1:end_year-start_year].plot()
        ess.agents["Nuclear"]["data"]["Expenses"][1:end_year-start_year].plot()
        ess.agents["Solar"]["data"]["Expenses"][1:end_year-start_year].plot()
        ess.agents["Wind"]["data"]["Expenses"][1:end_year-start_year].plot()
        ess.agents["Hydro"]["data"]["Expenses"][1:end_year-start_year].plot()
        ess.agents["Biomass"]["data"]["Expenses"][1:end_year-start_year].plot()
        ess.agents["Oil"]["data"]["Expenses"][1:end_year-start_year].plot()
        plt.show()
        
        #Profit
        ess.agents["Coal"]["data"]["Profit"][1:end_year-start_year].plot(title="Profit")
        ess.agents["Gas_CT"]["data"]["Profit"][1:end_year-start_year].plot()
        ess.agents["Gas_CC"]["data"]["Profit"][1:end_year-start_year].plot()
        ess.agents["Nuclear"]["data"]["Profit"][1:end_year-start_year].plot()
        ess.agents["Solar"]["data"]["Profit"][1:end_year-start_year].plot()
        ess.agents["Wind"]["data"]["Profit"][1:end_year-start_year].plot()
        ess.agents["Hydro"]["data"]["Profit"][1:end_year-start_year].plot()
        ess.agents["Biomass"]["data"]["Profit"][1:end_year-start_year].plot()
        ess.agents["Oil"]["data"]["Profit"][1:end_year-start_year].plot()
        plt.show()
        
        #Money
        ess.agents["Coal"]["data"]["Money"][1:end_year-start_year].plot(title="Money")
        ess.agents["Gas_CT"]["data"]["Money"][1:end_year-start_year].plot()
        ess.agents["Gas_CC"]["data"]["Money"][1:end_year-start_year].plot()
        ess.agents["Nuclear"]["data"]["Money"][1:end_year-start_year].plot()
        ess.agents["Solar"]["data"]["Money"][1:end_year-start_year].plot()
        ess.agents["Wind"]["data"]["Money"][1:end_year-start_year].plot()
        ess.agents["Hydro"]["data"]["Money"][1:end_year-start_year].plot()
        ess.agents["Biomass"]["data"]["Money"][1:end_year-start_year].plot()
        ess.agents["Oil"]["data"]["Money"][1:end_year-start_year].plot()
        plt.show()
        
        #Power demand
        ess.agents["Consumer 1"]["data"]["Power_demand"][0:end_year-start_year].plot(title="Power demand")
        ess.agents["Consumer 2"]["data"]["Power_demand"][0:end_year-start_year].plot()
        ess.agents["Consumer 3"]["data"]["Power_demand"][0:end_year-start_year].plot()
        ess.agents["Government"]["data"]["Total_demanded_power"][0:end_year-start_year].plot()
        plt.show()
    
        #irr plotten
        ess.agents["Coal"]["data"]["Irr"][1:end_year-start_year].plot(title="IRR")
        ess.agents["Gas_CT"]["data"]["Irr"][1:end_year-start_year].plot()
        ess.agents["Gas_CC"]["data"]["Irr"][1:end_year-start_year].plot()
        ess.agents["Solar"]["data"]["Irr"][1:end_year-start_year].plot()
        ess.agents["Wind"]["data"]["Irr"][1:end_year-start_year].plot()
        plt.show()        
    
    #electricity cost
    ess.agents["Government"]["data"]["Electricity_cost"][0:end_year-start_year].plot()
    
    plt.title("")  # Remove the title
    
    plt.xlabel("Year")  # Add x-axis label
    plt.ylabel("Electricity price [Euro/MWh]")  # Add y-axis label

    
    plt.show()
    
    #electricity cost pure
    ess.agents["Government"]["data"]["Electricity_cost_pure"][0:end_year-start_year].plot(title="Electricity price pure")
    plt.show()
    
    
    #LIBOR
    ess.agents["Bank"]["data"]["LIBOR_index"][0:end_year-start_year].plot(title="LIBOR index")
    plt.show()
    
    
    #define variables to be viewed in the varible explorer
    agent_list = ess.get_agent_list()
    Government=ess.agents["Government"]["data"]
    Bank=ess.agents["Bank"]["data"]
    Producer_Coal=ess.agents["Coal"]["data"]
    Producer_Gas_CT=ess.agents["Gas_CT"]["data"]
    Producer_Gas_CC=ess.agents["Gas_CC"]["data"]
    Producer_Nuclear=ess.agents["Nuclear"]["data"]
    Producer_Solar=ess.agents["Solar"]["data"]
    Producer_Wind=ess.agents["Wind"]["data"]
    Producer_Hydro=ess.agents["Hydro"]["data"]
    Producer_Biomass=ess.agents["Biomass"]["data"]
    Producer_Oil=ess.agents["Oil"]["data"]
    
    Storage=ess.agents["Storage"]["data"]
    Consumer_1=ess.agents["Consumer 1"]["data"]
    Consumer_2=ess.agents["Consumer 2"]["data"]
    Consumer_3=ess.agents["Consumer 3"]["data"]
    
    fix_costs = ess.Fix_costs
    
    coal_profile = ess.coal_profile
    gas_ct_profile = ess.gas_ct_profile
    gas_cc_profile = ess.gas_cc_profile
    
    #print final computing time
    executionTime = (time.time() - startTime)/60
    print('Execution time in seconds: ' + str(round(executionTime,2)))