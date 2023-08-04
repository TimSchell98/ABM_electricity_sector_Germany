
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 12:36:34 2023

@author: Tim Schell

Agent based modell of the electricity generation system of Germany.

This is the second and up to date version
"""

import time
startTime = time.time()
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypsa

class ESS:
    def __init__(self, start_year=2000, end_year=2020, governmental_funding = 5, battery_store_cap = 50000, battery_store_in_power = 2500,
                 battery_store_eff = 0.987, battery_store_out_power = 5000, co2_emissions_coal = 3.67, energy_content_coal = 29.75,
                 co2_emissions_gas =  2.343, energy_content_gas = 8.82,mmbtu_per_t_gas = 46.97, fuel_factor_uranium = 8.9, conversion_uranium = 19.5
                 ,enrichment_uranium = 67.1,fuel_fabrication_uranium = 300, energy_content_uranium = 360):
        """
        Initiate all the variables and lists
        """
        self.start_year = start_year
        self.end_year = end_year
        self.agents = {}

        #if a run has to be run again, the seed list can be imported, if not the seed list is safed, to ensure the run can be run again
        self.shuffle_seeds = []
        
        #coal specs
        #co2 emissions from one MWh of coal from the Umweltbundesamt
        self.co2_emissions_coal = co2_emissions_coal 
        #energy content in coal in MJ per tonne
        self.energy_content_coal = energy_content_coal
       
        #governmental funding for renewable energy
        self.Governmental_funding = governmental_funding
       
        #gas specs
        #co2 emissions from one MWh of gas from the Umweltbundesamt
        self.co2_emissions_gas = co2_emissions_gas
        #energy content in gas in MWh per tonne
        self.energy_content_gas = energy_content_gas
        #number of mmbtu per tonne of gas
        self.mmbtu_per_t_gas = mmbtu_per_t_gas
        
        #nuclear specs
        self.fuel_factor_uranium = fuel_factor_uranium #[kg uranium/kg fuel]
        self.conversion_uranium = conversion_uranium #[Euro/kg Uranium]
        self.enrichment_uranium = enrichment_uranium #[Euro/kg Uranium]
        self.fuel_fabrication_uranium = fuel_fabrication_uranium #[Euro/kg Fuel] 
        self.energy_content_uranium = energy_content_uranium #[MWh/kg Fuel]
        
        #import LIBOR index list
        self.LIBOR_index = pd.read_excel("LIBOR.xlsx")
        #import risk markup list
        self.Risk_markup = pd.read_excel("Risk_markup.xlsx")
        #import investment costs as CAPEX
        self.Capex = pd.read_excel("Capex_nrel.xlsx")
        
        #define battery specs
        self.battery_store_cap = battery_store_cap
        self.battery_store_in_power = battery_store_in_power
        self.battery_store_eff = battery_store_eff
        self.battery_store_out_power = battery_store_out_power
        
        #import co2 certificate costs and ressource cost as lists
        self.Co2_certificate_costs = pd.read_excel("co2_certificate_costs.xlsx")
        self.coal_costs = pd.read_excel("ressource_costs.xlsx", sheet_name = "coal")
        self.gas_costs = pd.read_excel("ressource_costs.xlsx", sheet_name = "gas")
        self.uranium_costs = pd.read_excel("ressource_costs.xlsx", sheet_name = "uranium")

        #import operational costs
        self.Operational_costs = pd.read_excel("Operational_costs.xlsx")
        
        #import tax and network charge 2009 - 2021
        self.tax = pd.read_excel("Strompreis.xlsx")

    def add_agent(self, agent_name, agent_function, columns):
        """
        Add an agent to the model with a descriptive name, name of function and data columns
        """
        self.agents[agent_name] = {
            'function': agent_function,
            'data': pd.DataFrame(columns=columns, index=np.arange(self.end_year-self.start_year))
        }

    def init_agents(self):
        """
        This function defines all the agents with the corresponding varibles
        Also the values of the varibles of the first timestep are initilized
        """
        
        #ToDo: Muss Braunkohle auch als agent rein? die Daten zu steinkohle sind schon sehr anders
        #ToDo: Kann die initierung der agenten auch als excel list rein? Wäre dann um einiges leichter diese anzupassen.

        #import agents names, functions and initial values
        agents_list = pd.read_excel("initial_values.xlsx")

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
            ess.add_agent(agent_name, getattr(ess, function_name), variable_names)
            
        #fill dataframes of agents with 0 instead of nan
        for agent_name, agent_dict in ess.agents.items():
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
        
        #create function name list for the create_agent_list randomizing function
        #ToDo: Import these from excel list
        self.essencial_list = [ess.agent0, ess.agent1]
        self.producer_list = [ess.agent2, ess.agent3, ess.agent4, ess.agent5, ess.agent6, ess.agent7, ess.agent8, ess.agent9, ess.agent10]
        self.storage_list = [ess.agent11]
        self.consumer_list = [ess.agent12, ess.agent13, ess.agent14]


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

    def sigmoid(self,x):
        """
        This function calculates the y value of a sigmoid function given the x value.
        """
        
        #Compute the y value using the sigmoid function
        y = 1 / (1 + np.exp(-10 * (x - 0.5)))
        
        return y
    
    def init_profiles(self):
        """
        This function imports the profiles that are used in the PyPSA function
        """
        #Load PV profile
        pv_profile = pd.read_excel('Generations_profile.xlsx', sheet_name='2017 Generationsprofil Solar')
        
        pv_generation_profile = pv_profile.iloc[:, 0].tolist()
        pv_profile[2:300].plot()
        plt.show()

        #Load wind profile
        wind_profile = pd.read_excel('Generations_profile.xlsx', sheet_name='2017 Generationsprofil wind')
        wind_generation_profile = wind_profile.iloc[:, 0].tolist()
        wind_profile[2:300].plot()
        plt.show()

        #ToDo: implement one load profile for every consumer
        #-> can be added als individual loads in pypsa

        #Load load profile
        load_profile_df = pd.read_excel('load_data_smard.xlsx')
        load_profile = load_profile_df.iloc[:, 0].tolist()
        load_profile_df[2:300].plot()
        plt.show()

        return pv_generation_profile, wind_generation_profile, load_profile
    
    def run_network(self, i, total_demand, pv_generation_profile, wind_generation_profile, load_factor_profile):
        """
        This function runs a PyPSA network simulation of the producers.
        The average electricity costs are calculated and given to the consumers in the next time step
        The total electricity supplied is also calculated
        """
        
        #Calculate load profile with total demand and factor profile
        load_profile = [round(value * total_demand, 4) for value in load_factor_profile]
        print("total demand:")
        print(sum(load_profile))

        #Create an empty network
        network = pypsa.Network()
        network.set_snapshots(range(len(load_factor_profile)))

        #Add bus
        network.add('Bus', name='Main_bus')

        
        #ToDo: implement other variables like start up time etc...#
        #committable = True, start_up_cost = XY, shut_down_cost = XY, stand_by_cost = XY
        #min_up_time: coal = 27, gas CT 1.5, Gas CC 20, 
        
        #Add producers as generators
        network.add('Generator', name='Producer_1', bus='Main_bus', p_nom=ess.agents["Producer 1"]["data"].loc[self.timestep-1, "Installed_power"], marginal_cost=ess.agents["Producer 1"]["data"].loc[self.timestep-1, "Marginal_cost"]) # , committable = True, min_up_time = 27
        network.add('Generator', name='Producer_2', bus='Main_bus', p_nom=ess.agents["Producer 2"]["data"].loc[self.timestep-1, "Installed_power"], marginal_cost=ess.agents["Producer 2"]["data"].loc[self.timestep-1, "Marginal_cost"]) # , committable = True, min_up_time = 1.5
        network.add('Generator', name='Producer_3', bus='Main_bus', p_nom=ess.agents["Producer 3"]["data"].loc[self.timestep-1, "Installed_power"], marginal_cost=ess.agents["Producer 3"]["data"].loc[self.timestep-1, "Marginal_cost"]) # , committable = True, min_up_time = 20
        network.add('Generator', name='Producer_4', bus='Main_bus', p_nom=ess.agents["Producer 4"]["data"].loc[self.timestep-1, "Installed_power"], marginal_cost=ess.agents["Producer 4"]["data"].loc[self.timestep-1, "Marginal_cost"])
        network.add('Generator', name='Producer_5', bus='Main_bus', p_nom=ess.agents["Producer 5"]["data"].loc[self.timestep-1, "Installed_power"], p_max_pu=pv_generation_profile,marginal_cost=ess.agents["Producer 5"]["data"].loc[self.timestep-1, "Marginal_cost"])
        network.add('Generator', name='Producer_6', bus='Main_bus', p_nom=ess.agents["Producer 6"]["data"].loc[self.timestep-1, "Installed_power"], p_max_pu=wind_generation_profile, marginal_cost=ess.agents["Producer 6"]["data"].loc[self.timestep-1, "Marginal_cost"])
        network.add('Generator', name='Producer_7', bus='Main_bus', p_nom=ess.agents["Producer 7"]["data"].loc[self.timestep-1, "Installed_power"], marginal_cost=ess.agents["Producer 7"]["data"].loc[self.timestep-1, "Marginal_cost"])
        network.add('Generator', name='Producer_8', bus='Main_bus', p_nom=ess.agents["Producer 8"]["data"].loc[self.timestep-1, "Installed_power"], marginal_cost=ess.agents["Producer 8"]["data"].loc[self.timestep-1, "Marginal_cost"])
        network.add('Generator', name='Producer_9', bus='Main_bus', p_nom=ess.agents["Producer 9"]["data"].loc[self.timestep-1, "Installed_power"], marginal_cost=ess.agents["Producer 9"]["data"].loc[self.timestep-1, "Marginal_cost"])
        
        
        #endless generator
        network.add('Generator', name='Endless_producer', bus='Main_bus', p_nom=0, p_nom_extendable = True, marginal_cost=100000)
    
        #Add storage battery
        network.add('Bus', name = 'Storagebus_battery')
        network.add('Store', name = 'Battery_storage', bus = 'Storagebus_battery', e_nom = ess.agents["Storage"]["data"].loc[self.timestep-1, "Battery_cap"])
        network.add('Link', name = 'Load_battery', bus0 = 'Main_bus',bus1 = 'Storagebus_battery', p_nom = ess.agents["Storage"]["data"].loc[self.timestep-1, "Battery_in"],efficiency = ess.agents["Storage"]["data"].loc[self.timestep-1, "Battery_eff"]**0.5)
        network.add('Link', name = 'Unload_battery', bus1 = 'Main_bus',bus0 = 'Storagebus_battery', p_nom = ess.agents["Storage"]["data"].loc[self.timestep-1, "Battery_out"],efficiency = ess.agents["Storage"]["data"].loc[self.timestep-1, "Battery_eff"]**0.5)

        #Add storage water pump
        network.add('Bus', name = 'Storagebus_pump')
        network.add('Store', name = 'Pump_storage', bus = 'Storagebus_pump', e_nom = ess.agents["Storage"]["data"].loc[self.timestep-1, "Pump_cap"])
        network.add('Link', name = 'Load_pump', bus0 = 'Main_bus',bus1 = 'Storagebus_pump', p_nom = ess.agents["Storage"]["data"].loc[self.timestep-1, "Pump_in"],efficiency = ess.agents["Storage"]["data"].loc[self.timestep-1, "Pump_eff"]**0.5)
        network.add('Link', name = 'Unload_pump', bus1 = 'Main_bus',bus0 = 'Storagebus_pump', p_nom = ess.agents["Storage"]["data"].loc[self.timestep-1, "Pump_out"],efficiency = ess.agents["Storage"]["data"].loc[self.timestep-1, "Pump_eff"]**0.5)

        #Add consumers as load
        network.add('Load', name='Sum_consumers', bus='Main_bus', p_set=load_profile)

        #Optimize network
        network.optimize(solver_name='gurobi')

        #plot some variables
        network.generators_t.p[2:100].plot(kind = 'bar')
        plt.show()
        network.buses_t.marginal_price["Main_bus"][2:300].plot()
        plt.show()
        network.links_t.p0[2:300].plot()
        plt.show()

        #calculate average electricity price
        Electricity_cost = round(sum(network.buses_t.marginal_price["Main_bus"])/(len(network.buses_t.marginal_price["Main_bus"])+1),4)
        
        print("Endless producer provided: ")
        print(sum(network.generators_t.p["Endless_producer"]))

        #create dataframes and fill with values, total generated power and max power
        data_generated_power = {'Producer 1': [round(sum(network.generators_t.p["Producer_1"]),2)],
                                'Producer 2': [round(sum(network.generators_t.p["Producer_2"]),2)],
                                'Producer 3': [round(sum(network.generators_t.p["Producer_3"]),2)],
                                'Producer 4': [round(sum(network.generators_t.p["Producer_4"]),2)],
                                'Producer 5': [round(sum(network.generators_t.p["Producer_5"]),2)],
                                'Producer 6': [round(sum(network.generators_t.p["Producer_6"]),2)],
                                'Producer 7': [round(sum(network.generators_t.p["Producer_7"]),2)],
                                'Producer 8': [round(sum(network.generators_t.p["Producer_8"]),2)],
                                'Producer 9': [round(sum(network.generators_t.p["Producer_9"]),2)]
                                }

        #info 99% because of rounding errors
        data_max_power = {
            'Producer 1': [round(max(network.generators_t.p["Producer_1"]), 2), (round(network.generators_t.p["Producer_1"], 1) >= round(ess.agents["Producer 1"]["data"].loc[self.timestep-1, "Installed_power"] * 0.99, 1)).sum()],
            'Producer 2': [round(max(network.generators_t.p["Producer_2"]), 2), (round(network.generators_t.p["Producer_2"], 1) >= round(ess.agents["Producer 2"]["data"].loc[self.timestep-1, "Installed_power"] * 0.99, 1)).sum()],
            'Producer 3': [round(max(network.generators_t.p["Producer_3"]), 2), (round(network.generators_t.p["Producer_3"], 1) >= round(ess.agents["Producer 3"]["data"].loc[self.timestep-1, "Installed_power"] * 0.99, 1)).sum()],
            'Producer 4': [round(max(network.generators_t.p["Producer_4"]), 2), (round(network.generators_t.p["Producer_4"], 1) >= round(ess.agents["Producer 4"]["data"].loc[self.timestep-1, "Installed_power"] * 0.99, 1)).sum()],
            'Producer 5': [round(max(network.generators_t.p["Producer_5"]), 2), (round(network.generators_t.p["Producer_5"], 1) >= round(ess.agents["Producer 5"]["data"].loc[self.timestep-1, "Installed_power"] * 0.99, 1)).sum()],
            'Producer 6': [round(max(network.generators_t.p["Producer_6"]), 2), (round(network.generators_t.p["Producer_6"], 1) >= round(ess.agents["Producer 6"]["data"].loc[self.timestep-1, "Installed_power"] * 0.99, 1)).sum()],
            'Producer 7': [round(max(network.generators_t.p["Producer_7"]), 2), (round(network.generators_t.p["Producer_7"], 1) >= round(ess.agents["Producer 7"]["data"].loc[self.timestep-1, "Installed_power"] * 0.99, 1)).sum()],
            'Producer 8': [round(max(network.generators_t.p["Producer_8"]), 2), (round(network.generators_t.p["Producer_8"], 1) >= round(ess.agents["Producer 8"]["data"].loc[self.timestep-1, "Installed_power"] * 0.99, 1)).sum()],
            'Producer 9': [round(max(network.generators_t.p["Producer_9"]), 2), (round(network.generators_t.p["Producer_9"], 1) >= round(ess.agents["Producer 9"]["data"].loc[self.timestep-1, "Installed_power"] * 0.99, 1)).sum()]
        }
        
        generated_power = pd.DataFrame(data = data_generated_power)
        max_power = pd.DataFrame(data = data_max_power)
        
        
        #network.generators_t.p in excel liste speichern mit aktuellem zeitschritt self.timestep
        #network.generators_t.p.to_excel("network.generators_{}.xlsx".format(self.timestep))
        

        return Electricity_cost, generated_power, max_power

    def run_sim(self,pv_generation_profile, wind_generation_profile, load_factor_profile):
        """
        Function that runs the simulation from the start year to the end year
        """
        #init dataframes to gather PyPSA Data
        data_generated_power = {'Producer 1': [0], 'Producer 2': [0], 'Producer 3': [0], 'Producer 4': [0], 'Producer 5': [0], 'Producer 6': [0], 'Producer 7': [0], 'Producer 8': [0], 'Producer 9': [0] }
        data_max_power = {'Producer 1': [0,0], 'Producer 2': [0,0], 'Producer 3': [0,0], 'Producer 4': [0,0], 'Producer 5': [0,0], 'Producer 6': [0,0], 'Producer 7': [0,0], 'Producer 8': [0,0], 'Producer 9': [0,0]}
        self.generated_power = pd.DataFrame(data = data_generated_power)
        self.max_power = pd.DataFrame(data = data_max_power)

        #loop from start year to end year, dont simulate first year (only start values)
        for self.time in range(self.start_year+1, self.end_year):
            #calculate timestep
            self.timestep = self.time - self.start_year
            print("\nTime:", self.time)
            print("Timestep:", self.timestep)
            
            #caculate total demand to use in PyPSA as load, Timestep-1 so that the inital values are used
            total_demand = ess.agents["Consumer 1"]["data"].loc[self.timestep-1, "Power_demand"] + ess.agents["Consumer 2"]["data"].loc[self.timestep-1, "Power_demand"] + ess.agents["Consumer 3"]["data"].loc[self.timestep-1, "Power_demand"]
            print("Total demand:")
            print(total_demand)
            
            #Run the network simulation and gather data
            self.Electricity_cost_pure, self.generated_power_temp, self.max_power = ess.run_network(self.timestep-1, total_demand, pv_generation_profile, wind_generation_profile, load_factor_profile)
            
            #gather data from PYPSA
            self.generated_power = self.generated_power.append(self.generated_power_temp, ignore_index = True)
            
            #add taxes and network costs to electricity price
            self.Electricity_cost = self.Electricity_cost_pure + ess.agents["Government"]["data"].loc[self.timestep-1, "Tax"] 
            
            #calulate average electricity price from the last 3 years
            #gather Electricity costs from the last 3 years, if not availible, gather avalible values
            if self.timestep < 3:
                Electricity_cost_3_years = ess.agents["Government"]["data"].loc[0:self.timestep-1, "Electricity_cost"]

            else:
                Electricity_cost_3_years = ess.agents["Government"]["data"].loc[self.timestep-3:self.timestep-1, "Electricity_cost"]

            #calculate average of last 3 values
            self.Average_electricity_cost = round(Electricity_cost_3_years.mean(),2)
            #calculate the proportion of the delta of the current price to the average price
            self.Electricity_cost_proportion = (self.Electricity_cost-self.Average_electricity_cost)/self.Average_electricity_cost
            
            #call investment/loan payback function
            #ToDo: impement function which act als loan storage, in this function the loans are listed and counted down
            #loan_list()
            
            #Randomize the order in which the agents are called to combat first mover advantage
            self.agent_list = ess.create_agent_list()
            
            #call agents in the order of the list
            for function_name in self.agent_list:
                function_name()


    def agent0(self):
        """
        This agent represents the government of authority agent
        In this function variables like tax can be changed
        """
        print("Goverment called")
        
        #update tax
        ess.agents["Government"]["data"].loc[self.timestep, "Tax"] = self.tax.iloc[self.timestep,0]

        #fill Electricity_cost with with calculated PyPSA data
        ess.agents["Government"]["data"].loc[self.timestep, "Electricity_cost"] = self.Electricity_cost
    
        #fill Electricity_cost_pure with with calculated PyPSA data
        ess.agents["Government"]["data"].loc[self.timestep, "Electricity_cost_pure"] = self.Electricity_cost_pure
    
        #update total emissions
        ess.agents["Government"]["data"].loc[self.timestep, "Total_emissions"] = ess.agents["Producer 1"]["data"].loc[self.timestep, "Total_emissions"]+ess.agents["Producer 2"]["data"].loc[self.timestep, "Total_emissions"]+ess.agents["Producer 3"]["data"].loc[self.timestep, "Total_emissions"]+ess.agents["Producer 4"]["data"].loc[self.timestep, "Total_emissions"] + ess.agents["Producer 5"]["data"].loc[self.timestep, "Total_emissions"] + ess.agents["Producer 6"]["data"].loc[self.timestep, "Total_emissions"] + ess.agents["Producer 7"]["data"].loc[self.timestep, "Total_emissions"] + ess.agents["Producer 8"]["data"].loc[self.timestep, "Total_emissions"]

        #update government total generated power
        ess.agents["Government"]["data"].loc[self.timestep, "Total_generated_power"] = ess.agents["Producer 1"]["data"].loc[self.timestep, "Generated_power_total"]+ ess.agents["Producer 2"]["data"].loc[self.timestep, "Generated_power_total"]+ ess.agents["Producer 3"]["data"].loc[self.timestep, "Generated_power_total"] + ess.agents["Producer 4"]["data"].loc[self.timestep, "Generated_power_total"] + ess.agents["Producer 5"]["data"].loc[self.timestep, "Generated_power_total"] + ess.agents["Producer 6"]["data"].loc[self.timestep, "Generated_power_total"] + ess.agents["Producer 7"]["data"].loc[self.timestep, "Generated_power_total"] + ess.agents["Producer 8"]["data"].loc[self.timestep, "Generated_power_total"]
        
        #update government total installed power
        ess.agents["Government"]["data"].loc[self.timestep, "Total_installed_power"] = ess.agents["Producer 1"]["data"].loc[self.timestep, "Installed_power"] + ess.agents["Producer 2"]["data"].loc[self.timestep, "Installed_power"] + ess.agents["Producer 3"]["data"].loc[self.timestep, "Installed_power"] + ess.agents["Producer 4"]["data"].loc[self.timestep, "Installed_power"] + ess.agents["Producer 5"]["data"].loc[self.timestep, "Installed_power"] + ess.agents["Producer 6"]["data"].loc[self.timestep, "Installed_power"] + ess.agents["Producer 7"]["data"].loc[self.timestep, "Installed_power"] + ess.agents["Producer 8"]["data"].loc[self.timestep, "Installed_power"]
        
        #update government total demanded power
        ess.agents["Government"]["data"].loc[self.timestep, "Total_demanded_power"] = ess.agents["Consumer 1"]["data"].loc[self.timestep, "Power_demand"] + ess.agents["Consumer 2"]["data"].loc[self.timestep, "Power_demand"] + ess.agents["Consumer 3"]["data"].loc[self.timestep, "Power_demand"]
        

    def agent1(self):
        """
        This agent represents a central bank, that provides loans to every other agent
        """
        print("Bank called")
        
        #update money
        ess.agents["Bank"]["data"].loc[self.timestep,"Money"] = ess.agents["Bank"]["data"].loc[self.timestep-1,"Money"]

        #update LIBOR
        ess.agents["Bank"]["data"].loc[self.timestep,"LIBOR_index"] = self.LIBOR_index.iloc[self.timestep,0]
        
        #update margin
        ess.agents["Bank"]["data"].loc[self.timestep,"Margin"] = ess.agents["Bank"]["data"].loc[self.timestep-1,"Margin"]

        #update interest rate
        ess.agents["Bank"]["data"].loc[self.timestep,"Interest_rate"] = ess.agents["Bank"]["data"].loc[self.timestep,"LIBOR_index"] + ess.agents["Bank"]["data"].loc[self.timestep,"Margin"]

    def agent2(self):
        """
        Function of producer 1.
        This producer resembels the combined coal power plants
        """
        print("Producer 1 called")
        
        #update installed_power, decrease because of deprication 2.5% per year -> life expectancy 40 years
        ess.agents["Producer 1"]["data"].loc[self.timestep, "Installed_power"] = ess.agents["Producer 1"]["data"].loc[self.timestep-1, "Installed_power"] * 0.975
        
        #update efficiency, increase by 0.1% every year
        #todo: function einbauen die dafür sorgt, dass der wirkungsgrad bis 41% steigt und danach so bleibt
        ess.agents["Producer 1"]["data"].loc[self.timestep, "Efficiency"] = ess.agents["Producer 1"]["data"].loc[self.timestep-1, "Efficiency"]*1.001    
    
        #update marginal cost
        
        #calaculate co2 costs
        #calculate the amount of mwh generated by 1 tonne coal
        mwh_t_coal = (self.energy_content_coal * ess.agents["Producer 1"]["data"].loc[self.timestep, "Efficiency"]) / 3.6
        
        #calculate the co2 emissions from one mwh of coal electricity
        co2_t_coal = self.co2_emissions_coal/mwh_t_coal

        #calculate co2 costs per MWh coal
        co2_costs = self.Co2_certificate_costs.iloc[self.timestep,0] * co2_t_coal

        #calculate ressource costs
        ressource_cost = self.coal_costs.iloc[self.timestep,0]/mwh_t_coal

        #calculate marginal costs  
        marginal_cost = ressource_cost + co2_costs + self.Operational_costs.loc[0,"Coal"]
        
        #write marginal cost in dataframe
        ess.agents["Producer 1"]["data"].loc[self.timestep, "Marginal_cost"] = marginal_cost

        #fill generated_power_total with calculated PyPSA data
        ess.agents["Producer 1"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.iloc[self.timestep,0]
        
        #calculate Income with calculated PyPSA data
        ess.agents["Producer 1"]["data"].loc[self.timestep, "Income"] = ess.agents["Producer 1"]["data"].loc[self.timestep, "Generated_power_total"]*self.Electricity_cost_pure

        #update money
        ess.agents["Producer 1"]["data"].loc[self.timestep, "Money"] = ess.agents["Producer 1"]["data"].loc[self.timestep-1, "Money"] + ess.agents["Producer 1"]["data"].loc[self.timestep, "Income"] 

        #investment function
        
        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy_total = ess.agents["Producer 1"]["data"].loc[self.timestep, "Installed_power"] * 365 * 24
        
        #calculate average load in the last time step
        ess.agents["Producer 1"]["data"].loc[self.timestep, "Load"] = round(ess.agents["Producer 1"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy_total,4)
        
        #calculate how much the installed power should be increased
        #ToDo: funktioniert noch nicht so gut
        power_increase = (0.15/8759) * self.max_power.iloc[1,0]
        print(power_increase)

        #calculate outlook of profit -> multiply number of times max power was provided and multiply that times the max power
        #then assume that 5% could be supplied more and calculate the profit of that energy amount
        profit_outlook = ((self.max_power.iloc[1,0] * self.max_power.iloc[1,0]) * power_increase)*self.Electricity_cost_pure
        #ToDo: villeicht kann hier eine funktion eingebaut werden, die anhand der anzahl der supplieten maximal leistungen entscheidet wie viel mehr verkauft werden könnte
        # -> wenn die anzahl an geliefertem maximal strom hochgeht geht auch der expected increase hoch -> funktion überlegen
        

        #caluclate investment costs of this expansion over a lifetime of 20 years
        investment_cost = ((ess.agents["Producer 1"]["data"].loc[self.timestep, "Installed_power"] * power_increase)* self.Capex.loc[self.timestep, "Coal"])/40
        
        #check if investment costs is lower than profit outlook
        if investment_cost < profit_outlook:
            #increase installed power by 5%
            #ToDo: zuerst kredit und zinz berechnen und checken ob das income für diesen zinz reicht.
            #ToDo: kredit in eine liste schreiben die abgehandelt wird
            ess.agents["Producer 1"]["data"].loc[self.timestep, "Installed_power"] = ess.agents["Producer 1"]["data"].loc[self.timestep, "Installed_power"] * (1 + power_increase)

        #calculate total emissions with calculated PyPSA data
        ess.agents["Producer 1"]["data"].loc[self.timestep, "Total_emissions"] = ess.agents["Producer 1"]["data"].loc[self.timestep, "Generated_power_total"] * co2_t_coal 
        
    
    def agent3(self):
        """
        Function of producer 2.
        This producer resembels the combined gas power plants Combustion Turbine
        """
        print("Producer 2 called")
        
        #update installed_power, decrease because of deprication 3.33% per year -> lifetime 30 years
        ess.agents["Producer 2"]["data"].loc[self.timestep, "Installed_power"] = ess.agents["Producer 2"]["data"].loc[self.timestep-1, "Installed_power"] * 0.967
        
        #update efficiency, increase by 0.25% every year
        ess.agents["Producer 2"]["data"].loc[self.timestep, "Efficiency"] = ess.agents["Producer 2"]["data"].loc[self.timestep-1, "Efficiency"]*1.0025
        
        #update marginal cost
        
        #co2 costs
        #calculate how many MWh one tonne of natural gas can generate
        mwh_t_gas = self.energy_content_gas*ess.agents["Producer 2"]["data"].loc[self.timestep, "Efficiency"]
        #calculate how many tonnes of co2 one mhw emmit
        t_co2_mwh = self.co2_emissions_gas/mwh_t_gas
        
        #calculate co2 costs
        co2_costs = t_co2_mwh*self.Co2_certificate_costs.iloc[self.timestep,0]
        
        #ressource costs
        #calculate costs per tonne of gas
        costs_t = self.gas_costs.iloc[self.timestep,0] * self.mmbtu_per_t_gas
        #calculate costs per mwh
        ressource_cost = costs_t/mwh_t_gas
        #ToDo: something is wrong with calculation, quick fix, times 2
        ressource_cost = ressource_cost*2

        #calculate marginal costs  
        marginal_cost = ressource_cost + co2_costs + self.Operational_costs.loc[0,"Gas_CT"]

        
        #write marginal cost in dataframe
        ess.agents["Producer 2"]["data"].loc[self.timestep, "Marginal_cost"] = marginal_cost
        
        #fill generated_power_total with calculated PyPSA data
        ess.agents["Producer 2"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.iloc[self.timestep,1]

        #calculate Income with calculated PyPSA data
        ess.agents["Producer 2"]["data"].loc[self.timestep, "Income"] = ess.agents["Producer 2"]["data"].loc[self.timestep, "Generated_power_total"]*self.Electricity_cost_pure

        #update money
        ess.agents["Producer 2"]["data"].loc[self.timestep, "Money"] = ess.agents["Producer 2"]["data"].loc[self.timestep-1, "Money"] + ess.agents["Producer 2"]["data"].loc[self.timestep, "Income"] 

        #calculate how much the installed power should be increased
        power_increase = (0.15/8759) * self.max_power.iloc[1,1]
        print(power_increase)

        #investment function
        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy_total = ess.agents["Producer 2"]["data"].loc[self.timestep, "Installed_power"] * 365 * 24

        #calculate load and fill data
        ess.agents["Producer 2"]["data"].loc[self.timestep, "Load"] = round(ess.agents["Producer 2"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy_total,4)
  
        #calculate outlook of profit -> multiply number of times max power was provided and multiply that times the max power
        #then assume that 5% could be supplied more and calculate the profit of that energy amount
        profit_outlook = ((self.max_power.iloc[1,1] * self.max_power.iloc[0,1]) * power_increase)*self.Electricity_cost_pure
        #caluclate investment costs of this expansion over a lifetime of 20 years -> ToDo: Hier ist noch kein zinz drauf und vorallem keine tilgung! -> muss hier überhaupt n Zinz/tilgung drauf?
        investment_cost = ((ess.agents["Producer 2"]["data"].loc[self.timestep, "Installed_power"] * power_increase)* self.Capex.loc[self.timestep, "Gas CT"])/30
        #check if investment cost is lower than profit outlook
        if investment_cost < profit_outlook:
            #increase installed power by 5%
            ess.agents["Producer 2"]["data"].loc[self.timestep, "Installed_power"] = ess.agents["Producer 2"]["data"].loc[self.timestep, "Installed_power"] * (1 + power_increase)
        
        print("Max Power Amount:")
        print(self.max_power.iloc[1,1])
        
        #calculate total emissions with calculated PyPSA data
        ess.agents["Producer 2"]["data"].loc[self.timestep, "Total_emissions"] = ess.agents["Producer 2"]["data"].loc[self.timestep, "Generated_power_total"] * t_co2_mwh
        
        
    def agent4(self):
        """
        Function of producer 3.
        This producer resembels the combined gas power plants Combined Cycle
        """
        print("Producer 3 called")
        
        #update installed_power, decrease because of deprication 3.33% per year -> lifetime 30 years
        ess.agents["Producer 3"]["data"].loc[self.timestep, "Installed_power"] = ess.agents["Producer 3"]["data"].loc[self.timestep-1, "Installed_power"] * 0.967
        
        #update efficiency, increase by 0.25% every year
        ess.agents["Producer 3"]["data"].loc[self.timestep, "Efficiency"] = ess.agents["Producer 3"]["data"].loc[self.timestep-1, "Efficiency"]*1.0025
        
        #update marginal cost
        
        #co2 costs
        #calculate how many MWh one tonne of natural gas can generate
        mwh_t_gas = self.energy_content_gas*ess.agents["Producer 3"]["data"].loc[self.timestep, "Efficiency"]
        #calculate how many tonnes of co2 one mhw emmit
        t_co2_mwh = self.co2_emissions_gas/mwh_t_gas
        
        #calculate co2 costs
        co2_costs = t_co2_mwh*self.Co2_certificate_costs.iloc[self.timestep,0]
        
        #ressource costs
        #calculate costs per tonne of gas
        costs_t = self.gas_costs.iloc[self.timestep,0] * self.mmbtu_per_t_gas
        #calculate costs per mwh
        ressource_cost = costs_t/mwh_t_gas
        #ToDo: something is wrong with calculation, quick fix, times 2
        ressource_cost = ressource_cost*2

        #calculate marginal costs  
        marginal_cost = ressource_cost + co2_costs + self.Operational_costs.loc[0,"Gas_CC"]

        
        #write marginal cost in dataframe
        ess.agents["Producer 3"]["data"].loc[self.timestep, "Marginal_cost"] = marginal_cost
        
        #fill generated_power_total with calculated PyPSA data
        ess.agents["Producer 3"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.iloc[self.timestep,2]

        #calculate Income with calculated PyPSA data
        ess.agents["Producer 3"]["data"].loc[self.timestep, "Income"] = ess.agents["Producer 3"]["data"].loc[self.timestep, "Generated_power_total"]*self.Electricity_cost_pure

        #update money
        ess.agents["Producer 3"]["data"].loc[self.timestep, "Money"] = ess.agents["Producer 3"]["data"].loc[self.timestep-1, "Money"] + ess.agents["Producer 3"]["data"].loc[self.timestep, "Income"] 

        #calculate how much the installed power should be increased
        power_increase = (0.15/8759) * self.max_power.iloc[1,2]
        print(power_increase)


        #investment function
        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy_total = ess.agents["Producer 3"]["data"].loc[self.timestep, "Installed_power"] * 365 * 24

        #calculate load and fill data
        ess.agents["Producer 3"]["data"].loc[self.timestep, "Load"] = round(ess.agents["Producer 3"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy_total,4)
  
        #calculate outlook of profit -> multiply number of times max power was provided and multiply that times the max power
        #then assume that 5% could be supplied more and calculate the profit of that energy amount
        profit_outlook = ((self.max_power.iloc[1,2] * self.max_power.iloc[0,2]) * power_increase)*self.Electricity_cost_pure
        #caluclate investment costs of this expansion over a lifetime of 20 years -> ToDo: Hier ist noch kein zinz drauf und vorallem keine tilgung! -> muss hier überhaupt n Zinz/tilgung drauf?
        investment_cost = ((ess.agents["Producer 3"]["data"].loc[self.timestep, "Installed_power"] * power_increase)* self.Capex.loc[self.timestep, "Gas CC"])/30
        #check if investment cost is lower than profit outlook
        if investment_cost < profit_outlook:
            #increase installed power by 5%
            ess.agents["Producer 3"]["data"].loc[self.timestep, "Installed_power"] = ess.agents["Producer 3"]["data"].loc[self.timestep, "Installed_power"] * (1 + power_increase)
        
        print("Max Power Amount:")
        print(self.max_power.iloc[1,2])
        
        #calculate total emissions with calculated PyPSA data
        ess.agents["Producer 3"]["data"].loc[self.timestep, "Total_emissions"] = ess.agents["Producer 3"]["data"].loc[self.timestep, "Generated_power_total"] * t_co2_mwh

        
        
    def agent5(self):
        """
        Function of Producer 4.
        This producer resembels the combined nuclear power plants
        """
        print("Producer 4 called")
        

        #update marginal cost
        #calculate ressource costs
        ressource_cost = (self.uranium_costs.iloc[self.timestep,0] * self.fuel_factor_uranium + self.fuel_factor_uranium * self.conversion_uranium + self.fuel_factor_uranium * self.enrichment_uranium + self.fuel_fabrication_uranium) / self.energy_content_uranium
        #calculate marginal costs
        marginal_cost = ressource_cost + self.Operational_costs.loc[0,"Nuclear"]
        #write marginal cost in dataframe
        ess.agents["Producer 4"]["data"].loc[self.timestep, "Marginal_cost"] = marginal_cost
        
        #update installed_power, generation decreased, because of nuclear phase out 21% a year
        ess.agents["Producer 4"]["data"].loc[self.timestep, "Installed_power"] = ess.agents["Producer 4"]["data"].loc[self.timestep-1, "Installed_power"]*0.79

        #fill generated_power_total with calculated PyPSA data
        ess.agents["Producer 4"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.iloc[self.timestep,3]

        #calculate Income with calculated PyPSA data
        ess.agents["Producer 4"]["data"].loc[self.timestep, "Income"] = ess.agents["Producer 4"]["data"].loc[self.timestep, "Generated_power_total"]*self.Electricity_cost_pure

        #update money
        ess.agents["Producer 4"]["data"].loc[self.timestep, "Money"] = ess.agents["Producer 4"]["data"].loc[self.timestep-1, "Money"] + ess.agents["Producer 4"]["data"].loc[self.timestep, "Income"] 

        #investment function
        
        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy_total = ess.agents["Producer 4"]["data"].loc[self.timestep-1, "Installed_power"] * 365 * 24
        
        #calculate average load
        average_load = round(ess.agents["Producer 4"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy_total,4)
        
        #fill data
        ess.agents["Producer 4"]["data"].loc[self.timestep, "Load"] = average_load
        
        #no investment function because of nuclear phase out
        
        #calculate total emissions with calculated PyPSA data, nuclear energy has no emissions in the operation
        ess.agents["Producer 4"]["data"].loc[self.timestep, "Total_emissions"] = 0
        
        #update efficiency, because of the nuclear phase-out stays the same
        ess.agents["Producer 4"]["data"].loc[self.timestep, "Efficiency"] = ess.agents["Producer 4"]["data"].loc[self.timestep-1, "Efficiency"]

        
    def agent6(self):
        """
        Function of Producer 5.
        This producer resembels the combined pv power plants
        """
        print("Producer 5 called")
        
        #update marginal cost
        ess.agents["Producer 5"]["data"].loc[self.timestep, "Marginal_cost"] = ess.agents["Producer 5"]["data"].loc[self.timestep-1, "Marginal_cost"]
        
        #update installed_power, decrease because of deprication 4% per year -> lifetime 25 years 
        ess.agents["Producer 5"]["data"].loc[self.timestep, "Installed_power"] = ess.agents["Producer 5"]["data"].loc[self.timestep-1, "Installed_power"]* 0.96

        #fill generated_power_total with calculated PyPSA data
        ess.agents["Producer 5"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.iloc[self.timestep,4]

        #calculate Income with calculated PyPSA data
        ess.agents["Producer 5"]["data"].loc[self.timestep, "Income"] = ess.agents["Producer 5"]["data"].loc[self.timestep, "Generated_power_total"]*self.Electricity_cost_pure

        #update money
        ess.agents["Producer 5"]["data"].loc[self.timestep, "Money"] = ess.agents["Producer 5"]["data"].loc[self.timestep-1, "Money"] + ess.agents["Producer 5"]["data"].loc[self.timestep, "Income"] 

        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy = [round(value * ess.agents["Producer 5"]["data"].loc[self.timestep, "Installed_power"], 4) for value in pv_generation_profile]
        theoratical_energy_total = sum(theoratical_energy)

        #calculate load and fill data
        ess.agents["Producer 5"]["data"].loc[self.timestep, "Load"] = round(ess.agents["Producer 5"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy_total,4)

        #calculate how much the installed power should be increased
        power_increase = (0.15/8759) * self.max_power.iloc[1,4]
        print(power_increase)

        #calculate outlook of profit -> multiply number of times max power was provided and multiply that times the max power
        #then assume that 5% could be supplied more and calculate the profit of that energy amount
        #calculate assumed power that can be sold with generation profile
        extended_power = [value * (ess.agents["Producer 5"]["data"].loc[self.timestep-1, "Installed_power"] * power_increase) for value in pv_generation_profile]
        extended_power_total = sum(extended_power)
        profit_outlook = extended_power_total * (self.Electricity_cost_pure + self.Governmental_funding)
        #caluclate investment costs of this expansion over a lifetime of 20 years -> ToDo: Hier ist noch kein zinz drauf und vorallem keine tilgung! -> muss hier überhaupt n Zinz/tilgung drauf?
        investment_cost = ((ess.agents["Producer 5"]["data"].loc[self.timestep, "Installed_power"] * power_increase)* self.Capex.loc[self.timestep, "Solar Utility"])/25

        if investment_cost < profit_outlook:
            ess.agents["Producer 5"]["data"].loc[self.timestep, "Installed_power"] = ess.agents["Producer 5"]["data"].loc[self.timestep, "Installed_power"] * (1+power_increase)

        print("Max Power Amount:")
        print(self.max_power.iloc[1,4])

        #update efficiency, increase by 1.8% every year
        ess.agents["Producer 5"]["data"].loc[self.timestep, "Efficiency"] = ess.agents["Producer 5"]["data"].loc[self.timestep-1, "Efficiency"]*1.018

        #calculate total emissions with calculated PyPSA data, PV has no emissions in operation
        ess.agents["Producer 5"]["data"].loc[self.timestep, "Total_emissions"] = 0
        
            
    def agent7(self):
        """
        Function of Producer 6.
        This producer resembels the combined wind power plants
        """
        print("Producer 6 called")
        
        #update installed_power, decrease because of deprication 4% per year -> life time 25 years
        ess.agents["Producer 6"]["data"].loc[self.timestep, "Installed_power"] = ess.agents["Producer 6"]["data"].loc[self.timestep-1, "Installed_power"] * 0.96
        
        #update marginal cost
        ess.agents["Producer 6"]["data"].loc[self.timestep, "Marginal_cost"] = ess.agents["Producer 6"]["data"].loc[self.timestep-1, "Marginal_cost"]
        
        #fill generated_power_total with calculated PyPSA data
        ess.agents["Producer 6"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.iloc[self.timestep,5]

        #calculate Income with calculated PyPSA data
        ess.agents["Producer 6"]["data"].loc[self.timestep, "Income"] = ess.agents["Producer 6"]["data"].loc[self.timestep, "Generated_power_total"]* (self.Electricity_cost_pure + self.Governmental_funding)

        #update money
        ess.agents["Producer 6"]["data"].loc[self.timestep, "Money"] = ess.agents["Producer 6"]["data"].loc[self.timestep-1, "Money"] + ess.agents["Producer 6"]["data"].loc[self.timestep, "Income"] 

        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy = [round(value * ess.agents["Producer 6"]["data"].loc[self.timestep, "Installed_power"], 4) for value in wind_generation_profile]
        theoratical_energy_total = sum(theoratical_energy)

        #calculate load and fill data
        ess.agents["Producer 6"]["data"].loc[self.timestep, "Load"] = round(ess.agents["Producer 6"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy_total,4)

        #calculate how much the installed power should be increased
        power_increase = (0.15/8759) * self.max_power.iloc[1,5]
        print(power_increase)

        #calculate outlook of profit -> multiply number of times max power was provided and multiply that times the max power
        #then assume that 5% could be supplied more and calculate the profit of that energy amount
        #calculate assumed power that can be sold with generation profile
        extended_power = [value * (ess.agents["Producer 6"]["data"].loc[self.timestep-1, "Installed_power"] * power_increase) for value in wind_generation_profile]
        extended_power_total = sum(extended_power)
        profit_outlook = extended_power_total * (self.Electricity_cost_pure+self.Governmental_funding)
        #caluclate investment costs of this expansion over a lifetime of 20 years -> ToDo: Hier ist noch kein zinz drauf und vorallem keine tilgung! -> muss hier überhaupt n Zinz/tilgung drauf?
        investment_cost = ((ess.agents["Producer 6"]["data"].loc[self.timestep, "Installed_power"] * power_increase)* self.Capex.loc[self.timestep, "Wind Onshore"])/25

        if investment_cost < profit_outlook:
            ess.agents["Producer 6"]["data"].loc[self.timestep, "Installed_power"] = ess.agents["Producer 6"]["data"].loc[self.timestep, "Installed_power"] * (1 + power_increase)
        
        print("Max Power Amount:")
        print(self.max_power.iloc[1,5])
                
        #update efficiency
        ess.agents["Producer 6"]["data"].loc[self.timestep, "Efficiency"] = ess.agents["Producer 6"]["data"].loc[self.timestep-1, "Efficiency"]

        #calculate total emissions with calculated PyPSA data, wind has no emissions in operation
        ess.agents["Producer 6"]["data"].loc[self.timestep, "Total_emissions"] = 0
    
    
    def agent8(self):
        """
        Function of Producer 7.
        This producer resembels the combined water power plants
        """
        print("Producer 7 called")
        
        #update marginal cost
        ess.agents["Producer 7"]["data"].loc[self.timestep, "Marginal_cost"] = ess.agents["Producer 7"]["data"].loc[self.timestep-1, "Marginal_cost"]
        
        #update installed_power
        ess.agents["Producer 7"]["data"].loc[self.timestep, "Installed_power"] = ess.agents["Producer 7"]["data"].loc[self.timestep-1, "Installed_power"]

        #fill generated_power_total with calculated PyPSA data
        ess.agents["Producer 7"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.iloc[self.timestep,6]

        #calculate Income with calculated PyPSA data
        ess.agents["Producer 7"]["data"].loc[self.timestep, "Income"] = ess.agents["Producer 7"]["data"].loc[self.timestep, "Generated_power_total"]*self.Electricity_cost_pure

        #update money
        ess.agents["Producer 7"]["data"].loc[self.timestep, "Money"] = ess.agents["Producer 7"]["data"].loc[self.timestep-1, "Money"] + ess.agents["Producer 7"]["data"].loc[self.timestep, "Income"] 

        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy = ess.agents["Producer 7"]["data"].loc[self.timestep, "Installed_power"] * 365 * 24
        
        #calculate load and fill data
        ess.agents["Producer 7"]["data"].loc[self.timestep, "Load"] = round(ess.agents["Producer 7"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy,4)

        #no inverstment function because capacity in germany has been reached

        #update efficiency
        ess.agents["Producer 7"]["data"].loc[self.timestep, "Efficiency"] = ess.agents["Producer 7"]["data"].loc[self.timestep-1, "Efficiency"]

        #calculate total emissions with calculated PyPSA data, hydro has no emissions in operation
        ess.agents["Producer 7"]["data"].loc[self.timestep, "Total_emissions"] = 0
    
        
    def agent9(self):
        """
        Function of Producer 8.
        This producer resembels the combined biomass power plants
        """
        print("Producer 8 called")
        
        #update marginal cost
        ess.agents["Producer 8"]["data"].loc[self.timestep, "Marginal_cost"] = self.Operational_costs.loc[0,"Biomass"]
        
        #update installed_power
        ess.agents["Producer 8"]["data"].loc[self.timestep, "Installed_power"] = ess.agents["Producer 8"]["data"].loc[self.timestep-1, "Installed_power"]

        #fill generated_power_total with calculated PyPSA data
        ess.agents["Producer 8"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.iloc[self.timestep,7]

        #calculate Income with calculated PyPSA data
        ess.agents["Producer 8"]["data"].loc[self.timestep, "Income"] = ess.agents["Producer 8"]["data"].loc[self.timestep, "Generated_power_total"]*self.Electricity_cost_pure

        #update money
        ess.agents["Producer 8"]["data"].loc[self.timestep, "Money"] = ess.agents["Producer 8"]["data"].loc[self.timestep-1, "Money"] + ess.agents["Producer 8"]["data"].loc[self.timestep, "Income"] 

        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy = ess.agents["Producer 8"]["data"].loc[self.timestep, "Installed_power"] * 365 * 24
        
        #calculate load and fill data
        ess.agents["Producer 8"]["data"].loc[self.timestep, "Load"] = round(ess.agents["Producer 8"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy,4)

        #ToDo: Investment function???

        #update efficiency
        ess.agents["Producer 8"]["data"].loc[self.timestep, "Efficiency"] = ess.agents["Producer 8"]["data"].loc[self.timestep-1, "Efficiency"]

        #calculate total emissions with calculated PyPSA data, esstimated that emissions is always 230 kg CO2 per MWh
        ess.agents["Producer 8"]["data"].loc[self.timestep, "Total_emissions"] = ess.agents["Producer 8"]["data"].loc[self.timestep, "Generated_power_total"] * 230
            
        
    def agent10(self):
        """
        Function of Producer 9.
        This producer resembels the combined oil power plants
        """
        print("Producer 9 called")
        
        #update marginal cost
        ess.agents["Producer 9"]["data"].loc[self.timestep, "Marginal_cost"] = ess.agents["Producer 9"]["data"].loc[self.timestep-1, "Marginal_cost"]
        
        #update installed_power
        ess.agents["Producer 9"]["data"].loc[self.timestep, "Installed_power"] = ess.agents["Producer 9"]["data"].loc[self.timestep-1, "Installed_power"]

        #fill generated_power_total with calculated PyPSA data
        ess.agents["Producer 9"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.iloc[self.timestep,8]

        #calculate Income with calculated PyPSA data
        ess.agents["Producer 9"]["data"].loc[self.timestep, "Income"] = ess.agents["Producer 9"]["data"].loc[self.timestep, "Generated_power_total"]*self.Electricity_cost_pure

        #update money
        ess.agents["Producer 9"]["data"].loc[self.timestep, "Money"] = ess.agents["Producer 9"]["data"].loc[self.timestep-1, "Money"] + ess.agents["Producer 9"]["data"].loc[self.timestep, "Income"] 

        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy = ess.agents["Producer 9"]["data"].loc[self.timestep, "Installed_power"] * 365 * 24
        
        #calculate load and fill data
        ess.agents["Producer 9"]["data"].loc[self.timestep, "Load"] = round(ess.agents["Producer 9"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy,4)

        #ToDo: Investment function???

        #update efficiency
        ess.agents["Producer 9"]["data"].loc[self.timestep, "Efficiency"] = ess.agents["Producer 9"]["data"].loc[self.timestep-1, "Efficiency"]

        #calculate total emissions with calculated PyPSA data, esstimated that emissions is always 890 kg CO2 per MWh
        ess.agents["Producer 9"]["data"].loc[self.timestep, "Total_emissions"] = ess.agents["Producer 9"]["data"].loc[self.timestep, "Generated_power_total"] * 890
         

    def agent11(self):
        """
        This agent represents the storage of the electricity grid.
        Currently there is only battery storage in the network.
        ToDo: Implement the other storages e.g. water pump storage etc.
        """
        print("Storage called")
 
        #update capacity   
        ess.agents["Storage"]["data"].loc[self.timestep, "Battery_cap"] = ess.agents["Storage"]["data"].loc[self.timestep-1, "Battery_cap"]

        #update power in   
        ess.agents["Storage"]["data"].loc[self.timestep, "Battery_in"] = ess.agents["Storage"]["data"].loc[self.timestep-1, "Battery_in"]

        #update power out     
        ess.agents["Storage"]["data"].loc[self.timestep, "Battery_out"] = ess.agents["Storage"]["data"].loc[self.timestep-1, "Battery_out"]

        #update efficiency
        ess.agents["Storage"]["data"].loc[self.timestep, "Battery_eff"] = ess.agents["Storage"]["data"].loc[self.timestep-1, "Battery_eff"]

        #ToDo: change increase to dynamic and not constant   
        #ToDo: Eine funktionalität einbauen, die die größe und speicherleistung des speichers erhöht, wenn eine zeit überbrückt werden muss.
        #-> wenn in einem zeitschritt viel erneuerbarer stom zur verfügung steht und in manchen momenten keiner dann erhöhe specs

        #water pump storage
        #update capacity  
        ess.agents["Storage"]["data"].loc[self.timestep, "Pump_cap"] = ess.agents["Storage"]["data"].loc[self.timestep-1, "Pump_cap"]

        #update power in
        ess.agents["Storage"]["data"].loc[self.timestep, "Pump_in"] = ess.agents["Storage"]["data"].loc[self.timestep-1, "Pump_in"]

        #update power out        
        ess.agents["Storage"]["data"].loc[self.timestep, "Pump_out"] = ess.agents["Storage"]["data"].loc[self.timestep-1, "Pump_out"]

        #update efficiency
        ess.agents["Storage"]["data"].loc[self.timestep, "Pump_eff"] = ess.agents["Storage"]["data"].loc[self.timestep-1, "Pump_eff"]


    def agent12(self):
        """
        Function of consumer 1.
        This consumer resembels the combined industrial consumers
        """
        print("Consumer 1 called")
        
        #ToDo: make demand increase depended of average_price, but "real" demand isnt just dependend of cost, how could i implement this?
        #maybe the demand has the same base level (because the demand doesnt realy change much in the last 20 years) only the cost de or increase the demand.
        #implement a electricity price, at which the consumers change there behaviour and invest in energy efficiency and own generatition
        
        #power demand stays the same
        ess.agents["Consumer 1"]["data"].loc[self.timestep, "Power_demand"] = ess.agents["Consumer 1"]["data"].loc[self.timestep-1, "Power_demand"]
        
        #update income, if not touched, next income same as this years
        ess.agents["Consumer 1"]["data"].loc[self.timestep, "Income"] = ess.agents["Consumer 1"]["data"].loc[self.timestep-1, "Income"]

        #substract electricity costs from income, ToDo: wird negativ
        ess.agents["Consumer 1"]["data"].loc[self.timestep, "Income"] = ess.agents["Consumer 1"]["data"].loc[self.timestep, "Income"] - ess.agents["Consumer 1"]["data"].loc[self.timestep, "Power_demand"]*ess.agents["Government"]["data"].loc[self.timestep-1, "Electricity_cost"]

        #add income to money, other expensise decrease income by 70%, Assumption
        ess.agents["Consumer 1"]["data"].loc[self.timestep, "Money"] = ess.agents["Consumer 1"]["data"].loc[self.timestep-1, "Money"]+ess.agents["Consumer 1"]["data"].loc[self.timestep, "Income"]*0.3

        #check if electricity price increase is to high
        #ToDo: wahrscheinlich is es besser nicht nur die kosten des letzten jahres zu nehmen sondern die durchschnittskosten der letzten 5 jahre
        if self.Electricity_cost_proportion > 1:
            print("Zu hohe Preissteigerung!")
            #ToDo: Hier können jetzt investitionen oder nachfrage änderungen eingebaut werden
            
            #demand decrease by 2%
            #ess.agents["Consumer 1"]["data"].loc[self.timestep, "Power_demand"] = ess.agents["Consumer 1"]["data"].loc[self.timestep-1, "Power_demand"]*0.98 

    def agent13(self):
        """
        Function of consumer 2.
        This consumer resembels the combined commerce consumers
        """
        print("Consumer 2 called")
        
        #power demand stays the same
        ess.agents["Consumer 2"]["data"].loc[self.timestep, "Power_demand"] = ess.agents["Consumer 2"]["data"].loc[self.timestep-1, "Power_demand"]

        #update income
        ess.agents["Consumer 2"]["data"].loc[self.timestep, "Income"] = ess.agents["Consumer 2"]["data"].loc[self.timestep-1, "Income"]

        #substract electricity costs from income
        ess.agents["Consumer 2"]["data"].loc[self.timestep, "Income"] = ess.agents["Consumer 2"]["data"].loc[self.timestep, "Income"] - ess.agents["Consumer 2"]["data"].loc[self.timestep, "Power_demand"]*ess.agents["Government"]["data"].loc[self.timestep-1, "Electricity_cost"]

        #get income, other expensise decrease income by 70%, Assumption
        ess.agents["Consumer 2"]["data"].loc[self.timestep, "Money"] = ess.agents["Consumer 2"]["data"].loc[self.timestep-1, "Money"]+ess.agents["Consumer 2"]["data"].loc[self.timestep, "Income"]*0.3

        #check if electricity price increase is to high
        if self.Electricity_cost_proportion > 1.5:
            print("Zu hohe Preissteigerung!")
            #ToDo: Hier können jetzt investitionen oder nachfrage änderungen eingebaut werden
            
            #demand decrease by 1%
            #ess.agents["Consumer 2"]["data"].loc[self.timestep, "Power_demand"] = ess.agents["Consumer 2"]["data"].loc[self.timestep-1, "Power_demand"]*0.99

    def agent14(self):
        """
        Function of consumer 3.
        This consumer resembels the combined private consumers
        """
        print("Consumer 3 called")

        #ToDo: der demand geht seit 2009 runter, aber durch e autos etc, wird der strombedarf in der zukunft steigen...
        #ToDo: Investment funktion einfügen, wenn der strompreis so hoch ist, dass sich ein investment in verringerter demand oder eigene erzeugung lohnt

        #power demand stays the same
        ess.agents["Consumer 3"]["data"].loc[self.timestep, "Power_demand"] = ess.agents["Consumer 3"]["data"].loc[self.timestep-1, "Power_demand"]
    
        #update income
        #Assumption average income of households increases by 3% every year
        ess.agents["Consumer 3"]["data"].loc[self.timestep, "Income"] = ess.agents["Consumer 3"]["data"].loc[self.timestep-1, "Income"]*1.03

        #substract electricity costs from income
        ess.agents["Consumer 3"]["data"].loc[self.timestep, "Income"] = ess.agents["Consumer 3"]["data"].loc[self.timestep, "Income"] - ess.agents["Consumer 3"]["data"].loc[self.timestep, "Power_demand"]*ess.agents["Government"]["data"].loc[self.timestep-1, "Electricity_cost"]

        #get income, other expensise decrease income by 80%, Assumption
        ess.agents["Consumer 3"]["data"].loc[self.timestep, "Money"] = ess.agents["Consumer 3"]["data"].loc[self.timestep-1, "Money"]+ess.agents["Consumer 3"]["data"].loc[self.timestep, "Income"]*0.2
 
        #check if electricity price increase is to high
        if self.Electricity_cost_proportion > 0.5:
            #ToDo: Hier können jetzt investitionen oder nachfrage änderungen eingebaut werden
            print("Zu hohe preissteigerung!")
            #demand decrease by 10%, for the current timestep, ToDo: Demand should increase again when consumer get used to price
            #ess.agents["Consumer 3"]["data"].loc[self.timestep, "Power_demand"] = ess.agents["Consumer 3"]["data"].loc[self.timestep-1, "Power_demand"]*0.9
            
        #abfrage ob eine investition sich lohnt
        #wenn strompreis * demand > investition über 20 jahre -> investieren in niedrigeren verbrauch/effizienz oder eigene erzeugung
        #-> das gleiche bei den anderen verbrauchern mit anderen trigger punkten
        
        electricity_cost_consumers = ess.agents["Consumer 3"]["data"].loc[self.timestep, "Power_demand"] * self.Electricity_cost
        
        investment_cost_3pers_generation_25_year = ((ess.agents["Consumer 3"]["data"].loc[self.timestep, "Power_demand"] * 0.03) * self.Capex.loc[self.timestep, "Solar Residential"])/25
        print("3% of demand:")
        print(ess.agents["Consumer 3"]["data"].loc[self.timestep, "Power_demand"] * 0.03)
        print("Capex of solar residential")
        print(self.Capex.loc[self.timestep, "Solar Residential"])
        
        
        print("Investment costs over 25 years for 3% of the demand:")
        print(investment_cost_3pers_generation_25_year)
        print("Electricity costs in this year total:")
        print(electricity_cost_consumers)
        
        if electricity_cost_consumers > investment_cost_3pers_generation_25_year:
            print("Invest!")

        #ToDo: implement own generation with generation profil

#init class
start_year=2009
end_year=2021
ess = ESS(start_year=start_year,end_year=end_year)

#init agents
ess.init_agents()

# Initialize profiles
pv_generation_profile, wind_generation_profile, load_profile = ess.init_profiles()

#print all the agents in the model
print("Agent List:", ess.get_agent_list())

#run simulation
ess.run_sim(pv_generation_profile, wind_generation_profile, load_profile)

#plot some variables

#Generated Power
ess.agents["Producer 1"]["data"]["Generated_power_total"][2:end_year-start_year-1].plot(title="Generated Power")
ess.agents["Producer 2"]["data"]["Generated_power_total"][2:end_year-start_year-1].plot()
ess.agents["Producer 3"]["data"]["Generated_power_total"][2:end_year-start_year-1].plot()
ess.agents["Producer 4"]["data"]["Generated_power_total"][2:end_year-start_year-1].plot()
ess.agents["Producer 5"]["data"]["Generated_power_total"][2:end_year-start_year-1].plot()
ess.agents["Producer 6"]["data"]["Generated_power_total"][2:end_year-start_year-1].plot()
ess.agents["Producer 7"]["data"]["Generated_power_total"][2:end_year-start_year-1].plot()
ess.agents["Producer 8"]["data"]["Generated_power_total"][2:end_year-start_year-1].plot()
ess.agents["Producer 9"]["data"]["Generated_power_total"][2:end_year-start_year-1].plot()
plt.show()

#installed power
ess.agents["Producer 1"]["data"]["Installed_power"][2:end_year-start_year-1].plot(title="Installed Power")
ess.agents["Producer 2"]["data"]["Installed_power"][2:end_year-start_year-1].plot()
ess.agents["Producer 3"]["data"]["Installed_power"][2:end_year-start_year-1].plot()
ess.agents["Producer 4"]["data"]["Installed_power"][2:end_year-start_year-1].plot()
ess.agents["Producer 5"]["data"]["Installed_power"][2:end_year-start_year-1].plot()
ess.agents["Producer 6"]["data"]["Installed_power"][2:end_year-start_year-1].plot()
ess.agents["Producer 7"]["data"]["Installed_power"][2:end_year-start_year-1].plot()
ess.agents["Producer 8"]["data"]["Installed_power"][2:end_year-start_year-1].plot()
ess.agents["Producer 9"]["data"]["Installed_power"][2:end_year-start_year-1].plot()
plt.show()

#Load
ess.agents["Producer 1"]["data"]["Load"][2:end_year-start_year-1].plot(title="Load")
ess.agents["Producer 2"]["data"]["Load"][2:end_year-start_year-1].plot()
ess.agents["Producer 3"]["data"]["Load"][2:end_year-start_year-1].plot()
ess.agents["Producer 4"]["data"]["Load"][2:end_year-start_year-1].plot()
ess.agents["Producer 5"]["data"]["Load"][2:end_year-start_year-1].plot()
ess.agents["Producer 6"]["data"]["Load"][2:end_year-start_year-1].plot()
ess.agents["Producer 7"]["data"]["Load"][2:end_year-start_year-1].plot()
ess.agents["Producer 8"]["data"]["Load"][2:end_year-start_year-1].plot()
ess.agents["Producer 9"]["data"]["Load"][2:end_year-start_year-1].plot()
plt.show()

#co2 Costs
ess.Co2_certificate_costs.plot(title="CO2 Costs")
plt.show()

#ressource costs
ess.gas_costs.plot(legend = True)
ess.coal_costs.plot()
ess.uranium_costs.plot()
plt.show()

#Marginal costs
ess.agents["Producer 1"]["data"]["Marginal_cost"][2:end_year-start_year-1].plot(title="Marginal costs")
ess.agents["Producer 2"]["data"]["Marginal_cost"][2:end_year-start_year-1].plot()
ess.agents["Producer 3"]["data"]["Marginal_cost"][2:end_year-start_year-1].plot()
ess.agents["Producer 4"]["data"]["Marginal_cost"][2:end_year-start_year-1].plot()
ess.agents["Producer 5"]["data"]["Marginal_cost"][2:end_year-start_year-1].plot()
ess.agents["Producer 6"]["data"]["Marginal_cost"][2:end_year-start_year-1].plot()
ess.agents["Producer 7"]["data"]["Marginal_cost"][2:end_year-start_year-1].plot()
ess.agents["Producer 8"]["data"]["Marginal_cost"][2:end_year-start_year-1].plot()
ess.agents["Producer 9"]["data"]["Marginal_cost"][2:end_year-start_year-1].plot()
plt.show()

#Total_emissions
ess.agents["Producer 1"]["data"]["Total_emissions"][2:end_year-start_year-1].plot(title="Emissions")
ess.agents["Producer 2"]["data"]["Total_emissions"][2:end_year-start_year-1].plot()
ess.agents["Producer 3"]["data"]["Total_emissions"][2:end_year-start_year-1].plot()
ess.agents["Producer 4"]["data"]["Total_emissions"][2:end_year-start_year-1].plot()
ess.agents["Producer 5"]["data"]["Total_emissions"][2:end_year-start_year-1].plot()
ess.agents["Producer 6"]["data"]["Total_emissions"][2:end_year-start_year-1].plot()
ess.agents["Producer 7"]["data"]["Total_emissions"][2:end_year-start_year-1].plot()
ess.agents["Producer 8"]["data"]["Total_emissions"][2:end_year-start_year-1].plot()
ess.agents["Producer 9"]["data"]["Total_emissions"][2:end_year-start_year-1].plot()
ess.agents["Government"]["data"]["Total_emissions"][2:end_year-start_year-1].plot()
plt.show()

#Money
ess.agents["Producer 1"]["data"]["Money"][2:end_year-start_year-1].plot(title="Money")
ess.agents["Producer 2"]["data"]["Money"][2:end_year-start_year-1].plot()
ess.agents["Producer 3"]["data"]["Money"][2:end_year-start_year-1].plot()
ess.agents["Producer 4"]["data"]["Money"][2:end_year-start_year-1].plot()
ess.agents["Producer 5"]["data"]["Money"][2:end_year-start_year-1].plot()
ess.agents["Producer 6"]["data"]["Money"][2:end_year-start_year-1].plot()
ess.agents["Producer 7"]["data"]["Money"][2:end_year-start_year-1].plot()
ess.agents["Producer 8"]["data"]["Money"][2:end_year-start_year-1].plot()
ess.agents["Producer 9"]["data"]["Money"][2:end_year-start_year-1].plot()
plt.show()

#Power demand
ess.agents["Consumer 1"]["data"]["Power_demand"][2:end_year-start_year-1].plot(title="Power demand")
ess.agents["Consumer 2"]["data"]["Power_demand"][2:end_year-start_year-1].plot()
ess.agents["Consumer 3"]["data"]["Power_demand"][2:end_year-start_year-1].plot()
ess.agents["Government"]["data"]["Total_demanded_power"][2:end_year-start_year-1].plot()
plt.show()

#electricity cost
ess.agents["Government"]["data"]["Electricity_cost"][2:end_year-start_year-1].plot(title="Electricity price")
plt.show()

#electricity cost pure
ess.agents["Government"]["data"]["Electricity_cost_pure"][2:end_year-start_year-1].plot(title="Electricity price pure")
plt.show()


#eLIBOR
ess.agents["Bank"]["data"]["LIBOR_index"][2:end_year-start_year-1].plot(title="LIBOR index")
plt.show()

#define variables to be viewed in the varible explorer
agent_list = ess.get_agent_list()
agent0=ess.agents["Government"]["data"]
agent1=ess.agents["Bank"]["data"]
agent2=ess.agents["Producer 1"]["data"]
agent3=ess.agents["Producer 2"]["data"]
agent4=ess.agents["Producer 3"]["data"]
agent5=ess.agents["Producer 4"]["data"]
agent6=ess.agents["Producer 5"]["data"]
agent7=ess.agents["Producer 6"]["data"]
agent8=ess.agents["Producer 7"]["data"]
agent9=ess.agents["Producer 8"]["data"]
agent10=ess.agents["Producer 9"]["data"]

agent11=ess.agents["Storage"]["data"]
agent12=ess.agents["Consumer 1"]["data"]
agent13=ess.agents["Consumer 2"]["data"]
agent14=ess.agents["Consumer 3"]["data"]

#print final computing time
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(round(executionTime,2)))