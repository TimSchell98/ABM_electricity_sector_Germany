# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:48:48 2023

@author: Tim Schell
"""

    def agent2(self):
        """
        Function of producer 1.
        This producer resembels the combined coal power plants
        """
        print("\nCoal called")
    
#check if initial installed power is higher than 0
if self.agents["Producer 1"]['data'].loc[self.timestep-1, 'Installed_power_initial'] > 0:
    #initial installed power is reduced by a fixed amount every year
    self.agents["Producer 1"]['data'].loc[self.timestep, 'Installed_power_initial'] = self.agents["Producer 1"]["data"].loc[self.timestep-1, 'Installed_power_initial'] - self.coal_deprication

#calculate the total installed power
self.agents["Producer 1"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Producer 1"]["data"].loc[self.timestep, "Installed_power_initial"] + self.agents["Producer 1"]["Investments"].loc[self.agents["Producer 1"]["Investments"]['Status'] == True, 'Block_size'].sum()

        #update efficiency, increase by 0.1% every year
        #todo: function einbauen die dafür sorgt, dass der wirkungsgrad bis 41% steigt und danach so bleibt
        self.agents["Producer 1"]["data"].loc[self.timestep, "Efficiency"] = self.agents["Producer 1"]["data"].loc[self.timestep-1, "Efficiency"] + 0.001    
        
        #update marginal cost
        #calculate co2 costs
        #calculate t co2 emittet by producing 1 MWh
        co2_mwh = self.co2_emissions_coal / self.agents["Producer 1"]["data"].loc[self.timestep, "Efficiency"]
        
        #calculate co2 costs
        co2_costs = self.Co2_certificate_costs.iloc[self.timestep,0] * co2_mwh
    
        #calculate mwh per tonne of coal
        mwh_t_coal = (self.energy_content_coal * self.agents["Producer 1"]["data"].loc[self.timestep, "Efficiency"])/3.6
    
        #calculate ressource costs
        ressource_cost = self.coal_costs.iloc[self.timestep,0]/mwh_t_coal
    
        #calculate marginal costs  
        marginal_cost = ressource_cost + co2_costs + self.Operational_costs.loc[0,"Coal"]
        
        #write marginal cost in dataframe
        self.agents["Producer 1"]["data"].loc[self.timestep, "Marginal_cost"] = marginal_cost

        #fill generated_power_total with calculated PyPSA data
        self.agents["Producer 1"]["data"].loc[self.timestep, "Generated_power_total"] = self.generated_power.loc["Producer 1"]
        
        #set income with the income of the last timestep from PyPSA
        self.agents["Producer 1"]["data"].loc[self.timestep, "Income"] = self.coal_income_total

        #set expenses from the fix costs
        self.agents["Producer 1"]["data"].loc[self.timestep, "Expenses"] = self.agents["Producer 1"]["data"].loc[self.timestep, "Expenses"]-self.Fix_costs.loc["Coal"]*self.agents["Producer 1"]["data"].loc[self.timestep, "Installed_power"]

        #calculate profit
        self.agents["Producer 1"]["data"].loc[self.timestep, "Profit"] = self.agents["Producer 1"]["data"].loc[self.timestep, "Income"] + self.agents["Producer 1"]["data"].loc[self.timestep, "Payback"] + self.agents["Producer 1"]["data"].loc[self.timestep, "Expenses"]

        #update money
        self.agents["Producer 1"]["data"].loc[self.timestep, "Money"] = self.agents["Producer 1"]["data"].loc[self.timestep-1, "Money"] + self.agents["Producer 1"]["data"].loc[self.timestep, "Profit"] 

        #investment function
        #calculate theoratical energy to calculate load of producer, installed power from the last timestep, because in PyPSA also the previous installed power is used
        theoratical_energy_total = self.agents["Producer 1"]["data"].loc[self.timestep, "Installed_power"] * 365 * 24
        
        #calculate average load in the last time step
        self.agents["Producer 1"]["data"].loc[self.timestep, "Load"] = round(self.agents["Producer 1"]["data"].loc[self.timestep, "Generated_power_total"]/theoratical_energy_total,4)
        
#check if load is bigger than 1
if self.agents["Producer 1"]["data"].loc[self.timestep, "Load"] > 1:
    #if yes set load to 1
    self.agents["Producer 1"]["data"].loc[self.timestep, "Load"] = 1
    
        #calculate outlook of profit
        income_outlook = sum((self.coal_profile * 100) * (self.marginal_costs-self.agents["Producer 1"]["data"].loc[self.timestep-1, "Marginal_cost"]))
        
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
        self.agents["Producer 1"]["data"].loc[self.timestep, "Irr"] = irr

        #calcualte shift, increases when interste rate rises
        shift = float((self.coal_shift-np.exp(150*((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"]+self.Risk_markup["Production Coal"])-0.15))).iloc[0])
        
        #calculate power_increase with sigmoid function
        power_increase = self.coal_max * self.sigmoid(irr,self.coal_slope,shift)

#round power increase to a block size
power_increase = (power_increase // self.coal_block_size) * self.coal_block_size

#check if power increase is more than 0
if power_increase > 0:
    #safe power increase in power increase dataframe
    self.agents["Producer 1"]['Investments'].loc[self.timestep, 'Block_size'] += power_increase
   
    #set lifetime
    self.agents["Producer 1"]['Investments'].loc[self.timestep, 'Lifetime'] = self.coal_lifetime
    
    #set status
    self.agents["Producer 1"]['Investments'].loc[self.timestep, 'Status'] = True
       
            #calculate interest rate 
            interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Coal"]).iloc[0])
            
            #amount
            amount = power_increase * self.Capex.loc[self.timestep, "Coal"]
        
            #calculate interest
            interest = float(amount * interest_rate)    
        
            #take on loan and write that into the loan list
            loans = {'Agent_name': ["Producer 1"], 'Active': [True], 'Runtime': [self.coal_lifetime], 'Amount': [amount], 'Interest_rate': [interest_rate], 'Payback': [interest]}
            self.loans_temp = pd.DataFrame(data = loans)
            
            #add to loans list
            self.loans = pd.concat([self.loans, self.loans_temp], ignore_index=True)
       
#calculate the total installed power
self.agents["Producer 1"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Producer 1"]["data"].loc[self.timestep, "Installed_power_initial"] + self.agents["Producer 1"]["Investments"].loc[self.agents["Producer 1"]["Investments"]['Status'] == True, 'Block_size'].sum()

        #calculate t co2 emittet by producing 1 MWh
        co2_mwh = self.co2_emissions_coal / self.agents["Producer 1"]["data"].loc[self.timestep, "Efficiency"]
        
        #calculate total emissions with calculated PyPSA data
        self.agents["Producer 1"]["data"].loc[self.timestep, "Total_emissions"] = self.agents["Producer 1"]["data"].loc[self.timestep, "Generated_power_total"] * co2_mwh 
