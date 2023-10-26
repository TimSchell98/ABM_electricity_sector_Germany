# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:48:31 2023

@author: Tim Schell
"""

    def agent2_prognosis(self):
        """
        Function of producer 1.
        This producer resembels the combined coal power plants
        """
        print("\nCoal prognosis")
        
        #update installed_power
        self.agents["Producer 1"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Producer 1"]["data"].loc[self.timestep-1, "Installed_power"]

        #update efficiency
        self.agents["Producer 1"]["data"].loc[self.timestep, "Efficiency"] = self.agents["Producer 1"]["data"].loc[self.timestep-1, "Efficiency"]
        
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
    
        #investment function
        #calculate outlook of profit
        income_outlook = sum((self.coal_profile * 100) * (self.marginal_costs-self.agents["Producer 1"]["data"].loc[self.timestep-1, "Marginal_cost"]))

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
        self.agents["Producer 1"]["data"].loc[self.timestep, "Irr prognose"] = irr
        
        #calcualte shift, increases when interste rate rises
        shift = float((self.coal_shift-np.exp(150*((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"]+self.Risk_markup["Production Coal"])-0.15)).iloc[0]))
        
        #calculate power_increase with sigmoid function
        power_increase = self.coal_max * self.sigmoid(irr,self.coal_slope,shift)
        
#round power increase to a block size
power_increase = (power_increase // self.coal_block_size) * self.coal_block_size       

#check if power increase is more than 0
if power_increase > 0:
#safe power increase in power increase dataframe
self.agents["Producer 1"]['Investments'].loc[self.timestep, 'Block_size'] +=  power_increase

#set lifetime
self.agents["Producer 1"]['Investments'].loc[self.timestep, 'Lifetime'] = self.coal_lifetime

#set status
self.agents["Producer 1"]['Investments'].loc[self.timestep, 'Status'] = True
       
            #calculate interest rate 
            interest_rate = float((self.agents["Bank"]["data"].loc[self.timestep-1, "Interest_rate"] + self.Risk_markup["Production Coal"]).iloc[0])
            
            #amount
            amount = power_increase * self.Capex.loc[self.timestep, "Coal"]
        
            #payback
            payback = float(amount * interest_rate)
        
            #take on loan and write that into the loan list
            loans = {'Agent_name': ["Producer 1"], 'Active': [True], 'Runtime': [40], 'Amount': [amount], 'Interest_rate': [interest_rate], 'Payback': [payback]}
            self.loans_temp = pd.DataFrame(data = loans)
            
            #add to loans list
            self.loans = pd.concat([self.loans, self.loans_temp], ignore_index=True)
       
#calculate the total installed power
self.agents["Producer 1"]["data"].loc[self.timestep, "Installed_power"] = self.agents["Producer 1"]["data"].loc[self.timestep-1, "Installed_power_initial"] + self.agents["Producer 1"]["Investments"].loc[self.agents["Producer 1"]["Investments"]['Status'] == True, 'Block_size'].sum()
