# ABM_electricity_sector_Germany
A agent based model of the German electricity sector

This model was created in the master thesis "Modeling the German electricity economy - Strategy analysis for overcoming crises and advancing the energy transition".

It is referenced in the paper *REFERENZ EINFÜGEN*

The main model can be found in the script "ABM_GES", the calibration script is named "calibration_ABM_GES".

The nessesary librarys are: PyPSA, Numpy financial, Pandas, Numpy and Matplotlib.

All nessesary datasets are available as Excel sheets.

The sources of these datasets are listed in the following:

1. Installed power 2009 - 2021: https://www.smard.de; https://energy-charts.info
2. Power consumption projection: V. Gustedt, B. Greve, C. Brehm, C. Halici, "Netzentwicklungsplan Strom 2037 mit Ausblick 2045, Version 2023; Zweiter Entwurf der Übertragungsnetzbetreiber", 2023
3. Power consumption 2009 - 2021: https://www.smard.de; Umwelt Bundesamt, "Stromverbrauch"
4. Electicity costs 2009 - 2021: https://energy-charts.info
5. Generation and load profile: https://data.open-power-system-data.org
6. OPEX: NREL, "2016 Annual Technology Baseline"; Trinomics, "Final Report Cost of Energy (LCOE): Energy costs, taxes and the impact of government interventions on investments"; iea, "Projected Costs of Generating Electricity"; Statista, "Fixed and variable costs for the operation and maintenance of new power plants in the United States as of 2021, by technology type"
7. CAPEX: NREL (National Renewable Energy Laboratory). 2023. "2023 Annual Technology Baseline." Golden, CO: National Renewable Energy Laboratory. https://atb.nrel.gov/. 
8. Ressource price projection (coal, gas), carbon dioxide certificate cost projection: F. Birol, R. Priddle, "World Energy Outlook 2016"
9. Coal price 2009 - 2021: Energy institute, "Energy Institute based on S&P Global Platts - Statistical Review of World Energy"
10. Gas price 2009 - 2021: Statista Research Department, Entwicklung des Kraftwerkpreises für Erdgas in Deutschland in den Jahren 1973 bis 2022 (in Euro je Tonne Steinkohleeinheit)
11. Uranium price 2009 - 2021: International Monetary Fund, "Uranium / US Dollar"
12. Carbon dioxide certificate price 2009 - 2021: Trading Economics, "EU Carbon Permits"
13. LIBOR 2009 - 2021: https://www.global-rates.com, LIBOR ZINSSATZ


Guide:
*GUIDE EINFÜGEN*


License:

You are free to use, modify, and distribute this code for any purpose, commercial or non-commercial, as long as you give appropriate credit to Tim Schell.

Attribution can be provided by mentioning Tim Schell in your project documentation or README file.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
