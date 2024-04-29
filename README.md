# ABM_electricity_sector_Germany
A agent based model of the German electricity sector

This model was created in the master thesis "Modeling the German electricity economy - Strategy analysis for overcoming crises and advancing the energy transition".

It is referenced in the paper "Crisis as a Chance - Strategy analysis with agent-based modeling for advancing the energy transition"

The main model can be found in the script "ABM_GES_6", the calibration script is named "analysis_ABM".

The nessesary librarys are: PyPSA, Numpy financial, Pandas, Numpy and Matplotlib.

All nessesary datasets are available as Excel sheets.

The sources of these datasets are listed in the following:

1. Installed power 2009 - 2021: https://www.smard.de; https://energy-charts.info
2. Power consumption projection: V. Gustedt, B. Greve, C. Brehm, C. Halici, "Netzentwicklungsplan Strom 2037 mit Ausblick 2045, Version 2023; Zweiter Entwurf der Übertragungsnetzbetreiber", 2023
3. Power consumption 2009 - 2021: https://www.smard.de; Umwelt Bundesamt, "Stromverbrauch"
4. Electicity costs 2009 - 2021: https://energy-charts.info
5. Generation and load profile: https://data.open-power-system-data.org
6. OPEX: NREL, "2016 Annual Technology Baseline"; Trinomics, "Final Report Cost of Energy (LCOE): Energy costs, taxes and the impact of government interventions on investments"; iea, "Projected Costs of Generating Electricity"; Statista, "Fixed and variable costs for the operation and maintenance of new power plants in the United States as of 2021, by technology type"
7. Ressource price projection, carbon dioxide certificate cost projection: F. Birol, R. Priddle, "World Energy Outlook 2016"
8. Coal price 2009 - 2021: Energy institute, "Energy Institute based on S&P Global Platts - Statistical Review of World Energy"
9. Gas price 2009 - 2021: Statista Research Department, Entwicklung des Kraftwerkpreises für Erdgas in Deutschland in den Jahren 1973 bis 2022 (in Euro je Tonne Steinkohleeinheit)
10. Carbon dioxide certificate cost: Trading Economics, "EU Carbon Permits"
11. Uranium costs 2009 - 2021: International Monetary Fund, "Uranium / US Dollar"
12. LIBOR 2009 - 2021: https://www.global-rates.com, LIBOR ZINSSATZ
