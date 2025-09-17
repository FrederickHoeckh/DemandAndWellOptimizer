Skripte:
Demand Estimator.py
	- estimates demand for a give date
	- requires temperature for every day, still to do: script to produce temperature timeseries
	- returns per head demand (example calculation on the bottom)

demand2well.py 
	- distributes demand from script above on wells, must be provided (see example calculations at the end)
	- restrictions of individual wells or BWV possible

changeWellsFromDiffs.py
	- check headfile (GW40.hds) for water levels below thresholds (can be defined in folder "WellData")
		so far thresholds are just some values for showcasing
	- pumping rates are adapted based on sensitivities (see folder WellData "hq(...).csv", only use the ones with "sc")
	- produces file "new_well_rates.csv" in folder WellRates