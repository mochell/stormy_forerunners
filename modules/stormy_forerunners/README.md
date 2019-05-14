# stormy_forerunners
The tool tracks storms by forerunners of Swell (Munk, 1947)

Any data from point observations of waves tha resolves dipersed swell arrivals can be used for this method. 
The module unitilized the dispersion slope and parameteric models of wave spectra (JONSWAP or Pierson Moskowitz) to contruct at two dimensional model function that is fitted to swell events in spectrograms. 

The fitting procedure has 4 stages that are 
- selecting events.
- preparation of the data
- model fitting
- collection and filtering of the results


requirements:
