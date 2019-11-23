# Stormy Forerunners
The tool tracks storms by forerunners of Swell inspired by Walter Munk's 1947 paper. It estimates the radial distance and initial time of the swell based on spectrograms. The spectrograms can be observed by wave buoys, seismic stations, or any other point observations of swell. 

The module combines the dispersion slope and parameteric models of wave spectra (JONSWAP or Pierson Moskowitz) to contruct at two dimensional model function that is fitted to swell events in spectrograms. 

The fitting procedure has 4 stages that are 
- event selection
- preparation 
- model fitting
- collection and filtering

requirements:
