# syntheticdata

Pipelines to support transformation of factories into training data.

## Components

FactoryLoader: Gets everything from raw/txt/av/manifest.json from the `data_files` key, returns data as Blueprint objects. (Load the contents of each file and pass it to the draftsman utils function that loads it). Could also load them from something else. Pfft.

RepresentationTransformation: Takes a Blueprint object and runs it through a function defined in `representation.py` to get a numpy array.

SimplicityFilter: Filters any blueprint that has anything other than the 4 primary machines. 

EntityPuncher: Given a factory and some parameters, generates a bunch of blueprints that correspond to a single entity being removed.

