Tests are built in the following way :  
We replicate directly the intended instance in a txt file, with `@` corresponding to robots' initial positions, `#` to obstacles, and positive floats to the weight of each cell.

Every cell is separated from the next one by a space or a linebreak. The grid needs to be rectangular to be correctly processed by `from_txt.py`