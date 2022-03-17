# CDCF2022
A very naive matlab implementation.
### dependency
This tracker works with Matconvnet toolbox, please install it.
### introduction 
We offer two entries: one for demo[I], one for VOT toolbox[II]. Some codes are copied from [Joao Henriques](https://www.robots.ox.ac.uk/~joao/) and the [Matconvnet toolbox](https://www.vlfeat.org/matconvnet/).
### structure
[I]: demo.m + run_CDCF.m + other functions
[II]:cdcf.m + vot.m + other functions
### tips
1.This demo entry support both gpu and cpu mode, which can be swithed in demo.m. But make sure that you have compiled the corresponding Matconvnet support, this matters.
2.All functions are given, just copy it into a new folder.
3.run_CDCF.m is not exactly an OTB entry, but it's not hard to change.
