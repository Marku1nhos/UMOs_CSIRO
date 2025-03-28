The purpose of this code is to match a satellite to a transient detected by CRACO.
It takes a fits rms image as input as well as the transients RADEC and detection time (mjd)
and overplots the satellites track on top of the image.

The code utilises query.py to fetch a recent TLE list, relevant to the detection time.
(Requires a space-track account)

There are unused unfunctions such as rotate_ra_dec from previous iterations of the program.
Also some redundancy that I havent cleaned up
