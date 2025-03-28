import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from sgp4.api import SGP4_ERRORS
import numpy as np
from datetime import datetime, timedelta, timezone
from skyfield.api import EarthSatellite, load, wgs84
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from astropy.io import fits

"""
The purpose of this code is to match a satellite to a transient detected by CRACO.
It takes the fits rms image as input as well as the transients RADEC and detection time (mjd)
and overplots the satellites track on top of the image.

The code utilises query.py to fetch a recent TLE list, relevant to the detection time.

There are unused unfunctions such as rotate_ra_dec from previous iterations of the program.
"""

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #


beam = 'b13'
prefix = 'SB070398.'


transient_ra = 313.45636
transient_dec = -80.28773
end_time = 60697.36836711

fits_file =f'{beam}.fits'
output_file = f"{prefix}{beam}.png"
p_title = f'{prefix}{fits_file}'
input_tle_file = "tle_data.csv"
output_candidates_file = f"candidates_{beam}.csv" 
celestial_r = 10000000
tolerance = 2.5


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

def which_integration(mjd):
    if mjd < 60645.48934975:
        integ_t = 0.0138
        frame_offset = 80
    else:
        integ_t = 0.00346
        frame_offset = 200
    
    return (integ_t * frame_offset) / 2


def mjd_to_datetime(mjd):
    # MJD epoch: midnight on November 17, 1858
    mjd_epoch = datetime(1858, 11, 17, tzinfo=timezone.utc)
    # Convert MJD to datetime
    return mjd_epoch + timedelta(days=mjd)

def rotate_ra_dec():
    """
        Performs a matrix transform to calcualte a unit vector in the direction
        of the source position

        Returns a scaled source position vector
    """

    # Unit vector pointing from Earth's center to observed RA/DEC transient location
    x = np.cos(np.radians(transient_dec)) * np.cos(np.radians(transient_ra))
    y = np.cos(np.radians(transient_dec)) * np.sin(np.radians(transient_ra))
    z = np.sin(np.radians(transient_dec))

    # Vector times set magnitute
    source_position = np.array([x, y, z]) * celestial_r

    # Multiply unit vector by a set magnitude
    return source_position

def get_telescope_vectors():
    # ASKAP telescope coordinates
    lat, lon, alt = -26.706798, 116.66104878, 247.6  # Latitude, longitude, altitude (in meters)
    telescope = wgs84.latlon(lat, lon, alt)
    at_time = mjd_to_datetime(end_time)
    # Convert datetime to Timescale object
    ts = load.timescale()
    time = ts.utc(at_time.year, at_time.month, at_time.day, at_time.hour, at_time.minute, at_time.second)

    return telescope.at(time).position.km, telescope.at(time).velocity.km_per_s

def angle_between_vectors(A, B):
    """
        Returns the angle (in degrees) between two vectors
    """
    # Normalize the vectors
    A_normalized = A / np.linalg.norm(A)
    B_normalized = B / np.linalg.norm(B)
    
    # Calculate the dot product
    dot_product = np.dot(A_normalized, B_normalized)
    
    # Calculate the cross product to determine the direction of the angle
    cross_product = np.cross(A_normalized, B_normalized)
    
    # Check the sign of the z-component of the cross product
    sign = np.sign(cross_product[2])  # This gives us the sign of the angle's direction
    
    # Calculate the angle in radians
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip to handle numerical precision issues
    
    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad*sign)
    
    # The signed angle direction
    return angle_deg

def propagate_tle(tle_line1, tle_line2):
    """
        Propagates input TLE up to end_time
        Returns final Cartesian position and velocity, and True Anomaly
    """
    at_time = mjd_to_datetime(end_time)
    ts = load.timescale()
    satellite = EarthSatellite(tle_line1, tle_line2, "TLE Satellite", ts)

    # Propagate to end time
    sf_time = ts.utc(at_time.year, at_time.month, at_time.day, at_time.hour, at_time.minute, at_time.second)
    geocentric = satellite.at(sf_time)
    # Get the satellite position in Cartesian coordinates (km)
    position = geocentric.position.km
    velocity = geocentric.velocity.km_per_s


    return position, velocity

def compute_apparent_v(telescope_to_sat, rel_v):
    """
        Calculates:
            - Distance between telescope and satellite
            - Satellites perpendicular velocity relative to the telescope
            - Relative angular velocity
        
        Returns the angular velocity of the satellite relative to the telescope (arcseconds)
    """
    # Distance between telescope and satellite
    distance = np.linalg.norm(telescope_to_sat)
    # Compute the perpendicular component of the relative velocity
    telescope_to_sat_u = telescope_to_sat / distance  # Unit vector of relative position
    v_parallel = np.dot(rel_v, telescope_to_sat_u) * telescope_to_sat_u  # Parallel velocity component
    v_perpendicular = rel_v - v_parallel  # Perpendicular velocity component
    # Compute the magnitude of the perpendicular velocity
    v_perpendicular_mag = np.linalg.norm(v_perpendicular)
    # Compute the angular velocity in radians per second
    omega_rad = v_perpendicular_mag / distance
    # Convert angular velocity to arcseconds per second
    angular_velocity_arcsec = omega_rad * (180 * 3600 / np.pi)

    return angular_velocity_arcsec

def process_tles():
    # Read TLE data
    tle_data = pd.read_csv(input_tle_file)

    # Validate TLE columns
    if 'TLE_LINE1' not in tle_data.columns or 'TLE_LINE2' not in tle_data.columns:
        raise ValueError("Input TLE file must have 'TLE_LINE1' and 'TLE_LINE2' columns")

    # Initialize output dataframe
    candidates = []
    # Get Earth-centered Cartesian (ECEF) coordinates in meters
    telescope_vector, telescope_vel = get_telescope_vectors()
    source_vector = rotate_ra_dec()
    telescope_to_source = source_vector - telescope_vector
    
    # Process TLEs  
    for idx, row in tle_data.iterrows():
        line1 = row['TLE_LINE1']
        line2 = row['TLE_LINE2']
        name = row['OBJECT_NAME']
        pos, sat_vel = propagate_tle(line1, line2)

        rel_v = sat_vel - telescope_vel

        telescope_to_sat = pos - telescope_vector

        angle = angle_between_vectors(telescope_to_sat, telescope_to_source)

        # Apply tolerance check
        if abs(angle) <= tolerance:

            ang_v = compute_apparent_v(telescope_to_sat, rel_v)

            candidate = {
                "Index": idx,
                "Object Name": name,
                "Angle": angle * 3600,
                "Relative Ang V": ang_v,
                "TLE1": line1,
                "TLE2": line2
            }
            candidates.append(candidate)

    # Save candidates to CSV

    if candidates:
        candidates_df = pd.DataFrame(candidates)
        candidates_df.to_csv(output_candidates_file, index=False)
        print(f"Candidates saved to {output_candidates_file}")
        return candidates
    else:
        print("No candidates found.")

def track_prop(candidates):
    end_time_dt = mjd_to_datetime(end_time)
    # Generate timestamps: [end_time - 2*time_step, ..., end_time + 2*time_step]
    timestamps = [
        end_time_dt - timedelta(seconds=2 * time_step_seconds),
        end_time_dt - timedelta(seconds=time_step_seconds),
        end_time_dt,
        end_time_dt + timedelta(seconds=time_step_seconds),
        end_time_dt + timedelta(seconds=2 * time_step_seconds),
    ]


    candidate_tracks = []

    # Load Skyfield timescale
    ts = load.timescale()

    # Convert timestamps to Skyfield time objects
    skyfield_times = [ts.from_datetime(t) for t in timestamps]

    # Propagate and convert to alt/az for each candidate
    for candidate in candidates:
        alt_az_points = []
        print(f"Processing {candidate['Object Name']} (Index: {candidate['Index']})")

        # Load satellite from TLE
        satellite = EarthSatellite(candidate["TLE1"], candidate["TLE2"], candidate["Object Name"], ts)

        for time in skyfield_times:
            # Propagate satellite position
            geocentric = satellite.at(time)

            # Get telescope vectors
            telescope_position, telescope_velocity = get_telescope_vectors()

            # Compute relative position vector (ECI)
            relative_position = geocentric.position.km - telescope_position

            # Convert to alt/az
            lat, lon, alt = -26.706798, 116.66104878, 247.6  # Latitude, longitude, altitude (in meters)
            observer = wgs84.latlon(lat, lon, alt).at(time)
            # Compute the satellite's apparent position as seen by the observer
            difference = geocentric - observer
            topocentric = difference
            alt, az, distance = topocentric.altaz()
            alt_az_points.append((alt, az, distance, candidate['Object Name']))

            # Print or store results
            #print(f"  Time: {time.utc_iso()} | Alt: {alt.degrees:.2f}° | Az: {az.degrees:.2f}° | Distance: {distance.km:.2f} km")

        ts = load.timescale()
        lat, lon, alt = -26.706798, 116.66104878, 247.6  # Telescope's latitude, longitude, altitude in meters
        observer = wgs84.latlon(lat, lon, alt)
        curr_cand_track = []
        # Convert Alt/Az to RA/Dec
        for (alt, az, dist, name), time in zip(alt_az_points, timestamps):
            # Convert datetime to Skyfield Time object
            skyfield_time = ts.utc(time)

            # Create observer at the specific time
            observer_at_time = observer.at(skyfield_time)

            # Ensure alt and az are floats
            alt = alt.degrees
            az = az.degrees

            # Convert Alt/Az to RA/Dec (ICRF)
            apparent = observer_at_time.from_altaz(alt_degrees=alt, az_degrees=az)
            ra, dec, _ = apparent.radec()

            curr_cand_track.append((ra, dec, dist, name))
            print(name, ra._degrees, dec.degrees, time, alt, az)

            # Print results
            #print(f"Alt: {alt:.2f}°, Az: {az:.2f}° at {skyfield_time.utc_iso()} -> RA: {ra.hours:.6f}h, Dec: {dec.degrees:.6f}°")
            
        candidate_tracks.append(curr_cand_track)
    
    return candidate_tracks
        

def draw_tracks(ctracks):
    # Load the FITS file and extract image data and WCS
    with fits.open(fits_file) as hdul:
        image_data = hdul[0].data  # Assuming the image data is in the primary HDU
        wcs = WCS(hdul[0].header)  # Load WCS information

    # Create a WCS-aware figure and axes for the full plot
    fig_full = plt.figure(figsize=(10, 8))
    # Add padding at the top for the title
    fig_full.subplots_adjust(top=0.9)  
    ax_full = fig_full.add_subplot(111, projection=wcs)

    # Add title to full plot
    ax_full.set_title(p_title, pad=20, fontsize=14)

    # Plot the FITS image
    ax_full.imshow(
        image_data, origin='lower', cmap='gray',
        vmin=np.percentile(image_data, 10), vmax=np.percentile(image_data, 99)
    )
    ax_full.set_xlabel("Right Ascension")
    ax_full.set_ylabel("Declination")
    ax_full.coords.grid(True, color="white", ls="dotted")
    ax_full.coords[0].set_format_unit(u.hourangle)  # RA in HMS
    ax_full.coords[0].set_major_formatter("hh:mm:ss")
    ax_full.coords[1].set_format_unit(u.deg)  # Dec in DMS
    ax_full.coords[1].set_major_formatter("dd:mm:ss")

    # Create a WCS-aware figure and axes for the limited plot
    fig_limited = plt.figure(figsize=(10, 8))
    # Add padding at the top for the title
    fig_limited.subplots_adjust(top=0.9)
    ax_limited = fig_limited.add_subplot(111, projection=wcs)

    # Add title to limited plot
    ax_limited.set_title(p_title, pad=20, fontsize=14)

    ax_limited.imshow(
        image_data, origin='lower', cmap='gray',
        vmin=np.percentile(image_data, 10), vmax=np.percentile(image_data, 99)
    )
    ax_limited.set_xlabel("Right Ascension")
    ax_limited.set_ylabel("Declination")
    ax_limited.coords.grid(True, color="white", ls="dotted")
    ax_limited.coords[0].set_format_unit(u.hourangle)  # RA in HMS
    ax_limited.coords[0].set_major_formatter("hh:mm:ss")
    ax_limited.coords[1].set_format_unit(u.deg)  # Dec in DMS
    ax_limited.coords[1].set_major_formatter("dd:mm:ss")

    # Get a colormap for distinct track colors
    cmap = plt.get_cmap("Set1")
    num_tracks = len(ctracks)

    for idx, c in enumerate(ctracks):
        pixel_coords = []
        pixel_coords_limited = []

        for ra, dec, dist, name in c:
            sky_coord = SkyCoord(ra=ra._degrees * u.deg, dec=dec.degrees * u.deg, frame="icrs")
            x, y = wcs.world_to_pixel(sky_coord)
            pixel_coords.append((x, y))
            if 0 <= x < image_data.shape[1] and 0 <= y < image_data.shape[0]:
                pixel_coords_limited.append((x, y))

            #calc_power(name, dist)

        # Plot the full track
        if pixel_coords:
            x_coords, y_coords = zip(*pixel_coords)
            ax_full.plot(x_coords, y_coords, marker="o", linestyle="-", label=f"{c[0][3]}", color=cmap(idx / num_tracks))
            # Add t0 label to the first point
            ax_full.annotate('t0', 
                           xy=(x_coords[0], y_coords[0]),
                           xytext=(10, 10),  # 10 points offset
                           textcoords='offset points',
                           color=cmap(idx / num_tracks),
                           fontsize=8,
                           bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        # Plot the limited track
        if pixel_coords_limited:
            x_coords, y_coords = zip(*pixel_coords_limited)
            ax_limited.plot(x_coords, y_coords, marker="o", linestyle="-", label=f"{c[0][3]}", color=cmap(idx / num_tracks))

        # Add legends and position them outside the figure
    ax_full.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize="small", title="Tracks", borderaxespad=0.
    )
    fig_full.tight_layout()
    fig_full.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig_full)
    print(f"Full image saved as {output_file}")

    ax_limited.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize="small", title="Tracks", borderaxespad=0.
    )
    fig_limited.tight_layout()
    limited_output_file = f"limited_{output_file}"
    fig_limited.savefig(limited_output_file, dpi=300, bbox_inches="tight")
    plt.close(fig_limited)
    print(f"Limited image saved as {limited_output_file}")

time_step_seconds = which_integration(end_time)
candidates = process_tles()
if candidates != None:
    ctracks = track_prop(candidates)
    draw_tracks(ctracks)
else:
    print('No candidates')


