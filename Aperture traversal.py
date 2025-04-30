import fitsio
import numpy as np
import sep
import glob
from natsort import natsorted
import concurrent.futures
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
from photutils.aperture import CircularAperture, CircularAnnulus
from scipy.ndimage import gaussian_filter1d
import logging
import gc
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import imageio  # New: for saving 16-bit PNG images
from astropy.io import fits  # Import for saving FITS files
from astropy.stats import sigma_clip
import tables

# Set up logging
logging.basicConfig(level=logging.INFO)

def save_fits(filename, data):
    """
    Save a 2D NumPy array as a FITS file.

    Parameters:
    - filename: Path to save the file.
    - data: 2D NumPy array to save.
    """
    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename, overwrite=True)
    logging.info(f"Saved FITS image: {filename}")

# Define plotting function to display the best apertures and save as 16-bit PNG
def plot_apertures_on_star(data, star, aperture_radius, inner_radius, outer_radius, output_dir, star_id, snr=None, sigma_x=None, sigma_y=None):

    x, y = star['x'], star['y']

    # Define the region size around the star (for display purposes)
    size = 64
    x_min = max(0, int(x) - size)
    x_max = min(data.shape[1], int(x) + size)
    y_min = max(0, int(y) - size)
    y_max = min(data.shape[0], int(y) + size)

    # Extract the sub-image
    sub_data = data[y_min:y_max, x_min:x_max]

    # Create the image
    fig, ax = plt.subplots()
    # Use percentiles to normalize the display
    ax.imshow(sub_data, cmap='gray', origin='lower', vmin=np.percentile(sub_data, 5), vmax=np.percentile(sub_data, 95))

    # Draw aperture circles
    aperture_circle = Circle((x - x_min, y - y_min), aperture_radius, color='red', fill=False, lw=1.5, label='Aperture')
    inner_circle = Circle((x - x_min, y - y_min), inner_radius, color='blue', fill=False, lw=1.5, label='Inner Ring')
    outer_circle = Circle((x - x_min, y - y_min), outer_radius, color='green', fill=False, lw=1.5, label='Outer Ring')

    # Draw ellipse for +/- one standard deviation
    if sigma_x is not None and sigma_y is not None:
        stddev_ellipse = Ellipse((x - x_min, y - y_min), width=2*sigma_x, height=2*sigma_y, angle=0, edgecolor='yellow', facecolor='none', lw=1.5, label='1σ')
        ax.add_patch(stddev_ellipse)

    ax.add_patch(aperture_circle)
    ax.add_patch(inner_circle)
    ax.add_patch(outer_circle)

    if snr is not None:
        ax.set_title(f"Star ID: {star_id}, SNR={snr:.2f}")
    else:
        ax.set_title(f"Star ID: {star_id}")
    ax.legend()

    # Save the image
    plot_filename = os.path.join(output_dir, f"aperture_plot_star_{int(star_id)}.png")
    fig.savefig(plot_filename, dpi=300)
    plt.close(fig)  # Close the figure to free memory
    logging.info(f"Saved aperture image: {plot_filename}")

# Read initial image and process the background
data_path = r"/public1/home/a8s000159/stacked/n_522_PID_172666001731860890997070_from_cal_172665985390888894877501_TrES5-0001.fit"

data = fitsio.read(data_path)
data = data.astype(np.float32)  # Ensure the data type is float32
bkg = sep.Background(data)
data_sub = data - bkg.back()

# Detect objects and generate segmentation map
threshold = 5.0  # Detection threshold

objects = sep.extract(data_sub, thresh=threshold, err=bkg.globalrms)

print(f"Number of objects detected: {len(objects)}")
logging.info(f"Number of objects detected: {len(objects)}")

# Create directories to save segmentation and cutout images
output_dir = "images2"
os.makedirs(output_dir, exist_ok=True)

cutout_dir = "cutouts2"
os.makedirs(cutout_dir, exist_ok=True)

# Initialize list to collect object information
stars_list = []

# Assign unique ID to each detected object
star_id = 0

# Iterate over the detected objects
for obj in objects:
    try:
        x_obj, y_obj = float(obj['x']), float(obj['y'])
        x_int, y_int = int(x_obj), int(y_obj)

        # Define the required region size
        required_size = 64  # Half size 64, i.e., region of 128x128

        # Check if there is enough region around the object
        if (x_int - required_size < 0 or x_int + required_size >= data_sub.shape[1] or
            y_int - required_size < 0 or y_int + required_size >= data_sub.shape[0]):
            # Skip this object if the region is not large enough (128x128)
            continue

        # Extract subimage (only if the region size condition is met)
        x_min = x_int - required_size
        x_max = x_int + required_size
        y_min = y_int - required_size
        y_max = y_int + required_size

        sub_data = data_sub[y_min:y_max, x_min:x_max]

        # Check if sub-image is valid
        if sub_data.size == 0:
            continue

        # Calculate FWHM
        a_obj, b_obj = obj['a'], obj['b']
        fwhm_x, fwhm_y = 2.3548 * a_obj, 2.3548 * b_obj
        fwhm_avg = (fwhm_x + fwhm_y) / 2.0

        # Get standard deviation
        sigma_x = fwhm_x / 2.355
        sigma_y = fwhm_y / 2.355

        # Set a minimum for FWHM
        fwhm_obj = max(fwhm_avg, 2.0)

        # Prevent FWHM from being too small or too large
        if fwhm_obj <= 0 or fwhm_obj > 20:
            continue

        # Save sub-image as a FITS file
        cutout_filename = os.path.join(cutout_dir, f"cutout_star_{star_id}.fits")
        save_fits(cutout_filename, sub_data)  # Save as FITS

        # Add object info to the list, including cutout filename
        stars_list.append({
            'star_id': star_id,
            'x': x_obj,
            'y': y_obj,
            'sigma_x': sigma_x,
            'sigma_y': sigma_y,
            'fwhm': fwhm_obj,
            'cutout_filename': cutout_filename
        })
        star_id += 1  # Update star ID

    except (RuntimeError, OptimizeWarning) as e:
        logging.warning(f"Failed to fit Gaussian for star at ({x_obj}, {y_obj}): {str(e)}")
        continue  # Ensure continue inside the loop

# Create DataFrame
stars_df = pd.DataFrame(stars_list)
print(f"Number of stars to process: {len(stars_df)}")
logging.info(f"Number of stars to process: {len(stars_df)}")

# Save stars_df including cutout_filename
stars_df.to_csv('stars_fwhm_cutouts.csv', index=False)

# Define aperture factors and new outer ring offsets
aperture_factors = np.arange(2.0, 9.0, 0.1)
inner_factors = np.arange(1.2, 2.0, 0.1)
outer_offsets = np.arange(3, 16, 1)

# Get all FITS file paths and sort them
image_files = natsorted(glob.glob('/public1/home/a8s000159/aligned/*.fit'))
print("Number of image files:", len(image_files))
logging.info(f"Number of image files: {len(image_files)}")

# Check if image files are found
if not image_files:
    logging.error("No image files found. Please check the directory path and file extensions.")
    import sys
    sys.exit(1)

# Prepare parameter list, include stars_df instead of stars_list
args_list = [(filename, stars_df.to_dict('records'), aperture_factors, inner_factors, outer_offsets) for filename in image_files]

def process_file(args):
    filename, stars, aperture_factors, inner_factors, outer_offsets = args
    try:
        # Read the image and perform background subtraction
        data = fitsio.read(filename).astype(np.float32)
        bkg = sep.Background(data)
        data_sub = data - bkg.back()
        err = bkg.globalrms

        # Initialize a list to store the current file's flux data
        flux_data = []

        # Split the star list into smaller batches
        batch_size = 4000  # Adjust based on memory
        for i in range(0, len(stars), batch_size):
            stars_batch = stars[i:i + batch_size]
            for star in stars_batch:
                star_id = star['star_id']  # Get star_id
                x = star['x']
                y = star['y']
                fwhm = star['fwhm']

                position = (x, y)

                for aperture_factor in aperture_factors:
                    aperture_radius = aperture_factor * fwhm / 2.0

                    for inner_factor in inner_factors:
                        inner_radius = inner_factor * aperture_radius

                        for outer_offset in outer_offsets:
                            outer_radius = inner_radius + outer_offset

                            # No need to check inner_radius < outer_radius because outer_offset > 0
                            # Perform photometry
                            flux, flux_err, flag = sep.sum_circle(
                                data_sub, np.array([x]), np.array([y]), np.array([aperture_radius]),
                                err=err, gain=1.0
                            )
                            inner_flux, _, _ = sep.sum_circle(
                                data_sub, np.array([x]), np.array([y]), np.array([inner_radius]),
                                err=err, gain=1.0
                            )
                            outer_flux, _, _ = sep.sum_circle(
                                data_sub, np.array([x]), np.array([y]), np.array([outer_radius]),
                                err=err, gain=1.0
                            )

                            # Calculate background flux correction
                            sky_background = (outer_flux[0] - inner_flux[0]) / (np.pi * (outer_radius**2 - inner_radius**2))
                            corrected_flux = flux[0] - sky_background * (np.pi * aperture_radius**2)

                            # Store flux values, including star_id
                            flux_entry = {
                                'star_id': star_id,  # Add star_id
                                'aperture_factor': aperture_factor,
                                'inner_factor': inner_factor,
                                'outer_offset': outer_offset,
                                'aperture_radius': aperture_radius,
                                'inner_radius': inner_radius,
                                'outer_radius': outer_radius,
                                'corrected_flux': corrected_flux
                            }
                            flux_data.append(flux_entry)

        # Convert the current file's flux data to a DataFrame and return
        if flux_data:
            df = pd.DataFrame(flux_data)
            # Free up memory
            del data, bkg, data_sub, err
            gc.collect()
            return df  # Return DataFrame
        else:
            # Free up memory
            del data, bkg, data_sub, err
            gc.collect()
            return None

    except Exception as e:
        logging.error(f"Error processing file {filename}: {str(e)}")
        return None

# Process all files in parallel, limiting max workers to control memory usage
max_workers = 96  # Adjust based on system memory
with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    results = list(executor.map(process_file, args_list))

# Filter out None results
flux_dfs = [df for df in results if df is not None]

# Check if there are results
if not flux_dfs:
    logging.error("No flux data was processed. Exiting.")
    import sys
    sys.exit(1)

# Combine all DataFrames into one large DataFrame
flux_df = pd.concat(flux_dfs, ignore_index=True)

print(f"Number of flux measurements: {len(flux_df)}")
logging.info(f"Number of flux measurements: {len(flux_df)}")

# Ensure flux_df contains 'star_id' column
if 'star_id' not in flux_df.columns:
    logging.error("flux_df is missing 'star_id' column. Please check if 'process_file' function correctly added 'star_id'.")
    raise KeyError("flux_df is missing 'star_id' column.")

# Initialize list to store optimal aperture information
optimal_apertures = []

# Calculate the best aperture combination for each star
unique_stars = flux_df['star_id'].unique()

for star_id in unique_stars:
    # Filter flux measurements for the current star
    star_flux = flux_df[flux_df['star_id'] == star_id]

    if star_flux.empty:
        continue

    # Calculate SNR: mean_flux / std_flux
    grouped = star_flux.groupby(['aperture_factor', 'inner_factor', 'outer_offset'])
    snr_df = grouped['corrected_flux'].agg(['mean', 'std'])
    snr_df['snr'] = snr_df['mean'] / snr_df['std']

    # Remove cases where std_flux is 0 or negative
    snr_df = snr_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['snr'])

    if snr_df.empty:
        continue

    # Find the combination with the highest SNR
    optimal_params = snr_df['snr'].idxmax()
    optimal_snr = snr_df.loc[optimal_params, 'snr']
    optimal_mean_flux = snr_df.loc[optimal_params, 'mean']
    optimal_std_flux = snr_df.loc[optimal_params, 'std']
    optimal_aperture_factor, optimal_inner_factor, optimal_outer_offset = optimal_params

    # Get corresponding fwhm and cutout filename
    star_info = stars_df[stars_df['star_id'] == star_id]
    if star_info.empty:
        continue
    fwhm = star_info['fwhm'].values[0]
    cutout_filename = star_info['cutout_filename'].values[0]

    # Calculate aperture radii
    optimal_aperture_radius = optimal_aperture_factor * fwhm / 2.0
    optimal_inner_radius = optimal_inner_factor * optimal_aperture_radius
    optimal_outer_radius = optimal_inner_radius + optimal_outer_offset

    optimal_apertures.append({
        'star_id': star_id,
        'x': star_info['x'].values[0],
        'y': star_info['y'].values[0],
        'aperture_radius': optimal_aperture_radius,
        'inner_radius': optimal_inner_radius,
        'outer_radius': optimal_outer_radius,
        'snr': optimal_snr,
        'mean_flux': optimal_mean_flux,
        'std_flux': optimal_std_flux,
        'cutout_filename': cutout_filename
    })

    # Call plot function to save the image
    star_info_dict = star_info.iloc[0].to_dict()
    plot_apertures_on_star(
        data_sub, star_info_dict,
        optimal_aperture_radius, optimal_inner_radius, optimal_outer_radius,
        output_dir, star_id, optimal_snr
    )

    # Free memory promptly
    del star_flux
    gc.collect()

# Save the optimal aperture parameters to DataFrame
optimal_apertures_df = pd.DataFrame(optimal_apertures)
output_csv_path = "optimal_apertures.csv"
optimal_apertures_df.to_csv(output_csv_path, index=False)

