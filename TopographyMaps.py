"""
Run Raman Map Generation
========================

This script uses the Raman analysis toolkit to load a Raman map from .txt,
process the data cube, and visualize the total intensity map.
"""

from RamanMap_toolkit import load_raman_txt, plot_raman_map

if __name__ == "__main__":
    # Example file path (adjust to your actual location)
    file_path = "data/St kC CLs/Map St kC CL 14 Region 2.txt"

    # Load the data cube and spectral axis
    data_cube, raman_shift = load_raman_txt(file_path)

    # Visualize the total intensity Raman map
    plot_raman_map(data_cube, title="Total Raman Intensity – St kC CL 14 Region 2")
