"""
Run Raman Map Generation
========================

This script uses the Raman analysis toolkit to load a Raman map from .txt,
process the data cube, and visualize the total intensity map.
"""

import RamanMap_toolkit as rm

if __name__ == "__main__":

    # Load raw map
    file_path = "data/St kC CLs/Map St kC CL 14 Region 2.txt"
    raw_map = rm.load_raman_txt(file_path)

    # Preprocess the map
    processed_maps = rm.preprocess_maps([raw_map], region=(250, 1800), win_len=15)
    processed_map = processed_maps[0]

    # Plot total intensity map
    rm.plot_raman_map(processed_map, title="Total Raman Intensity – Preprocessed")
