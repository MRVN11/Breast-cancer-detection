import os
import pandas as pd
import pathlib as path

def main() -> None:

    csv_root = ".data/CBIS_data/Ori_csv"
    Img_root = "data/CBIS_data/CBIS_DDSM"
    csv_output_path = "data/CBIS_data/CBIS_DDSM_mask"
