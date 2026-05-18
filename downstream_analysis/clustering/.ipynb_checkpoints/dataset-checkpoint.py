import os
from glob import glob
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, DataLoader


class VideoMaskDataset(Dataset):
    def __init__(self, project_dir):
        self.project_dir = project_dir

        # Get mask sql DBs
        mask_sqlite_files = glob(os.path.join(self.project_dir, "videos", "**", "masks.sqlite"))
        
        # Stratify by camera view
        project_folder_names = [Path(x).parent.name for x in mask_sqlite_files]

        # if split x \in project_folder_names by '_', you get [hash, camera_view, ...]
        # so in order to get camera_view, take x.split('_')[1]
        camera_view_per_project = [x.split('_')[1] for x in project_folder_names]

        self.project_folders = project_folder_names
        self.camera_views    = camera_view_per_project
        self.mask_sql_dbs    = mask_sqlite_files
        