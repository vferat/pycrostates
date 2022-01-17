import pkg_resources
import pooch
import os
from ...utils._config import get_config


def load_data(subject, condition):
    config = get_config()
    path = config['PREPROCESSED_LEMON_DATASET_PATH']
    fetcher = pooch.create(
                path=path,
                base_url="https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed/",
                version=None,
                registry=None)
    registry = pkg_resources.resource_stream('pycrostates',
                                             os.path.join('datasets',
                                                          'lemon',
                                                          'data',
                                                          'PREPROCESSED_LEMON_registry.txt'))
    fetcher.load_registry(registry)

    filename_set = f'sub-{subject}_{condition}.set'
    filename_fdt = f'sub-{subject}_{condition}.fdt'
    output_path_set = fetcher.fetch(filename_set)
    output_path_fdt = fetcher.fetch(filename_fdt)
    return(output_path_set)
