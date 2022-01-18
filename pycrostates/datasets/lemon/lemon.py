import pkg_resources
import pooch
import os
from ...utils._config import get_config


def load_data(subject_id, condition):
    """Get path to local copy of preprocessed EEG recording
    from the mind-brain-body dataset of MRI, EEG, cognition, emotion, 
    and peripheral physiology in young and old adults dataset files.
    If there is no local copy of the recording, this will fetch it from
    the online repository and store it on disk.

    Parameters
    ----------
    subject_id : str
        The subject id to use.
        For example '010276'.
        The list of available subjects can be found
        at <https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/>.
    condition : str
        Can be 'EO' for resting state, eyes open condition 
        or 'EC' for  resting state, eyes closed condition.
    Returns
    -------
    path : str
        path to local copy of the requested recording .

    References
    ----------
    .. [1] `Babayan, A., Erbey, M., Kumral, D., Reinelt, J. D., Reiter, A. M., Röbbig, J., ... & Villringer, A. (2019).
       "A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults."
       Scientific data, 6(1), 1-21.
       <https://doi.org/10.1038/sdata.2018.308>`_
    """
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

    filename_set = f'sub-{subject_id}_{condition}.set'
    filename_fdt = f'sub-{subject_id}_{condition}.fdt'
    output_path_set = fetcher.fetch(filename_set)
    output_path_fdt = fetcher.fetch(filename_fdt)
    return(output_path_set)
