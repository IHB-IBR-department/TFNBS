import os
import shutil
import pickle
from pathlib import Path
import tempfile

def get_cache_dir(cache_dir: str | None = None) -> Path:
    """Resolve the cache directory for Neurosynth/NiMARE datasets.
    
    Checks in order:
    1. Provided ``cache_dir`` argument.
    2. Environment variable ``CONNINFPY_NIMARE_CACHE``.
    3. Default ``~/.conninfpy/nimare/``.
    """
    if cache_dir is not None:
        return Path(cache_dir)
    env_val = os.getenv("CONNINFPY_NIMARE_CACHE")
    if env_val:
        return Path(env_val)
    return Path.home() / ".conninfpy" / "nimare"

def fetch_neurosynth_dataset(cache_dir: str | None = None, *, force: bool = False):
    """One-shot fetch + Coordinate -> Dataset conversion.
    
    Downloads the Neurosynth database from NiMARE's online repository,
    converts it to a cached ``nimare.Dataset`` instance, and pickles it
    to disk. Writes atomically to prevent corruption on interruption.
    """
    try:
        import nimare
        from nimare.extract import fetch_neurosynth
        from nimare.io import convert_neurosynth_to_dataset
    except ImportError:
        raise ImportError("conninfpy.decode requires 'pip install conninfpy[decode]'")

    c_dir = get_cache_dir(cache_dir)
    c_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = c_dir / "neurosynth_dataset.pkl"

    if pkl_path.exists() and not force:
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError):
            # If the pickle is corrupted, rebuild
            pass

    # Fetch raw files
    raw_dir = c_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    db_files = fetch_neurosynth(data_dir=str(raw_dir), version="7", active_filters=["glove"])
    if not db_files:
        raise RuntimeError("Failed to fetch Neurosynth dataset from NiMARE.")
        
    files = db_files[0]

    # Convert coordinates & features to NiMARE Dataset (with fallback for NiMARE version compatibility)
    try:
        dataset = convert_neurosynth_to_dataset(
            coordinates_file=files['coordinates'],
            metadata_file=files['metadata'],
            annotations_files=files['features']
        )
    except TypeError:
        dataset = convert_neurosynth_to_dataset(
            text_file=files['features'],
            coordinate_file=files['coordinates']
        )

    # Write atomically
    temp_fd, temp_path = tempfile.mkstemp(dir=str(c_dir), suffix=".tmp")
    try:
        with os.fdopen(temp_fd, 'wb') as tmp:
            pickle.dump(dataset, tmp)
        os.replace(temp_path, pkl_path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

    return dataset


def fetch_neuroquery_dataset(cache_dir: str | None = None, *, force: bool = False):
    """Fetch a compact NeuroQuery source and cache it as a NiMARE dataset.

    NeuroQuery ships several large, overlapping vocabulary matrices. The
    combined TF-IDF ``neuroquery6308`` source is sufficient for decoding and
    keeps the first conversion practical for a desktop Streamlit session.
    """
    try:
        import nimare
        from nimare.extract import fetch_neuroquery
        from nimare.io import convert_neurosynth_to_dataset
    except ImportError:
        raise ImportError("conninfpy.decode requires 'pip install conninfpy[decode]'")

    c_dir = get_cache_dir(cache_dir)
    c_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = c_dir / "neuroquery_dataset.pkl"

    if pkl_path.exists() and not force:
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError):
            pass

    # Fetch raw files
    raw_dir = c_dir / "raw_nq"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    db_files = fetch_neuroquery(
        data_dir=str(raw_dir),
        version="1",
        source="combined",
        vocab="neuroquery6308",
        type="tfidf",
    )
    if not db_files:
        raise RuntimeError("Failed to fetch NeuroQuery dataset from NiMARE.")
        
    files = db_files[0]

    # Convert coordinates & features to NiMARE Dataset (with fallback for NiMARE version compatibility)
    try:
        dataset = convert_neurosynth_to_dataset(
            coordinates_file=files['coordinates'],
            metadata_file=files['metadata'],
            annotations_files=files['features']
        )
    except TypeError:
        dataset = convert_neurosynth_to_dataset(
            text_file=files['features'],
            coordinate_file=files['coordinates']
        )

    # Write atomically
    temp_fd, temp_path = tempfile.mkstemp(dir=str(c_dir), suffix=".tmp")
    try:
        with os.fdopen(temp_fd, 'wb') as tmp:
            pickle.dump(dataset, tmp)
        os.replace(temp_path, pkl_path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

    return dataset
