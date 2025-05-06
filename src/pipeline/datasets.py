import hashlib
import os
from torch.utils.data import Dataset
import pickle
from src.pipeline import FactoryLoader


datasets = {
    'av-redscience': 'txt/av',
    'factorio-tech-json': 'json/factorio-tech',
    'factorio-tech': 'csv/factorio-tech',
    'factorio-codex': 'csv/factorio-codex',
    'idan': 'csv/idan_blueprints.csv',
}

class ChunkedDiskCachedDatasetWrapper(Dataset):
    def __init__(self, base_dataset, cache_dir='dataset_cache', chunk_size=100, force_rebuild=False):
        self.base_dataset = base_dataset
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        
        # Create a unique signature for this dataset
        dataset_signature = f"{type(base_dataset).__name__}_{len(base_dataset)}"
        self.signature = hashlib.md5(dataset_signature.encode()).hexdigest()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Store length
        self.length = len(base_dataset)
        
        # Initialize cache structures
        self.chunk_map = {}  # Maps index to (chunk_id, offset)
        self.cached_chunks = {}  # Maps chunk_id to file path
        self.loaded_chunks = {}  # In-memory cache of loaded chunks
        
        # Create chunk mapping
        for i in range(self.length):
            chunk_id = i // chunk_size
            offset = i % chunk_size
            self.chunk_map[i] = (chunk_id, offset)
        
        # Scan existing cache
        self._scan_cache()
        
        # Clear if forced
        if force_rebuild:
            self._clear_cache()

    @classmethod
    def from_cache(cls, cache_dir='dataset_cache'):
        """
        Create a dataset instance directly from cache without the original dataset.
        
        Args:
            cache_dir: Directory where the dataset is cached
            
        Returns:
            A new instance of ChunkedDiskCachedDatasetWrapper loaded from cache
        """
        # Create a new instance without calling __init__
        instance = cls.__new__(cls)
        instance.cache_dir = cache_dir
        instance.loaded_chunks = {}
        
        # Find all chunk files
        chunk_files = []
        for filename in os.listdir(cache_dir):
            if filename.endswith(".pkl") and "chunk_" in filename:
                chunk_files.append(os.path.join(cache_dir, filename))
        
        if not chunk_files:
            raise ValueError(f"No cached chunks found in {cache_dir}")
        
        # Extract signature from filenames
        # Format: {signature}_chunk_{chunk_id}.pkl
        signature = os.path.basename(chunk_files[0]).split('_chunk_')[0]
        instance.signature = signature
        
        # Sort chunk files by number and extract IDs
        chunk_files_with_ids = []
        for file_path in chunk_files:
            # Extract chunk ID from filename
            filename = os.path.basename(file_path)
            chunk_id = int(filename.split('_chunk_')[1].split('.')[0])
            chunk_files_with_ids.append((chunk_id, file_path))
        
        chunk_files_with_ids.sort()  # Sort by chunk ID
        
        # Load the first chunk to figure out chunk size
        with open(chunk_files_with_ids[0][1], 'rb') as f:
            first_chunk = pickle.load(f)
            chunk_size = len(first_chunk)
        
        # If there's only one chunk, length is just that chunk's length
        if len(chunk_files_with_ids) == 1:
            total_length = len(first_chunk)
        else:
            # Otherwise, check the last chunk (might be smaller)
            with open(chunk_files_with_ids[-1][1], 'rb') as f:
                last_chunk = pickle.load(f)
            # Calculate total length
            total_length = chunk_size * (len(chunk_files_with_ids) - 1) + len(last_chunk)
        
        # Initialize other required fields
        instance.length = total_length
        instance.chunk_size = chunk_size
        instance.cached_chunks = {chunk_id: path for chunk_id, path in chunk_files_with_ids}
        
        # Create chunk mapping
        instance.chunk_map = {}
        for i in range(total_length):
            chunk_id = i // chunk_size
            offset = i % chunk_size
            instance.chunk_map[i] = (chunk_id, offset)
        
        # No base dataset needed
        instance.base_dataset = None
        
        return instance
    
    def _get_chunk_path(self, chunk_id):
        return os.path.join(self.cache_dir, f"{self.signature}_chunk_{chunk_id}.pkl")
    
    def _scan_cache(self):
        """Scan the cache directory to find already cached chunks."""
        prefix = f"{self.signature}_chunk_"
        for filename in os.listdir(self.cache_dir):
            if filename.startswith(prefix) and filename.endswith(".pkl"):
                try:
                    chunk_id = int(filename[len(prefix):-4])
                    self.cached_chunks[chunk_id] = os.path.join(self.cache_dir, filename)
                except ValueError:
                    continue
    
    def _clear_cache(self):
        """Clear all cached chunks."""
        for chunk_id, path in self.cached_chunks.items():
            if os.path.exists(path):
                os.remove(path)
        self.cached_chunks = {}
        self.loaded_chunks = {}
    
    def _load_chunk(self, chunk_id):
        """Load a chunk into memory or build it if not cached."""
        if chunk_id in self.loaded_chunks:
            return
            
        chunk_path = self._get_chunk_path(chunk_id)
        
        # If cached on disk, load it
        if chunk_id in self.cached_chunks:
            with open(chunk_path, 'rb') as f:
                self.loaded_chunks[chunk_id] = pickle.load(f)
            return
            
        # Otherwise, build the chunk
        chunk_data = []
        start_idx = chunk_id * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.length)
        
        for i in range(start_idx, end_idx):
            item = self.base_dataset[i]
            chunk_data.append(item)
        
        # Save to disk and memory
        with open(chunk_path, 'wb') as f:
            pickle.dump(chunk_data, f)
        
        self.cached_chunks[chunk_id] = chunk_path
        self.loaded_chunks[chunk_id] = chunk_data
        
        # Simple LRU - keep only 3 chunks in memory
        if len(self.loaded_chunks) > 3:
            oldest = next(iter(self.loaded_chunks))
            if oldest != chunk_id:  # Don't remove the one we just loaded
                del self.loaded_chunks[oldest]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError("Index out of bounds")
            
        # Get chunk info
        chunk_id, offset = self.chunk_map[idx]
        
        # Ensure chunk is loaded
        self._load_chunk(chunk_id)
        
        # Return the item
        return self.loaded_chunks[chunk_id][offset]
    

def load_dataset(dataset_name: str='av-redscience',
                  **kwargs):
    """ dataset_name: The name of a prepared dataset. 
    """
    return FactoryLoader(datasets[dataset_name], **kwargs)