from src.pipeline import FactoryLoader

datasets = {
    'av-redscience': 'txt/av',
    'factorio-tech-json': 'json/factorio-tech',
    'factorio-tech': 'csv/factorio-tech',
    'factorio-codex': 'csv/factorio-codex',
    'idan': 'csv/idan_blueprints.csv',
}

def load_dataset(dataset_name: str='av-redscience',
                  **kwargs):
    """ dataset_name: The name of a prepared dataset. 
    """
    return FactoryLoader(datasets[dataset_name])