
from src.model import BinaryMatrixTransformCNN, AttentiveBinaryMatrixTransformCNN, FactorioCNN_PixelOutput
from src.training import train_model
import itertools
from src.training import *
from src.pipeline.datasets import RotationalDataset, prepare_and_save_seed_dataset, MatrixTupleDataset
import warnings
warnings.filterwarnings("ignore", message="Unknown entity 'ee-infinity-loader'")

# You should be able to just run this.
filename = 'data/processed/paths.npz'
dims = (20,20)

def get_dataloader(filename, **kwargs):
    matrix_dataset = MatrixTupleDataset(filename)
    rotational_datalist = RotationalDataset(matrix_dataset)
    print(f"Dataset size is: {len(rotational_datalist)}")

    dataloader = DataLoader(
        rotational_datalist,
        batch_size=kwargs.get('batch_size', 256), 
        collate_fn=collate_numpy_matrices,
        num_workers=0,
        pin_memory=True
    )
    return dataloader


if __name__ == "__main__":
    if not os.path.exists(filename):
        prepare_and_save_seed_dataset(filename, dims, repr_version=5, seed_paths=10)

    dataloader = get_dataloader(filename)

    train_dataloader, test_dataloader = split_dataloader(dataloader, val_split=0.2)
    train_dataloader, val_dataloader = split_dataloader(train_dataloader, val_split=0.2)

    input_size = dims[0]

    # underscore as to not shadow str
    for str_, bias in itertools.product([0.7, 1, 1.05], [-0.1, 0, 0.1]):
        moodel = FactorioCNN_PixelOutput(4, 4, 5, 3, 3, 2, 32, 21, 
                                        presence_gate_strength=str_,
                                        presence_gate_bias=bias)
        moodel.filename += f"_pgst{str_}_bias{bias}"
        train_model(moodel, train_loader=train_dataloader,
                    val_loader=val_dataloader,
                    integrity_weight=0.0, num_epochs=20, device='cuda')
        del(moodel)  # great farm in the sky