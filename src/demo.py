""" demo.py
Has functions for our demo.
"""
import torch

from src.representation import Factory, split_matrix_into_entities
from src.model import BinaryMatrixTransformCNN, FactorioCNN_PixelOutput, FactorioCNN_Repr5
from src.visualization import print_matrix_channels, print_factory, visualize_factory_matrix

input_size=20

model = FactorioCNN_Repr5()

model_name = 'factorio_unet_pgst1_bias-0.5.pt'
model.load_state_dict(torch.load(f'models/{model_name}'))


def produce_factory(input_str, number_of_fixes):
    F = Factory.from_str(input_str)
    O = F.get_tensor(dims=(input_size, input_size))
    for _ in range(number_of_fixes):
        O = model.predict(O)
    return O

if __name__ == "__main__":
    input_str = "0eNq1l2+PojAQxr9LX7cbC5Q/JvdJjDEVu9pcablSzjPG735TjUp2wW3Z7EvL8JuHYXhmPKOt6kVrpXZoeUayNrpDy9UZdXKvufJnmjcCLRHvOtFsldR70vD6ILUgFF0wknon/qElveCRm5zlumuNdWQrlBuEJ5c1RkI76aS4Jbz+OG1032yFBR6eYGDUmg5uM9rnAVSZvjGMTmhJKK3eGOTYSSvqWwRNvK4P7OTBlroT1sHZZyp7UkugjmDScIlZrMQsnM1i2Sycncey83B2EcsuHuyu4UoRoSDcypq0RonX4ifeYBmstoquchXQZFXxpUS6CNcY/bZo+FdWRb8u+vzMhCBSv0sNl4kyfDdai/JlAozcqfUs07u29wLfpYKi3szjbkKPjLVpW2FJq7jzrfGn5wqSwwVtLDQPHNWmabnlzoBU9Av5B7h70zco67uuTWN2HnE8SCeU7BwaK1Ea19LDGk31y9M5Rh37pX/k17pD1eW11Lx3puE+lnS1FLoWpOX1b3QP2XwqyJggFilo2Gk/IigPsf+BY2cTlS4ix0j2savLMWoZQi2+VlfNd8sJZLIIsbQADo20xqDCJcn8b2lKaBrlYMNFJH3hYFJ7AxvLl83aKNKwpYfNWikC4fmsnSIQXsxaKgLh83aAQHg1a3iHwdPFrOEdCKezh3c6c3gPtnILBr8X3JLjQQj1/ckL146gx6dbFZhhMEe2xisYvwyDb7A1BEBwA3c9/w5h9BcUXp+A5UmVVRXL2YKyRXa5/AcWwkZE"
    O = produce_factory(input_str, 10)
    O = O[0].detach().cpu().permute(1, 2, 0).numpy().astype(int)
    O_split = split_matrix_into_entities(O)
    O_viz = visualize_factory_matrix(O_split)
    print_factory(O_viz)