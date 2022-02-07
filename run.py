from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
import tensorflow as tf

from simple_ggan import get_QM9_dataset, process_QM9_graph, sample_molecules
from simple_ggan import BaseGenerator, BaseDiscriminator, GraphWGAN


###############################################################################
# Parameters
###############################################################################
atom_mapping = {"C": 0,
                0: "C",
                "N": 1,
                1: "N",
                "O": 2,
                2: "O",
                "F": 3,
                3: "F"}

bond_mapping = {"SINGLE": 0,
                0: Chem.BondType.SINGLE,
                "DOUBLE": 1,
                1: Chem.BondType.DOUBLE,
                "TRIPLE": 2,
                2: Chem.BondType.TRIPLE,
                "AROMATIC": 3,
                3: Chem.BondType.AROMATIC}

NUM_ATOMS = 9  # Maximum number of atoms on each molecule
ATOM_DIM = 4 + 1  # Number of atom types
BOND_DIM = 4 + 1  # Number of bond types
LATENT_DIM = 32  # Size of the latent space


###############################################################################
# Dataset
###############################################################################
dataset = get_QM9_dataset()
adjacency_tensor, feature_tensor = process_QM9_graph(dataset, NUM_ATOMS,
                                                     ATOM_DIM, BOND_DIM,
                                                     atom_mapping,
                                                     bond_mapping, amount=2)

print(f"adjacency_tensor shape: {adjacency_tensor.shape}")
print(f"feature_tensor shape: {feature_tensor.shape}")


###############################################################################
# Create Generator
###############################################################################
generator = BaseGenerator(dense_units=[128, 256, 512],
                          dropout_rate=0.2,
                          latent_dim=LATENT_DIM,
                          adjacency_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
                          feature_shape=(NUM_ATOMS, ATOM_DIM))
# generator.summary()


###############################################################################
# Create Discriminator
###############################################################################
discriminator = BaseDiscriminator(gconv_units=[64, 32],
                                  dense_units=[128, 16],
                                  dropout_rate=0.2,
                                  adjacency_shape=(BOND_DIM, NUM_ATOMS,
                                                   NUM_ATOMS),
                                  feature_shape=(NUM_ATOMS, ATOM_DIM))
# discriminator.summary()


###############################################################################
# Create WGAN and train
###############################################################################

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="./model/checkpoint/epoch_{epoch:02d}.ckpt",
    save_weights_only=True,
    save_freq=209200,
    monitor="loss",
    mode="min",
    save_best_only=False)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir="./model/tensorboard/",
                                             histogram_freq=100)

logger = tf.keras.callbacks.CSVLogger(
    "./model/log.csv", separator=",", append=True)


wgan = GraphWGAN(generator, discriminator,
                 discriminator_steps=1, generator_steps=1)

wgan.compile(optimizer_generator=tf.keras.optimizers.Adam(1e-3),
             optimizer_discriminator=tf.keras.optimizers.Adam(1e-3))

hist = wgan.fit([adjacency_tensor, feature_tensor], epochs=2500, batch_size=32,
                callbacks=[checkpoint, tensorboard, logger])


###############################################################################
# Sample novel molecules
###############################################################################
gen_molecules = sample_molecules(wgan.generator, 64,
                                 LATENT_DIM, BOND_DIM, ATOM_DIM,
                                 atom_mapping, bond_mapping)

gen_smiles = [Chem.MolToSmiles(sml, isomericSmiles=False)
              for sml in gen_molecules if sml is not None]
with open("./model/generated_samples.txt", "w") as f:
    for item in gen_smiles:
        f.write("%s\n" % item)

img = MolsToGridImage([mol for mol in gen_molecules if mol is not None],
                      molsPerRow=8, subImgSize=(200, 200),
                      legends=tuple(gen_smiles))
img.save("./model/generated_samples.png")
