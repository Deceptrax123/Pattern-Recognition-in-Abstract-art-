# Art Generation

Made a generative model that learns the color patterns of various abstract art pieces to generate its own patterns

## Table of Contents Used

| Subject                  | Link                     |
| ------------------------ | ------------------------ |
| Dataset                  | [Abstract-Art-Dataset]() |
| Reference Research paper | [DCGAN Paper]()          |

## Description of Files

| File               | Functionality                                     |
| ------------------ | ------------------------------------------------- |
| Discriminator.py   | Model architecture of the discriminator           |
| Generator.py       | Model architecture of the generator               |
| Data_generator.py  | Dataset class                                     |
| Initialize.py      | Initializer class                                 |
| Random_walk_art.py | Implementation of random walk on the Latent space |
| Training_loops.py  | Training the model                                |
| Transforms.py      | Just messed around with a bunch of transforms     |
| Models             | Contains the .pt model state files                |

## Random Walk

As per the DCGAN paper , A random walk on the latent space was implemented to capture logical relationships between the color patterns generated

This is analogous to contextual relationships between words that were capture by word embeddings such as Glove and word2vec. On performing vector arithmetic we can see that a certain color shade can be obtained from an arithemetic operation of other color patterns.

## Implementation Notes

- Used BCEwithLogits Loss function of Pytorch instead of a sigmoid layer in discriminator for stable results
- Adams Beta 1 parameter changed to 0.5 as per the DCGAN paper
- Learning rate set to 0.0002
- No pooling layers, only strided convolutions used as per the DCGAN paper
