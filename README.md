# NEAT-Flappy-Bird
## Overview
NEAT-Flappy-Bird is an Artificial Intelligence (AI) project that uses the NEAT (NeuroEvolution of Augmenting Topologies) Python module to train an AI agent to play the classic game Flappy Bird. The project demonstrates how machine learning and evolutionary algorithms can be used to create an AI system capable of mastering a challenging game like Flappy Bird.

## Dependencies
Before running the project, you need to have the following dependencies installed:

- Python 3.x
- Pygame
- NEAT Python module

## How it Works
**Gameplay:** The AI agent aims to navigate a bird through a series of pipes by tapping a key or clicking the screen to make the bird jump. The objective is to keep the bird flying as long as possible without colliding with the pipes.

**NEAT Algorithm:** NEAT (NeuroEvolution of Augmenting Topologies) is a genetic algorithm designed to evolve artificial neural networks. In this project, NEAT is used to evolve the AI agent's neural network to improve its performance in the game.

**Genetic Evolution:** The NEAT algorithm starts with a population of AI agents, each represented by a neural network with random weights. Through genetic evolution, the agents undergo a process of selection, reproduction, and mutation over multiple generations.

**Fitness Function:** A fitness function is defined to evaluate the performance of each AI agent. The fitness function is based on the distance covered by the bird and the number of pipes successfully passed.

**Crossover and Mutation:** The most successful AI agents are selected as parents for the next generation. Crossover and mutation operations are applied to create new offspring with modified neural networks.

**Training:** The AI agents continue to play the game in successive generations, and their neural networks gradually improve through the genetic evolution process.

**Adaptation:** As the training progresses, the AI agents learn to time their jumps, avoid collisions, and achieve higher scores, demonstrating the effectiveness of the NEAT algorithm.
