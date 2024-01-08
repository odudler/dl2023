# dl2023
Deep Learning Project 2023 Autumn Semester ETH

## Main steps in the project:
- created "connect four" environment
- implemented a DQN and CQL Agent for learning connect four
- implemented a Decision Transformer agent learning offline and online
- implemented a minimax agent for training and evaluation purposes

## File/Folder descriptions:
- agents: Folder containing all agent implementations (except the DT agent)
- dt_model: Folder containing stored DT Agent models
- saved_models: Folder containing all stored CQL/DQN models
- trajectories: Folder containing stored trajectories (potentially files too large for github)

- board.py: Implementation of the connect four board for use in the environment
- env.py: Implementation of the environment used by the agents
- env_opp_integrated: Implementation of the environment where the opponent is an integrated part of the environment instead of a second seperate player/agent
- networks.py: Implementation of the networks used in the various agents
- notebook.ipynb: Notebook used for training and evaluating the DQN and CQL Agents using the default environment
- notebook_opp_integrated: Notebook used for training and evaluating the DQN and CQL Agents using the env_opp_integrated environment
- trainer.py: Implementation of the Trainer used for training the DQN and CQL Agents
- dt_notebook.ipynb: Notebook containing the Implementation, Training and Evaluation of the DT Agent. Contains code for generating and storing trajectories used for training. Contains code for offline and online training of DT Agents.
- transformer_utils.py: Contains helper functions for the DT Agent code.
- utils.py: Contains helper functions for the DQN / CQL Agents and the implemented environment

## Dependencies (used non-standard python packages/libraries)
- transformers (pip install transformers)
- pytorch (pip install torch)