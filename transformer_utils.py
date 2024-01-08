import torch
import numpy as np
import random
from env import Env

# Import Transformers
from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments

#Based on a DT by huggingface, implements proper loss and masking
class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        # Use Cross Entropy Loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(action_preds, action_targets)

        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)

#Caluclate Return to Go given the episode rewards
def calculate_RTG(rewards):
    """
    Given an array of rewards, calculate the corresponding "Return-to-go" array!
    Example: If rewards are [1,2,3,4] then the result should be [10,9,7,4]
    because initially the total reward we get is 10 in this case.
    """
    RTGs = []
    for i in range(0,len(rewards)):
        # Sum up all the rewards occuring in the current timestep until the end!
        RTGs.append(sum(rewards[i:]))
    return RTGs

# Function that gets an action from the model using autoregressive prediction with a window of the previous max_len timesteps.
# This function is also directly taken from the Huggingface blog post
def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    # This implementation does not condition on past rewards

    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    states = states[:, -model.config.max_length :]
    actions = actions[:, -model.config.max_length :]
    returns_to_go = returns_to_go[:, -model.config.max_length :]
    timesteps = timesteps[:, -model.config.max_length :]
    padding = model.config.max_length - states.shape[1]
    # pad all tokens to sequence length
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat([torch.zeros((1, padding, model.config.state_dim)), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, model.config.act_dim)), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)

    state_preds, action_preds, return_preds = model.original_forward(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )

    return action_preds[0, -1]



# Creates One Trajectory, given the model to use. Used for online learning
def create_trajectory(model, RTG, env, opponent, render, explore):

    _states = []
    _actions = []
    _rewards = []
    _dones = []
    finished = -1

    max_ep_len = 21
    device = "cpu"
    state_dim = 42
    act_dim = 7
    env.reset()

    _state = env.get_state()
    state = np.array(_state).flatten()
    
    target_return = torch.tensor(RTG, device=device, dtype=torch.float32).reshape(1, 1)
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    timestep = 0
    while finished == -1:

        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        _state = env.get_state()
        action = get_action(
            model,
            states,
            actions,
            rewards,
            target_return,
            timesteps,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()
        action_taken = np.argmax(action)
        #Explore
        if random.random() < explore:
            action_taken = env.random_valid_action()


        valid, reward, finished = env.step(int(action_taken), 1)

        _states.append(np.ravel(_state))
        _actions.append(int(action_taken))
        _rewards.append(reward)

        state = np.array(env.get_state()).flatten()
        #env.render_pretty()

        if finished != -1:
            break

        opp_action = opponent.act(env.field)
        #print(opp_action)
        valid, _, finished = env.step(int(opp_action), 2)
        state = np.array(env.get_state()).flatten()

        if finished == 2:
            reward = -1
            _rewards[-1] = reward

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        pred_return = target_return[0, -1] - reward
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (timestep + 1)], dim=1)

        timestep += 1
        #env.render_pretty()

    dones = ([False] * (len(_rewards)-1)) + [True]
   
    assert len(_states) == len(_actions)
    assert len(_actions) == len(_rewards)
    assert len(dones) == len(_rewards)
    length = len(_states)
    RTGs = calculate_RTG(_rewards)
    traj = [length, _states, _actions, _rewards, RTGs, dones]

    if render:
        env.render_pretty()

    return traj


# Method trains decision transformer model and optionally stores it
def train_model(model, training_args, dataset, collator, store_path=None):

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    if store_path is not None:
        trainer.save_model(f"dt_model/{store_path}")

    return model

def evaluate_model(model, opponent, episodes, target_return_start, render, agent_start=None):
    # This cell is supposed to evaluate the Decision Transformer against a MinimaxAgent (or RandomAgent) for a fixed amount of Episodes
    env = Env()
    max_ep_len = 21
    device = "cpu"
    model = model.to("cpu")

    games_won = 0
    games_lost = 0

    for ii in range(0,episodes):

        state_dim = env.field.num_columns * env.field.num_rows
        act_dim = 7
        # Create the decision transformer model
        episode_return, episode_length = 0, 0
        env.reset()
        state = np.array(env.get_state()).flatten()
        target_return = torch.tensor(target_return_start, device=device, dtype=torch.float32).reshape(1, 1)
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)

        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        for t in range(max_ep_len):
            if agent_start == False:
                # opp_action = opponent.act(env.get_state_inverted(), deterministic=False) # THIS FOR CQL & DQN AGENTS
                opp_action = opponent.act(env.field) # THIS FOR RANDOM & MINIMAX AGENTS
                #print(f"Opponent Action: {opp_action}")
                #print(env.field.field)
                valid, _, finished = env.step(int(opp_action), 2)
                state = np.array(env.get_state()).flatten()

                if finished != -1:
                    if finished == 1:
                        games_won += 1
                    if finished == 2:
                        games_lost += 1
                    break
            # if agent_start == True or (agent_start == None and random.choice[(True, False)]):

            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = get_action(
                model,
                states,
                actions,
                rewards,
                target_return,
                timesteps,
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()
            #print(f"Action chosen: {action}")
            action_taken = np.argmax(action)
            #print(action_taken)

            #state, reward, done, _ = env.step(action)
            # valid, reward, finished = env.step(int(action), AGENT)
            valid, reward, finished = env.step(int(action_taken), 1)
            #print(f"My reward: {reward}, move was valid: {valid}, is game finished: {finished}")
            #env.render_console()
            state = np.array(env.get_state()).flatten()
            #env.render_pretty()

            if finished != -1:
                if finished == 1:
                    games_won += 1
                if finished == 2:
                    games_lost += 1
                break
            
            if agent_start == True:
                # opp_action = opponent.act(env.get_state_inverted(), deterministic=False) # THIS FOR CQL & DQN AGENTS
                opp_action = opponent.act(env.field) # THIS FOR RANDOM & MINIMAX AGENTS
                #print(f"Opponent Action: {opp_action}")
                #print(env.field.field)
                valid, _, finished = env.step(int(opp_action), 2)
                state = np.array(env.get_state()).flatten()

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            pred_return = target_return[0, -1] - reward
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

            episode_return += reward
            episode_length += 1

            if finished != -1:
                if finished == 1:
                    games_won += 1
                if finished == 2:
                    games_lost += 1
                break
        if render:
            env.render_pretty()

    print(f"Score: Agent {games_won} - {games_lost} Opponent. There were {episodes - games_lost - games_won} Ties")
        #env.render_pretty()