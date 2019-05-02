import gym
import math
import random
import statistics
import tensorflow as tf

env = gym.make("CartPole-v1")

eps = 1.0
states = []
rewards = []
reward_mean = []
next_states = []
actions = []


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape=(4,)))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dense(3, activation="linear"))

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

def train_model(states, actions, rewards, next_states):
    size = len(next_states)

    #tf_states = tf.tensor2d(states, shape=[states.length, 26]);


# for epi in range(150):
for epi in range(1):
    st = env.reset()

    env.render()

    reward = 0
    step = 0

    # while step < 400:
    while step < 400:

        # act = pickAction(st, eps)
        act = env.action_space.sample() # your agent here (this takes random actions)

        st2, reward, done, info = env.step(act)

        if done:
            reward = int(-1)
        else:
            reward = int(1)

        mask = [0, 0]
        mask[act] = 1

        index = math.floor(random.uniform(0, 1) * len(states))

        states.insert(index, st)
        rewards.insert(index, [reward])
        reward_mean.insert(index, reward)
        next_states.insert(index, st2)
        actions.insert(index, mask)

        if len(states) > 10000:
            states.pop(0)
            rewards.pop(0)
            reward_mean.pop(0)
            next_states.pop(0)
            actions.pop(0)

        st = st2
        step += 1

        if done:
            break

    eps = max(0.1, eps*0.99)

    if epi % 5 == 0:
        print("---------------");
        print("rewards mean", statistics.mean(reward_mean))
        print("episode", epi);
        train_model(states, actions, rewards, next_states)
        #await tf.nextFrame();

env.close()
