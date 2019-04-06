# coding:utf-8
from unityagents import UnityEnvironment
import numpy as np
from network.DQN import DQNAgent
import matplotlib.pyplot as plt
import tensorflow as tf
import time

env = UnityEnvironment(file_name="../environment/Banana_Windows_x86_64_0.4/Banana.exe") #* 환경 불러오기 
path = "../result/banana/" # 결과 저장할 주소
# get the default brain
brain_name = env.brain_names[0] #* brain_name 불러오기
brain = env.brains[brain_name] #* brain 불러오기

env_info = env.reset(train_mode=True)[brain_name] #* env reset으로 env_info 불러오기
action_size = brain.vector_action_space_size #* action size 불러오기
state = env_info.vector_observations[0] #* state 불러오기
state_size = len(state) #* state size 구하기
####################################
total_scores = []
scores = []
batch_size = 64
mean = 0
count = 0
eps = 1.0
eps_end = 0.01
decay = 0.999
max_t = 1000
gamma = 0.99
alpha = 1e-4
tua = 1e-3
max_memory_size = 50000
train = True # True: 트레이닝 False: 모델 불러오기
#################################### 하이퍼 파라미터들
with tf.Session() as session:
    brain_agent = DQNAgent(session, state_size, action_size, max_memory_size, gamma, alpha, tua) # DQN 알고리즘 
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver() # 모델 세이브 함수
    while mean < 13 and train:
        env_info = env.reset(train_mode=True)[brain_name] #* env reset으로 env_info 불러오기
        score = 0
        time_b = time.time()
        loss = 0
        for i in range(max_t):
            if np.random.random() > eps:
                action = np.argmax(brain_agent.choose_action(state), axis=1) 
            else:
                action = np.random.choice(action_size)
            env_info = env.step(action)[brain_name] #* 환경에서 1step 진행하기
            next_state = env_info.vector_observations[0] #* action후 환경으로부터 next state, reward, done 반환받기
            reward = env_info.rewards[0] #* 반환받은 reward
            done = env_info.local_done[0] #* 반환받은 done
            score += reward
            brain_agent.store(state, action, reward, next_state, [done])
            state = next_state
            if brain_agent.step % 4 == 0:
                loss += brain_agent.learn(batch_size)
            if done:
                break
        scores.append(score)
        total_scores.append(score)
        eps = max(eps * decay, eps_end)
        print("\rEpisode: {},\tCurr Score: {},\tAverage Score: {:.2f},\tLoss:{:.4},\tEPS:{:.4},\tTime: {:.4}".format(count, score, np.mean(scores), loss/250.0, eps, time.time()-time_b), end="")
        if count % 100 == 0 and count > 0:
            mean = np.mean(scores)
            print("\rEpisode: {}, \tAverage Score: {:.2f}".format(count, mean))
            scores.clear()
        count += 1

    if train:
        saver.save(session, path)
        fig = plt.figure()
        plt.plot(range(len(total_scores)), total_scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
    else:
        saver.restore(session, path)

    saver.restore(session, path)
    for _ in range(10):
        done = False
        env_info = env.reset(train_mode=False)[brain_name] #* env reset으로 env_info 불러오기
        score = 0
        state = env_info.vector_observations[0] #* state 불러오기
        while not done:

            action = brain_agent.action = np.argmax(brain_agent.choose_action(state), axis=1)
            env_info = env.step(action)[brain_name] #* 환경에서 1step 진행하기
            next_state = env_info.vector_observations[0] #* action후 환경으로부터 next state, reward, done 반환받기
            reward = env_info.rewards[0] #* 반환받은 reward
            done = env_info.local_done[0] #* 반환받은 done
            score += reward
            state = next_state
        print("Score is ", score)
