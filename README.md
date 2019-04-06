# ml-agent

## Getting Started
Running environment.
#### 1. Python
> The python version is 3.6
#### 2. UnityEnvironment
>```text
>pip install mlagents==0.4
>```
#### 3. GPU-Environment:
> 
>CUDA 9.0
>
#### 4. Pytorch-GPU
>torch-1.0.0-cp36-cp36m-win_amd64 
>```text
>pip3 install https://download.pytorch.org/whl/cu90/torch-1.0.0-cp36-cp36m-win_amd64.whl
>pip3 install torchvision
>```

## ml-agents 구조
#### ml-agnets import 하기
>```text
> # ml-agents 0.5버전 이상
> from mlagents.envs import UnityEnvironment
>
> # ml-agents 0.4버전 이하
> from unityagents import UnityEnvironment
>```
#### 환경 불러오기
>```text
> env = UnityEnvironment(file_name="(불러올 환경 주소)")
> ex) env = UnityEnvironment(file_name="../environment/Banana_Windows_x86_64_0.4/Banana.exe")
>```
#### brain 불러오기
>```text
> brain_name = env.brain_names[0]
> brain = env.brains[brain_name]
>```
#### 환경 reset하기 (환경정보를 반환해줌)
>```text
> env_info = env.reset(train_mode=True)[brain_name]
>```
#### action size, state size 불러오기
>```text
> # ml-agents 0.4버전 이하
> action_size = brain.vector_action_space_size (액션 으로 반환)
>
> # ml-agents 0.5버전 이상
> action_size = brain.vector_action_space_size[0] ([액션] 으로 반환)
>
>state = env_info.vector_observations[0]
>state_size = len(state)
>```
#### 환경에서 1step 진행하고 next state, reward, done 반환받기
>```text
> env_info = env.step(action)[brain_name]
> next_state = env_info.vector_observations[0]
> reward = env_info.rewards[0]
> done = env_info.local_done[0]
>```
#### 주의사황
> 신경망을 작성할떄 리스트형식은 입력이 안되므로 state size와 action size를 대입할때 주의하자

> 정확한 이유를 모르겠지만 cmd에서 git clone하면 환경이 실행하는 오류가 발생하는데 만약 코드를 실행시켰는데 "The Unity environment took too long to respond" 오류가 뜨면 zip 파일로 다운받아서 사용하시면 작동됩니다.

