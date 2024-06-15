# Quantifying-and-Mitigating-Prompt-Overfitting
This repository contains the code used for our paper: Reinforcement Learning for Aligning Large Language Models Agents with Interactive Environments: Quantifying and Mitigating Prompt Overfitting
![screenshot](Figures.png)
# Installation steps
1. Create Python Environment
 
       conda create -n dlp python=3.10.8; conda activate EMNLP
2. Install packages required

       pip install -r requirements.txt
3. Install Simulation Environment
* BabyAI-Text:

       pip install blosc; cd babyai-text/babyai; pip install -e .; cd ..
       cd gym-minigrid; pip install -e.; cd ..
       pip install -e .
* TWC:
  
       pip install textworld
       git clone https://github.com/IBM/commonsense-rl
       cd commonsense-rl/game_generation
       # Create TWC Levels 
       python twc_make_game.py --level medium --num_games 10000
       cd ../..
4. Install Lamorel

       git clone https://github.com/flowersteam/lamorel.git; cd lamorel/lamorel; pip install -e .; cd ../..
# Training a Language Model
## TWC:
### Train single strategy $P_i$:
        python3 -m lamorel_launcher.launch --config-path "./experiments/TWC/configs/custom/" 
                   --config-name "local_gpu_config"        
                   rl_script_args.path="/experiments/TWC/main_pi.py"
                   rl_script_args.output_dir="." 
                   lamorel_args.accelerate_args.machine_rank=0 
                   rl_script_args.gradient_batch_size=4 
                   lamorel_args.llm_args.model_path="google/flan-t5-small"  
                   rl_script_args.prompt_id=0 # P_i number from 0 to 3 
                   rl_script_args.seed= 1   
                   lamorel_args.llm_args.model_type="seq2seq"
### Train single strategy $P_{all}$:
        python3 -m lamorel_launcher.launch --config-path "./experiments/TWC/configs/custom/" 
                   --config-name "local_gpu_config"        
                   rl_script_args.path="/experiments/TWC/main_pAll.py"
                   rl_script_args.output_dir="." 
                   lamorel_args.accelerate_args.machine_rank=0 
                   rl_script_args.gradient_batch_size=4 
                   lamorel_args.llm_args.model_path="google/flan-t5-small"  
                   rl_script_args.prompt_id=0 # useless we train on all prompts 
                   rl_script_args.seed= 1   
                   lamorel_args.llm_args.model_type="seq2seq"
### Train single strategy $Contrastive_{P_i}$:
        python3 -m lamorel_launcher.launch --config-path "./experiments/TWC/configs/custom/" 
                   --config-name "local_gpu_config"        
                   rl_script_args.path="/experiments/TWC/main_Contrastive.py"
                   rl_script_args.output_dir="." 
                   lamorel_args.accelerate_args.machine_rank=0 
                   rl_script_args.gradient_batch_size=4 
                   lamorel_args.llm_args.model_path="google/flan-t5-small"  
                   rl_script_args.prompt_id=0 # P_i number from 0 to 3 , Contrastive applied to all others
                   rl_script_args.seed= 1   
                   lamorel_args.llm_args.model_type="seq2seq"
## BabyAI-Text:
### Train single strategy $P_i$:
        python3 -m lamorel_launcher.launch --config-path "./experiments/TWC/configs/custom/" 
                   --config-name "local_gpu_config"        
                   rl_script_args.path="/experiments/BabyAI-Text/main_pi.py"
                   rl_script_args.output_dir="." 
                   lamorel_args.accelerate_args.machine_rank=0 
                   rl_script_args.gradient_batch_size=4 
                   lamorel_args.llm_args.model_path="google/flan-t5-small"  
                   rl_script_args.prompt_id=0 # P_i number from 0 to 3 
                   rl_script_args.seed= 1   
                   lamorel_args.llm_args.model_type="seq2seq"
### Train single strategy $P_{all}$:
        python3 -m lamorel_launcher.launch --config-path "./experiments/BabyAI-Text/configs/custom/" 
                   --config-name "local_gpu_config"        
                   rl_script_args.path="/experiments/TWC/main_pAll.py"
                   rl_script_args.output_dir="." 
                   lamorel_args.accelerate_args.machine_rank=0 
                   rl_script_args.gradient_batch_size=4 
                   lamorel_args.llm_args.model_path="google/flan-t5-small"  
                   rl_script_args.prompt_id=0 # useless we train on all prompts 
                   rl_script_args.seed= 1   
                   lamorel_args.llm_args.model_type="seq2seq"
### Train single strategy $Contrastive_{P_i}$:
        python3 -m lamorel_launcher.launch --config-path "./experiments/TWC/configs/custom/" 
                   --config-name "local_gpu_config"        
                   rl_script_args.path="/experiments/BabyAI-Text/main_Contrastive.py"
                   rl_script_args.output_dir="." 
                   lamorel_args.accelerate_args.machine_rank=0 
                   rl_script_args.gradient_batch_size=4 
                   lamorel_args.llm_args.model_path="google/flan-t5-small"  
                   rl_script_args.prompt_id=0 # P_i number from 0 to 3 , Contrastive applied to all others
                   rl_script_args.seed= 1   
                   lamorel_args.llm_args.model_type="seq2seq"
