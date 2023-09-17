# Traffic-Diffusion
### Creating a Robust Real-World Traffic Sign Attack.
<br>

![alt text](pics/main_pic_from_paper.jpg?raw=true)

<br>
Our final dataset is located in the following folder:
`datasets/larger_images`

### Running experiments:
#### 1. RFLA Attack: <br>
##### Our physical method: <br>
- run `attacks/RFLA/our_pos_reflect_attack.py` file. 
  You can edit config file `attacks/RFLA/config.yml` to change the attack settings.


#### 2. Shadow Attack: <br>
##### Our physical method: <br>
- run Lisa attack untargeted: <br>
`bash run_attack_experiments/shadowAttack/our_attack/run_our_attack_lisa.sh` 
- run Gtsrb attack untargeted: <br>
`bash run_attack_experiments/shadowAttack/our_attack/run_our_attack_gtsrb.sh` 
- ##### paper method only: <br>
    - run Lisa attack: <br>
    `bash run_attack_experiments/shadowAttack/regular_attack/run_simple_physical_attack.sh`
  <br>You can change inside the file to GTSRB model.
- You can also run the Shadow attack directly by executing `our_shadow_attack.py` file.  

#### 3. PGD Attack: <br>
- Run PGD attack by running `attacks/our_pgd_attack.py` file

* Attack experiments are saved in `experiments` folder.



