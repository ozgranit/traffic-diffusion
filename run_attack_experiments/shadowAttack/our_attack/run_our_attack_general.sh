# Our Attack:
python our_shadow_attack.py \
    --shadow_level 0.43 \
    --input_data_name "LARGER_IMAGES" \
    --attack_model_name "LISA" \
    --attack_type "physical" \
    --image_label None \
    --polygon 3 \
    --n_try 1 \
    --target_model "normal" \
    --iter_num 200 \
    --crop_size 32 \
    --output_dir 'experiments/shadowAttack' \
    --our_attack True \
    --with_EOT False \
    --untargeted_only False \
    --ensemble False \
    --transform_num 0 \
    --plot_pairs False
