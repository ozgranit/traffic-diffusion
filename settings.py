GENERATED_IMAGES_TYPES_TRAIN = ['midday', 'cloud', 'rain']
GENERATED_IMAGES_TYPES_TEST = ['night', 'snow', 'dawn']
GENERATED_IMAGES_TYPES_ALL = GENERATED_IMAGES_TYPES_TRAIN + GENERATED_IMAGES_TYPES_TEST
DF_RESULTS_COLUMNS = ['img_name', 'label', 'clean_pred_label', 'clean_pred_prob',
                      'adv_normal_pred_label', 'adv_normal_pred_prob',
                      'adv_special_pred_label', 'adv_special_pred_prob']
for type in GENERATED_IMAGES_TYPES_TEST:
    gen_columns = [f'gen_{type}_clean_pred_label', f'gen_{type}_clean_pred_prob',
                   f'gen_{type}_normal_pred_label', f'gen_{type}_normal_pred_prob',
                   f'gen_{type}_special_pred_label', f'gen_{type}_special_pred_prob']
    DF_RESULTS_COLUMNS+=gen_columns

    gen_columns = [f'gen_{type}_2_clean_pred_label', f'gen_{type}_2_clean_pred_prob',
                   f'gen_{type}_2_normal_pred_label', f'gen_{type}_2_normal_pred_prob',
                   f'gen_{type}_2_special_pred_label', f'gen_{type}_2_special_pred_prob']
    DF_RESULTS_COLUMNS+=gen_columns

DF_RESULTS_COLUMNS += ['total_generated_imgs']
DF_RESULTS_COLUMNS += ['total_adv_gen_attack_succeeded_normal', 'avg_prob_gen_attack_succeeded_normal', 'total_gen_failed_normal']
DF_RESULTS_COLUMNS += ['total_adv_gen_attack_succeeded_special', 'avg_prob_gen_attack_succeeded_special', 'total_gen_failed_special']
