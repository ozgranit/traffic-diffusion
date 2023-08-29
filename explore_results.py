import os
import pandas as pd
def explore_total_physical_attack_results(results_path):
    parent_dir = os.path.split(results_path)[0]
    output_path = os.path.join(parent_dir, 'result_summary.csv')
    results = pd.read_csv(results_path)
    output_df = pd.DataFrame(columns=['total_inputs(diff orig images_orig)', 'total_generated_imgs', "total_test_images",
                                      'total_adv_gen_attack_succeeded_normal', 'avg_prob_gen_attack_succeeded_normal', 'acc_normal', 'count_greater_than_zero_normal',
                                      'total_adv_gen_attack_succeeded_special', 'avg_prob_gen_attack_succeeded_special', 'acc_special', 'count_greater_than_zero_special'])
    total_inputs = len(results)
    print("total inputs (different orig images_orig): ", total_inputs)
    total_gen_test = results['total_generated_imgs'][0].item()
    print("total_generated_imgs: ", total_gen_test)
    total_test_images_variations = len(results) * total_gen_test
    print("total_test_image_variations: ", total_test_images_variations)

    # normal
    total_succeeded_normal = int(results['total_adv_gen_attack_succeeded_normal'].sum())
    print("total_adv_gen_attack_succeeded_normal: ", total_succeeded_normal)

    succeeded_normal_avg = "{:.2f}".format(results['avg_prob_gen_attack_succeeded_normal'].mean())
    print("avg_prob_gen_attack_succeeded_normal: ", succeeded_normal_avg)

    acc_normal = "{:.2f}".format((total_succeeded_normal / total_test_images_variations) * 100)
    print("acc_normal: ", acc_normal)

    count_greater_than_zero_normal = (results['total_adv_gen_attack_succeeded_normal'] > 0).sum()
    print("count_greater_than_zero_normal (succeeded): ", count_greater_than_zero_normal)
    print('-'*15)
    # special
    total_succeeded_special = int(results['total_adv_gen_attack_succeeded_special'].sum())
    print("total_adv_gen_attack_succeeded_special: ", total_succeeded_special)

    succeeded_special_avg = "{:.2f}".format(results['avg_prob_gen_attack_succeeded_special'].mean())
    print("avg_prob_gen_attack_succeeded_special: ", succeeded_special_avg)

    acc_special = "{:.2f}".format((total_succeeded_special / total_test_images_variations) * 100)
    print("acc_special: ", acc_special)

    count_greater_than_zero_special = (results['total_adv_gen_attack_succeeded_special'] > 0).sum()
    print("count_greater_than_zero_special (succeeded): ", count_greater_than_zero_special)

    output_df.loc[0] = [total_inputs, total_gen_test, total_test_images_variations, total_succeeded_normal, succeeded_normal_avg, acc_normal, count_greater_than_zero_normal,
                        total_succeeded_special, succeeded_special_avg, acc_special, count_greater_than_zero_special]
    output_df.to_csv(output_path)

if __name__ == "__main__":
    path = r'/tmp/pycharm_project_534/larger_images/physical_attack/results.csv'
    explore_total_physical_attack_results(path)