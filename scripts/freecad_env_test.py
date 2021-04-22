import numpy as np
from freecad_env import freecad_env


def get_action_list(fc_env):
    if (fc_env.action_type == 'circle'):
        radius_num = 4
        x_circle = np.linspace(fc_env.min_vals[fc_env.column_index],
                               fc_env.max_vals[fc_env.column_index],
                               num=fc_env.NUM_CS)
        y_circle = np.linspace(fc_env.min_vals[fc_env.column_index + 1],
                               fc_env.max_vals[fc_env.column_index + 1],
                               num=int(fc_env.NUM_CS * (fc_env.obj_dim_w / fc_env.obj_dim_l)))
        r_circle = np.linspace(fc_env.EPSILON * 10,
                               abs(fc_env.max_vals[fc_env.column_index] - fc_env.min_vals[fc_env.column_index]),
                               num=radius_num)[1:]
        c_circle = (x_circle, y_circle)
        region_values_list = (c_circle, r_circle)
    elif (fc_env.action_type == 'rectangle'):
        w_rectangle = fc_env.obj_dims[fc_env.column_index] / fc_env.NUM_CS
        coor_rectangle = np.linspace(fc_env.min_vals[fc_env.column_index],
                                     fc_env.max_vals[fc_env.column_index],
                                     num=fc_env.NUM_CS)[:-1]

        region_values_list = (coor_rectangle, w_rectangle)
    return region_values_list


def select_action_list(fc_env, region_values_list, indx):
    if (fc_env.action_type == 'circle'):
        indx_x, indx_y, indx_r = indx
        c_circle, r_circle = region_values_list
        x_circle, y_circle = c_circle
        region_values = (np.array(x_circle[indx_x], y_circle[indx_y]), r_circle[indx_r])
    elif (fc_env.action_type == 'rectangle'):
        coor_rectangle, w_rectangle = region_values_list

        region_values = (coor_rectangle[indx], w_rectangle)

    return region_values


if __name__ == '__main__':
    mode = 'test_simple'

    PATH = ''
    fc_env = freecad_env(force_value=1e4, num_actions=5, action_type='rectangle', flag_rand_action=0)
    fc_env.flag_save = 1
    num_samples = 4  # number of episodes
    count = 0
    max_repeat_crash = 5
    save_path = './results/samples_'

    if (mode == 'run_loop_random'):
        while count < num_samples:
            region_values_vec = []
            force_dir_str_vec = []
            crash_counter = 0
            while (fc_env.count_action < fc_env.num_actions):

                if (len(region_values_vec) > fc_env.count_action):
                    region_values = region_values_vec[fc_env.count_action]
                    force_dir_str = force_dir_str_vec[fc_env.count_action]
                else:
                    region_values, force_dir_str = fc_env.get_random_action()
                    region_values_vec.append(region_values)
                    force_dir_str_vec.append(force_dir_str)
                flag_break = fc_env.run_step(region_values=region_values, force_dir_str=force_dir_str)

                if (flag_break == 1) and (crash_counter < max_repeat_crash):
                    fc_env.clear_doc()
                    fc_env.initialize_doc()
                    crash_counter += 1
                elif (crash_counter >= max_repeat_crash):
                    break

            if (fc_env.flag_save == 1) and (fc_env.count_action == fc_env.num_actions):
                pickle_dict = fc_env.create_pickle_dict()
                fc_env.save_result_info(save_path + 'pickle_meta_data_{:03d}'.format(count), pickle_dict)
                fc_env.save_result_all(save_path + '{:03d}'.format(count))
                count += 1

    elif (mode == 'test_simple'):
        region_values_list = get_action_list(fc_env)
        region_values_vec = []
        force_dir_str_vec = []
        crash_counter = 0
        indx_region = 0  # if it's circle it should be a tuple of size 3 (indx_x, indx_y, indx_r), e.g. (0, 0, 0)
        force_dir_str_list = ['top'] * fc_env.num_actions  # or a different list, e.g. ['top', 'bottom', 'top'] -> if num_actions is 3
        while (fc_env.count_action < fc_env.num_actions):

            print('here from {} actions {} is running'.format(fc_env.num_actions, fc_env.count_action))
            if (len(region_values_vec) > fc_env.count_action):
                region_values = region_values_vec[fc_env.count_action]
                force_dir_str = force_dir_str_vec[fc_env.count_action]
            else:
                region_values = select_action_list(fc_env, region_values_list, indx_region)
                force_dir_str = force_dir_str_list[fc_env.count_action]  # it could be given as constant, e.g. 'top'
                print('region_values: {}'.format(region_values), 'force_dir_str: {}'.format(force_dir_str))
                region_values_vec.append(region_values)
                force_dir_str_vec.append(force_dir_str)
            flag_break = fc_env.run_step(region_values=region_values, force_dir_str=force_dir_str)

            if (flag_break == 1) and (crash_counter < max_repeat_crash):
                print('crash: ', crash_counter)
                fc_env.clear_doc()
                fc_env.initialize_doc()
                crash_counter += 1
            elif (crash_counter >= max_repeat_crash):
                break

        if (fc_env.flag_save == 1) and (fc_env.count_action == fc_env.num_actions):
            pickle_dict = fc_env.create_pickle_dict()
            fc_env.save_result_info(save_path + 'pickle_meta_data_{:03d}'.format(count), pickle_dict)
            fc_env.save_result_all(save_path + '{:03d}'.format(count))
