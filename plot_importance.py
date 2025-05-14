import matplotlib.pyplot as plt
import os
def read_importance(lamda, type):
        file_path = f'/home/dh/Grace/Grace/IND_Result/lamda_{lamda}/{type}_importance.txt'
        with open(file_path, 'r') as file:
            imp_index = [int(line.strip()) for line in file]
        org_index = sorted(range(len(imp_index)), key=lambda k: imp_index[k])
        return imp_index, org_index

def plt_importance(lamda, type):
    max_list = []
    min_list = []

    imp_index, org_index = read_importance(lamda, type)
    idx_list = [_ for _ in range(len(imp_index))]
    for idx in imp_index:
        with open(f'/home/dh/Grace/Grace/IND_Result/lamda_{lamda}/{type}_min_max.txt', 'r') as file:
            for line in file:
                if f'channel_{idx:03d}:' in line:
                    parts = line.split(':')[-1].strip().strip('[]').split(',')
                    min_val = float(parts[0].strip())
                    max_val = float(parts[1].strip())
                    min_list.append(min_val)
                    max_list.append(max_val)
                    #with open(f'/home/dh/Grace/Grace/IND_Result/lamda_{lamda}/{type}_length.txt', 'a') as file1:
                    #     file1.write(f'{max_val-min_val}\n')
                    break
    #print(min_list)
    #print(max_list)
    fig2 = plt.figure()
    plt.plot(idx_list, min_list, label='min')
    plt.plot(idx_list, max_list, label='max')
    plt.fill_between(idx_list, min_list, max_list, color='gray', alpha=0.5)
    plt.xlabel('Channel Index')
    plt.ylabel('Value')
    plt.title('Channel Importance')
    plt.grid()
    plt.legend()
    output_dir = f"./IND_Result/lamda_{lamda}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig2.savefig(f"{output_dir}/Channel_Importance_{type}_{lamda}.png")

lamda = 1024
plt_importance(lamda, 'mv')
plt_importance(lamda, 'res')
     
