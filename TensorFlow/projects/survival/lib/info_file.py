
def create_info_file(loss_and_metrics,training_model_folder, mask, mixed_data, mixed_data_info, batch_size_train, batch_size_val, epochs, reduced_size, alpha=0, beta=0, gamma=0, censure_val=[], num_convolutions=0):
    """
        A function to create a file summing up different parameters chosen 
    """

    file_path = os.path.join(training_model_folder, 'info_network')
    f = open(file_path, "w+")
    f.write("training_model_folder : "+ training_model_folder +"\n")
    f.write("mask : "+ str(mask) +"\n")
    f.write("mixed_data : "+ str(mixed_data) +"\n")
    f.write("mixed_data_info : ")
    for i in mixed_data_info: f.write(i+" ")
    f.write("\n")
    f.write("batch_size_train : "+ str(batch_size_train)+"\n")
    f.write("batch_size_val : "+ str(batch_size_val)+"\n")
    f.write("epochs : "+ str(epochs)+"\n")
    f.write("reduced_size : "+ str(reduced_size)+"\n")
    f.write("alpha : "+ str(alpha)+"\n")
    f.write("beta : "+ str(beta)+"\n")
    f.write("gamma : "+ str(gamma)+"\n")
    f.write('-----------------------------------------------------------------\n')
    f.write(f'Loss and metrics per fold\n')
    for i in loss_and_metrics.keys():
        f.write('------------------------------------------------------------------\n')
        f.write(f'> {i}: ')
        for j in range(len(loss_and_metrics[i])):
            f.write(f'{loss_and_metrics[i][j]}\n')
        f.write('------------------------------------------------------------------\n')  

    f.write(f'Loss and metrics per fold average and meean\n')
    f.write('------------------------------------------------------------------\n')
    for i in loss_and_metrics.keys():
        f.write(f'> {i}: {np.mean(loss_and_metrics[i])}(+- {np.std(loss_and_metrics[i])})\n')
    f.write('------------------------------------------------------------------\n')   
    for i in censure_val:
        print("censure validation fold  : ", i)
    f.write(f'{num_convolutions}\n')
