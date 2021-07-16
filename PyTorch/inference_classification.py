from Projects.Classification.infer_classification import inference_classification

#model_path = '../torch_files/model.pth' #path from DeepOncology
model_path = '../../torch_files/model.pth'
CT_path = '/media/m-056285chu-toulousefr/c6a4c10e-9316-4b95-8c6e-1d87e1dce435/lysarc/NIFTI/GAINED_NIFTI/11011101051014/pet0/1.2.276.0.7230010.3.2.212610_nifti_CT.nii'
CT_paths = ['/media/m-056285chu-toulousefr/c6a4c10e-9316-4b95-8c6e-1d87e1dce435/lysarc/NIFTI/GAINED_NIFTI/11011101051014/pet0/1.2.276.0.7230010.3.2.212610_nifti_CT.nii', 
'/media/m-056285chu-toulousefr/c6a4c10e-9316-4b95-8c6e-1d87e1dce435/lysarc/NIFTI/GAINED_NIFTI/11011101051014/pet0/1.2.276.0.7230010.3.2.212610_nifti_CT.nii']
output_shape= (256,256,1024)
angle=0
result = inference_classification(model_path, CT_path, output_shape, angle)
print(result)