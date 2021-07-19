
import torch
from torch.nn.functional import softmax 
import torchmetrics
from  torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os 
import numpy as np 
from sklearn.model_selection import train_test_split 

from .loader_classification import Rescale_transform_array
from .loader_classification import Transformed_dataset
from .lib.json_to_csv_reader_nifti import JSON_TO_CSV
from .lib.csv_to_array import CSV_TO_ARRAYS

from Models.classification_model import net_classification

def train_classification(json_path, nifti_directory, csv_directory, training_model_folder) : 

    model_path = training_model_folder +'/runs/'
    writer = SummaryWriter(model_path)

    objet = JSON_TO_CSV(json_path) 
    objet.result_csv(nifti_directory, csv_directory)
    print(objet.csv_result_path)
    prep_objet = CSV_TO_ARRAYS(objet.csv_result_path)
    print(prep_objet.dataset)
    train_dataset, test_dataset = train_test_split(prep_objet.dataset[0:], random_state = 42, test_size = 0.10) #random state 
    train_dataset, val_dataset = train_test_split(train_dataset, random_state = 42, test_size = 0.20)
    
    print("Size training dataset : ", len(train_dataset))
    print("Size validation dataset : ", len(val_dataset))
    train_batch_size = 100 
    val_batch_size = 100 

    train_transformed_dataset = Transformed_dataset(train_dataset,
                                            transform=Rescale_transform_array((256,256,1024)))
    train_dataloader = DataLoader(train_transformed_dataset, batch_size=train_batch_size,
                            shuffle=True, num_workers=0, drop_last=True)

    val_transformed_dataset = Transformed_dataset(val_dataset,
                                            transform=Rescale_transform_array((256,256,1024)))
    val_dataloader = DataLoader(val_transformed_dataset, batch_size=val_batch_size,
                            shuffle=True, num_workers=4, drop_last=True)

    train_num_batches = int(len(train_transformed_dataset)/train_batch_size)
    val_num_batches = int(len(val_transformed_dataset)/val_batch_size)


    model = net_classification().float()
    #model.to(device) #to run on gpu 
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)
    criterion = nn.CrossEntropyLoss()
    metric = torchmetrics.Accuracy()

    nb_epochs = 35
    loss_finale = []
    accuracy_finale = []
    min_valid_loss = np.inf
    for epoch in range(nb_epochs):
        l = 0
        a = 0
        print("CHANGEMENT D'EPOCH : ", epoch)
        running_loss = 0
        running_accuracy_head = 0.
        running_accuracy_leg = 0.
        running_accuracy_right_arm = 0.
        running_accuracy_left_arm = 0.
        print('Chargement images training ...')
        for i, data in enumerate(train_dataloader, 0):
            
            inputs = data['image']
            head_label = data['head']
            leg_label = data['leg']
            right_arm_label = data['head']
            left_arm_label = data['left_arm']
            #inputs = (data['image']).to(device) #to run on gpu 
            #labels = (data['label']).to(device) #to run on gpu 

            optimizer.zero_grad() #zero the parameter gradients 
            head_output, leg_output, right_arm_output, left_arm_output = model(inputs.float())
            head_loss = criterion(head_output, head_label)
            leg_loss = criterion(leg_output, leg_label)
            right_arm_loss = criterion(right_arm_output, right_arm_label)
            left_arm_loss = criterion(left_arm_output, left_arm_label)
            loss = head_loss+leg_loss+right_arm_loss+left_arm_loss
            loss.backward()
            optimizer.step()
            accuracy_head = (metric(F.softmax(head_output, dim=1), head_label))
            accuracy_leg = metric(F.softmax(leg_output, dim=1), leg_label)
            accuracy_right_arm = metric(F.softmax(right_arm_output, dim=1), right_arm_label)
            accuracy_left_arm = metric(F.softmax(left_arm_output, dim=1), left_arm_label)
            accuracy = (accuracy_head.item()+accuracy_leg.item()+accuracy_right_arm.item()+accuracy_left_arm.item())/4

            #print statistics 
            running_loss += loss.item()
            l +=loss.item()
            a += accuracy
            running_accuracy_head += accuracy_head.item()
            running_accuracy_leg += accuracy_leg.item()
            running_accuracy_right_arm += accuracy_right_arm.item()
            running_accuracy_left_arm += accuracy_left_arm.item()
            running_accuracy = (running_accuracy_head+ running_accuracy_leg+ running_accuracy_right_arm+running_accuracy_left_arm)/4
            print('[%d, %5d] loss: %.3f' %
                    (epoch+1, i+1, loss.item()))
            print('[%d, %5d] accuracy: %.3f' %
                (epoch+1, i+1, accuracy))
        val_loss =0.
        val_accuracy=0
        model.eval() 
        print('Chargement images validation ...')
        for i, data in enumerate(val_dataloader, 0):
            inputs = data['image']
            head_label = data['head']
            leg_label = data['leg']
            right_arm_label = data['head']
            left_arm_label = data['left_arm']

            head_output, leg_output, right_arm_output, left_arm_output = model(inputs.float())
            head_loss = criterion(head_output, head_label)
            leg_loss = criterion(leg_output, leg_label)
            right_arm_loss = criterion(right_arm_output, right_arm_label)
            left_arm_loss = criterion(left_arm_output, left_arm_label)
            loss = head_loss+leg_loss+right_arm_loss+left_arm_loss
            
            accuracy_head = (metric(F.softmax(head_output, dim = 1), head_label))
            accuracy_leg = metric(F.softmax(leg_output, dim = 1), leg_label)
            accuracy_right_arm = metric(F.softmax(right_arm_output, dim=1), right_arm_label)
            accuracy_left_arm = metric(F.softmax(left_arm_output, dim=1), left_arm_label)
            accuracy = (accuracy_head.item()+accuracy_leg.item()+accuracy_right_arm.item()+accuracy_left_arm.item())/4
            
            val_loss +=loss.item()
            val_accuracy+= accuracy
            print('[%d, %5d] Validation loss: %.3f' %
                    (epoch+1, i+1, loss.item()))
            print('[%d, %5d] Validation accuracy: %.3f' %
                (epoch+1, i+1, accuracy))

            
        val_loss= val_loss/val_num_batches
        val_accuracy = val_accuracy /val_num_batches
        print('Val loss ici :', val_loss)
        print('val accuracy ici', val_accuracy)
        writer.add_scalars('Loss', {
                            'training loss': running_loss / train_num_batches, 
                            'validation_loss':  val_loss}, 
                            epoch *len(val_dataloader)+i)
        writer.add_scalars('Accuracy',{
                            'training_accuracy':running_accuracy / train_num_batches, 
                            'validation_accuracy': val_accuracy}, 
                            epoch *len(val_dataloader)+i)
        
        running_loss = 0.
        if min_valid_loss>val_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}---->{val_loss:.6f}) \t Saving The Model')
            min_valid_loss = val_loss
            #Saving state dict 
            torch.save(model.state_dict(), training_model_folder)
        loss_finale.append(l/train_num_batches)
        accuracy_finale.append(a/train_num_batches)

        print('\n#####################################\n')
        print('Loss epoch ', epoch, ' : ', l/train_num_batches)
        print('Accuracy epoch ', epoch, ' : ',a/train_num_batches)
        print('\n#####################################\n')
    print('Finished training')
    print(loss)
    print(accuracy)
    
    return 0

if __name__ == "__main__":
    train_classification()