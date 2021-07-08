#METHODE SPECIFIQUES A MA CLASSIFICATION
import os
import numpy as np 
import matplotlib.pyplot as plt 

"""
methods to decode predictions, truths and show true/false predictions 
"""

def decodage_predictions(inference):
    """function to decode inference results from classification model
    Args:
        inference ([list]): [ [(list_predictions_head,size=(nombre_inference,2)), (list_predictions_legs,size=(nombre_inference,3)), (list_predictions_right_arm,size=(nombre_inference,2)), (list_predictions_left_arm,size=(nombre_inference,2))]]

    Returns:
        [list]: [return the results with assoiated value]
    """
    result = []
    for i in range(len(inference[0])):
        #i = number of inferences
        sub_result = []
        for j in range(0,4) : 
            #j = head,legs, right_arm, left_arm
            a = inference[j][i].tolist()
            maxi = np.max(a)
            index = a.index(maxi)
            if j == 0 : #head 
                if index == 0 : 
                    sub_result.append('Vertex')
                else : 
                    sub_result.append('Eye / Mouth')
            elif j == 1 : #leg 
                if index == 0 : 
                    sub_result.append('Hips')
                elif index == 1 : 
                    sub_result.append('Knee')
                else : 
                    sub_result.append('Foot')

            elif j == 2 : #right arm 
                if index == 0 : 
                    sub_result.append('down')
                else : 
                    sub_result.append('up')

            elif j == 3 : #left arm
                if index == 0 : 
                    sub_result.append('down')
                else : 
                    sub_result.append('up')
        result.append(sub_result)
    return result


def decodage_truth(array) : 
    """decode truth label array

    Args:
        array ([type]): [array of size (number_of_labelled_img, 4)]

    Returns:
        [list]: [return the list of decoded truth label]
    """
    truth = []
    for i in range(array.shape[0]): 
        sub = []
        liste = array[i].tolist()
        #head
        if liste[0] == 0 : 
            sub.append('Vertex')
        else : 
            sub.append('Eye / Mouth')
        #leg 
        if liste[1] == 0 : 
            sub.append('Hips')
        if liste[1] == 1 : 
            sub.append('Knee')
        if liste[1] == 2 : 
            sub.append('Foot')
        #right arm 
        if liste[2] == 0 : 
            sub.append('down')
        if liste[2] == 1 : 
            sub.append('up')
        #left arm 
        if liste[3] == 0 : 
            sub.append('down')
        if liste[3] == 1 : 
            sub.append('up')

        truth.append(sub)

    return truth 
        

def affichage(array, liste_pred_label, liste_true_label, directory):
    """generate inference image.jpeg with true and predict labels, and save them in True or False folder.

    Args:
        array ([np.ndarray]): [np.ndarray of shape (number of inference, size_y, size_x, 1)]
        liste_pred_label ([list]): [ [[head1, legs1, arm1, arm1], [head2, legs2, arm2, arm2],... ] ]
        liste_true_label ([list]): [ [[head1, legs1, arm1, arm1], [head2, legs2, arm2, arm2],... ] ]
        directory ([str]): [where to save predictions images/results]
    """

    true = directory+'/predictions/true'
    false = directory+'/predictions/false'
    os.makedirs(false)
    os.makedirs(true)
    for i in range(len(liste_array)):
        image = array[i][:,:,0] #2D
        image = np.rot90(image, k=2)
        f = plt.figure(figsize=(10,16))
        axes = plt.gca()
        axes.set_axis_off()
        plt.imshow(image, cmap='gray')
        plt.title("pred : {}, truth : {}".format(liste_pred_label[i], liste_true_label[i]))
        #plt.show()

        if liste_pred_label[i] == liste_true_label[i] : 
            filename = true+'/'+str(i)+'.jpeg'
            f.savefig(filename, bbox_inches='tight', origin='lower') 
            plt.close()

        else : 
            filename = false+'/'+str(i)+'.jpeg'
            f.savefig(filename, bbox_inches='tight', origin='lower') 
            plt.close()


