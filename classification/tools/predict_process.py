#METHODE SPECIFIQUES A MA CLASSIFICATION

import numpy as np 
import matplotlib.pyplot as plt 

"""
methods to decode predictions, truths and show true/false predictions 
"""

def decodage_predictions(array):
    """[summary]

    Args:
        array ([ndarray]): [ [number_of_predictions, ]]

    Returns:
        [type]: [description]
    """
    result = []
    for i in range(len(array[0])):
        sub_result = []
        for j in range(0,4) : 
            a = array[j][i].tolist()
            maxi = np.max(a)
            index = a.index(maxi)
            if j == 0 : #head 
                if index == 0 : 
                    sub_result.append('Vertex')
                #elif index == 1  : 
                    #sub_result.append('Eye')
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
    truth = []
    for i in range(array.shape[0]): 
        sub = []
        liste = array[i].tolist()
        #head
        if liste[0] == 0 : 
            sub.append('Vertex')
        else : 
            sub.append('Eye / Mouth')
        #if liste[0] == 1 : 
        #    sub.append('Eye')
        #if liste[0] == 2 : 
            #sub.append('Mouth')
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
        

def affichage(liste_array, liste_pred_label, liste_true_label, directory):
    for i in range(len(liste_array)):
        image = liste_array[i][:,:,0]
        f = plt.figure(figsize=(5,8))
        axes = plt.gca()
        axes.set_axis_off()
        plt.imshow(image, cmap='gray')
        plt.title("pred : {}, truth : {}".format(liste_pred_label[i], liste_true_label[i]))
        plt.show()

        if liste_pred_label[i] == liste_true_label[i] : 
            filename = directory + '/predictions/true'+'/'+str(i)+'.png'
            f.savefig(filename, bbox_inches='tight') 
            plt.close()

        else : 
            filename = directory + '/predictions/false'+'/'+str(i)+'.png'
            f.savefig(filename, bbox_inches='tight') 
            plt.close()


