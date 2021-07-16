


def encoding_instance(liste):
    #upper Limit 
    if liste[2] == 'Vertex' : 
        head = 0
    if liste[2] == 'Eye'  or liste[2] == 'Mouth' : 
        head = 1

    #lower Limit
    if liste[3] == 'Hips' : 
        leg = 0
    if liste[3] == 'Knee': # or liste[3] == 'Foot': 
        leg = 1
    if liste[3] == 'Foot':
        leg = 2

    #right Arm 
    if liste[4] == 'down' : 
        right_arm = 0
    if liste[4] == 'up' : 
        right_arm =1

    #left Arm 
    if liste[5] == 'down' : 
        left_arm = 0
    if liste[5] == 'up' : 
        left_arm = 1

    return [head, leg, right_arm, left_arm]