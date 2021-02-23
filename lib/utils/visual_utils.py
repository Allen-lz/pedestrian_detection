import numpy as np
def draw_boxes(img, boxes):
    width = img.shape[1]
    height = img.shape[0]
    persons = []
    for i in range(len(boxes)):
        one_box = boxes[i]
        one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
                    min(one_box[2], width - 1), min(one_box[3], height - 1)])
        x1, y1, x2, y2 = np.array(one_box[:4]).astype(int)
        person_i = img[y1:y2, x1:x2, :]
        persons.append(person_i)
    return persons

