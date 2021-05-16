import tensorflow.keras.backend as K


def f1_weighted(label, pred):
    num_classes = 5
    label = K.cast(K.flatten(label), 'int32')
    true = K.one_hot(label, num_classes)
    pred_labels = K.argmax(pred, axis=-1)
    pred = K.one_hot(pred_labels, num_classes)

    ground_positives = K.sum(true, axis=0) + K.epsilon()  # = TP + FN
    pred_positives = K.sum(pred, axis=0) + K.epsilon()  # = TP + FP
    true_positives = K.sum(true * pred, axis=0) + K.epsilon()  # = TP

    precision = true_positives / pred_positives
    recall = true_positives / ground_positives

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives)
    weighted_f1 = K.sum(weighted_f1)

    return weighted_f1
