def main():

    import numpy as np
    from keras import models
    from keras.models import Sequential
    from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras import optimizers
    from keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import classification_report, precision_recall_curve, roc_curve
    import matplotlib.pyplot as plt

    basepath = "C:/Users/vaish/Documents/Detection_of_blood_grp_using_fingerprint_img_100%_Code[1]/Detection of blood grp using fingerprint img 100% Code"
    
    # Initializing the CNN
    classifier = Sequential()
    
    # Step 1 - Convolution Layer
    classifier.add(Convolution2D(32, 1, 1, input_shape=(64, 64, 3), activation='relu'))
    
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Adding second convolution layer
    classifier.add(Convolution2D(32, 1, 1, activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Adding third convolution layer
    classifier.add(Convolution2D(64, 1, 1, activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Step 3 - Flattening
    classifier.add(Flatten())
    
    # Step 4 - Full Connection
    classifier.add(Dense(256, activation='relu'))
    classifier.add(Dropout(0.8))
    classifier.add(Dense(8, activation='softmax'))  # Change class no.
    
    # Compiling the CNN
    classifier.compile(
        optimizer=optimizers.SGD(lr=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Data preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory(
        basepath + '/training',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical'
    )
    
    test_set = test_datagen.flow_from_directory(
        basepath + '/testing',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        shuffle=False  # Disable shuffle for evaluation purposes
    )
    
    steps_per_epoch = int(np.ceil(training_set.samples / 32))
    val_steps = int(np.ceil(test_set.samples / 32))
    
    # Training the model
    model = classifier.fit_generator(
        training_set,
        steps_per_epoch=steps_per_epoch,
        epochs=6000,
        validation_data=test_set,
        validation_steps=val_steps
    )
    
    # Saving the model
    classifier.save(basepath + '/Group.h5')
    
    # Evaluate the model
    scores = classifier.evaluate(test_set, verbose=1)
    B = "Testing Accuracy: %.2f%%" % (scores[1] * 100)
    print(B)
    
    scores = classifier.evaluate(training_set, verbose=1)
    C = "Training Accuracy: %.2f%%" % (scores[1] * 100)
    print(C)
    
    msg = B + '\n' + C
    
    # Plot accuracy
    plt.plot(model.history['accuracy'])
    plt.plot(model.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(basepath + "/accuracy.png", bbox_inches='tight')
    plt.show()
    
    # Plot loss
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(basepath + "/loss.png", bbox_inches='tight')
    plt.show()

    # Predict labels for test set
    Y_pred = classifier.predict(test_set, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)  # Convert predictions to class labels
    
    # Classification report
    print('Classification Report:')
    print(classification_report(test_set.classes, y_pred, target_names=list(test_set.class_indices.keys())))

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(test_set.classes, Y_pred[:, 1], pos_label=1)  # Assuming binary classification
    
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(basepath + "/precision_recall_curve.png", bbox_inches='tight')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(test_set.classes, Y_pred[:, 1], pos_label=1)  # Assuming binary classification
    
    plt.plot(fpr, tpr, marker='.')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(basepath + "/roc_curve.png", bbox_inches='tight')
    plt.show()
    
    
    
    return msg
