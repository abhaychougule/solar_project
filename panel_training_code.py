import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def train_panel_model(train_dir, val_dir):
    """
    Trains a CNN model for detecting solar panels using the provided training and validation directories.

    Yields:
        str: Logs and progress updates for frontend.
    """
    try:
        class_names = ["Solar_Panel", "No_Panel"]
        with open('labels.txt', 'w') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")
        yield "✅ Labels file written: labels.txt\n"

        img_height, img_width = 150, 150
        batch_size = 32

        yield "📊 Initializing data generators...\n"
        train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(rescale=1.0/255.0)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical'
        )

        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical'
        )

        yield "🧠 Building CNN model...\n"
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(len(class_names), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        yield "⚙️ Model compiled successfully.\n"

        yield "🚀 Starting training...\n"
        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=val_generator,
            validation_steps=val_generator.samples // batch_size,
            epochs=20
        )

        save_path = os.path.join("Classifiers", "solar_panel_detection_model.h5")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        yield f"✅ Model saved to: {save_path}\n"
        yield "🎉 Panel Detection Training Completed!\n"

    except Exception as e:
        yield f"❌ Error during training: {str(e)}\n"
