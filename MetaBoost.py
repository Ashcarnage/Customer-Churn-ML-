import tensorflow as tf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# # Data preprocessing
def preprocess_data(data_path):
    data = pd.read_csv(data_path)
    
    # Convert date and extract features
    data['date_of_registration'] = pd.to_datetime(data['date_of_registration'])
    data['year_of_registration'] = data['date_of_registration'].dt.year
    data['month_of_registration'] = data['date_of_registration'].dt.month
    data['day_of_registration'] = data['date_of_registration'].dt.day
    data = data.drop(columns=['date_of_registration'])

    
    # Encode categorical variables
    le = LabelEncoder()
    for col in ['telecom_partner', 'gender', 'state', 'city']:
        data[col] = le.fit_transform(data[col])
    
    # Scale numerical features
    scaler = MinMaxScaler()
    numerical_cols = ['age', 'pincode', 'num_dependents', 'estimated_salary', 'calls_made', 'sms_sent', 'data_used']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data[:13000]  # Limit to 5000 samples as in the original code

def load_and_preprocess_data(data):
    y = np.array(data['churn'])
    X = np.array(data.drop(columns=['churn']))
    return train_test_split(X, y, test_size=0.2, random_state=42)

class IntegratedModel(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.ann = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(256, activation='swish',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.04),
            tf.keras.layers.Dense(128, activation='swish',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.03),
            # tf.keras.layers.LayerNormalization(epsilon=1e-5),
            tf.keras.layers.Dense(64, activation='swish',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.02),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.meta_learner = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(180, activation='swish',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.03),
            tf.keras.layers.Dense(128, activation='swish',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.02),
            # tf.keras.layers.LayerNormalization(epsilon=1e-5),
            tf.keras.layers.Dense(52, activation='swish',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.02),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs, xgb_probs):
        ann_probs = self.ann(inputs)
        ann_probs = tf.reshape(ann_probs, [-1, 1])
        xgb_probs = tf.reshape(xgb_probs[:,1], [-1, 1]) 
        combined = tf.concat([xgb_probs, ann_probs], axis=1)
        return self.meta_learner(combined)

def train_step(model, optimizer, x, y, xgb_probs):
    with tf.GradientTape() as tape:
        predictions = model(x, xgb_probs)
        loss = tf.keras.losses.binary_crossentropy(np.array([y]).T, predictions)
        loss += sum(model.losses)
        loss = tf.reduce_mean(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions

def train_model(model, xgb, X_train, y_train, X_val, y_val, epochs=120, batch_size=125):
    with tf.device("/GPU:0"):
        optimizer = tf.keras.optimizers.Adam(learning_rate=8e-5)
        train_loss = tf.keras.metrics.Mean()
        train_accuracy = tf.keras.metrics.BinaryAccuracy()
        train_precision  = tf.keras.metrics.Precision()
        train_recall = tf.keras.metrics.Recall()
        val_loss = tf.keras.metrics.Mean()
        val_accuracy = tf.keras.metrics.BinaryAccuracy()
        val_precision = tf.keras.metrics.Precision()
        val_recall = tf.keras.metrics.Recall()


        # Early stopping
        best_val_accuracy = 0
        patience = 5
        patience_counter = 0

        # History tracking
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'train_precision': [],
            'train_recall': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': []
        }
        val_true_labels = []  
        val_pred_labels = []  

        # Train XGBoost
        xgb.fit(X_train, y_train)
        # Prepare data for TensorFlow
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)


        for epoch in range(epochs):
            # print(f"\nEpoch {epoch+1}/{epochs}")
            # train_loss.reset_states()
            # train_accuracy.reset_states()
            progress_bar = tqdm(train_dataset, desc=rf'Epoch {epoch+1}/{epochs}')
            for step, (x_batch, y_batch) in enumerate(progress_bar):
                xgb_probs = xgb.predict_proba(x_batch.numpy())
                xgb_probs = tf.convert_to_tensor(xgb_probs, dtype=tf.float32)
                
                loss, predictions = train_step(model, optimizer, x_batch, y_batch, xgb_probs)
                train_loss.update_state(loss)
                train_accuracy.update_state(y_batch, predictions)
                train_precision.update_state(y_batch, predictions)
                train_recall.update_state(y_batch, predictions)
                # Display results in progress bar
                progress_bar.set_postfix(
                    loss=train_loss.result().numpy(), 
                    train_accuracy=train_accuracy.result().numpy(), 
                    train_precision=train_precision.result().numpy(),
                    train_recall=train_recall.result().numpy()
                )
            # Validation loop
            # val_loss.reset_states()
            # val_accuracy.reset_states()

            for x_batch, y_batch in val_dataset:
                xgb_probs = xgb.predict_proba(x_batch.numpy())
                xgb_probs = tf.convert_to_tensor(xgb_probs, dtype=tf.float32)
                val_predictions = model(x_batch, xgb_probs)
                val_predictions = np.squeeze(val_predictions)
                
                batch_val_loss = tf.keras.losses.binary_crossentropy(y_batch, val_predictions)
                val_true_labels.extend(y_batch.numpy())  # True labels
                val_pred_labels.extend(val_predictions)
                
                val_loss.update_state(batch_val_loss)
                val_accuracy.update_state(y_batch, val_predictions)
                val_precision.update_state(y_batch, val_predictions)
                val_recall.update_state(y_batch, val_predictions)

            # Log metrics to history
            history['train_loss'].append(train_loss.result().numpy())
            history['train_accuracy'].append(train_accuracy.result().numpy())
            history['train_precision'].append(train_precision.result().numpy())
            history['train_recall'].append(train_recall.result().numpy())
            history['val_loss'].append(val_loss.result().numpy())
            history['val_accuracy'].append(val_accuracy.result().numpy())
            history['val_precision'].append(val_precision.result().numpy())
            history['val_recall'].append(val_recall.result().numpy())

            print(
                f'Epoch {epoch+1} - '
                f'Loss: {train_loss.result():.4f}, '
                f'Accuracy: {train_accuracy.result():.4f}, '
                f'Val Loss: {val_loss.result():.4f}, '
                f'Val Accuracy: {val_accuracy.result():.4f}'
            )
                # XGBoost only performance
            xgb_pred = xgb.predict(X_test)
            xgb_accuracy = accuracy_score(y_test, xgb_pred)

            print(f"XGBoost only accuracy: {xgb_accuracy:.4f}")
            
            # Early stopping
            current_val_accuracy = val_accuracy.result()
            if current_val_accuracy >= best_val_accuracy:
                best_val_accuracy = current_val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
        return history, val_true_labels, val_pred_labels
                
            # accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_batch, predictions)
            # print(f"Epoch {epoch}: loss = {loss.numpy().mean():.4f}, accuracy = {accuracy.numpy().mean():.4f}")
            # print(f'Epoch {epoch+1}, Loss: {epoch_loss_avg.result().numpy()}, '
            # f'Train Accuracy: {train_accuracy.result().numpy() * 100:.2f}%, ')
def evaluate_model(model, xgb, X_test, y_test):

    xgb_probs = xgb.predict_proba(X_test)
    xgb_probs = tf.convert_to_tensor(xgb_probs, dtype=tf.float32)
    predictions = model(X_test, xgb_probs)
    accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_test, predictions)
    print(f"Test accuracy: {accuracy.numpy().mean():.4f}")

if __name__ == "__main__":
    data_path = "/Users/ayushbhakat/Desktop/sem-5/ML/miniproject/telecom_churn.csv"
    data = preprocess_data(data_path)
    X_temp, X_test, y_temp, y_test = load_and_preprocess_data(data)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    
    num_classes = len(np.unique(y_train))
    print(num_classes)
    input_shape = X_train.shape[1:]
    print(input_shape,num_classes)

    xgb_params = {
        'objective': 'binary:logistic',
       # 'num_class': num_classes,
        'learning_rate': 0.3,
        'max_depth': 8,
        'n_estimators': 220
    }
    model = IntegratedModel(input_shape, num_classes)
    model.summary()
    xgb = XGBClassifier(**xgb_params)
    history = train_model(model, xgb, X_train, y_train,X_val,y_val)
    evaluate_model(model, xgb, X_test, y_test)
    import pickle 
    import os
    history_file = os.path.join("/Users/ayushbhakat/Desktop/sem-5/ML/miniproject",'history',"MLModelStats.pkl")
    model.save(os.path.join("/Users/ayushbhakat/Desktop/sem-5/ML/miniproject",'Model',"Model1.h5"))
    # model.save(os.path.join("/Users/ayushbhakat/Desktop/sem-5/ML/miniproject",'Model',"Model1"))
    # print(history)
    with open(history_file, 'wb') as f:
        pickle.dump(history,f)

# import tensorflow as tf
# import pandas as pd
# import numpy as np
# from xgboost import XGBClassifier
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, roc_auc_score
# from tqdm import tqdm

# class IntegratedModel(tf.keras.Model):
#     def __init__(self, input_shape):
#         super().__init__()
#         # Modified ANN architecture for binary classification with L2 regularization
#         self.ann = tf.keras.Sequential([
#             tf.keras.layers.Input(shape=input_shape),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#             tf.keras.layers.Dropout(0.3),
#             tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#             tf.keras.layers.Dropout(0.1),
#             tf.keras.layers.Dense(1, activation='sigmoid')  # Changed to sigmoid for binary classification
#         ])
#         # Modified meta-learner for binary classification with L2 regularization
#         self.meta_learner = tf.keras.Sequential([
#             tf.keras.layers.Input(shape=(2,)),  # Changed input shape for binary case
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#             tf.keras.layers.Dropout(0.3),
#             tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Dense(1, activation='sigmoid')  # Changed to sigmoid for binary classification
#         ])

#     @tf.function
#     def call(self, inputs, xgb_probs):
#         ann_probs = self.ann(inputs)
#         # Reshape probabilities for concatenation
#         ann_probs = tf.reshape(ann_probs, [-1, 1])
#         xgb_probs = tf.reshape(xgb_probs[:, 1], [-1, 1])  # Take only positive class probability
#         combined = tf.concat([xgb_probs, ann_probs], axis=1)
#         return self.meta_learner(combined)

# def train_step(model, optimizer, x, y, xgb_probs):
#     with tf.GradientTape() as tape:
#         predictions = model(x, xgb_probs)
#         loss = tf.keras.losses.binary_crossentropy(np.array([y]).T, predictions)
#         loss += sum(model.losses)  # Add regularization losses
#         loss = tf.reduce_mean(loss)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss, predictions

# def train_model(model, xgb, X_train, y_train, X_val, y_val, epochs=15, batch_size=32):
#     with tf.device("/GPU:0"):
#         optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        
#         # Training metrics
#         train_loss = tf.keras.metrics.Mean()
#         train_accuracy = tf.keras.metrics.BinaryAccuracy()
        
#         # Validation metrics
#         val_loss = tf.keras.metrics.Mean()
#         val_accuracy = tf.keras.metrics.BinaryAccuracy()
        
#         # Early stopping
#         best_val_accuracy = 0
#         patience = 10
#         patience_counter = 0
        
#         # Train XGBoost
#         xgb.fit(X_train, y_train)
        
#         # Prepare datasets
#         train_dataset = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
#                          .shuffle(1000).batch(batch_size))
        
#         val_dataset = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
#                        .batch(batch_size))
        
#         for epoch in range(epochs):
#             # Training loop
#             # train_loss.reset_states()
#             # train_accuracy.reset_states()
            
#             progress_bar = tqdm(train_dataset, desc=f'Epoch {epoch+1}/{epochs}')
            
#             for x_batch, y_batch in progress_bar:
#                 xgb_probs = xgb.predict_proba(x_batch.numpy())
#                 xgb_probs = tf.convert_to_tensor(xgb_probs, dtype=tf.float32)
#                 loss, predictions = train_step(model, optimizer, x_batch, y_batch, xgb_probs)
#                 train_loss.update_state(loss)
#                 train_accuracy.update_state(y_batch, predictions)
#                 progress_bar.set_postfix(
#                     loss=f'{train_loss.result():.4f}',
#                     accuracy=f'{train_accuracy.result():.4f}'
#                 )
            
#             # Validation loop
#             # val_loss.reset_states()
#             # val_accuracy.reset_states()
            
#             for x_batch, y_batch in val_dataset:
#                 xgb_probs = xgb.predict_proba(x_batch.numpy())
#                 xgb_probs = tf.convert_to_tensor(xgb_probs, dtype=tf.float32)
#                 val_predictions = model(x_batch, xgb_probs)
#                 val_predictions = np.squeeze(val_predictions)
#                 batch_val_loss = tf.reduce_mean(
#                     tf.keras.losses.binary_crossentropy(y_batch, val_predictions))
                
#                 val_loss.update_state(batch_val_loss)
#                 val_accuracy.update_state(y_batch, val_predictions)

#             print(
#                 f'Epoch {epoch+1} - '
#                 f'Loss: {train_loss.result():.4f}, '
#                 f'Accuracy: {train_accuracy.result():.4f}, '
#                 f'Val Loss: {val_loss.result():.4f}, '
#                 f'Val Accuracy: {val_accuracy.result():.4f}'
#             )
            
#             # Early stopping logic
#             current_val_accuracy = val_accuracy.result()
            
#             if current_val_accuracy > best_val_accuracy:
#                 best_val_accuracy = current_val_accuracy
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
            
#             if patience_counter >= patience:
#                 print(f'Early stopping triggered after {epoch+1} epochs')
#                 break

# def evaluate_model(model, xgb, X_test, y_test):
#     xgb_probs = xgb.predict_proba(X_test)
#     xgb_probs = tf.convert_to_tensor(xgb_probs, dtype=tf.float32)
#     predictions = model(X_test, xgb_probs)

#     # Calculate metrics
#     accuracy = tf.keras.metrics.binary_accuracy(y_test, predictions)
#     # auc = roc_auc_score(y_test.numpy(), predictions.numpy())
    
#     print(f"Test accuracy: {accuracy.numpy().mean():.4f}")
#     # print(f"Test AUC-ROC: {auc:.4f}")

#     # XGBoost only performance
#     xgb_pred = xgb.predict(X_test)
#     xgb_accuracy = accuracy_score(y_test.numpy(), xgb_pred)

#     print(f"XGBoost only accuracy: {xgb_accuracy:.4f}")

# if __name__ == "__main__":
#     # Your data loading code here...
#     data_path = "/Users/ayushbhakat/Desktop/sem-5/ML/miniproject/telecom_churn.csv"
#     data = preprocess_data(data_path)
#     X_temp, X_test, y_temp, y_test = load_and_preprocess_data(data)
#     X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    
#     num_classes = len(np.unique(y_train))
#     print(num_classes)
#     input_shape = X_train.shape[1:]
#     print(input_shape,num_classes)

    
#     # # Split data into train, validation and test sets
#     # X_temp, X_test, y_temp, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2,
#     #                                                   random_state=42)
    
#     # X_train, X_val, y_train, y_val = train_test_split(X_temp.numpy(), y_temp.numpy(),
#     #                                                   test_size=0.2,
#     #                                                   random_state=42)

#     input_shape = X_train.shape[1:]
    
#     xgb_params = {
#       'objective': 'binary:logistic',  # Changed for binary classification
#       'learning_rate': 0.1,
#       'max_depth': 6,
#       'n_estimators': 100,
#       'eval_metric': 'auc'
#     }
    
#     model = IntegratedModel(input_shape)
    
#     xgb_model = XGBClassifier(**xgb_params)

#     train_model(model=model,
#                xgb=xgb_model,
#                X_train=X_train,
#                y_train=y_train,
#                X_val=X_val,
#                y_val=y_val)

#     evaluate_model(model=model,
#                    xgb=xgb_model,
#                    X_test=X_test,
#                    y_test=y_test)
