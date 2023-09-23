import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

#Embedding projections
def project_embeddings(
    embeddings, num_projection_layers, projection_dims, dropout_rate
):
    projected_embeddings = keras.layers.Dense(units=projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = tf.nn.gelu(projected_embeddings)
        x = keras.layers.Dense(projection_dims)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Add()([projected_embeddings, x])
        projected_embeddings = keras.layers.LayerNormalization()(x)
    return projected_embeddings

#Create image feature extractor
def create_vision_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    #Import pre-trained RESNET
    resnet_v2 = keras.applications.ResNet50V2(include_top=False, weights="imagenet", pooling="avg")
    for layer in resnet_v2.layers:
        layer.trainable = trainable
    #Model input
    image_1 = keras.Input(shape=(128, 128, 3), name="image_1")
    image_2 = keras.Input(shape=(128, 128, 3), name="image_2")
    preprocessed_1 = keras.applications.resnet_v2.preprocess_input(image_1)
    preprocessed_2 = keras.applications.resnet_v2.preprocess_input(image_2)
    #Image feature extraction
    embeddings_1 = resnet_v2(preprocessed_1)
    embeddings_2 = resnet_v2(preprocessed_2)
    embeddings = keras.layers.Concatenate()([embeddings_1, embeddings_2])
    #Model output
    outputs = project_embeddings(embeddings, num_projection_layers, projection_dims, dropout_rate)
    #Construct model
    return keras.Model([image_1, image_2], outputs, name="vision_encoder")

#Text feature extraction model
def create_text_encoder(
    num_projection_layers, projection_dims, dropout_rate,bert_model_path, trainable=False
):
    #Pre-trained BERT Model
    bert = hub.KerasLayer(bert_model_path, name="bert",)
    bert.trainable = trainable
    #Model input
    bert_input_features = ["input_type_ids", "input_mask", "input_word_ids"]
    inputs = {
        feature: keras.Input(shape=(128,), dtype=tf.int32, name=feature)
        for feature in bert_input_features}
    #Word embeddings
    embeddings = bert(inputs)["pooled_output"]
    outputs = project_embeddings(embeddings, num_projection_layers, projection_dims, dropout_rate)
    #Model construction
    return keras.Model(inputs, outputs, name="text_encoder")


#Create final model
def multimodal_model(bert_model_path,
    num_projection_layers=1,
    projection_dims=256,
    dropout_rate=0.1,
    vision_trainable=False,
    text_trainable=False,
):
    #Input images
    image_1 = keras.Input(shape=(128, 128, 3), name="image_1")
    image_2 = keras.Input(shape=(128, 128, 3), name="image_2")

    # Input text
    bert_input_features = ["input_type_ids", "input_mask", "input_word_ids"]
    text_inputs = {
        feature: keras.Input(shape=(128,), dtype=tf.int32, name=feature)
        for feature in bert_input_features}

    #Image and text feature extraction
    vision_encoder = create_vision_encoder(num_projection_layers, projection_dims, dropout_rate, vision_trainable)
    text_encoder = create_text_encoder(num_projection_layers, projection_dims, dropout_rate, bert_model_path, text_trainable)
    #Embedding projections.
    vision_projections = vision_encoder([image_1, image_2])
    text_projections = text_encoder(text_inputs)
    #Feature fusion
    concatenated = keras.layers.Concatenate()([vision_projections, text_projections])
    #Output layer
    outputs = keras.layers.Dense(3, activation="softmax")(concatenated)
    return keras.Model([image_1, image_2, text_inputs], outputs)