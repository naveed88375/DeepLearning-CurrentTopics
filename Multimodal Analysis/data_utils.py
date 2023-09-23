import tensorflow as tf

def prepare_datasets(batch_size, bert_preprocess_model, train_df, val_df, test_df):

  #Convert pandas entries into tensorflow dataset
  def dataframe_to_dataset(dataframe):
      columns = ["image_1_path", "image_2_path", "text_1", "text_2", "label_idx"]
      dataframe = dataframe[columns].copy()
      labels = dataframe.pop("label_idx")
      ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
      ds = ds.shuffle(buffer_size=len(dataframe))
      return ds

  #Image size
  resize = (128, 128)
  #Bert features
  bert_input_features = ["input_word_ids", "input_type_ids", "input_mask"]

  #preprocess images
  def preprocess_image(image_path):
      extension = tf.strings.split(image_path)[-1]
      image = tf.io.read_file(image_path)
      if extension == b"jpg":
          image = tf.image.decode_jpeg(image, 3)
      else:
          image = tf.image.decode_png(image, 3)
      image = tf.image.resize(image, resize)
      return image

  #process text using bert preprocessor
  def preprocess_text(text_1, text_2):
      text_1 = tf.convert_to_tensor([text_1])
      text_2 = tf.convert_to_tensor([text_2])
      output = bert_preprocess_model([text_1, text_2])
      output = {feature: tf.squeeze(output[feature]) for feature in bert_input_features}
      return output

  #Processed images and texts
  def preprocess_text_and_image(sample):
      image_1 = preprocess_image(sample["image_1_path"])
      image_2 = preprocess_image(sample["image_2_path"])
      text = preprocess_text(sample["text_1"], sample["text_2"])
      return {"image_1": image_1, "image_2": image_2, "text": text}

  #Dataset creation
  def prepare_dataset(dataframe, training=True):
      ds = dataframe_to_dataset(dataframe)
      if training:
          ds = ds.shuffle(len(dataframe))
      ds = ds.map(lambda x, y: (preprocess_text_and_image(x), y)).cache()
      ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
      return ds

  #Train, validation and test datasets
  train_ds = prepare_dataset(train_df)
  validation_ds = prepare_dataset(val_df, training=False)
  test_ds = prepare_dataset(test_df, training=False)

  return train_ds, validation_ds, test_ds