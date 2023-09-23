import pandas as pd
import matplotlib.pyplot as plt

def data_viz(idx, df):
    #Read image and text pairs
    current_row = df.iloc[idx]
    image_1 = plt.imread(current_row["image_1_path"])
    image_2 = plt.imread(current_row["image_2_path"])
    text_1 = current_row["text_1"]
    text_2 = current_row["text_2"]
    label = current_row["label"]
    #Plot the pairs
    plt.subplot(1, 2, 1)
    plt.imshow(image_1)
    plt.axis("off")
    plt.title("Image One")
    plt.subplot(1, 2, 2)
    plt.imshow(image_1)
    plt.axis("off")
    plt.title("Image Two")
    plt.show()
    #Print the corresponding text
    print(f"Text one: {text_1}")
    print(f"Text two: {text_2}")
    print(f"Label: {label}")