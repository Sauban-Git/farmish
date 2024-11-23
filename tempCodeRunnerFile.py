cropn = label_encoder.inverse_transform(prediction.astype(int))
# print(cropn)

# best_crop_index = prediction.argmax()
# best_crop = label_encoder.inverse_transform([best_crop_index])