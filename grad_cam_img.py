def grad_Cam_1(image_path,eff_model)
    if 1==1:
        img = img_p = cv2.imread(image_path)
      
        ###
        # ---- Grad-CAM Function ----
        def get_gradcam(model, img_array, last_conv_layer_name, pred_index=None):
            grad_model = tf.keras.models.Model(
                [model.inputs],
                [model.get_layer(last_conv_layer_name).output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]

            grads = tape.gradient(class_channel, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]

            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            return heatmap.numpy()


        # ---- Preprocess Input Image ----
            # ---- Overlay Grad-CAM on Original Image ----
        def overlay_gradcam(img, heatmap, alpha=0.4, cmap=cv2.COLORMAP_JET):
            heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cmap)
            img = np.array(img)
            superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
            return superimposed_img


        # ---- Run Grad-CAM ----
        #img_path = "E:/project_new/Project_MCN_code/sample_images/test_image.png"  # change this
        def preprocess_image_g(img_path, target_size=(300, 300)):
            img = keras.preprocessing.image.load_img(img_path, target_size=target_size)
            img_array = keras.preprocessing.image.img_to_array(img)
            
            # Ensure 3 channels
            if img_array.shape[-1] == 1:
                img_array = np.repeat(img_array, 3, axis=-1)
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Normalize and convert to float32
            img_array = img_array / 255.0
            img_array = img_array.astype(np.float32)
            return img, img_array


        img1, img_array = preprocess_image_g(img_path, target_size=(300, 300))
        # Find last conv layer name (EfficientNetB3 usually: "top_conv")
        last_conv_layer_name = "top_conv"

        heatmap = get_gradcam(eff_model, img_array, last_conv_layer_name)
        superimposed_img = overlay_gradcam(img1, heatmap)
        cv2.imwrite("./output_YOLOV11/Grad_cam_PRED.png", superimposed_img)
        #################
        # Copy image
    shutil.copy("./output_YOLOV11/Grad_cam_PRED.png", "./result.jpg")
