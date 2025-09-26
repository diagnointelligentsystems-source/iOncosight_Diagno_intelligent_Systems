    ########################## segmentation model
def seg_code(img_p,yolov11,image_path,predicted_proba_DL,predicted_value,sel_ens_M1, sel_ens_M2, sel_ens_M3, scaled_ens_M1, scaled_ens_M2, scaled_ens_M3, ens_MCN):
    import os
    import sys
    import cv2
    import torch
    import pandas as pd
    from PIL import Image
    import traceback
    import psutil
    import shutil
    output_path = "./images_YOLOV11/V11_input.png"
    try: #if 1==1:#predicted_value[0]!=1:
        print('ex 1_1', flush=True)
        # Load best model
        # Class mapping
        class_names = {0: "Mass", 1: "COPD", 2: "Normal"}
        # Fixed colors (BGR for OpenCV)
        class_colors = {
            "Mass": (0, 0, 255),       # Red
            "COPD": (0, 165, 255),     # Orange
            "Normal": (0, 255, 0)      # Green
        }
        
        # Transparency factor
        alpha = 0.4
        results=[]
        # Load best model
        cwd = os.getcwd()
        #print("Current working directory:", cwd)
        # Set directory
        #from ultralytics import YOLO
        model = yolov11 #YOLO(yolov11)#"./yolov11_seg_MCN_best.pt")
        # Input/output
        ############################
        output_path = "./images_YOLOV11/V11_input.png"
        # Make sure the output folder exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Copy image
        shutil.copy(image_path, output_path)
        print('ex 1_2', flush=True)
        output_path = "./output_YOLOV11/V11_SEG_PRED.png"
        print('ex 1_2_1', flush=True)
        # Run inference
        img_samp = cv2.imread(image_path)
        print('image_path :',image_path, flush=True)
        print("img_samp shape:", img_samp.shape, flush=True)
        # -------------------- Feature extraction hook --------------------
        features_dict = {}
        # -------------------- Inference function --------------------
        # -------------------- Main inference wrapper --------------------
        torch.set_num_threads(2)

        # ---------- Load image ----------
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"❌ Image not found: {image_path}")

        # Ensure RGB (YOLO expects 3 channels)
        if len(img.shape) != 3 or img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(f"✅ orig img shape: {img.shape}", flush=True)

        # ---------- Setup hook for feature extraction ----------
        features_dict = {}

        def hook_fn(module, input, output):
            pooled = torch.mean(output[0], dim=(1, 2))  # Global Average Pooling
            features_dict['feat'] = pooled.detach().cpu().numpy()

        try:
            hook = model.model.model[10].register_forward_hook(hook_fn)
            print("✅ Hook registered", flush=True)
        except Exception as e:
            print(f"❌ Failed to register hook: {e}", flush=True)
            hook = None

        # ---------- Run inference ----------
        results = model(image_path, conf=0.2, iou=0.5, imgsz=512, device="cpu")
        #print("Raw results:", results, flush=True)
        print("Results length:", len(results), flush=True)

        result = None
        if results and len(results) > 0:
            result = results[0]
            boxes = getattr(result, "boxes", None)
            masks = getattr(result, "masks", None)

            if boxes is not None and len(boxes.xyxy) > 0:
                print("✅ Detections found", flush=True)
                #print("Boxes (x1, y1, x2, y2, conf, class_id):", flush=True)
                #print(boxes.xyxy, flush=True)
                class_ids = boxes.cls
                print("Predicted classes:", [result.names[int(c)] for c in class_ids], flush=True)
            else:
                print("⚠️ No boxes found", flush=True)

            if masks is not None:
                print("Masks available", flush=True)
            else:
                print("No masks (detection-only model)", flush=True)
        else:
            print("⚠️ No detections returned", flush=True)

        # ---------- Feature extraction ----------
        feat = features_dict.get('feat', None)
        ens_ML_MCN_output = 60
        conf_ML = predicted_proba_DL  # comes from earlier context

        if feat is not None:
            print(f"✅ Feature shape: {feat.shape}", flush=True)

            row = [os.path.basename(image_path), 10] + feat.tolist()
            df = pd.DataFrame([row], columns=['filename', 'label'] + [f'feat_{i}' for i in range(len(feat))])
            df.to_csv('./yolov11_MCN_whole_features_test.csv', index=False)
            print("✅ Features saved to CSV", flush=True)

            import ens_modelling_MCN_test_fn
            ens_ML_MCN_output, predicted_proba1 = ens_modelling_MCN_test_fn.ens_ML_MCN(
                sel_ens_M1, sel_ens_M2, sel_ens_M3, scaled_ens_M1, scaled_ens_M2, scaled_ens_M3, ens_MCN
            )

            predicted_proba1 = predicted_proba1[0]
            conf_ML = predicted_proba1[ens_ML_MCN_output] * 100

            cv2.imwrite(output_path, img_p)
            print("ex 3", flush=True)
        else:
            print("⚠️ Feature was not extracted.", flush=True)

        # ---------- Cleanup hook ----------
        if hook is not None:
            hook.remove()
            print("ℹ️ Hook removed", flush=True) 
        print('ex 2', flush=True)
        ######### ML results

        ################3 changing label confidence score
        # -------- Step 1: Apply segmentation masks (without darkening background) --------
        class_names = {0: "Mass", 1: "COPD", 2: "Normal"}

        if result.masks is not None and ens_ML_MCN_output < 2 and (masks is not None):  # ✅ safe check
            # -------- Step 1: Overlay masks --------
            for mask, cls_id in zip(result.masks.data, result.boxes.cls):
                cls_id_scalar = int(cls_id.item() if torch.is_tensor(cls_id) else cls_id) 
                cls_name = class_names[cls_id_scalar]
                color = class_colors[cls_name]

                mask = mask.cpu().numpy().astype(np.uint8)
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

                # Extract the region of interest (ROI) where mask==1
                roi = img[mask == 1]

                # Create same-shape color array
                color_arr = np.full_like(roi, color, dtype=np.uint8)

                # Blend only masked region
                blended = cv2.addWeighted(roi, 1 - alpha, color_arr, alpha, 0)

                # Put back blended pixels
                img[mask == 1] = blended

            copd_p = 0
            # -------- Step 2: Draw bounding boxes + labels (with white background) --------
            for box, cls_id, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                cls_id_scalar = int(cls_id.item() if torch.is_tensor(cls_id) else cls_id)
                cls_name = class_names[cls_id_scalar]
                color = class_colors[cls_name]

                print("cls_id_scalar", cls_id_scalar, flush=True)

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Build label with confidence
                conf = conf_ML  # ⚠️ you overwrite here — intentional?
                label = f"{cls_name} {conf:.0f}%"
                label_w = cls_name

                # Get text size
                (font_w, font_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

                # White background rectangle
                cv2.rectangle(
                    img,
                    (x1, y1 - font_h - baseline),
                    (x1 + font_w, y1),
                    (255, 255, 255), -1
                )

                # Text on top of white background
                if ens_ML_MCN_output == 2:
                    cv2.putText(img, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 0, 0), 2)
                else:
                    if label_w == "COPD":
                        copd_p = 1
                    cv2.putText(img, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                color, 2)

        # -------- Step 3: Save and show result --------
        if ens_ML_MCN_output == 0 or ens_ML_MCN_output == 1:  #
            cv2.imwrite(output_path, img)
        else:
            cv2.imwrite(output_path, img_p)
        #print(f"Saved to {output_path}")

        # Show with matplotlib (correct colors)
##        plt.figure(figsize=(7, 7))
##        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
##        plt.title("Detected regions")
##        plt.axis("off")
##        plt.tight_layout()
        #plt.show()

        ############## segmented region
        print('ex 4')
        if 1==1:#ens_ML_MCN_output==0: #
            # ----------------------------
            # CLASS NAMES
            # ----------------------------
            class_names = {0: "Mass", 1: "COPD", 2:"Normal"}


            # ==================== helpers ====================

            def feret_diameters_from_contour(cnt):
                if cnt.ndim == 3 and cnt.shape[1] == 1:
                    cnt = cnt[:, 0, :]
                cnt = cnt.astype(np.float32)

                area = float(cv2.contourArea(cnt))
                if len(cnt) < 3:
                    return dict(area=area, major_len=0.0, minor_len=0.0,
                                major_p1=(0, 0), major_p2=(0, 0), major_angle_deg=0.0,
                                minor_p1=(0, 0), minor_p2=(0, 0), minor_angle_deg=0.0)

                hull = cv2.convexHull(cnt)
                if hull.ndim == 3:
                    hull = hull[:, 0, :]
                P = hull.astype(np.float32)
                M = len(P)

                # --- Max Feret ---
                if M > 600:
                    step = int(M / 600) + 1
                    P_major = P[::step]
                else:
                    P_major = P

                A = P_major[:, None, :]
                B = P_major[None, :, :]
                diff = A - B
                D2 = (diff ** 2).sum(-1)
                i, j = np.unravel_index(np.argmax(D2), D2.shape)
                p1 = tuple(P_major[i].astype(float))
                p2 = tuple(P_major[j].astype(float))
                major_len = float(np.sqrt(D2[i, j]))
                major_angle_deg = float((np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) + 180) % 180)

                # --- Min Feret (rotating calipers) ---
                rect = cv2.minAreaRect(P)
                (cx, cy), (w, h), angle = rect
                if w < h:
                    min_len = w
                    min_angle_deg = angle
                else:
                    min_len = h
                    min_angle_deg = angle + 90
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                dists = [np.linalg.norm(box[(k + 1) % 4] - box[k]) for k in range(4)]
                kmin = int(np.argmin(dists))
                q1 = tuple(box[kmin])
                q2 = tuple(box[(kmin + 1) % 4])

                return dict(
                    area=area,
                    major_len=major_len, major_p1=p1, major_p2=p2, major_angle_deg=major_angle_deg,
                    minor_len=float(min_len), minor_p1=q1, minor_p2=q2, minor_angle_deg=min_angle_deg
                )

            print('ex 5', flush=True)
            # ==================== main ====================

            def process_segmentation(image_path, results,predicted_value):
                copd_p=0
                orig_img = np.array(Image.open(image_path).convert("RGB"))
                H, W = orig_img.shape[:2]

                if results[0].masks is None:
                    print("No segmentation detected.", flush=True)
                    return copd_p

                masks = results[0].masks.data.cpu().numpy()
                boxes = results[0].boxes
                class_ids = boxes.cls.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()

                region_rows = []
                overlay = orig_img.copy()
                red = np.array([255, 0, 0], dtype=np.uint8)
                alpha = 0.35

                os.makedirs("./output_poly_feret", exist_ok=True)

                for idx, m in enumerate(masks, start=0):  # keep index aligned with boxes
                    cls_id233 = class_ids[idx]
                    label ='none'
                    conf=0
                    # 🚨 skip anything that is not class 0
                    #print('cls_id233',cls_id233)
                    if cls_id233 > 1:
                        continue
                    m_resized = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)  ########### resize to original
                    mask_bin = (m_resized > 0.5).astype(np.uint8)
                    if mask_bin.sum() == 0:
                        continue

                    m_idx = mask_bin.astype(bool)
                    if cls_id233 == 0 and predicted_value[0]!=1:
                        overlay[m_idx] = (alpha * np.array([255, 0, 0]) + (1 - alpha) * overlay[m_idx]).astype(np.uint8)
                    cnts, _ = cv2.findContours((mask_bin * 255).astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # ✅ Case 2: bounding box with label (class 1)
                    if cls_id233 == 1:
                        # get box coords and clamp to image
                        box = boxes.xyxy[idx].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
                        x1, y1, x2, y2 = box
                        x1 = max(0, int(x1))
                        y1 = max(0, int(y1))
                        x2 = min(W - 1, int(x2))
                        y2 = min(H - 1, int(y2))
                        if x2 <= x1 or y2 <= y1:
                            continue

                        # percentage confidence (use your confidences array)
                        #conf_ML = confidences[idx] * 100  # convert to percent if conf is 0..1
                        label_p = f"COPD ({conf_ML:.0f})%"
                        label = f"COPD"
                        if label == f"COPD":
                            if copd_p==0:
                                copd_p=1
                        # colors (BGR)
                        sandal = (255, 204, 102)  # sandal yellow (RGB)
                        dark_orange = (255, 255, 255)  # dark orange (RGB)
                        bbox_border_color = (255, 140, 0)

                        # ---------- 1) Fill the bbox with semi-transparent sandal ----------
                        alpha_fill = 0.35
                        sub = overlay[y1:y2, x1:x2].copy()
                        if sub.size != 0:
                            sandal_rect = np.full(sub.shape, sandal, dtype=np.uint8)
                            cv2.addWeighted(sandal_rect, alpha_fill, sub, 1 - alpha_fill, 0, sub)
                            overlay[y1:y2, x1:x2] = sub

                        # optional: draw bbox border (keeps a visible outline)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), bbox_border_color, 2)

                        # ---------- 2) Draw an opaque label bar on top INSIDE the bbox ----------
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        text_thickness = 2
                        padding = 6

                        # auto-shrink font if label is wider than bbox
                        (text_w, text_h), baseline = cv2.getTextSize(label_p, font, font_scale, text_thickness)
                        box_width = x2 - x1
                        while (text_w + 2 * padding) > box_width and font_scale > 0.25:
                            font_scale -= 0.05
                            (text_w, text_h), baseline = cv2.getTextSize(label_p, font, font_scale, text_thickness)

                        # label rectangle coordinates (inside top of bbox)
                        label_x1 = x1
                        label_x2 = x1 + text_w + 2 * padding
                        if label_x2 > x2:
                            label_x2 = x2
                        label_y1 = y1
                        label_y2 = y1 + text_h + 2 * padding + baseline

                        # clamp vertical coords
                        if label_y2 > H:
                            label_y2 = H
                            label_y1 = max(0, label_y2 - (text_h + 2 * padding + baseline))

                        # draw opaque sandal label bar
                        cv2.rectangle(overlay,
                                      (int(label_x1), int(label_y1)),
                                      (int(label_x2), int(label_y2)),
                                      sandal,
                                      thickness=-1)

                        # ---------- 3) Put the text inside the label bar (dark orange) ----------
                        text_org = (int(label_x1 + padding), int(label_y2 - baseline - padding))
                        cv2.putText(overlay, label_p, text_org, font, font_scale, dark_orange, text_thickness, cv2.LINE_AA)

                    if not cnts:
                        continue
                    cnt = max(cnts, key=cv2.contourArea)

                    # # Approx polygon
                    # epsilon = 0.01 * cv2.arcLength(cnt, True)
                    # approx = cv2.approxPolyDP(cnt, epsilon, True)
                    # print('len(approx)',len(approx))
                    # if len(approx) < 10:
                    #     print(f"Skipping region {idx + 1}: not a polygon")
                    #     continue
                    if cls_id233 == 0 and predicted_value[0]!=1:
                        stats = feret_diameters_from_contour(cnt)

                        # --- Class + Confidence ---
                        cls_id = class_ids[idx]
                        conf = confidences[idx]
                        label = class_names.get(cls_id, str(cls_id))

                        # --- Draw major axis line ---
                        p1 = tuple(map(int, stats['major_p1']))
                        p2 = tuple(map(int, stats['major_p2']))
                        cv2.line(overlay, p1, p2, (0, 255, 0), 2)

                    # --- Label ---
                    def draw_text_with_background(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
                                                  font_scale=0.6, text_color=(255, 0, 0),  # Red text
                                                  bg_color=(255, 0, 255), thickness=2, padding=4):  # Yellow background

                        # Get text size
                        (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                        # Coordinates for background rectangle
                        x, y = org
                        cv2.rectangle(img, (x, y - h - baseline - padding),
                                      (x + w + padding * 2, y + baseline + padding),
                                      bg_color, -1)

                        # Put text over the background
                        cv2.putText(img, text, (x + padding, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

                        return img
                    if cls_id233 <2:
                        x, y, w, h = cv2.boundingRect(cnt)
                        # --- Place text near region instead of top-left ---
                        # Conversion factors from your resized DICOM (640x640)
                        px_to_cm = 0.034365    # 0.0531  # cm per pixel
                        px_to_cm2 = 0.034365*0.033193  #0.00114 #0.00292  # cm² per pixel
                        if cls_id233 ==0 and predicted_value[0]!=1:
                            txt = [
                                f"{label} ({conf_ML:.0f}%)",
                                f"Area: {stats['area'] * px_to_cm2:.2f} cm2",
                                f"Length: {stats['major_len'] * px_to_cm:.2f} cm",
                                # f"MinFeret: {stats['minor_len'] * px_to_cm:.2f} cm"
                            ]
                            # Start y above the bounding box (or inside if too close to top)
                            y0 = max(y - 10, 20)
                            dy = 30
                            print("overlay Image shape:", overlay.shape, flush=True)
                            for i, t in enumerate(txt):
                                yy = y0 + i * dy
                                if i == 0:  # label + confidence → red text on yellow background
                                    draw_text_with_background(
                                        overlay, t, (x + w, yy),
                                        text_color=(255, 0, 0),  # red
                                        bg_color=(255, 255, 0)  # yellow
                                    )
                                else:  # white plain text for stats
                                    cv2.putText(
                                        overlay, t, (x + w, yy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 255, 255), 2, cv2.LINE_AA
                                    )
                    # --- Save row ---
                    # Conversion factors (for your resized 640x640 case)
                    #0.034365×0.033193 cm 1024*1024
                    px_to_mm_x, px_to_mm_y = 0.34365 ,0.33193       #0.550, 0.531  (640*640)
                    px_to_cm_x, px_to_cm_y = px_to_mm_x / 10, px_to_mm_y / 10
                    px_to_mm2 = px_to_mm_x * px_to_mm_y
                    px_to_cm2 = px_to_mm2 / 100

                    region_rows.append({
                        "Region": idx + 1,
                        "Class": label,
                        #"Confidence": float(conf),
                        #"Area_px2": stats['area'],
                        #"Area_cm2": stats['area'] * px_to_cm2,
                        #"MaxFeret_px": stats['major_len'],
                        #"MaxFeret_cm": stats['major_len'] * ((px_to_cm_x + px_to_cm_y) / 2),  # avg if spacing not square
                        #"MinFeret_px": stats['minor_len'],
                        #"MinFeret_cm": stats['minor_len'] * ((px_to_cm_x + px_to_cm_y) / 2),
                        #"MaxFeret_angle_deg": stats['major_angle_deg'],
                        #"MinFeret_angle_deg": stats['minor_angle_deg'],
                    })
                plt.figure(figsize=(7, 7))
                plt.imshow(overlay)
                plt.title("Overlay + Polygon Feret Diameters + Class")
                plt.axis("off")
                plt.tight_layout()
                #plt.show()

                #cv2.imwrite("./output_poly_feret/overlay_with_class.png",
                            #cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                cv2.imwrite("./output_YOLOV11/V11_SEG_PRED.png",  cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                df = pd.DataFrame(region_rows)
                df.to_csv(os.path.join(current_dir, "output_poly_feret", "region_stats_with_class.csv"), index=False)
                #print(df)
                return copd_p

            copd_p=process_segmentation(image_path, results,predicted_value)
            print('ex 6', flush=True)
    #else:
     #   print('predicted output: Non-Lung Cancer')
    #print('classfier_output',predicted_value[0])
    except Exception as e:
        print("⚠️ An error occurred in process_segmentation:", flush=True)
        traceback.print_exc()   # prints full traceback with line number
        raise  # re-raise so the real error propagates
    #############3
    # Extract all class values
    #df = pd.DataFrame(region_rows)
    #df.to_csv("./output_poly_feret/region_stats_with_class.csv", index=False)
    #print(df)
    has_mass = has_copd = False
    if results[0].masks is not None and os.path.exists(os.path.join(current_dir, "output_poly_feret", "region_stats_with_class.csv")):
        from pandas.errors import EmptyDataError

        csv_path = os.path.join(current_dir, "output_poly_feret", "region_stats_with_class.csv")

        df = None  # initialize
        has_mass = has_copd = False
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                print("CSV loaded successfully:", csv_path, flush=True)
            except EmptyDataError:
                print("CSV file exists but is empty:", csv_path, flush=True)
        else:
            print("CSV file does not exist:", csv_path, flush=True)

        # ✅ Only run this block if df was successfully loaded
        if df is not None and not df.empty:
            # Check if Class column exists before accessing
            if "Class" in df.columns:
                has_mass = (df["Class"] == "Mass").any()
                has_copd = (df["Class"] == "COPD").any()
            else:
                has_mass = has_copd = False
                print("⚠️ 'Class' column not found in CSV", flush=True)

            if predicted_value[0] == 0:
                max_confidence_ML = predicted_proba_DL
                imp_result = "Lung Cancer"
                if has_mass and (has_copd or copd_p == 1):
                    imp_result = "Lung Cancer + Mass + COPD"
                    max_confidence_ML = conf_ML
                elif has_mass:
                    imp_result = "Lung Cancer + Mass"
                    max_confidence_ML = conf_ML
                elif has_copd or (copd_p == 1):
                    imp_result = "Lung Cancer + COPD"
                    max_confidence_ML = conf_ML
            else:
                imp_result = "Non-Lung Cancer"
                max_confidence_ML = predicted_proba_DL
                if has_copd:
                    imp_result = "COPD (High Risk for Lung Cancer)"
                    max_confidence_ML = conf_ML
        else:
            # fallback if df missing/empty
            if predicted_value[0] == 0:
                imp_result = 'Lung Cancer'
                max_confidence_ML = predicted_proba_DL
                shutil.copy("./output_YOLOV11/Grad_cam_PRED.png", "./result.jpg")
                imp_image_out = "./result.jpg"
            else:
                imp_result = 'Non-Lung Cancer'
                max_confidence_ML = predicted_proba_DL
                shutil.copy("./output_YOLOV11/Grad_cam_PRED.png", "./result.jpg")
                imp_image_out = "./result.jpg"
        print('ex 7')
        if imp_result != 'Non-Lung Cancer':
            imp_image_out2 = "./output_YOLOV11/Grad_cam_PRED.png"
            imp_image_out1 = "./output_YOLOV11/V11_SEG_PRED.png"
            imp_image_out = "./result.jpg"  # output_YOLOV11/concat_PRED.png"
            ### concat image
            # Read images
            img1 = cv2.imread(imp_image_out1)
            img2 = cv2.imread(imp_image_out2)

            # Make sure both images have the same height
            if img1.shape[0] != img2.shape[0]:
                # Resize second image height same as first
                img2 = cv2.resize(img2, (int(img2.shape[1] * img1.shape[0] / img2.shape[0]), img1.shape[0]))

            # Resize final image to width = 1024, keep aspect ratio
            # h, w = img2.shape[:2]
            # new_w = 1024
            # new_h = int(h * (new_w / w))
            # img1 = cv2.resize(img1,  (new_w, new_h))
            # Concatenate horizontally
            concat_img_resized = np.vstack((img2, img1))

            # Save output
            cv2.imwrite(imp_image_out, concat_img_resized)

    else:
        if predicted_value[0]==0:
            imp_result='Lung Cancer'
            max_confidence_ML = predicted_proba_DL
            imp_image_out = "./result.jpg"#"./output_YOLOV11/Grad_cam_PRED.png"
        else:
            if copd_p==1:
                max_confidence_ML = conf_ML
                imp_result = " COPD (High Risk for Lung Cancer)"
                imp_image_out2 = "./output_YOLOV11/Grad_cam_PRED.png"
                imp_image_out1 = "./output_YOLOV11/V11_SEG_PRED.png"
                imp_image_out = "./result.jpg"#output_YOLOV11/concat_PRED.png"
                ### concat image
                # Read images
                img1 = cv2.imread(imp_image_out1)
                img2 = cv2.imread(imp_image_out2)

                # Make sure both images have the same height
                if img1.shape[0] != img2.shape[0]:
                    # Resize second image height same as first
                    img2 = cv2.resize(img2, (int(img2.shape[1] * img1.shape[0] / img2.shape[0]), img1.shape[0]))

                # Resize final image to width = 1024, keep aspect ratio
                #h, w = img2.shape[:2]
                #new_w = 1024
                #new_h = int(h * (new_w / w))
                #img1 = cv2.resize(img1,  (new_w, new_h))
                # Concatenate horizontally
                concat_img_resized= np.vstack((img2, img1))

                # Save output
                cv2.imwrite(imp_image_out, concat_img_resized)
                #imp_result=='COPD'

            else:
                imp_result = 'Non-Lung Cancer'
                max_confidence_ML = predicted_proba_DL
                imp_image_out="./result.jpg"#"./output_YOLOV11/Grad_cam_PRED.png"
    print('ex 8', flush=True)
    if imp_result=='Lung Cancer':
      shutil.copy("./output_YOLOV11/Grad_cam_PRED.png", "./result.jpg")
    plt.close('all')

    return imp_result, max_confidence_ML



