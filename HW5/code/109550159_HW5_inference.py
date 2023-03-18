import os
import csv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

for T in ["task1", "task2", "task3"]:
    f = open("log_demo_result.txt", "w")
    f.close()
    os.system(f"python demo.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --image_folder test2/{T} --saved_model ./saved_models/best_norm_ED.pth")
    f = open("./log_demo_result.txt")
    R = f.read()
    f.close()

    write_list = []
    if not os.path.exists('submission.csv'):
        write_list.append(["filename", "label"])

    for row in R.split('\n'):
        if row.startswith("test2"):
            filename, result, _ = row.split("\t")
            filename = (filename.split("/")[1]).replace("\\","/")
            print(result)
            result = result.rstrip()
            write_list.append((filename, result))

    with open('submission.csv', 'a', newline = "") as csvfile:
        writer = csv.writer(csvfile)
        for filename, ans in write_list:
            writer.writerow([filename, ans])