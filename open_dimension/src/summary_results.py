import pandas as pd
import glob
import os

#  calculate mean ± std
def mean_std(series):
    return f"{series.mean():.4f} ± {series.std():.4f}"

# hanle the 
def process_file(filepath):
    df = pd.read_csv(filepath)
    rows = []
    
    # row 2-6
    group1 = df.iloc[0:5]
    # print(group1)
    row1 = {
        "Dataset": group1["Dataset"].iloc[0],
        "Algorithm": group1["Algorithm"].iloc[0],
        "ContainerShape": group1["ContainerShape"].iloc[0],
        "BinSizeFactor": group1["BinSizeFactor"].iloc[0],
        "Evaluation": group1["Evaluation"].iloc[0],
        "NestingStrategy": group1["NestingStrategy"].iloc[0],
        "Selection_range": group1["Selection_range"].iloc[0],
        "Time": mean_std(group1["Time"]),
        "Best_BinsUsed": mean_std(group1["Best_BinsUsed"]),
        "Best_U": mean_std(group1["Best_U"]),
        "Best_U_star": mean_std(group1["Best_U_star"]),
        "Ori_BinsUsed": mean_std(group1["Ori_BinsUsed"]),
        "Ori_U": mean_std(group1["Ori_U"]),
        "Ori_U_star": mean_std(group1["Ori_U_star"])
    }
    rows.append(row1)
    
    best_u = group1["Best_U"].mean()
    ori_u = group1["Ori_U"].mean()
    u_improve = (best_u - ori_u) / ori_u * 100

    best_ustar = group1["Best_U_star"].mean()
    ori_ustar = group1["Ori_U_star"].mean()
    ustar_improve = (best_ustar - ori_ustar) / ori_ustar * 100
    row1["U_improvement (%)"] = f"{u_improve:.4f}"
    row1["U_star_improvement (%)"] = f"{ustar_improve:.4f}"

    # row 7-11
    group2 = df.iloc[5:11]
    # print(group2)
    row2 = {
        "Dataset": group2["Dataset"].iloc[0],
        "Algorithm": group2["Algorithm"].iloc[0],
        "ContainerShape": group2["ContainerShape"].iloc[0],
        "BinSizeFactor": group2["BinSizeFactor"].iloc[0],
        "Evaluation": group2["Evaluation"].iloc[0],
        "NestingStrategy": group2["NestingStrategy"].iloc[0],
        "Selection_range": group2["Selection_range"].iloc[0],
        "Time": mean_std(group2["Time"]),
        "Best_BinsUsed": mean_std(group2["Best_BinsUsed"]),
        "Best_U": mean_std(group2["Best_U"]),
        "Best_U_star": mean_std(group2["Best_U_star"]),
        "Ori_BinsUsed": mean_std(group2["Ori_BinsUsed"]),
        "Ori_U": mean_std(group2["Ori_U"]),
        "Ori_U_star": mean_std(group2["Ori_U_star"])
    }
    rows.append(row2)
    
    best_u = group2["Best_U"].mean()
    ori_u = group2["Ori_U"].mean()
    u_improve = (best_u - ori_u) / ori_u * 100

    best_ustar = group2["Best_U_star"].mean()
    ori_ustar = group2["Ori_U_star"].mean()
    ustar_improve = (best_ustar - ori_ustar) / ori_ustar * 100
    row2["U_improvement (%)"] = f"{u_improve:.4f}"
    row2["U_star_improvement (%)"] = f"{ustar_improve:.4f}"

    return rows


#list_datasets_name = ["chess"]

list_datasets_name = ["chess","engine","liu","Merged1_normal","Merged2_normal","Merged3_normal",
                        "Merged4_normal","Merged5_normal", "St_05_Example3_normal",
                        "st04_Example2_normal","st04_Example3_normal","st04_Example4_normal","st04_Example5_normal",
                        "st05_e2"]

folder_path = "D:/Carlos_project/3D_packing_voxel_ILS/result/numerical" 

csv_files = []
for each_name in list_datasets_name:
    file_path = os.path.join(folder_path, f"{each_name}_results.csv")
    if os.path.exists(file_path):
        csv_files.append(file_path)
    else:
        print(f"⚠️ Warning: 文件未找到 {file_path}")
        
# for each_name in list_datasets_name:
#     csv_files = glob.glob(os.path.join(folder_path, f"{each_name}_results.csv"))

all_rows = []
for file in csv_files:
    all_rows.extend(process_file(file))

summary_df = pd.DataFrame(all_rows)
summary_df.to_csv("all_summary_formatted.csv", index=False, encoding="utf-8-sig")
print("Read finished!!")
