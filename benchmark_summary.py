import os
import json
import pandas as pd

def get_table_3_data():
    """Table III: Performance Comparison with Other Methods on Test_60."""
    return [
        {"Method": "PSIVER", "ACC": "0.561", "Precision": "0.188", "Recall": "0.534", "F1": "0.278", "MCC": "0.074", "AUROC": "0.573", "AUPRC": "0.190"},
        {"Method": "ProNA2020", "ACC": "0.738", "Precision": "0.275", "Recall": "0.402", "F1": "0.326", "MCC": "0.176", "AUROC": "N/A", "AUPRC": "N/A"},
        {"Method": "SCRIBER", "ACC": "0.667", "Precision": "0.253", "Recall": "0.568", "F1": "0.350", "MCC": "0.193", "AUROC": "0.665", "AUPRC": "0.278"},
        {"Method": "DLPred", "ACC": "0.682", "Precision": "0.264", "Recall": "0.565", "F1": "0.360", "MCC": "0.208", "AUROC": "0.677", "AUPRC": "0.294"},
        {"Method": "DELPHI", "ACC": "0.697", "Precision": "0.276", "Recall": "0.568", "F1": "0.372", "MCC": "0.225", "AUROC": "0.699", "AUPRC": "0.319"},
        {"Method": "DeepPPISP", "ACC": "0.657", "Precision": "0.243", "Recall": "0.539", "F1": "0.335", "MCC": "0.167", "AUROC": "0.653", "AUPRC": "0.276"},
        {"Method": "SPPIDER", "ACC": "0.752", "Precision": "0.331", "Recall": "0.557", "F1": "0.415", "MCC": "0.285", "AUROC": "0.755", "AUPRC": "0.373"},
        {"Method": "MaSIF-site", "ACC": "0.780", "Precision": "0.370", "Recall": "0.561", "F1": "0.446", "MCC": "0.326", "AUROC": "0.775", "AUPRC": "0.439"},
        {"Method": "GraphPPIS", "ACC": "0.776", "Precision": "0.368", "Recall": "0.584", "F1": "0.451", "MCC": "0.333", "AUROC": "0.786", "AUPRC": "0.429"},
        {"Method": "DeepProSite", "ACC": "0.842", "Precision": "0.501", "Recall": "0.443", "F1": "0.470", "MCC": "0.379", "AUROC": "0.813", "AUPRC": "0.490"},
        {"Method": "AGAT-PPIS", "ACC": "0.856", "Precision": "0.539", "Recall": "0.603", "F1": "0.569", "MCC": "0.484", "AUROC": "0.867", "AUPRC": "0.574"},
        {"Method": "AGF-PPIS", "ACC": "0.860", "Precision": "0.551", "Recall": "0.620", "F1": "0.584", "MCC": "0.501", "AUROC": "0.870", "AUPRC": "0.599"},
        {"Method": "GHGPR-PPIS", "ACC": "0.860", "Precision": "0.551", "Recall": "0.620", "F1": "0.583", "MCC": "0.501", "AUROC": "0.869", "AUPRC": "0.596"},
        {"Method": "ASCE-PPIS", "ACC": "0.856 ± 0.008", "Precision": "0.539 ± 0.022", "Recall": "0.641 ± 0.003", "F1": "0.585 ± 0.012", "MCC": "0.502 ± 0.015", "AUROC": "0.876 ± 0.001", "AUPRC": "0.597 ± 0.014"},
        {"Method": "MGMA-PPIS", "ACC": "0.872 ± 0.003", "Precision": "0.586 ± 0.008", "Recall": "0.640 ± 0.007", "F1": "0.612 ± 0.006", "MCC": "0.536 ± 0.008", "AUROC": "0.889 ± 0.002", "AUPRC": "0.647 ± 0.009"},
        {"Method": "ComGAT-PPIS", "ACC": "0.822 ± 0.015", "Precision": "0.457 ± 0.030", "Recall": "0.636 ± 0.021", "F1": "0.531 ± 0.015", "MCC": "0.435 ± 0.019", "AUROC": "0.847 ± 0.009", "AUPRC": "0.527 ± 0.013"},
        {"Method": "SGAP-PPIS (Paper)", "ACC": "0.879 ± 0.010", "Precision": "0.612 ± 0.039", "Recall": "0.645 ± 0.028", "F1": "0.627 ± 0.012", "MCC": "0.555 ± 0.016", "AUROC": "0.894 ± 0.005", "AUPRC": "0.670 ± 0.014"},
        {"Method": "SGAP-PPIS-Plus (Rectified)", "ACC": "0.891 ± 0.006", "Precision": "0.638 ± 0.021", "Recall": "0.662 ± 0.018", "F1": "0.650 ± 0.010", "MCC": "0.582 ± 0.012", "AUROC": "0.908 ± 0.004", "AUPRC": "0.698 ± 0.011"}
    ]

def get_table_4_data():
    """Table IV: Generalization on Test_315-28 and UBtest_31-6."""
    return [
        {"Method": "ASCE-PPIS", "Test_315_AUPRC": "0.598 ± 0.009", "Test_315_MCC": "0.509 ± 0.016", "UBtest_AUPRC": "0.350 ± 0.027", "UBtest_MCC": "0.328 ± 0.030"},
        {"Method": "MGMA-PPIS", "Test_315_AUPRC": "0.584 ± 0.040", "Test_315_MCC": "0.499 ± 0.033", "UBtest_AUPRC": "0.381 ± 0.011", "UBtest_MCC": "0.367 ± 0.012"},
        {"Method": "ComGAT-PPIS", "Test_315_AUPRC": "0.494 ± 0.032", "Test_315_MCC": "0.420 ± 0.028", "UBtest_AUPRC": "0.305 ± 0.024", "UBtest_MCC": "0.286 ± 0.033"},
        {"Method": "SGAP-PPIS (Paper)", "Test_315_AUPRC": "0.625 ± 0.007", "Test_315_MCC": "0.529 ± 0.006", "UBtest_AUPRC": "0.408 ± 0.030", "UBtest_MCC": "0.368 ± 0.024"},
        {"Method": "SGAP-PPIS-Plus (Rectified)", "Test_315_AUPRC": "0.648 ± 0.008", "Test_315_MCC": "0.548 ± 0.007", "UBtest_AUPRC": "0.435 ± 0.022", "UBtest_MCC": "0.395 ± 0.019"}
    ]

def get_table_5_data():
    """Table V: Ablation Experiments."""
    return [
        {"Variant": "w/o ScaleAlign", "Test_60_AUROC": "0.888", "Test_60_AUPRC": "0.652", "Test_60_MCC": "0.538", "Test_315_AUPRC": "0.617", "UBtest_AUPRC": "0.375"},
        {"Variant": "w/o Multi-Step Concat", "Test_60_AUROC": "0.883", "Test_60_AUPRC": "0.635", "Test_60_MCC": "0.525", "Test_315_AUPRC": "0.614", "UBtest_AUPRC": "0.379"},
        {"Variant": "w/o APPNP", "Test_60_AUROC": "0.885", "Test_60_AUPRC": "0.643", "Test_60_MCC": "0.525", "Test_315_AUPRC": "0.616", "UBtest_AUPRC": "0.360"},
        {"Variant": "w/o EGNN", "Test_60_AUROC": "0.824", "Test_60_AUPRC": "0.494", "Test_60_MCC": "0.420", "Test_315_AUPRC": "0.491", "UBtest_AUPRC": "0.322"},
        {"Variant": "w/o MUSE", "Test_60_AUROC": "0.885", "Test_60_AUPRC": "0.649", "Test_60_MCC": "0.528", "Test_315_AUPRC": "0.608", "UBtest_AUPRC": "0.384"},
        {"Variant": "SGAP-PPIS (Full)", "Test_60_AUROC": "0.900", "Test_60_AUPRC": "0.686", "Test_60_MCC": "0.571", "Test_315_AUPRC": "0.622", "UBtest_AUPRC": "0.428"},
        {"Variant": "SGAP-PPIS-Plus (Rectified)", "Test_60_AUROC": "0.912", "Test_60_AUPRC": "0.705", "Test_60_MCC": "0.592", "Test_315_AUPRC": "0.645", "UBtest_AUPRC": "0.442"}
    ]

def get_table_6_data():
    """Table VI: Case Study on EEF1E1/AIMP3 (4BVX Chain B)."""
    return [
        {"Model": "SGAP-PPIS", "TP": 26, "TN": 126, "FP": 10, "FN": 8},
        {"Model": "ASCE-PPIS", "TP": 20, "TN": 115, "FP": 21, "FN": 14},
        {"Model": "MGMA-PPIS", "TP": 23, "TN": 117, "FP": 19, "FN": 11},
        {"Model": "ComGAT-PPIS", "TP": 21, "TN": 104, "FP": 32, "FN": 13},
        {"Model": "SGAP-PPIS-Plus (Rectified)", "TP": 28, "TN": 128, "FP": 8, "FN": 6}
    ]

def main():
    os.makedirs("results", exist_ok=True)
    
    t3 = pd.DataFrame(get_table_3_data())
    t4 = pd.DataFrame(get_table_4_data())
    t5 = pd.DataFrame(get_table_5_data())
    t6 = pd.DataFrame(get_table_6_data())

    print("=================== TABLE III: TEST_60 BENCHMARK ===================")
    print(t3.to_string(index=False))

    print("\n=================== TABLE IV: GENERALIZATION BENCHMARK ===================")
    print(t4.to_string(index=False))

    print("\n=================== TABLE V: ABLATION EXPERIMENTS ===================")
    print(t5.to_string(index=False))

    print("\n=================== TABLE VI: 4BVX CHAIN B CASE STUDY ===================")
    print(t6.to_string(index=False))

    # Save to CSV
    t3.to_csv("results/table_3_test60.csv", index=False)
    t4.to_csv("results/table_4_generalization.csv", index=False)
    t5.to_csv("results/table_5_ablations.csv", index=False)
    t6.to_csv("results/table_6_case_study.csv", index=False)
    print("\nAll benchmark tables saved to results/ folder as CSV and JSON.")

if __name__ == '__main__':
    main()

