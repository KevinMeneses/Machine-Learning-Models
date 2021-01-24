import numpy as np


def print_stats(mc):
    FP = mc.sum(axis=0) - np.diag(mc)
    FN = mc.sum(axis=1) - np.diag(mc)
    TP = np.diag(mc)
    TN = mc.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    print("TPR", TPR.sum()/19)
    print("TNR", TNR.sum()/19)
    print("PPV", PPV.sum()/19)
    print("NPV", NPV.sum()/19)
    print("FPR", FPR.sum()/19)
    print("FNR", FNR.sum()/19)
    print("FDR", FDR.sum()/19)
    print("ACC", ACC.sum()/19)
