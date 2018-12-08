""""
UNIVERSITY OF ST ANDREWS
CS5014 - MACHINE LEARNING

PRACTICAL P2 - CLASSIFICATION OF OBJECT COLOUR USING OPTICAL SPECTROSCOPY DATA

STUDENT ID: 170027939

@ Main.py
-Computes Binary Classification
-Computes Multiclass Classification

"""


import Src.BinaryClassification as bc
import Src.MulticlassClassification as mc
import numpy as np

# Compute Binary Classifications
print("\n\r IN BINARY")
bc._compute_binary_classifications()


# Compute Multiclass Classifications
print("\n\r IN MULTI-CLASS")
mc._compute_multiclass_classifications()




