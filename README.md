# MLHD25
Bone Age Prediction from Hand X-Rays
![image](https://github.com/user-attachments/assets/8f942124-d684-4152-b039-34009ff20c42)




Ideas:
        
    1. Multi-Input CNN
        process gender as a one-hot encoded vector
        Concatenate CNN features with the gender vector 
        
    OR
  
    1. Dual-Stream Gender-Aware Network: one branch for males, one for females (better)
        Train with shared weights for early layers, but allow separate feature extraction at later stages.
        Introduce a gender classifier as an auxiliary task to ensure gender-based feature separation.
        Fusion at the final regression layer.

        
    2. Temporal regression via RNNs
         bone growth as a time-series-like problem by extracting sequential features (via CNN)
         Pass the features through an LSTM or GRU 
         Predict bone age as a sequence-to-sequence output
         e.g., predict small increments leading to the final age

         
    3. Graph-Based (hard)
         bone keypoint detection (nodes = keypoints, edges = anatomical connections)
         pass through a GCN for feature extraction
         finally use a CNN


    4. Vision Transformer w patches (very hard)
        divide the x-ray into anatomically meaningful patches instead of simple square patches (how?)
        use bone-specific positional embeddings ( growth plate areas with different weights)
        feed patch embeddings into a ViT
        combine the ViT output with a CNN-based backbone for final reg


    5. CNN with attention?
