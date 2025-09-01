# TrustCL
This repository contains the code of our ICDM'2025 paper Trusted Collective Learning for Conflictive Multi-View Decision-Making.

# 1.  Virtual Environment
Input the following code to create the virtual environment TrustCL. <br>
```markdown
conda env create -f environment.yml
```

# 2. Project Directory
```markdown
.
├── data       
│   └── dcdb.csv  
├── environment.yaml 
├── data.py                  
├── main.py                                    
├── models.py                  
└── loss_function.py                   
```
# 3. Train and predict
```markdown
python main.py
```
# Dataset
BCDB-2K is a real-world multi-disciplinary consultation dataset, recording the physical indicators of patients with breast cancer and 2000 instances of doctors’ decision data.
