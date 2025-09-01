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
BCDB-2K is a subset extracted from a real-world multi-disciplinary consultation dataset, recording the physical indicators of patients with breast cancer and 2000 instances of doctors’ decision data. We performed data de-identification on this dataset, which digitized the content of the dataset, to ensure that patient privacy is not compromised. There are 6 views in total, with View_1 to View_6 representing the decisions of six different views (i.e., doctors), respectively, and Final_view being the final decision after Multi-Disciplinary medical discussion. Attribute_0 to Attribute_45 are the attribute values of these patients. The complete BCDB contains 9015 instances. If you want to obtain the complete dataset and follow our work, please contact zhu\_nj@shu.edu.cn and cite this paper.
