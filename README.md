## SGCL-DTI:
Supervised Co-contrastive Graph Learning for Drug-Target Interaction Prediction
### Quick start
We provide an example script to run experiments on our dataset: 

- Run `./SGCL-DTI/main.py`: predict drug-target interactions. 

### All process
 -Run `./main.py`   You can run the entire model


### Code and data

#### 
- `CLaugmentdti.py`: data augment for graph contrastive learning
- `modeltestdtiseed.py`: SGCL model
- `utilsdeiseed.py`: tool kit
- `main.py`: use the dataset to run SGCL-DTI 
- `GCNLayer.py`: a GCL layers 

#### data sample `data/heter` directory
- `drug.txt`: list of drug names
- `protein.txt`: list of protein names
- `disease.txt`: list of disease names
- `se.txt`: list of side effect names
- `drug_dict_map`: a complete ID mapping between drug names and DrugBank ID
- `protein_dict_map`: a complete ID mapping between protein names and UniProt ID
- `mat_drug_se.txt` 		: Drug-SideEffect association matrix
- `mat_protein_protein.txt` : Protein-Protein interaction matrix
- `mat_protein_drug.txt` 	: Protein-Drug interaction matrix
- `mat_drug_protein.txt`: Drug_Protein interaction matrix
- `mat_drug_drug.txt` 		: Drug-Drug interaction matrix
- `mat_protein_disease.txt` : Protein-Disease association matrix
- `mat_drug_disease.txt` 	: Drug-Disease association matrix
- `Similarity_Matrix_Drugs.txt` 	: Drug similarity scores based on chemical structures of drugs
- `Similarity_Matrix_Proteins.txt` 	: Protein similarity scores based on primary sequences of proteins

<h3>
<red>If you use our code, please cite our paper at the same time. Thank you!!!</red>

Yang Li, Guanyu Qiao, Xin Gao, Guohua Wang, Supervised graph co-contrastive learning for drug–target interaction prediction, Bioinformatics, Volume 38, Issue 10, 15 May 2022, Pages 2847–2854, https://doi.org/10.1093/bioinformatics/btac164
</h3>
