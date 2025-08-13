# ProMEP
Zero-shot prediction of mutation effects on protein function with multimodal deep representation learning
<<<<<<< HEAD

## Quick Start 
As a prerequisite, you must have  [SE(3)-Transformers](https://github.com/NVIDIA/DeepLearningExamples/tree/master/DGLPyTorch/DrugDiscovery/SE3Transformer) installed to use this repository.

Dependences
```python
conda install --yes --file requirements.txt
```

## Usage 
Generate per-residue representations
```python
python inference.py --task ec --outfile embeddings.h5
```

Calculate log-ratio heuristic under the constraints of both sequence and structure

```python
python inference_dms.py --task ec --outfile fitness_prediction.h5
```

Zero-shot prediction of mutation effects
```python
python predict_mutation_effects.py testdata/fitness_prediction.h5 
```

## ProMEP-guided protein engineering
To guide protein engineering, users need to generate the 'fitness_prediction.h5' file following the above instructions and provide the raw sequence. Then run 1_dms_scanning.py to:
1) generate the virtual single-point saturation mutagenesis library
2) calculate fitness scores for all mutants
3) and rank them accordingly

# examples
```python
cd examples
python 1_dms_scanning.py
```
Protein mutants sorted by fitness score will be stored in 'dms_data/scanning-cas9.csv', while the fitness score for each mutant will be recorded in 'score_data/cas9-score.csv'.

=======
>>>>>>> origin/main
